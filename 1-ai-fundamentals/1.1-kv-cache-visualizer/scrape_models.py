#!/usr/bin/env python3
"""
scrape_models.py — scrape LLM model info and auto-generate KV cache presets.

Two modes:
  1. Ollama scrapes https://ollama.com/search (HTMX-based) for model listings.
  2. Hugging Face fetches config.json from a HF repo and maps it to ModelConfig.

Primary use:  python scrape_models.py preset --model Llama-3.1-8B
               python scrape_models.py preset --ollama --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "https://ollama.com/search"
REQUEST_DELAY = 0.5  # seconds between pagination requests
REQUEST_TIMEOUT = 15

# HTMX headers — Ollama's search is an HTMX-driven page and only returns model
# cards when the client sends these.
HTMX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/125.0.0.0 Safari/537.36",
    "HX-Request": "true",
    "HX-Target": "search",
    "Accept": "text/html,*/*",
}

# Known Ollama → HuggingFace repo mappings.
# Keys are lowercase Ollama names (without tag/version suffixes like :latest).
# Values are HF repo IDs.
# Mapping: canonical preset name → HF repo ID, plus Ollama name aliases.
# Both preset names (e.g. "Llama-3.1-8B") and Ollama names (e.g. "llama3.1:8b")
# resolve through this dict.
OLLAMA_TO_HF: Dict[str, str] = {
    # --- Meta / Llama ---
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.1:70b": "meta-llama/Llama-3.1-70B",
    "llama-3.1-405b": "meta-llama/Llama-3.1-405B",
    "llama3.1:405b": "meta-llama/Llama-3.1-405B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B",
    # --- Mistral ---
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral:8x7b": "mistralai/Mixtral-8x7B-v0.1",
    # --- Qwen ---
    "qwen-2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    "qwen-2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen-2.5-32b": "Qwen/Qwen2.5-32B",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B",
    "qwen2.5:72b": "Qwen/Qwen2.5-72B",
    # --- DeepSeek ---
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # --- Gemma ---
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma2:9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
    "gemma2:27b": "google/gemma-2-27b",
    # --- Phi ---
    "phi-3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-medium-128k": "microsoft/Phi-3-medium-128k-instruct",
    "phi3:medium": "microsoft/Phi-3-medium-128k-instruct",
    # --- CodeLlama ---
    "codellama-34b": "codellama/CodeLlama-34b-hf",
    "codellama:34b": "codellama/CodeLlama-34b-hf",
    # --- Cohere ---
    "command-r-plus": "CohereForAI/c4ai-command-r-plus",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OllamaModel:
    """Surface-level model info from Ollama."""
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    pulls: str = ""
    tags_count: str = ""
    url: str = ""


@dataclass
class HFConfig:
    """Parsed HuggingFace config.json mapped to ModelConfig fields."""
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_ctx: int
    n_experts: int = 0
    n_active_experts: int = 0
    hf_repo: str = ""
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def to_preset_code(self, indent: int = 0) -> str:
        """Generate Python code to register this model as a preset."""
        pad = " " * indent
        expert_lines = ""
        if self.n_experts > 0:
            expert_lines = f"\n{pad}    n_experts={self.n_experts},"
            expert_lines += f"\n{pad}    n_active_experts={self.n_active_experts},"
        return (
            f"ModelPresets.register(ModelConfig(\n"
            f'{pad}    name="{self.name}",\n'
            f"{pad}    n_layers={self.n_layers},\n"
            f"{pad}    n_heads={self.n_heads},\n"
            f"{pad}    n_kv_heads={self.n_kv_heads},\n"
            f"{pad}    head_dim={self.head_dim},\n"
            f"{pad}    max_ctx={self.max_ctx},{expert_lines}\n"
            f"{pad}))"
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Ollama scraper
# ---------------------------------------------------------------------------

def fetch_ollama_page(url: str, page: int = 1) -> Optional[BeautifulSoup]:
    """Fetch one page of Ollama search results with HTMX headers."""
    params = {"page": page} if page > 1 else {}
    try:
        resp = requests.get(
            url, params=params, headers=HTMX_HEADERS, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        # Debug: check if we got actual model cards
        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.find_all("a", href=lambda h: h and h.startswith("/library/"))
        if not cards:
            print(f"  ⚠  Page {page}: no model cards found (got {len(resp.text)} bytes)",
                  file=sys.stderr)
        return soup
    except requests.RequestException as e:
        print(f"  ✗  Error fetching page {page}: {e}", file=sys.stderr)
        return None


def parse_ollama_models(soup: BeautifulSoup) -> List[OllamaModel]:
    """Parse model cards from an Ollama search page."""
    models: List[OllamaModel] = []
    cards = soup.find_all("a", href=lambda h: h and h.startswith("/library/"))

    for card in cards:
        # Model name from the <span x-test-search-response-title>
        name_span = card.find("span", {"x-test-search-response-title": True})
        if not name_span:
            # Fallback: h2 text
            h2 = card.find("h2")
            name = h2.get_text(strip=True) if h2 else "unknown"
        else:
            name = name_span.get_text(strip=True)

        # Description from <p class="max-w-lg break-words ...">
        desc_p = card.find("p", class_=lambda c: c and "break-words" in str(c))
        description = desc_p.get_text(strip=True) if desc_p else ""

        # Capability tags from <span x-test-capability>
        capability_spans = card.find_all("span", {"x-test-capability": True})
        capabilities = [s.get_text(strip=True) for s in capability_spans]

        # Pulls count — find the "Pulls" label span, then get preceding number
        pulls = ""
        for sp in card.find_all("span"):
            if sp.get_text(strip=True) == "Pulls":
                # The number is in the previous sibling span
                parent = sp.parent
                if parent:
                    num_span = parent.find("span", class_=lambda c: c is None or "hidden" not in str(c))
                    if num_span:
                        pulls = num_span.get_text(strip=True)
                break

        # Tags count — find the "Tags" label span
        tags_count = ""
        for sp in card.find_all("span"):
            if sp.get_text(strip=True) == "Tags":
                parent = sp.parent
                if parent:
                    num_span = parent.find("span", class_=lambda c: c is None or "hidden" not in str(c))
                    if num_span:
                        tags_count = num_span.get_text(strip=True)
                break

        href = card.get("href", "")
        url = f"https://ollama.com{href}" if href else ""

        models.append(OllamaModel(
            name=name,
            description=description,
            capabilities=capabilities,
            pulls=pulls,
            tags_count=tags_count,
            url=url,
        ))

    return models


def scrape_ollama(limit: int = 10) -> List[OllamaModel]:
    """Scrape ollama.com/search with pagination."""
    print(f"Scraping Ollama (limit={limit})...", file=sys.stderr)
    all_models: List[OllamaModel] = []
    page = 1

    while len(all_models) < limit:
        soup = fetch_ollama_page(OLLAMA_URL, page=page)
        if not soup:
            break
        models = parse_ollama_models(soup)
        if not models:
            print(f"  No more models found at page {page}.", file=sys.stderr)
            break
        all_models.extend(models)
        print(f"  Page {page}: {len(models)} models (total: {len(all_models)})",
              file=sys.stderr)
        if len(models) < 30:  # fewer than full page = last page
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return all_models[:limit]


# ---------------------------------------------------------------------------
# HuggingFace config.json fetcher
# ---------------------------------------------------------------------------

HF_RAW_URL = "https://huggingface.co/{repo}/raw/main/config.json"


def _hf_headers() -> Dict[str, str]:
    """Return headers including HF token if available."""
    headers = {"User-Agent": "ModelScraper/2.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_hf_config(hf_repo: str) -> Optional[Dict[str, Any]]:
    """Fetch config.json from a HuggingFace repo."""
    url = HF_RAW_URL.format(repo=hf_repo)
    try:
        resp = requests.get(url, headers=_hf_headers(), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  ✗  Failed to fetch {url}: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"  ✗  Invalid JSON from {url}: {e}", file=sys.stderr)
        return None


def _merge_search(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dicts and search in nested 'text_config' / 'llm_config' too.

    Many multi-modal models (Qwen3.5, LLaVA, etc.) nest LLM architecture params
    in a `text_config` or `llm_config` sub-object.  This helper searches each
    dict AND its nested sub-dicts.
    """
    merged: Dict[str, Any] = {}
    for d in dicts:
        if not d:
            continue
        merged.update(d)
        # Also pull up values from common nested sections
        for nested_key in ("text_config", "llm_config", "language_model",
                           "transformer_config", "model_config"):
            nested = d.get(nested_key)
            if isinstance(nested, dict):
                merged.update(nested)
    return merged


def parse_hf_config(hf_repo: str, raw: Dict[str, Any]) -> HFConfig:
    """Map a HuggingFace config.json to our ModelConfig schema.

    This handles the common Transformers config keys.  Different model
    families use slightly different key names, so we try multiple aliases.
    Also handles nested configs (text_config, llm_config) common in
    multi-modal models.
    """
    # Flatten nested config sections so we search both top-level and sub-objects
    cfg = _merge_search(raw)

    # Model name — try several common naming conventions
    name = raw.get("_name_or_path", hf_repo.split("/")[-1])

    # Core architecture params
    n_layers = _get_int(cfg, [
        "num_hidden_layers", "n_layer", "num_layers",
        "decoder_num_hidden_layers", "encoder_num_hidden_layers",
    ])
    n_heads = _get_int(cfg, [
        "num_attention_heads", "n_head", "num_heads",
        "decoder_attention_heads", "encoder_attention_heads",
    ])
    n_kv_heads = _get_int(cfg, [
        "num_key_value_heads", "n_kv_head", "multi_query_group_num",
        "num_attention_heads",  # fallback: MHA uses n_heads for KV too
    ])

    hidden_size = _get_int(cfg, [
        "hidden_size", "d_model", "n_embd", "dim", "n_embed",
    ])

    # head_dim might be explicit or derived
    head_dim = _get_int(cfg, ["head_dim"])
    if not head_dim and hidden_size and n_heads:
        head_dim = hidden_size // n_heads

    # Context length
    max_ctx = _get_int(cfg, [
        "max_position_embeddings", "max_sequence_length",
        "n_positions", "model_max_length", "max_seq_len",
        "seq_len", "sliding_window",
    ])

    # MoE params
    n_experts = _get_int(cfg, [
        "num_local_experts", "num_experts", "n_experts",
        "num_routed_experts",
    ])
    n_active = _get_int(cfg, [
        "num_experts_per_tok", "top_k", "num_experts_per_token",
        "activated_experts",
    ])

    return HFConfig(
        name=name,
        n_layers=n_layers or 0,
        n_heads=n_heads or 0,
        n_kv_heads=n_kv_heads or 0,
        head_dim=head_dim or 0,
        max_ctx=max_ctx or 0,
        n_experts=n_experts or 0,
        n_active_experts=n_active or 0,
        hf_repo=hf_repo,
        raw_config=raw,
    )


def _get_int(config: Dict[str, Any], keys: List[str]) -> Optional[int]:
    """Try multiple keys to find an integer value in a config dict."""
    for key in keys:
        val = config.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return None


def _hf_repo_exists(repo_id: str) -> bool:
    """Quickly check if a HF repo exists (HEAD request to config.json)."""
    url = HF_RAW_URL.format(repo=repo_id)
    try:
        r = requests.head(url, headers=_hf_headers(), timeout=5)
        return r.status_code in (200, 401)  # 401 = exists but gated
    except requests.RequestException:
        return False


def resolve_hf_repo(model_name: str) -> Optional[str]:
    """Resolve an Ollama model name to a HF repo ID.

    Tries direct lookup, then heuristic patterns.
    """
    # Direct lookup (case-insensitive)
    key = model_name.lower().strip()
    if key in OLLAMA_TO_HF:
        return OLLAMA_TO_HF[key]

    # Try stripping ":latest" or just the tag
    base = key.split(":")[0]
    if base in OLLAMA_TO_HF:
        return OLLAMA_TO_HF[base]

    # Heuristic: if it looks like "org/model" already, use it as-is
    if "/" in model_name and not model_name.startswith("/"):
        return model_name

    # Check if it starts with a well-known org prefix
    known_orgs = {
        "llama": "meta-llama",
        "mistral": "mistralai",
        "mixtral": "mistralai",
        "qwen": "Qwen",
        "deepseek": "deepseek-ai",
        "gemma": "google",
        "phi": "microsoft",
        "codellama": "codellama",
        "command": "CohereForAI",
        "falcon": "tiiuae",
        "bloom": "bigscience",
        "starcoder": "bigcode",
        "dbrx": "databricks",
        "olmo": "allenai",
    }
    # Try fuzzy matching against all known keys (strip hyphens/dots/colons)
    normalised_key = re.sub(r"[-.:]", "", key)
    for ol_key, hf_repo in OLLAMA_TO_HF.items():
        normalised_ol = re.sub(r"[-.:]", "", ol_key)
        if normalised_key == normalised_ol:
            return hf_repo
        # Also try matching just the prefix (model name without variant)
        if normalised_key.startswith(normalised_ol):
            return hf_repo

    # Check if it starts with a well-known org prefix
    for prefix, org in known_orgs.items():
        if key.startswith(prefix):
            parts = key.split(":")
            model_part = parts[0]
            # Try to find a matching entry in OLLAMA_TO_HF that starts with this prefix
            for ol_key, hf_repo in OLLAMA_TO_HF.items():
                if ol_key.startswith(prefix) and ol_key.split(":")[0] == model_part:
                    return hf_repo
            # Last-resort heuristic: try to construct the HF name
            model_suffix = model_part[len(prefix):].lstrip("-")
            if model_suffix:
                candidate = f"{org}/{prefix.capitalize()}{model_suffix}"
                if _hf_repo_exists(candidate):
                    return candidate

            candidate = f"{org}/{model_part.capitalize()}"
            if _hf_repo_exists(candidate):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Preset generation
# ---------------------------------------------------------------------------

def generate_preset_from_hf(
    model_name: str,
    hf_repo: Optional[str] = None,
) -> Optional[HFConfig]:
    """Generate a preset entry from a model name, resolving via HF."""
    if not hf_repo:
        hf_repo = resolve_hf_repo(model_name)
    if not hf_repo:
        print(f"  ✗  Could not resolve {model_name!r} to a HF repo. "
              f"Use --hf-repo to specify.", file=sys.stderr)
        return None

    print(f"  Fetching config from HF: {hf_repo} ...", file=sys.stderr)
    raw = fetch_hf_config(hf_repo)
    if not raw:
        return None

    config = parse_hf_config(hf_repo, raw)
    # Override name with our canonical name
    # If model_name looks like an HF path (contains /), strip the org prefix
    canonical = model_name
    if canonical and "/" in canonical:
        canonical = canonical.split("/")[-1]
    if canonical:
        config.name = canonical
    return config


def autodetect_model_name_from_ollama(ollama_name: str) -> str:
    """Convert an Ollama model name to a canonical preset name.

    E.g. "llama3.1:8b" -> "Llama-3.1-8B"
         "qwen2.5:32b" -> "Qwen-2.5-32B"
         "/library/laguna-xs-2.1" -> "Laguna-Xs-2.1"
    """
    # Remove leading path components (use removeprefix, NOT lstrip!)
    name = ollama_name.strip()
    if name.startswith("/library/"):
        name = name[len("/library/"):]

    # If it has a version tag like :8b, replace colon with hyphen
    name = re.sub(r':(\d+)([bkm])(?:\b|$)', r'-\1\2', name, flags=re.IGNORECASE)
    name = re.sub(r':', '-', name)

    # Known capitalisation overrides (lowercase -> canonical)
    cap_overrides = {
        "deepseek": "DeepSeek",
        "codellama": "CodeLlama",
    }
    # Capitalise each hyphen-separated segment
    # Skip pure numeric segments (like "2.5") — they stay as-is
    def _cap(seg: str) -> str:
        if not seg:
            return seg
        # If it's purely numeric or a version like "2.5", leave it
        if re.match(r'^[\d.]+$', seg):
            return seg
        seg_lower = seg.lower()
        if seg_lower in cap_overrides:
            return cap_overrides[seg_lower]
        return seg.capitalize()

    parts = name.replace("_", "-").split("-")
    formatted = "-".join(_cap(p) for p in parts)

    # Capitalise size suffixes (8b -> 8B, 70b -> 70B)
    formatted = re.sub(
        r'(\d+)([bkm])(?=\b|$|-)',
        lambda m: m.group(1) + m.group(2).upper(),
        formatted,
        flags=re.IGNORECASE
    )
    return formatted


# ---------------------------------------------------------------------------
# Append to presets.py
# ---------------------------------------------------------------------------

def append_to_presets_file(config: HFConfig, presets_path: Optional[str] = None):
    """Append a generated preset entry to the presets.py file."""
    if presets_path is None:
        presets_path = str(
            Path(__file__).parent / "calculator" / "presets.py"
        )

    code = config.to_preset_code()

    # Check if this model name already exists in the file
    presets_file = Path(presets_path)
    if presets_file.exists():
        content = presets_file.read_text()
        if f'name="{config.name}"' in content:
            print(f"  ⚠  Model {config.name!r} already exists in {presets_path} "
                  f"— skipping append.", file=sys.stderr)
            print(f"\nGenerated code (add manually if needed):\n{code}")
            return

    # Append before the last empty line or at end
    with open(presets_path, "a") as f:
        f.write("\n\n")
        f.write(code)
        f.write("\n")

    print(f"  ✓  Appended {config.name!r} to {presets_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def do_list(args: argparse.Namespace):
    """List models from Ollama."""
    models = scrape_ollama(limit=args.limit)
    if args.json:
        print(json.dumps([asdict(m) for m in models], indent=2))
    else:
        print(f"\n{'Model':<25} {'Capabilities':<25} {'Pulls':<10}")
        print("-" * 70)
        for m in models:
            caps = ",".join(m.capabilities) if m.capabilities else "-"
            print(f"{m.name:<25} {caps:<25} {m.pulls:<10}")


def do_preset(args: argparse.Namespace):
    """Generate/update presets.py entries."""
    models_to_process: List[Tuple[str, Optional[str]]] = []

    if args.ollama:
        # Fetch from Ollama then resolve each to HF
        ollama_models = scrape_ollama(limit=args.limit)
        models_to_process = [
            (autodetect_model_name_from_ollama(m.name), None)
            for m in ollama_models
        ]
    elif args.model:
        models_to_process = [(args.model, args.override_repo)]
    elif args.hf_repo:
        models_to_process = [(args.hf_repo.split("/")[-1], args.hf_repo)]
    else:
        print("Specify --model, --ollama, or --hf-repo.", file=sys.stderr)
        sys.exit(1)

    generated: List[HFConfig] = []
    for model_name, hf_repo in models_to_process:
        print(f"\nProcessing: {model_name}", file=sys.stderr)
        config = generate_preset_from_hf(model_name, hf_repo=hf_repo)
        if config:
            generated.append(config)
            print(f"  ✓  {config.name}: "
                  f"{config.n_layers}L, {config.n_heads}Q, "
                  f"{config.n_kv_heads}KV, {config.head_dim}d, "
                  f"ctx={config.max_ctx}",
                  file=sys.stderr)
            print(f"\n{config.to_preset_code()}")

    if not generated:
        print("No presets generated.", file=sys.stderr)
        sys.exit(1)

    if args.append is not None and len(generated) == 1:
        path = args.append if isinstance(args.append, str) else None
        append_to_presets_file(generated[0], presets_path=path)

    # --stats summary
    if args.stats:
        print(f"\n{'Model':<30} {'Layers':<8} {'Q Heads':<8} {'KV Heads':<8} "
              f"{'Head Dim':<9} {'Max Ctx':<10} {'MoE':<8} {'HF Repo'}")
        print("-" * 100)
        for cfg in generated:
            moe = f"{cfg.n_experts}x{cfg.n_active_experts}" if cfg.n_experts else ""
            print(f"{cfg.name:<30} {cfg.n_layers:<8} {cfg.n_heads:<8} "
                  f"{cfg.n_kv_heads:<8} {cfg.head_dim:<9} "
                  f"{cfg.max_ctx:<10} {moe:<8} {cfg.hf_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape model info and generate KV cache presets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- list ---
    list_p = sub.add_parser("list", help="List models from Ollama")
    list_p.add_argument("--limit", type=int, default=10)
    list_p.add_argument("--json", action="store_true", help="Output as JSON")

    # --- preset ---
    preset_p = sub.add_parser("preset", help="Generate preset entries")
    src = preset_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Model name (e.g. 'Qwen-2.5-7B')")
    src.add_argument("--ollama", action="store_true",
                     help="Scrape Ollama for model names, then resolve")
    src.add_argument("--hf-repo", help="HF repo directly (e.g. 'Qwen/Qwen2.5-7B')")
    preset_p.add_argument("--limit", type=int, default=5,
                          help="Max models to process (with --ollama)")
    preset_p.add_argument("--override-repo", metavar="HF_REPO",
                          help="Override HF repo (with --model)")
    preset_p.add_argument("--append", nargs="?", const=True, metavar="PATH",
                          help="Append generated preset to presets.py")
    preset_p.add_argument("--stats", action="store_true",
                          help="Show summary table of generated configs")

    args = parser.parse_args()

    if args.command == "list":
        do_list(args)
    elif args.command == "preset":
        do_preset(args)


if __name__ == "__main__":
    main()
