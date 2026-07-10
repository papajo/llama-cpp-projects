"""
Runner — sends a prompt through multiple sampler configs and collects
the outputs side-by-side.

Usage::

    from sampler_ablation.config import preset_greedy, preset_creative
    from sampler_ablation.runner import run_ablation

    configs = [preset_greedy(), preset_creative()]
    results = run_ablation(
        configs=configs,
        prompt="Write a haiku about Python.",
        base_url="http://127.0.0.1:8080",
    )
    for r in results:
        print(f"[{r.label}]\\n{r.output}\\n")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx

from .config import SamplerConfig


@dataclass
class AblationResult:
    """Output from one sampler config for a single prompt."""

    config: SamplerConfig
    prompt: str

    output: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def label(self) -> str:
        return self.config.label

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @property
    def output_words(self) -> list[str]:
        return self.output.split()

    @property
    def output_word_count(self) -> int:
        return len(self.output_words)


def run_ablation(
    configs: List[SamplerConfig],
    prompt: str,
    base_url: str = "http://127.0.0.1:8080",
    system_prompt: Optional[str] = None,
    max_retries: int = 2,
    client_timeout: float = 300.0,
    progress_cb: Optional[Callable] = None,
) -> List[AblationResult]:
    """
    Run a prompt through multiple sampler configs.

    Args:
        configs: List of sampler configurations to test.
        prompt: The prompt to send.
        base_url: llama.cpp server URL.
        system_prompt: Optional system instruction.
        max_retries: Retries on transient HTTP errors.
        client_timeout: HTTP client timeout in seconds.
        progress_cb: Optional callback ``(done, total, label)``.

    Returns:
        One ``AblationResult`` per config (in the same order).
    """
    total = len(configs)
    results: List[AblationResult] = []

    with httpx.Client(
        base_url=base_url,
        timeout=client_timeout,
    ) as client:

        for i, config in enumerate(configs):
            label = f"[{i + 1}/{total}] {config.label}"
            if progress_cb:
                progress_cb(i, total, config.label)

            result = _run_single(
                client=client,
                config=config,
                prompt=prompt,
                system_prompt=system_prompt,
                max_retries=max_retries,
            )
            results.append(result)

        if progress_cb:
            progress_cb(total, total, "Done!")

    return results


def _run_single(
    client: httpx.Client,
    config: SamplerConfig,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_retries: int = 2,
) -> AblationResult:
    """Execute one API call with retries."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: Dict[str, Any] = {
        "messages": messages,
        "stream": False,
    }
    body.update(config.to_request_body())

    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            response = client.post("/v1/chat/completions", json=body)
            elapsed = time.time() - start

            if response.status_code >= 500 and attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue

            if response.status_code >= 400:
                return AblationResult(
                    config=config,
                    prompt=prompt,
                    elapsed_s=elapsed,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            output = _extract_content(data)
            usage = data.get("usage", {})

            return AblationResult(
                config=config,
                prompt=prompt,
                output=output,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                elapsed_s=elapsed,
            )

        except httpx.TimeoutException as e:
            elapsed = time.time() - start
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return AblationResult(
                config=config,
                prompt=prompt,
                elapsed_s=elapsed,
                error=f"Timeout: {e}",
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return AblationResult(
                config=config,
                prompt=prompt,
                elapsed_s=elapsed,
                error=f"RequestError: {e}",
            )

    return AblationResult(
        config=config,
        prompt=prompt,
        error="Exhausted retries",
    )


def _extract_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")
