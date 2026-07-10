"""
Sweep orchestrator — runs multiple parameter configurations against
llama.cpp's ``/v1/chat/completions`` endpoint and collects results.

Usage::

    from budget_sweep.budget import SweepConfig
    from budget_sweep.sweep import run_sweep

    config = SweepConfig(
        base_url="http://127.0.0.1:8080",
        param_grid={"max_tokens": [64, 128, 256], "temperature": [0.0, 0.7]},
        n_runs=2,
    )
    results = run_sweep(config)
    print(f"Completed {len(results)} runs")
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from .budget import SweepConfig
from .metrics import SweepResult


def run_sweep(
    config: SweepConfig,
    progress_cb=None,
    max_retries: int = 2,
    client_timeout: float = 300.0,
) -> List[SweepResult]:
    """
    Execute a full sweep experiment.

    Iterates over all parameter combinations × prompts × runs and returns
    a flat list of ``SweepResult`` objects (failures included).

    Args:
        config: The sweep configuration.
        progress_cb: Optional callback ``(current, total, label)``.
        max_retries: Number of retries on transient HTTP errors.
        client_timeout: HTTP client timeout in seconds.

    Returns:
        List of ``SweepResult`` (one per API call).
    """
    param_configs = config.iter_configs()
    results: List[SweepResult] = []
    total = config.n_total_runs
    done = 0

    with httpx.Client(
        base_url=config.base_url,
        timeout=client_timeout,
    ) as client:

        for params in param_configs:
            for prompt in config.prompts:
                prompt_label = (
                    prompt[:50] + "..." if len(prompt) > 50 else prompt
                )

                for run_id in range(config.n_runs):
                    label = (
                        f"[{done + 1}/{total}] "
                        f"{_param_label(params)} | "
                        f"{prompt_label} (#{run_id})"
                    )
                    if progress_cb:
                        progress_cb(done, total, label)

                    result = _run_single(
                        client=client,
                        params=params,
                        prompt=prompt,
                        run_id=run_id,
                        system_prompt=config.system_prompt,
                        max_retries=max_retries,
                    )
                    results.append(result)
                    done += 1

        if progress_cb:
            progress_cb(done, total, "Done!")

    return results


def _run_single(
    client: httpx.Client,
    params: Dict[str, Any],
    prompt: str,
    run_id: int,
    system_prompt: Optional[str] = None,
    max_retries: int = 2,
) -> SweepResult:
    """Execute one API call with retries."""
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    body: Dict[str, Any] = {
        "messages": messages,
        "stream": False,
    }
    # Copy only recognised params to the body
    recognised = {
        "max_tokens", "temperature", "top_p", "top_k", "min_p",
        "repeat_penalty", "frequency_penalty", "presence_penalty",
        "seed", "n_keep", "n_ctx",
    }
    for k, v in params.items():
        if k in recognised and v is not None:
            body[k] = v

    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            response = client.post("/v1/chat/completions", json=body)
            elapsed = time.time() - start

            if response.status_code >= 500 and attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue

            if response.status_code >= 400:
                return SweepResult(
                    params=params,
                    prompt=prompt,
                    run_id=run_id,
                    output="",
                    elapsed_s=elapsed,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            output = _extract_content(data)
            usage = data.get("usage", {})

            return SweepResult(
                params=params,
                prompt=prompt,
                run_id=run_id,
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
            return SweepResult(
                params=params,
                prompt=prompt,
                run_id=run_id,
                output="",
                elapsed_s=elapsed,
                error=f"Timeout: {e}",
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return SweepResult(
                params=params,
                prompt=prompt,
                run_id=run_id,
                output="",
                elapsed_s=elapsed,
                error=f"RequestError: {e}",
            )

    # Should not reach here
    return SweepResult(
        params=params,
        prompt=prompt,
        run_id=run_id,
        output="",
        error="Exhausted retries",
    )


def _extract_content(data: Dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible response."""
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def _param_label(params: Dict[str, Any]) -> str:
    """Short human label for a parameter combo."""
    parts: List[str] = []
    if "temperature" in params:
        parts.append(f"T={params['temperature']}")
    if "max_tokens" in params:
        parts.append(f"tok={params['max_tokens']}")
    if "top_p" in params:
        parts.append(f"p={params['top_p']}")
    for k, v in params.items():
        if k not in ("temperature", "max_tokens", "top_p"):
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "default"
