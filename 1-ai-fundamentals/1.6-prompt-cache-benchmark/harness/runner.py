"""
Benchmark runner for prompt caching.

Connects to a llama-server instance, sends prompts with and without
caching, and measures time-to-first-token (TTFT).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .metrics import CacheBenchmarkResult, PromptResult, estimate_tokens
from .scenarios import CacheScenario, get_scenario

logger = logging.getLogger("cache-benchmark")


class CacheBenchmarkRunner:
    """
    Runs cache benchmarks against a llama-server instance.
    
    For each scenario:
      1. Send all prompts WITHOUT cache (cache_prompt=False) — measure TTFT
      2. Wait for cache to clear (or reset)
      3. Send all prompts WITH cache (cache_prompt=True) — measure TTFT
      4. Compare and report
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        model_path: str = "",
        n_predict: int = 50,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_path = model_path
        self.n_predict = n_predict
        self.temperature = temperature
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()

    def health_check(self) -> bool:
        """Check if llama-server is running."""
        try:
            r = self.client.get(f"{self.server_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    def _send_completion(
        self,
        prompt: str,
        cache_prompt: bool = True,
    ) -> Dict[str, Any]:
        """Send a single completion request and return the full response."""
        payload = {
            "prompt": prompt,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "cache_prompt": cache_prompt,
        }
        start = time.time()
        response = self.client.post(
            f"{self.server_url}/completion",
            json=payload,
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            raise RuntimeError(
                f"llama-server returned {response.status_code}: {response.text}"
            )

        data = response.json()
        data["_request_time_ms"] = elapsed_ms
        return data

    def run_scenario(
        self,
        scenario: CacheScenario,
        warmup_runs: int = 1,
    ) -> CacheBenchmarkResult:
        """
        Run a complete cache benchmark for a scenario.
        
        Returns a CacheBenchmarkResult with both cached and uncached runs.
        """
        result = CacheBenchmarkResult(
            scenario_name=scenario.name,
            description=scenario.description,
        )

        # Warmup
        for prompt in scenario.prompts[:warmup_runs]:
            try:
                self._send_completion(prompt, cache_prompt=True)
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

        # ── Uncached runs ────────────────────────────────────────────
        logger.info(f"Running {len(scenario.prompts)} uncached prompts...")
        for i, prompt in enumerate(scenario.prompts):
            try:
                data = self._send_completion(prompt, cache_prompt=False)
                timings = data.get("timings", {})

                pr = PromptResult(
                    prompt_index=i,
                    prompt_length_chars=len(prompt),
                    ttft_ms=timings.get("prompt_ms", data.get("_request_time_ms", 0)),
                    prompt_tokens=timings.get("prompt_n", estimate_tokens(prompt)),
                    prompt_processing_ms=timings.get("prompt_ms", 0),
                    predicted_per_second=timings.get("predicted_per_second", 0),
                    cached=False,
                )
                result.runs_uncached.append(pr)
            except Exception as e:
                logger.error(f"Uncached prompt {i} failed: {e}")

        # Small delay to let cache settle
        time.sleep(0.5)

        # ── Cached runs ──────────────────────────────────────────────
        logger.info(f"Running {len(scenario.prompts)} cached prompts...")
        for i, prompt in enumerate(scenario.prompts):
            try:
                data = self._send_completion(prompt, cache_prompt=True)
                timings = data.get("timings", {})

                pr = PromptResult(
                    prompt_index=i,
                    prompt_length_chars=len(prompt),
                    ttft_ms=timings.get("prompt_ms", data.get("_request_time_ms", 0)),
                    prompt_tokens=timings.get("prompt_n", estimate_tokens(prompt)),
                    prompt_processing_ms=timings.get("prompt_ms", 0),
                    predicted_per_second=timings.get("predicted_per_second", 0),
                    cached=True,
                )
                result.runs_cached.append(pr)
            except Exception as e:
                logger.error(f"Cached prompt {i} failed: {e}")

        result.compute()
        return result

    def run_all_scenarios(
        self,
        scenario_names: Optional[List[str]] = None,
        warmup_runs: int = 1,
    ) -> Dict[str, CacheBenchmarkResult]:
        """Run benchmarks for multiple scenarios."""
        names = scenario_names or [s["name"] for s in get_scenario_list()]
        results = {}

        for name in names:
            scenario = get_scenario(name)
            if scenario is None:
                logger.warning(f"Unknown scenario: {name}")
                continue
            logger.info(f"Benchmarking scenario: {scenario.name}")
            try:
                result = self.run_scenario(scenario, warmup_runs)
                results[name] = result
                logger.info(
                    f"  Speedup: {result.speedup_factor:.2f}x "
                    f"(cached: {result.avg_ttft_cached_ms:.1f}ms, "
                    f"uncached: {result.avg_ttft_uncached_ms:.1f}ms)"
                )
            except Exception as e:
                logger.error(f"Scenario {name} failed: {e}")

        return results


def get_scenario_list() -> List[Dict]:
    """Get list of available scenarios."""
    from .scenarios import list_scenarios
    return list_scenarios()
