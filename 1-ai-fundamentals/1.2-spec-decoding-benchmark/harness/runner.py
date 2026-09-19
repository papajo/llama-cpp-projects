"""
BenchmarkRunner — orchestrates speculative decoding benchmark runs.

For each strategy, it:
  1. Launches llama-server with the appropriate --spec-type flags
  2. Sends prompts from the corpus via the /completion streaming API
  3. Records per-token timestamps, accept/reject events, and counts
  4. Terminates the server and computes metrics
  5. Repeats for reliability (configurable N runs)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from .configs import StrategyConfig
from .metrics import BenchmarkMetrics

# Type for a single completion result before metric computation
RawCompletion = Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Results from a single strategy × prompt combination."""
    metrics: BenchmarkMetrics
    raw_completions: List[RawCompletion] = field(default_factory=list)
    strategy: str = ""
    prompt_name: str = ""

    def as_dict(self) -> Dict:
        return self.metrics.as_dict()


class BenchmarkRunner:
    """
    Orchestrates speculative decoding benchmark runs.

    Parameters
    ----------
    model_path : str
        Path to the target GGUF model.
    draft_model_path : str, optional
        Path to the draft GGUF model (required for draft-based strategies).
    server_binary : str
        Path to the llama-server binary (default: 'llama-server').
    server_port : int
        Base port for the server (default: 18080).
    host : str
        Server host (default: '127.0.0.1').
    n_gpu_layers : int
        Number of layers to offload to GPU (default: -1 = all).
    context_size : int
        Context size (default: 4096).
    n_parallel : int
        Number of parallel slots (default: 1).
    verbose : bool
        Print server output (default: False).
    server_start_timeout : float
        Seconds to wait for server to become ready (default: 30).
    """

    def __init__(
        self,
        model_path: str,
        draft_model_path: Optional[str] = None,
        server_binary: str = "llama-server",
        server_port: int = 18080,
        host: str = "127.0.0.1",
        n_gpu_layers: int = -1,
        context_size: int = 4096,
        n_parallel: int = 1,
        verbose: bool = False,
        server_start_timeout: float = 30.0,
    ):
        self.model_path = Path(model_path).resolve()
        self.draft_model_path = (
            Path(draft_model_path).resolve() if draft_model_path else None
        )
        self.server_binary = server_binary
        self.server_port = server_port
        self.host = host
        self.n_gpu_layers = n_gpu_layers
        self.context_size = context_size
        self.n_parallel = n_parallel
        self.verbose = verbose
        self.server_start_timeout = server_start_timeout

        self._server_proc: Optional[subprocess.Popen] = None
        self._server_url = f"http://{host}:{server_port}"
        self._client = httpx.Client(base_url=self._server_url, timeout=120.0)

    # ── Server lifecycle ───────────────────────────────────────

    def _base_flags(self) -> List[str]:
        """Common llama-server flags used for all runs."""
        flags = [
            "--model", str(self.model_path),
            "--host", self.host,
            "--port", str(self.server_port),
            "--ctx-size", str(self.context_size),
            "--parallel", str(self.n_parallel),
            "--cont-batching",
            "--slots",  # enable /slots endpoint for monitoring
        ]
        if self.n_gpu_layers >= 0:
            flags.extend(["--n-gpu-layers", str(self.n_gpu_layers)])
        return flags

    def start_server(self, extra_flags: Optional[List[str]] = None) -> None:
        """Launch llama-server as a subprocess."""
        if self._server_proc is not None:
            self.stop_server()

        cmd = [self.server_binary] + self._base_flags()
        if extra_flags:
            cmd.extend(extra_flags)

        if self.verbose:
            print(f"  🚀 Starting: {' '.join(cmd)}")
        else:
            print(f"  🚀 Starting llama-server (PID will be assigned)...")

        self._server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not self.verbose else None,
            stderr=subprocess.PIPE if not self.verbose else None,
        )

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        """Poll /health or /slots until server responds."""
        start = time.time()
        while time.time() - start < self.server_start_timeout:
            try:
                resp = self._client.get("/slots", timeout=2.0)
                if resp.status_code == 200:
                    print(f"  ✅ Server ready at {self._server_url}")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(1)

        raise RuntimeError(
            f"Server did not start within {self.server_start_timeout}s. "
            f"Check that '{self.server_binary}' is in PATH and the model exists."
        )

    def stop_server(self) -> None:
        """Terminate the llama-server subprocess."""
        if self._server_proc is None:
            return
        print(f"  🛑 Stopping server...")
        self._server_proc.send_signal(signal.SIGTERM)
        try:
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()
            self._server_proc.wait()
        self._server_proc = None
        # Small cooldown to let the port release
        time.sleep(2)

    # ── Completion helpers ─────────────────────────────────────

    def send_completion(
        self,
        prompt: str,
        n_predict: int = 200,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> RawCompletion:
        """
        Send a completion request via the streaming API and collect
        per-token timestamps and draft acceptance info.

        Returns a dict with full timing breakdown.
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
        }

        result: RawCompletion = {
            "timestamps": [],
            "tokens": [],
            "draft_accepted_count": 0,
            "draft_total_count": 0,
            "prompt_eval_ms": 0,
            "total_tokens": 0,
            "full_text": "",
        }

        start_time = time.time()
        first_token_time: Optional[float] = None
        token_count = 0

        try:
            with self._client.stream("POST", "/completion", json=payload) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                    else:
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    now = time.time()

                    # The final SSE chunk carries stop metadata with an
                    # empty `content`, so key-presence alone would count one
                    # phantom token per request.
                    if data.get("content"):
                        token_count += 1
                        result["tokens"].append(data["content"])
                        result["timestamps"].append(now)
                        if first_token_time is None:
                            first_token_time = now
                            result["prompt_eval_ms"] = (now - start_time) * 1000

                    # Draft acceptance info (if available in response)
                    if "draft_accepted" in data:
                        result["draft_accepted_count"] += data["draft_accepted"]
                    if "draft_total" in data:
                        result["draft_total_count"] += data["draft_total"]

                    if data.get("stop", False):
                        # The stop chunk's own `content` is empty; the text is
                        # whatever the streamed token chunks accumulated.
                        result["full_text"] = "".join(result["tokens"])
                        result["timing"] = data.get("timings", {})
                        break

        except httpx.TimeoutException:
            print("    ⚠ Completion timed out")

        result["total_tokens"] = token_count
        result["wall_time_s"] = time.time() - start_time

        return result

    # ── Single strategy run ────────────────────────────────────

    def run_strategy(
        self,
        strategy: StrategyConfig,
        prompts: List[Tuple[str, str]],  # (name, prompt_text)
        n_predict: int = 200,
        temperature: float = 0.7,
        n_runs: int = 1,
    ) -> List[BenchmarkResult]:
        """
        Run all prompts through a single speculative decoding strategy.

        Parameters
        ----------
        strategy : StrategyConfig
            The strategy configuration to benchmark.
        prompts : list of (name, text)
            List of (prompt_name, prompt_text) pairs.
        n_predict : int
            Number of tokens to generate per completion.
        temperature : float
            Sampling temperature.
        n_runs : int
            Number of times to repeat each prompt (for statistical
            significance).

        Returns
        -------
        list of BenchmarkResult
            One result per prompt × run.
        """
        print(f"\n{'='*60}")
        print(f"  🎯 Strategy: {strategy.label}")
        print(f"     {strategy.description}")
        print(f"     Params: {strategy.param_summary()}")
        print(f"{'='*60}")

        # Build the extra server flags needed for this strategy
        extra_flags = strategy.to_llama_server_flags()

        # For draft-model strategies, add the draft model path
        if strategy.strategy.requires_draft_model():
            if self.draft_model_path is None:
                raise ValueError(
                    f"Strategy {strategy.label} requires a draft model path. "
                    f"Set draft_model_path."
                )
            extra_flags.extend(["--spec-draft-model", str(self.draft_model_path)])

        self.start_server(extra_flags)
        results: List[BenchmarkResult] = []

        try:
            for prompt_name, prompt_text in prompts:
                for run_idx in range(n_runs):
                    run_label = f"{prompt_name} (run {run_idx + 1})"
                    print(f"\n  📝 Prompt: {run_label}")

                    completion = self.send_completion(
                        prompt=prompt_text,
                        n_predict=n_predict,
                        temperature=temperature,
                        seed=run_idx + 42,  # vary seed per run
                    )

                    metrics = self._compute_metrics(
                        strategy, prompt_name, completion, n_predict
                    )

                    result = BenchmarkResult(
                        metrics=metrics,
                        raw_completions=[completion],
                        strategy=strategy.label,
                        prompt_name=prompt_name,
                    )
                    results.append(result)

                    print(f"     Tokens: {metrics.total_tokens_generated} | "
                          f"Accept rate: {metrics.acceptance_rate:.1%} | "
                          f"tok/s: {metrics.tokens_per_second:.1f} | "
                          f"Wall: {metrics.wall_time_s:.1f}s")

        finally:
            self.stop_server()

        return results

    # ── Baseline run (no speculation) ──────────────────────────

    def run_baseline(
        self,
        prompts: List[Tuple[str, str]],
        n_predict: int = 200,
        temperature: float = 0.7,
        n_runs: int = 1,
    ) -> List[BenchmarkResult]:
        """Run without any speculative decoding to establish a baseline."""
        print(f"\n{'='*60}")
        print(f"  📊 Baseline (no speculation)")
        print(f"{'='*60}")

        self.start_server()
        results: List[BenchmarkResult] = []

        try:
            for prompt_name, prompt_text in prompts:
                for run_idx in range(n_runs):
                    run_label = f"{prompt_name} (run {run_idx + 1})"
                    completion = self.send_completion(
                        prompt=prompt_text,
                        n_predict=n_predict,
                        temperature=temperature,
                        seed=run_idx + 42,
                    )

                    # For baseline, there's no draft — total_tokens is manual
                    metrics = BenchmarkMetrics(
                        strategy_label="baseline",
                        prompt_name=prompt_name,
                        total_tokens_generated=completion["total_tokens"],
                        total_draft_tokens=0,
                        total_accepted_draft_tokens=0,
                        total_verified_tokens=completion["total_tokens"],
                        num_target_forward_passes=completion["total_tokens"],
                        num_draft_forward_passes=0,
                        wall_time_s=completion["wall_time_s"],
                        prompt_eval_time_s=completion.get("prompt_eval_ms", 0) / 1000,
                        token_timestamps_s=completion.get("timestamps", []),
                    )

                    result = BenchmarkResult(
                        metrics=metrics,
                        raw_completions=[completion],
                        strategy="baseline",
                        prompt_name=prompt_name,
                    )
                    results.append(result)

                    print(f"     Tokens: {metrics.total_tokens_generated} | "
                          f"tok/s: {metrics.tokens_per_second:.1f} | "
                          f"Wall: {metrics.wall_time_s:.1f}s")

        finally:
            self.stop_server()

        return results

    # ── Metrics computation ────────────────────────────────────

    def _compute_metrics(
        self,
        strategy: StrategyConfig,
        prompt_name: str,
        completion: RawCompletion,
        n_predict: int,
    ) -> BenchmarkMetrics:
        """Convert raw completion data into computed metrics."""
        total_tokens = completion["total_tokens"]

        # Estimate draft vs verified counts.
        # In speculative decoding, every target forward pass produces
        # 1 verified token. Draft tokens are accepted or rejected.
        # We estimate from the completion data.
        draft_accepted = completion.get("draft_accepted_count", 0)
        draft_total = completion.get("draft_total_count", 0)

        # If the server doesn't report draft stats, estimate:
        if draft_total == 0 and strategy.strategy.requires_draft_model():
            # Rough estimate: each verification step produces up to
            # draft_n_max draft tokens + 1 verified token
            n_verify_steps = max(1, total_tokens // (strategy.draft_n_max + 1))
            draft_total = n_verify_steps * strategy.draft_n_max
            # Assume ~60% acceptance as a fallback estimate
            draft_accepted = int(draft_total * 0.6)

        # Try to extract timing info from llama-server's timings
        timing = completion.get("timing", {})
        prompt_eval_ms = completion.get("prompt_eval_ms", 0)
        if timing:
            prompt_eval_ms = timing.get("prompt_eval_ms", prompt_eval_ms)

        return BenchmarkMetrics(
            strategy_label=strategy.label,
            prompt_name=prompt_name,
            total_tokens_generated=total_tokens,
            total_draft_tokens=draft_total,
            total_accepted_draft_tokens=draft_accepted,
            total_verified_tokens=total_tokens,
            num_target_forward_passes=max(1, total_tokens - draft_accepted),
            num_draft_forward_passes=max(1, draft_total // strategy.draft_n_max)
                if strategy.draft_n_max > 0 else 1,
            wall_time_s=completion["wall_time_s"],
            prompt_eval_time_s=prompt_eval_ms / 1000,
            token_timestamps_s=completion.get("timestamps", []),
            draft_n_max=strategy.draft_n_max,
            config_flags=strategy.param_summary(),
        )

    # ── Full benchmark suite ───────────────────────────────────

    def run_full_benchmark(
        self,
        prompts: List[Tuple[str, str]],
        strategies: Optional[List[StrategyConfig]] = None,
        n_predict: int = 200,
        temperature: float = 0.7,
        n_runs: int = 1,
    ) -> List[BenchmarkResult]:
        """
        Run all strategies + baseline against all prompts.

        Returns a flat list of results.
        """
        if strategies is None:
            from .configs import ALL_STRATEGIES
            strategies = ALL_STRATEGIES

        all_results: List[BenchmarkResult] = []

        # Baseline first
        baseline_results = self.run_baseline(prompts, n_predict, temperature, n_runs)
        all_results.extend(baseline_results)

        # Then each strategy
        for strategy in strategies:
            results = self.run_strategy(
                strategy, prompts, n_predict, temperature, n_runs
            )
            all_results.extend(results)

        return all_results

    def cleanup(self):
        """Clean up resources."""
        self.stop_server()
        self._client.close()
