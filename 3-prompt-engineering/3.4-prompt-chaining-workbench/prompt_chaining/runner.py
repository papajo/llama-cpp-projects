"""
Chain runner — executes a ``PromptChain`` step-by-step against a
llama.cpp server, passing context between steps.

Usage::

    from prompt_chaining.chain import PromptChain, ChainStep
    from prompt_chaining.runner import run_chain

    chain = PromptChain(
        name="test",
        steps=[
            ChainStep(name="step1", user_prompt="Say hello: {{input}}"),
            ChainStep(name="step2", user_prompt="Repeat: {{step_1}}"),
        ],
    )
    result = run_chain(chain, input_text="world", base_url="http://127.0.0.1:8080")
    for sr in result.steps:
        print(f"[{sr.step_name}] {sr.output}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx

from .chain import ChainStep, PromptChain


@dataclass
class StepResult:
    """Result of executing one chain step."""

    step_name: str
    step_index: int
    output: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None
    rendered_system: str = ""
    rendered_user: str = ""

    @property
    def is_ok(self) -> bool:
        return self.error is None


@dataclass
class ChainResult:
    """Complete result of executing a ``PromptChain``."""

    chain_name: str
    steps: List[StepResult]
    input_text: str
    total_elapsed_s: float = 0.0

    @property
    def ok_steps(self) -> List[StepResult]:
        return [s for s in self.steps if s.is_ok]

    @property
    def failed_steps(self) -> List[StepResult]:
        return [s for s in self.steps if not s.is_ok]

    @property
    def all_ok(self) -> bool:
        return all(s.is_ok for s in self.steps)

    @property
    def last_output(self) -> str:
        """Return the output of the last successful step."""
        for s in reversed(self.steps):
            if s.is_ok and s.output:
                return s.output
        return ""

    def context(self) -> Dict[str, str]:
        """Build a context dict from step outputs for template resolution."""
        ctx: Dict[str, str] = {"input": self.input_text}
        for s in self.steps:
            if s.is_ok:
                ctx[f"step_{s.step_index}"] = s.output
                ctx[s.step_name] = s.output
        return ctx


def run_chain(
    chain: PromptChain,
    input_text: str,
    base_url: str = "http://127.0.0.1:8080",
    max_retries: int = 2,
    client_timeout: float = 300.0,
    progress_cb: Optional[Callable] = None,
    stop_on_error: bool = True,
) -> ChainResult:
    """Execute a ``PromptChain`` step by step.

    Args:
        chain: The chain to execute.
        input_text: Initial input (available as ``{{input}}``).
        base_url: llama.cpp server URL.
        max_retries: Retries per step on transient errors.
        client_timeout: HTTP client timeout per step.
        progress_cb: ``(step_index, total_steps, step_name, status)``.
        stop_on_error: If ``True``, stop on first error.

    Returns:
        ``ChainResult`` with per-step results.
    """
    start_time = time.time()
    context: Dict[str, str] = {"input": input_text}
    step_results: List[StepResult] = []

    with httpx.Client(base_url=base_url, timeout=client_timeout) as client:
        for i, step in enumerate(chain.steps, start=1):
            if progress_cb:
                progress_cb(i, len(chain.steps), step.name, "running")

            # Render with current context
            rendered = step.render(context)
            sr = _execute_step(
                client=client,
                step=step,
                step_index=i,
                rendered=rendered,
                temperature=step.temperature if step.temperature is not None else chain.temperature,
                max_tokens=step.max_tokens if step.max_tokens is not None else chain.max_tokens,
                max_retries=max_retries,
            )
            step_results.append(sr)

            if sr.is_ok:
                # Add output to context for next step
                context[f"step_{i}"] = sr.output
                context[step.name] = sr.output
                if progress_cb:
                    progress_cb(i, len(chain.steps), step.name, "ok")
            else:
                if progress_cb:
                    progress_cb(i, len(chain.steps), step.name, f"error: {sr.error}")
                if stop_on_error:
                    break

    total_elapsed = time.time() - start_time
    return ChainResult(
        chain_name=chain.name,
        steps=step_results,
        input_text=input_text,
        total_elapsed_s=total_elapsed,
    )


def _execute_step(
    client: httpx.Client,
    step: ChainStep,
    step_index: int,
    rendered: Dict[str, str],
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> StepResult:
    """Execute one chain step against the API with retries."""
    messages: List[Dict[str, str]] = []
    if rendered["system"]:
        messages.append({"role": "system", "content": rendered["system"]})
    messages.append({"role": "user", "content": rendered["user"]})

    body: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            response = client.post("/v1/chat/completions", json=body)
            elapsed = time.time() - start

            if response.status_code >= 500 and attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue

            if response.status_code >= 400:
                return StepResult(
                    step_name=step.name,
                    step_index=step_index,
                    elapsed_s=elapsed,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    rendered_system=rendered["system"],
                    rendered_user=rendered["user"],
                )

            data = response.json()
            output = _extract_content(data)
            usage = data.get("usage", {})

            return StepResult(
                step_name=step.name,
                step_index=step_index,
                output=output,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                elapsed_s=elapsed,
                rendered_system=rendered["system"],
                rendered_user=rendered["user"],
            )

        except httpx.TimeoutException as e:
            elapsed = time.time() - start
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return StepResult(
                step_name=step.name,
                step_index=step_index,
                elapsed_s=elapsed,
                error=f"Timeout: {e}",
                rendered_system=rendered["system"],
                rendered_user=rendered["user"],
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return StepResult(
                step_name=step.name,
                step_index=step_index,
                elapsed_s=elapsed,
                error=f"RequestError: {e}",
                rendered_system=rendered["system"],
                rendered_user=rendered["user"],
            )

    return StepResult(
        step_name=step.name,
        step_index=step_index,
        error="Exhausted retries",
        rendered_system=rendered["system"],
        rendered_user=rendered["user"],
    )


def _extract_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")
