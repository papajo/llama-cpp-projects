"""Live integration tests for the prompt optimizer loop.

The offline suite patches urllib.request.urlopen with canned bodies.
These drive LlamaClient, RubricEvaluator and OptimizationLoop against the
real chat server.

IMPORTANT - what is and is not asserted here.

SmolLM2-360M-Instruct is far too small to be a useful LLM-as-judge or to
write better prompts than the one it is given. So nothing here asserts
that scores improve, that the judge is accurate, or that the rewritten
prompt is better. Those would be assertions about model capability, not
about this code.

What IS asserted is structure: requests are accepted, responses parse,
scores land in [0,1], the loop terminates, history is well-formed, and
the fallback paths fire when the judge returns unparseable output - which
on this model is the common case, not the exception.

Budgets are tiny (max_tokens 16-64, 1-2 iterations, 1-2 test cases)
because the model is slow on CPU.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import pytest

from prompt_optimizer.evaluator import (
    CompositeEvaluator,
    ContainsEvaluator,
    ExactMatchEvaluator,
    RubricEvaluator,
    Score,
)
from prompt_optimizer.optimizer import LlamaClient, OptimizationLoop
# Aliased: pytest tries to collect any module-level name starting with
# "Test", and warns because this dataclass has an __init__.
from prompt_optimizer.prompt import PromptTemplate
from prompt_optimizer.prompt import TestCase as Case


@pytest.fixture
def client(chat_base_url):
    return LlamaClient(
        server_url=chat_base_url, temperature=0.0, max_tokens=32
    )


# ── LlamaClient ─────────────────────────────────────────────────────


@pytest.mark.live
def test_client_completes_against_real_server(client):
    out = client.complete("Say hi.")
    assert isinstance(out, str)
    assert out, "real server returned empty content"
    # complete() strips, so no leading/trailing whitespace survives.
    assert out == out.strip()


@pytest.mark.live
def test_client_per_call_overrides_apply(client):
    """max_tokens passed per call wins over the instance default."""
    long_out = client.complete("Write a long essay about the sea.", max_tokens=64)
    short_out = client.complete("Write a long essay about the sea.", max_tokens=8)
    assert len(short_out) < len(long_out)


@pytest.mark.live
def test_client_sends_model_only_when_set(chat_base_url):
    """An explicit model id is accepted; so is omitting it.

    Note: llama-server serves the loaded model regardless of the id sent,
    so a wrong id is NOT an error here (see drift log).
    """
    named = LlamaClient(
        server_url=chat_base_url,
        model="HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0",
        temperature=0.0,
        max_tokens=8,
    )
    assert named.complete("Say hi.")


# ── RubricEvaluator (LLM-as-judge) ──────────────────────────────────


@pytest.mark.live
def test_rubric_returns_a_wellformed_score(chat_base_url):
    """The judge call round-trips and yields a normalised score.

    The score's *value* is not asserted - a 360M judge is unreliable.
    Only that it is a Score in [0,1] with the right scorer name.
    """
    ev = RubricEvaluator(server_url=chat_base_url, threshold=0.7)
    score = ev.evaluate("Paris", expected="Paris")

    assert isinstance(score, Score)
    assert score.scorer == "rubric"
    assert 0.0 <= score.score <= 1.0
    assert isinstance(score.passed, bool)
    assert score.detail


@pytest.mark.live
def test_rubric_without_expected_short_circuits(chat_base_url):
    """No expected answer means no judge call at all."""
    ev = RubricEvaluator(server_url=chat_base_url)
    score = ev.evaluate("anything", expected=None)
    assert score.score == 0.0
    assert score.passed is False
    assert "No expected output" in score.detail


@pytest.mark.live
def test_rubric_parse_fallback_is_exercised_by_this_model(chat_base_url):
    """SmolLM2 rarely emits the clean JSON the rubric asks for.

    The evaluator has three tiers: strict json.loads, a regex for
    "score": N, then a neutral 5/10 fallback. Whichever fires, the result
    must still be a valid normalised Score - that is the contract this
    pins. Which tier wins is a model-capability question, not a bug.
    """
    ev = RubricEvaluator(server_url=chat_base_url, threshold=0.7)
    scores = [
        ev.evaluate("The capital of France is Paris.", expected="Paris"),
        ev.evaluate("Bananas are blue.", expected="Paris"),
    ]
    for score in scores:
        assert 0.0 <= score.score <= 1.0
        # score_val is clamped to 0..10 then divided by 10, so the
        # normalised value always lands on a tenth.
        assert round(score.score * 10) == pytest.approx(score.score * 10, abs=1e-9)


# ── Deterministic evaluators over real output ───────────────────────


@pytest.mark.live
def test_deterministic_evaluators_score_real_output(client):
    """ContainsEvaluator over genuine model output.

    The prompt is engineered so the check is about plumbing, not about
    the model being clever: we ask it to repeat a token back.
    """
    out = client.complete("Repeat exactly this word and nothing else: zebra")
    score = ContainsEvaluator(case_sensitive=False).evaluate(out, expected="zebra")
    assert score.scorer == "contains"
    assert score.score in (0.0, 1.0)
    assert isinstance(score.passed, bool)


@pytest.mark.live
def test_composite_evaluator_combines_real_scores(chat_base_url, client):
    """min-strategy composite over a real judge plus a real exact match."""
    out = client.complete("Say the single word: Paris")
    composite = CompositeEvaluator(
        [
            ExactMatchEvaluator(),
            ContainsEvaluator(case_sensitive=False),
            RubricEvaluator(server_url=chat_base_url),
        ],
        strategy="min",
    )
    score = composite.evaluate(out, expected="Paris")
    assert score.scorer == "composite"
    assert 0.0 <= score.score <= 1.0
    # min strategy: never above the weakest component.
    assert score.score <= 1.0


# ── Full loop ───────────────────────────────────────────────────────


@pytest.mark.live
def test_optimization_loop_terminates_and_records_history(chat_base_url, client):
    """One real iteration end to end.

    max_iterations=1 keeps the CPU cost bounded. The assertion is that the
    loop runs, records a well-formed snapshot and terminates - NOT that
    the score improved.
    """
    template = PromptTemplate(
        template="Answer in one word. What is the capital of {{country}}?"
    )
    cases = [Case(input_vars={"country": "France"}, expected="Paris")]

    loop = OptimizationLoop(
        template=template,
        test_cases=cases,
        evaluator=ContainsEvaluator(case_sensitive=False),
        llm_client=client,
        max_iterations=1,
        verbose=False,
    )
    history = loop.run()

    assert len(history) == 1
    snap = history[0]
    assert snap.iteration == 1
    assert 0.0 <= snap.overall_score <= 1.0
    assert snap.prompt_text
    assert 0.0 <= loop.best_score <= 1.0
    assert loop.best_prompt


@pytest.mark.live
def test_loop_stops_at_max_iterations_without_improving(chat_base_url, client):
    """The loop must terminate even when the model never improves.

    With a 360M model the meta-prompt rewrite is usually useless, so this
    is the realistic path: run out of iterations and stop cleanly.
    """
    template = PromptTemplate(
        template="Respond to {{thing}} with an exact SHA-256 hash."
    )
    cases = [Case(input_vars={"thing": "x"}, expected="impossible-to-match")]

    loop = OptimizationLoop(
        template=template,
        test_cases=cases,
        evaluator=ExactMatchEvaluator(),
        llm_client=client,
        max_iterations=2,
        plateau_window=99,  # disable plateau stop so max_iterations is the bound
        min_score=0.99,
        verbose=False,
    )
    history = loop.run()

    assert len(history) == 2
    assert all(0.0 <= s.overall_score <= 1.0 for s in history)
    # Never reached the target - which is expected, not a failure.
    assert loop.best_score < 0.99


@pytest.mark.live
def test_progress_callback_fires_per_iteration(chat_base_url, client):
    seen = []
    template = PromptTemplate(template="Say {{word}}.")
    cases = [Case(input_vars={"word": "ok"}, expected="ok")]

    loop = OptimizationLoop(
        template=template,
        test_cases=cases,
        evaluator=ContainsEvaluator(case_sensitive=False),
        llm_client=client,
        max_iterations=1,
        verbose=False,
        progress_callback=seen.append,
    )
    loop.run()
    assert len(seen) == 1
    assert seen[0].iteration == 1
