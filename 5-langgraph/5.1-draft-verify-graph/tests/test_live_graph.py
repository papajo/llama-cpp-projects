"""Live integration tests for the draft-verify reflexion loop.

These exercise the same code paths as the mocked tests in test_llm.py /
test_nodes.py / test_graph.py, but against the real llama-server on
$LLM_CHAT_BASE_URL instead of a patched urllib.request.urlopen.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Assertions are STRUCTURAL only. SmolLM2-360M is a 360M-parameter model: it
does not reliably emit JSON, follow a rubric, or self-critique. Its output
quality is not a property of this code, so nothing here asserts on meaning.
"""

from __future__ import annotations

import pytest

from draft_verify_graph.graph import DraftVerifyGraph
from draft_verify_graph.llm import LlamaClient, LlamaError
from draft_verify_graph.nodes import (
    DraftNode,
    ImproveNode,
    VerificationResult,
    VerifyNode,
)
from draft_verify_graph.reporting import result_to_markdown


@pytest.fixture
def client(chat_base_url):
    """A LlamaClient pointed at the real chat server."""
    return LlamaClient(chat_base_url, timeout=180)


# ---------------------------------------------------------------------------
# LlamaClient — the HTTP layer the mocked tests patch out
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_complete_returns_text(client, chat_model):
    """complete() extracts content from a real response body."""
    out = client.complete(
        [{"role": "user", "content": "Say hi"}],
        model=chat_model,
        max_tokens=8,
    )
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.live
def test_complete_honours_max_tokens(client, chat_model):
    """A small max_tokens really does bound the generation."""
    out = client.complete(
        [{"role": "user", "content": "Count from one to one hundred slowly."}],
        model=chat_model,
        max_tokens=8,
    )
    # 8 tokens can never be a 100-number list; bound generously in characters.
    assert len(out) < 200


@pytest.mark.live
def test_complete_ignores_unknown_model(client):
    """Real llama-server serves its loaded model whatever `model` says.

    It does NOT 404 on an unknown model id, so complete() returns normally.
    Recorded as drift: a mocked test asserting an error here would be wrong.
    """
    out = client.complete(
        [{"role": "user", "content": "Hi"}],
        model="definitely-not-a-real-model",
        max_tokens=8,
    )
    assert isinstance(out, str)


@pytest.mark.live
def test_complete_raises_on_unreachable_server():
    """The URLError branch of complete(), against a closed port."""
    client = LlamaClient("http://127.0.0.1:1", timeout=5)
    with pytest.raises(LlamaError):
        client.complete([{"role": "user", "content": "hi"}], max_tokens=4)


@pytest.mark.live
def test_complete_raises_on_server_error(embed_base_url):
    """A 4xx/5xx body has no `choices`, so complete() raises LlamaError.

    The embed server cannot do logits computation, so asking it for a chat
    completion is a real, non-synthetic error path: it returns HTTP 500 with
    {"error": {...}}. urlopen turns that into an HTTPError (an OSError
    subclass), which complete() wraps.
    """
    client = LlamaClient(embed_base_url, timeout=30)
    with pytest.raises(LlamaError):
        client.complete([{"role": "user", "content": "hi"}], max_tokens=4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_draft_node_produces_text(client, chat_model):
    node = DraftNode(client=client, model=chat_model)
    draft = node.run("Name one colour.")
    assert isinstance(draft, str)
    assert draft.strip()


@pytest.mark.live
def test_verify_node_returns_wellformed_result(client, chat_model):
    """VerifyNode always yields a valid VerificationResult.

    SmolLM2-360M will almost certainly NOT return the requested JSON, so this
    normally lands in _parse()'s regex fallback. That is the point: the
    contract is that the node degrades to a well-formed result either way.
    """
    node = VerifyNode(client=client, model=chat_model)
    result = node.run("Name one colour.", "Blue.")

    assert isinstance(result, VerificationResult)
    assert isinstance(result.score, int)
    assert 0 <= result.score <= 10, "score must be clamped into range"
    assert result.verdict in ("pass", "fail")
    assert isinstance(result.issues, list)
    assert all(isinstance(i, str) for i in result.issues)
    assert isinstance(result.passed, bool)


@pytest.mark.live
def test_improve_node_produces_text(client, chat_model):
    node = ImproveNode(client=client, model=chat_model)
    out = node.run("Name one colour.", "Blue.", ["Too terse."])
    assert isinstance(out, str)
    assert out.strip()


# ---------------------------------------------------------------------------
# Whole graph
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_graph_runs_to_completion(client, chat_model):
    """The reflexion loop terminates and accounts for its calls correctly.

    max_iterations=1 keeps this to two real calls (draft + verify): the loop
    returns as soon as `iteration == self.max_iterations`, whatever the score.
    """
    graph = DraftVerifyGraph(
        draft_node=DraftNode(client=client, model=chat_model),
        verify_node=VerifyNode(client=client, model=chat_model),
        improve_node=ImproveNode(client=client, model=chat_model),
        max_iterations=1,
    )
    result = graph.run("Name one colour.")

    assert result.task == "Name one colour."
    assert result.final_output.strip()
    assert result.num_iterations == 1
    assert result.total_calls == 2, "one draft + one verify"
    assert isinstance(result.passed, bool)

    rec = result.iterations[0]
    assert rec.iteration == 1
    assert rec.draft == result.final_output
    assert rec.improved is None, "last iteration never improves"


@pytest.mark.live
def test_graph_report_renders(client, chat_model):
    """reporting.result_to_markdown survives real (messy) model output."""
    graph = DraftVerifyGraph(
        draft_node=DraftNode(client=client, model=chat_model),
        verify_node=VerifyNode(client=client, model=chat_model),
        improve_node=ImproveNode(client=client, model=chat_model),
        max_iterations=1,
    )
    md = result_to_markdown(graph.run("Name one colour."))

    assert md.startswith("# Draft-Verify Graph Report")
    assert "## Iteration 1" in md
    assert "## Final Output" in md
