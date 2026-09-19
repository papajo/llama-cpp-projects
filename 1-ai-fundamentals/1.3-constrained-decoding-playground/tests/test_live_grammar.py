"""Live integration tests for constrained decoding.

Exercises the same GBNF / JSON-schema paths the offline suite covers, but
against a real llama-server, and feeds the real /completion payload through
GrammarDebugger.

SmolLM2-360M is far too small to produce good content, so every assertion
here is structural: does the grammar actually constrain the token stream,
and does our parser read the server's fields correctly.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.grammar import GBNFParser, GrammarDebugger  # noqa: E402
from backend.schema_converter import SchemaConverter  # noqa: E402


def _completion(base_url, **body):
    import urllib.request

    req = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


@pytest.mark.live
def test_digit_grammar_constrains_output(chat_base_url):
    """A digits-only GBNF grammar admits nothing but digits."""
    data = _completion(
        chat_base_url,
        prompt="Pick a number:",
        n_predict=8,
        grammar="root ::= [0-9]+",
        temperature=0.0,
    )
    assert data["content"], "grammar-constrained generation produced no text"
    assert data["content"].strip().isdigit(), repr(data["content"])


@pytest.mark.live
def test_grammar_trace_reads_real_fields(chat_base_url):
    """analyse_completion parses a real n_probs payload.

    Regression guard for the token/text field drift: token_text must be
    populated from the server's "token" key.
    """
    grammar = "root ::= [0-9]+"
    data = _completion(
        chat_base_url,
        prompt="Pick a number:",
        n_predict=6,
        n_probs=5,
        grammar=grammar,
        temperature=0.0,
    )
    assert data.get("completion_probabilities"), "server returned no n_probs data"

    result = GrammarDebugger.analyse_completion(data, grammar_text=grammar)

    assert result.total_tokens > 0
    assert result.grammar_used == grammar
    assert result.full_text == data["content"]
    # Every step must carry the token string the server reported.
    assert all(step.token_text for step in result.constrained_steps)
    for step, raw in zip(result.constrained_steps, data["completion_probabilities"]):
        assert step.token_text == raw["token"]
        assert step.token_id == raw["id"]
        assert step.num_candidates == len(raw["top_logprobs"])
    assert 0.0 <= result.mask_rate <= 1.0


@pytest.mark.live
def test_grammar_forces_low_probability_tokens(chat_base_url):
    """The masking heuristic fires when a grammar overrides the model.

    Unconstrained, the model would not answer a word-shaped prompt with a
    digit; the grammar forces one, so the chosen token should rank below
    the unconstrained favourite at least once.
    """
    grammar = "root ::= [0-9]+"
    data = _completion(
        chat_base_url,
        prompt="The capital of France is",
        n_predict=6,
        n_probs=10,
        grammar=grammar,
        temperature=0.0,
    )
    probs = data["completion_probabilities"]
    ranks = []
    for entry in probs:
        ordered = sorted(entry["top_logprobs"], key=lambda c: c["logprob"], reverse=True)
        ranks.append(next(
            (i for i, c in enumerate(ordered) if c["id"] == entry["id"]), -1
        ))
    # At least the first token must have been dragged off the top choice.
    assert any(r != 0 for r in ranks), f"grammar appears not to bind: ranks={ranks}"


@pytest.mark.live
@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(
            {"type": "object", "properties": {"ok": {"type": "boolean"}},
             "required": ["ok"]},
            id="boolean",
        ),
        pytest.param(
            {"type": "object",
             "properties": {"c": {"type": "string", "enum": ["red", "green"]}},
             "required": ["c"]},
            id="enum",
        ),
        pytest.param(
            {"type": "object",
             "properties": {"first_name": {"type": "string"}},
             "required": ["first_name"]},
            id="string-underscored-key",
        ),
        pytest.param(
            {"type": "object",
             "properties": {"xs": {"type": "array", "items": {"type": "integer"}}},
             "required": ["xs"]},
            id="array",
        ),
        pytest.param(
            {"type": "object",
             "properties": {"ok": {"type": "boolean"},
                            "c": {"type": "string", "enum": ["r", "g"]}},
             "required": ["ok", "c"]},
            id="two-properties",
        ),
    ],
)
def test_json_schema_grammar_round_trip(chat_base_url, schema):
    """A converted JSON Schema must actually drive the real sampler.

    Only bounded schemas are used: an unbounded integer lets SmolLM2-360M
    emit digits until it hits the token cap, which is a model-size artefact
    rather than a grammar defect.
    """
    gbnf = SchemaConverter().convert(schema)
    assert GBNFParser.validate(gbnf) == [], gbnf

    data = _completion(
        chat_base_url,
        prompt="Emit the object:",
        n_predict=120,
        grammar=gbnf,
        temperature=0.0,
    )
    parsed = json.loads(data["content"])
    assert set(schema["required"]) <= set(parsed)


@pytest.mark.live
def test_native_json_schema_param_is_supported(chat_base_url):
    """llama-server also accepts `json_schema` directly on /completion."""
    data = _completion(
        chat_base_url,
        prompt="Emit an object with an integer field n:",
        n_predict=32,
        json_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        temperature=0.0,
    )
    parsed = json.loads(data["content"])
    assert isinstance(parsed["n"], int)


@pytest.mark.live
def test_invalid_grammar_is_rejected_by_server(chat_base_url):
    """A malformed GBNF is a 400, and our validator agrees it is bad."""
    import urllib.error

    bad = "root ::= [0-9"
    assert GBNFParser.validate(bad), "validator should flag the unclosed class"

    with pytest.raises(urllib.error.HTTPError) as exc:
        _completion(chat_base_url, prompt="hi", n_predict=4, grammar=bad)
    assert exc.value.code == 400
    body = json.loads(exc.value.read().decode())
    assert body["error"]["type"] == "invalid_request_error"
