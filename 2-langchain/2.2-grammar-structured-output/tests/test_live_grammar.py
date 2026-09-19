"""Live integration tests for 2.2 against the real chat server.

This is the project whose central claim is directly checkable against a real
server: llama.cpp's grammar constraint means the output is *guaranteed*
schema-conformant, with no retries. The mocked tests can only ever assert that
the parser parses whatever JSON the mock handed back -- they cannot demonstrate
the guarantee. These tests do.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Note that the guarantee is exactly what makes this testable with a 360M model:
SmolLM2 has no idea what it is doing, but the sampler physically cannot emit a
token outside the grammar, so the *shape* is always right even when the
*values* are nonsense. Field values are therefore never asserted -- only types,
keys, and schema conformance.
"""

from __future__ import annotations

import json
from typing import List, Literal, Optional

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from parser import GrammarOutputParser, GrammarStructuredOutput


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150)
    email: Optional[str] = None


class Review(BaseModel):
    rating: int = Field(ge=1, le=5)
    summary: str
    verified: bool = False


class Team(BaseModel):
    """Nested + list types, to push the grammar harder."""

    team_name: str
    members: List[Person]


class Ticket(BaseModel):
    """Enum-constrained field."""

    title: str
    priority: Literal["low", "medium", "high"]


@pytest.fixture
def parser(chat_base_url):
    # max_tokens well under n_ctx (2048); the grammar stops at the closing brace.
    p = GrammarOutputParser(base_url=chat_base_url, max_tokens=256)
    yield p
    p.close()


# ---------------------------------------------------------------------------
# The core guarantee
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_returns_a_valid_model_instance(parser):
    result = parser.invoke("Extract info: John Smith is 30 years old", schema=Person)
    assert isinstance(result, Person)
    assert isinstance(result.name, str)
    assert isinstance(result.age, int)
    assert 0 <= result.age <= 150


@pytest.mark.live
def test_json_schema_must_be_an_object_not_a_string(parser, chat_base_url):
    """Regression guard for the bug this live pass found.

    The parser used to send `json_schema` as `json.dumps(schema)`. A real
    llama-server rejects that with HTTP 400 -- so every structured-output call
    failed against a real server while the offline suite stayed green, because
    no mocked test inspected the request body. See drift-rag.md.
    """
    schema = parser.compile_schema(Person)
    client = httpx.Client(base_url=chat_base_url, timeout=60.0)
    try:
        # The old, broken form.
        bad = client.post(
            "/completion",
            json={"prompt": "x", "json_schema": json.dumps(schema), "n_predict": 8},
        )
        assert bad.status_code == 400
        assert "must be an object" in bad.json()["error"]["message"]

        # The corrected form.
        good = client.post(
            "/completion",
            json={"prompt": "x", "json_schema": schema, "n_predict": 32},
        )
        assert good.status_code == 200
    finally:
        client.close()


@pytest.mark.live
def test_output_is_valid_json_without_any_retry(parser):
    """Grammar-constrained output parses first time, every time.

    Five different prompts, no retry logic anywhere in the parser. If the
    grammar were not enforced, a 360M model would produce unparseable output
    for at least one of these.
    """
    prompts = [
        "Extract info: Alice is 41",
        "Extract info: nothing useful here at all",
        "Extract info: 12345",
        "Extract info: Bob Bobson, aged twenty",
    ]
    for prompt in prompts:
        result = parser.invoke(prompt, schema=Person)
        assert isinstance(result, Person), f"failed for {prompt!r}"
        assert isinstance(result.age, int)


@pytest.mark.live
def test_empty_prompt_generates_nothing_and_raises_clearly(parser):
    """The grammar constrains what IS generated; it does not force generation.

    With an empty prompt the real server returns `content: ""` with
    `tokens_predicted: 0` and `stop_type: "none"` -- it emits no tokens at all,
    so there is nothing for the grammar to make conformant. The parser's
    "should never happen" guard is what catches this, and it produces a clear
    error rather than a JSONDecodeError traceback. Documented rather than
    "fixed": raising here is the correct behaviour for a degenerate input.
    """
    with pytest.raises(ValueError, match="not valid JSON"):
        parser.invoke("", schema=Person)


@pytest.mark.live
def test_message_list_prompt_form(parser):
    result = parser.invoke(
        [{"role": "user", "content": "Bob is 25"}], schema=Person
    )
    assert isinstance(result, Person)


# ---------------------------------------------------------------------------
# Schema features, enforced by the real grammar
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_bool_and_int_types_are_respected(parser):
    result = parser.invoke(
        "Extract a review: five stars, loved it, verified buyer", schema=Review
    )
    assert isinstance(result, Review)
    assert isinstance(result.rating, int)
    assert 1 <= result.rating <= 5
    assert isinstance(result.verified, bool)
    assert isinstance(result.summary, str)


@pytest.mark.live
def test_enum_field_is_constrained_to_allowed_values(parser):
    """A Literal becomes a grammar alternation, so the value cannot be invalid.

    This holds even though the model has no real grasp of priority -- the
    sampler simply cannot emit anything else.
    """
    result = parser.invoke(
        "Extract a ticket: the server is on fire, extremely urgent",
        schema=Ticket,
    )
    assert isinstance(result, Ticket)
    assert result.priority in {"low", "medium", "high"}


@pytest.mark.live
def test_nested_model_with_a_list(parser):
    """Nested objects inside an array still come back conformant."""
    result = parser.invoke(
        "Extract the team: the Rockets, with Ann aged 30 and Bo aged 41",
        schema=Team,
    )
    assert isinstance(result, Team)
    assert isinstance(result.team_name, str)
    assert isinstance(result.members, list)
    for m in result.members:
        assert isinstance(m, Person)
        assert isinstance(m.age, int)


@pytest.mark.live
def test_optional_field_is_str_or_none(parser):
    result = parser.invoke("Extract info: Dana is 22, no email given", schema=Person)
    assert result.email is None or isinstance(result.email, str)


# ---------------------------------------------------------------------------
# The compiled schema itself
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_compiled_schema_is_accepted_by_the_server(parser, chat_base_url):
    """Every schema this project compiles must be one llama-server can convert.

    A schema the compiler happily produces but the server's GBNF converter
    rejects would fail only at runtime.
    """
    client = httpx.Client(base_url=chat_base_url, timeout=60.0)
    try:
        for model in (Person, Review, Team, Ticket):
            schema = parser.compile_schema(model)
            assert schema["type"] == "object"
            resp = client.post(
                "/completion",
                json={"prompt": "Extract: test", "json_schema": schema,
                      "n_predict": 128, "temperature": 0.0},
            )
            assert resp.status_code == 200, (
                f"server rejected the compiled schema for {model.__name__}: "
                f"{resp.text[:300]}"
            )
            # And the constrained output really is valid JSON.
            json.loads(resp.json()["content"])
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Native /completion response shape
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_native_completion_response_shape(parser, chat_base_url):
    """What /completion actually returns, versus the 5-key canned response."""
    client = httpx.Client(base_url=chat_base_url, timeout=60.0)
    try:
        resp = client.post(
            "/completion",
            json={
                "prompt": "Extract: Alice is 30",
                "json_schema": parser.compile_schema(Person),
                "n_predict": 64,
                "temperature": 0.0,
                "cache_prompt": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Keys the mock models.
        for key in ("content", "tokens_predicted", "tokens_evaluated", "model"):
            assert key in data, f"missing {key}"
        assert isinstance(data["content"], str)
        # Keys the mock omits entirely.
        assert "stop" in data
        assert "timings" in data
        assert "generation_settings" in data
        # The grammar shows up in the echoed settings, proving it was applied.
        assert data["generation_settings"]["grammar"]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# The chain wrapper
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_chain_formats_template_and_returns_model(chat_base_url):
    chain = GrammarStructuredOutput(
        schema=Person,
        prompt_template="Extract info: {input}",
        base_url=chat_base_url,
        max_tokens=256,
    )
    try:
        result = chain.invoke({"input": "John Smith is 30 years old"})
        assert isinstance(result, Person)
        assert isinstance(result.name, str)
    finally:
        chain.close()


@pytest.mark.live
def test_chain_with_system_prompt(chat_base_url):
    chain = GrammarStructuredOutput(
        schema=Review,
        prompt_template="Extract a review: {input}",
        base_url=chat_base_url,
        system_prompt="You extract structured data.",
        max_tokens=256,
    )
    try:
        result = chain.invoke({"input": "four stars, pretty good"})
        assert isinstance(result, Review)
        assert 1 <= result.rating <= 5
    finally:
        chain.close()


@pytest.mark.live
def test_chain_template_variables(chat_base_url):
    chain = GrammarStructuredOutput(
        schema=Person,
        prompt_template="Extract {what}: {input}",
        base_url=chat_base_url,
    )
    try:
        assert set(chain._get_template_variables()) == {"what", "input"}
    finally:
        chain.close()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_parse_and_parse_result_are_not_supported(parser):
    with pytest.raises(NotImplementedError):
        parser.parse("{}")
    with pytest.raises(NotImplementedError):
        parser.parse_result(["{}"])


@pytest.mark.live
def test_unreachable_server_raises(chat_base_url):
    p = GrammarOutputParser(base_url="http://127.0.0.1:1", request_timeout=5.0)
    try:
        with pytest.raises(httpx.HTTPError):
            p.invoke("Extract: test", schema=Person)
    finally:
        p.close()
