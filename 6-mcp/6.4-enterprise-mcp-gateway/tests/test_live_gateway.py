"""Live integration tests for the enterprise MCP gateway.

The gateway is a pure policy layer — auth, rate limiting, validation, audit —
so the offline tests wrap trivial lambdas. These tests wrap a handler that
makes a REAL llama-server chat call, which is what the gateway exists to
protect. That makes the deny paths meaningful: a denied request must never
reach the upstream model, and an allowed one must return the model's output
unchanged.

Real calls are kept to a minimum (max_tokens=8, and the deny-path tests make
no call at all by design).

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from gateway.core import (
    ApiKeyAuthProvider,
    AuditLog,
    EnterpriseGateway,
    GatewayConfig,
    RateLimiter,
    RequestValidator,
    ToolSpec,
)


@pytest.fixture
def llm_tool(chat_base_url, chat_model):
    """A tool handler that really calls llama-server, counting its invocations."""
    calls: list[str] = []

    def ask_model(prompt: str = "Say hi") -> str:
        calls.append(prompt)
        payload = {
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
        }
        req = urllib.request.Request(
            f"{chat_base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]

    ask_model.calls = calls
    return ask_model


# ---------------------------------------------------------------------------
# Allowed path — the real upstream is reached
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_allowed_request_reaches_the_model(llm_tool):
    """An authorised call passes through and returns real model output."""
    auth = ApiKeyAuthProvider()
    auth.add_key("sk-live-test", {"ask_model"})
    gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

    wrapped = gw.wrap_handler(llm_tool, "ask_model")
    out = wrapped(prompt="Say hi", _client_id="tester", _token="sk-live-test")

    assert len(llm_tool.calls) == 1, "handler must be invoked exactly once"
    assert not out.startswith("Error "), out
    assert out.strip(), "real model returned empty content"

    entries = gw.config.audit_log.recent()
    assert len(entries) == 1
    assert entries[0].allowed is True
    assert entries[0].tool_name == "ask_model"


@pytest.mark.live
def test_wrapped_handler_strips_gateway_kwargs(llm_tool):
    """_client_id/_token must not be forwarded to the handler.

    ask_model() takes only `prompt`, so a leaked kwarg would raise TypeError.
    """
    gw = EnterpriseGateway(GatewayConfig(require_auth=False))
    wrapped = gw.wrap_handler(llm_tool, "ask_model")

    out = wrapped(prompt="Say hi", _client_id="tester", _token="ignored")
    assert not out.startswith("Error "), out
    assert len(llm_tool.calls) == 1


# ---------------------------------------------------------------------------
# Deny paths — the real upstream must NOT be reached
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_bad_token_never_reaches_the_model(llm_tool):
    auth = ApiKeyAuthProvider()
    auth.add_key("sk-live-test", {"ask_model"})
    gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

    wrapped = gw.wrap_handler(llm_tool, "ask_model")
    out = wrapped(prompt="Say hi", _client_id="attacker", _token="sk-wrong")

    assert out.startswith("Error 401:")
    assert llm_tool.calls == [], "denied request must not hit llama-server"
    assert gw.config.audit_log.total_denied == 1


@pytest.mark.live
def test_token_not_authorized_for_tool(llm_tool):
    """A valid key scoped to a different tool is still denied."""
    auth = ApiKeyAuthProvider()
    auth.add_key("sk-live-test", {"some_other_tool"})
    gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

    wrapped = gw.wrap_handler(llm_tool, "ask_model")
    out = wrapped(prompt="Say hi", _client_id="tester", _token="sk-live-test")

    assert out.startswith("Error 401:")
    assert llm_tool.calls == []


@pytest.mark.live
def test_rate_limit_stops_the_second_call(llm_tool):
    """One real call gets through; the next is refused before the network."""
    limiter = RateLimiter()
    limiter.set_limit("tester", max_requests=1, window_seconds=60)
    gw = EnterpriseGateway(
        GatewayConfig(rate_limiter=limiter, require_auth=False)
    )
    wrapped = gw.wrap_handler(llm_tool, "ask_model")

    first = wrapped(prompt="Say hi", _client_id="tester")
    second = wrapped(prompt="Say hi", _client_id="tester")

    assert not first.startswith("Error ")
    assert second.startswith("Error 429:")
    assert len(llm_tool.calls) == 1, "rate-limited call must not hit llama-server"


@pytest.mark.live
def test_validation_failure_stops_the_call(llm_tool):
    """A request failing its ToolSpec never reaches the model."""
    validator = RequestValidator()
    validator.add_spec(
        ToolSpec(
            name="ask_model",
            required_params={"prompt"},
            param_types={"prompt": str},
        )
    )
    gw = EnterpriseGateway(
        GatewayConfig(request_validator=validator, require_auth=False)
    )
    wrapped = gw.wrap_handler(llm_tool, "ask_model")

    missing = wrapped(_client_id="tester")
    assert missing.startswith("Error 400:")

    wrong_type = wrapped(prompt=1234, _client_id="tester")
    assert wrong_type.startswith("Error 400:")

    assert llm_tool.calls == []


# ---------------------------------------------------------------------------
# Audit over a mixed real/denied workload
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_audit_summary_over_mixed_traffic(llm_tool):
    """One allowed real call plus two denials, correctly accounted."""
    auth = ApiKeyAuthProvider()
    auth.add_key("sk-live-test", {"ask_model"})
    gw = EnterpriseGateway(GatewayConfig(auth_provider=auth, audit_log=AuditLog()))
    wrapped = gw.wrap_handler(llm_tool, "ask_model")

    wrapped(prompt="Say hi", _client_id="good", _token="sk-live-test")
    wrapped(prompt="Say hi", _client_id="bad", _token="sk-nope")
    wrapped(prompt="Say hi", _client_id="bad", _token="sk-also-nope")

    log = gw.config.audit_log
    assert log.total_requests == 3
    assert log.total_denied == 2
    assert len(llm_tool.calls) == 1

    summary = gw.audit_summary()
    assert "## Gateway Audit Summary" in summary
    assert "Total requests: 3" in summary
    assert "Total denied: 2" in summary

    denied_clients = {e.client_id for e in log.denied()}
    assert denied_clients == {"bad"}
