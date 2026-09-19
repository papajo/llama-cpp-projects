"""Tests for enterprise gateway — auth, rate-limit, audit, validation."""

import time

import pytest

from gateway.core import (
    AllowAllAuthProvider,
    ApiKeyAuthProvider,
    AuditLog,
    EnterpriseGateway,
    GatewayConfig,
    GatewayResult,
    RateLimiter,
    RequestValidator,
    ToolSpec,
)


class TestApiKeyAuth:
    def test_valid_key(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-123", allowed_tools={"read_file", "write_file"})
        ok, reason = auth.authenticate("sk-123", "read_file")
        assert ok is True

    def test_invalid_key(self):
        auth = ApiKeyAuthProvider()
        ok, reason = auth.authenticate("bad-key", "read_file")
        assert ok is False
        assert "Invalid" in reason

    def test_key_not_authorized_for_tool(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-123", allowed_tools={"read_file"})
        ok, reason = auth.authenticate("sk-123", "delete_all")
        assert ok is False
        assert "not authorized" in reason

    def test_empty_allowed_set_means_all(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-123")  # no restrictions
        ok, reason = auth.authenticate("sk-123", "any_tool")
        assert ok is True

    def test_allow_all(self):
        auth = AllowAllAuthProvider()
        ok, reason = auth.authenticate("anything", "any_tool")
        assert ok is True


class TestRateLimiter:
    def test_no_limit_set(self):
        limiter = RateLimiter()
        ok, reason = limiter.check("client_1")
        assert ok is True

    def test_under_limit(self):
        limiter = RateLimiter()
        limiter.set_limit("client_1", 5, 60)
        for _ in range(3):
            ok, reason = limiter.check("client_1")
            assert ok is True

    def test_over_limit(self):
        limiter = RateLimiter()
        limiter.set_limit("client_1", 2, 60)
        assert limiter.check("client_1")[0] is True
        assert limiter.check("client_1")[0] is True
        ok, reason = limiter.check("client_1")
        assert ok is False
        assert "Rate limit" in reason

    def test_per_client(self):
        limiter = RateLimiter()
        limiter.set_limit("client_a", 1, 60)
        limiter.set_limit("client_b", 1, 60)
        assert limiter.check("client_a")[0] is True
        assert limiter.check("client_b")[0] is True  # different client, not blocked
        assert limiter.check("client_a")[0] is False  # a is now blocked


class TestAuditLog:
    def test_record_and_count(self):
        log = AuditLog()
        log.record("client1", "read_file", {"path": "/tmp"}, True, "OK")
        assert log.total_requests == 1
        assert log.total_denied == 0

    def test_denied_count(self):
        log = AuditLog()
        log.record("c1", "tool", {}, True, "OK")
        log.record("c1", "tool", {}, False, "not allowed")
        assert log.total_denied == 1

    def test_recent(self):
        log = AuditLog()
        for i in range(20):
            log.record(f"c{i}", "tool", {}, True)
        recent = log.recent(5)
        assert len(recent) == 5

    def test_denied_filter(self):
        log = AuditLog()
        log.record("c1", "tool", {}, False, "auth fail")
        log.record("c2", "tool", {}, True)
        denied = log.denied()
        assert len(denied) == 1
        assert denied[0].client_id == "c1"


class TestRequestValidator:
    def test_missing_required(self):
        validator = RequestValidator()
        validator.add_spec(ToolSpec(
            name="write_file",
            required_params={"path", "content"},
        ))
        ok, reason = validator.validate("write_file", {"path": "/tmp"})
        assert ok is False
        assert "Missing required" in reason

    def test_wrong_param_type(self):
        validator = RequestValidator()
        validator.add_spec(ToolSpec(
            name="repeat",
            param_types={"count": int},
        ))
        ok, reason = validator.validate("repeat", {"count": "not_int"})
        assert ok is False
        assert "should be int" in reason

    def test_param_too_long(self):
        validator = RequestValidator()
        validator.add_spec(ToolSpec(name="echo", max_param_length=5))
        ok, reason = validator.validate("echo", {"msg": "hello world"})
        assert ok is False
        assert "exceeds max length" in reason

    def test_unknown_tool_passes(self):
        validator = RequestValidator()
        ok, reason = validator.validate("unknown", {"anything": 1})
        assert ok is True

    def test_valid_request(self):
        validator = RequestValidator()
        validator.add_spec(ToolSpec(name="add", param_types={"x": int, "y": int}))
        ok, reason = validator.validate("add", {"x": 1, "y": 2})
        assert ok is True


class TestEnterpriseGateway:
    def test_auth_failure(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-valid")
        gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))
        result = gw.check_request(
            tool_name="read_file",
            params={"path": "/tmp"},
            client_id="alice",
            token="sk-invalid",
        )
        assert result.allowed is False
        assert result.status_code == 401

    # ── Auth bypass regression (see drift-graph.md, "Real bugs found") ──
    #
    # `check_request` used to gate its auth block on `require_auth and token`,
    # so an empty token skipped authentication entirely: presenting NO
    # credential was safer than presenting a wrong one. These four tests pin
    # the whole truth table.

    def test_require_auth_denies_empty_token(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-valid")
        gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

        result = gw.check_request(
            tool_name="read_file",
            params={"path": "/tmp"},
            client_id="anonymous",
            token="",
        )

        assert result.allowed is False
        assert result.status_code == 401
        assert result.reason == "Missing authentication token"

        # The denial is audited exactly like any other auth failure.
        assert gw.config.audit_log.total_requests == 1
        assert gw.config.audit_log.total_denied == 1
        entry = gw.config.audit_log.denied()[0]
        assert entry.client_id == "anonymous"
        assert entry.tool_name == "read_file"
        assert entry.allowed is False

    def test_require_auth_denies_missing_token_argument(self):
        """`token` defaults to "", so omitting it entirely must also deny."""
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-valid")
        gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

        result = gw.check_request("read_file", {"path": "/tmp"}, "anonymous")

        assert result.allowed is False
        assert result.status_code == 401

    def test_require_auth_allows_valid_token(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-valid")
        gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

        result = gw.check_request(
            tool_name="read_file",
            params={"path": "/tmp"},
            client_id="alice",
            token="sk-valid",
        )

        assert result.allowed is True
        assert result.status_code == 200
        assert gw.config.audit_log.total_denied == 0

    def test_require_auth_denies_invalid_token(self):
        auth = ApiKeyAuthProvider()
        auth.add_key("sk-valid")
        gw = EnterpriseGateway(GatewayConfig(auth_provider=auth))

        result = gw.check_request(
            tool_name="read_file",
            params={"path": "/tmp"},
            client_id="alice",
            token="sk-wrong",
        )

        assert result.allowed is False
        assert result.status_code == 401
        assert result.reason == "Invalid API key"

    def test_no_require_auth_allows_empty_token(self):
        """require_auth=False stays an open passthrough."""
        gw = EnterpriseGateway(GatewayConfig(require_auth=False))

        result = gw.check_request("read_file", {"path": "/tmp"}, "anonymous", "")

        assert result.allowed is True
        assert result.status_code == 200
        assert gw.config.audit_log.total_denied == 0

    def test_rate_limit(self):
        limiter = RateLimiter()
        limiter.set_limit("bob", 1, 60)
        gw = EnterpriseGateway(GatewayConfig(rate_limiter=limiter, require_auth=False))
        result1 = gw.check_request("tool", {}, "bob")
        assert result1.allowed is True
        result2 = gw.check_request("tool", {}, "bob")
        assert result2.allowed is False
        assert result2.status_code == 429

    def test_validation_failure(self):
        validator = RequestValidator()
        validator.add_spec(ToolSpec(name="write", required_params={"data"}))
        gw = EnterpriseGateway(GatewayConfig(
            request_validator=validator,
            require_auth=False,
        ))
        result = gw.check_request("write", {"path": "/tmp"})
        assert result.allowed is False
        assert result.status_code == 400

    def test_all_passed(self):
        gw = EnterpriseGateway(GatewayConfig(require_auth=False))
        result = gw.check_request("any_tool", {"x": 1}, "client_1")
        assert result.allowed is True
        assert result.status_code == 200

    def test_audit_records_all(self):
        gw = EnterpriseGateway(GatewayConfig(require_auth=False))
        gw.check_request("t1", {}, "c1")
        gw.check_request("t2", {}, "c2")
        gw.check_request("t3", {}, "c3")
        assert gw.config.audit_log.total_requests == 3

    def test_wrap_handler_allowed(self):
        def handler(x: int) -> str:
            return f"Result: {x}"

        gw = EnterpriseGateway(GatewayConfig(require_auth=False))
        wrapped = gw.wrap_handler(handler, "test_tool")
        result = wrapped(x=42, _client_id="c1")
        assert result == "Result: 42"

    def test_wrap_handler_denied(self):
        limiter = RateLimiter()
        limiter.set_limit("c1", 0, 60)  # always blocked
        gw = EnterpriseGateway(GatewayConfig(rate_limiter=limiter, require_auth=False))

        def handler(x: int) -> str:
            return "should not reach"

        wrapped = gw.wrap_handler(handler, "test_tool")
        result = wrapped(x=42, _client_id="c1")
        assert "Error 429" in result

    def test_audit_summary(self):
        gw = EnterpriseGateway(GatewayConfig(require_auth=False))
        gw.check_request("t1", {}, "c1")
        gw.check_request("t2", {}, "c2")
        summary = gw.audit_summary()
        assert "Total requests: 2" in summary
        assert "Total denied: 0" in summary
