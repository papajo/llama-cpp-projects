"""Enterprise MCP gateway — auth, rate-limit, audit, validation middleware."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ── Auth ───────────────────────────────────────────────────────

AuthResult = Tuple[bool, str]  # (allowed, reason)


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, token: str, tool_name: str) -> AuthResult:
        """Check if a token is allowed to call a tool."""
        ...


@dataclass
class ApiKeyAuthProvider(AuthProvider):
    """Simple API-key auth with per-tool permissions."""

    valid_keys: Dict[str, Set[str]] = field(default_factory=dict)

    def add_key(self, key: str, allowed_tools: Optional[Set[str]] = None):
        self.valid_keys[key] = allowed_tools or set()

    def authenticate(self, token: str, tool_name: str) -> AuthResult:
        if token not in self.valid_keys:
            return False, "Invalid API key"
        allowed = self.valid_keys[token]
        if allowed and tool_name not in allowed:
            return False, f"Key not authorized for tool '{tool_name}'"
        return True, ""


@dataclass
class AllowAllAuthProvider(AuthProvider):
    def authenticate(self, token: str, tool_name: str) -> AuthResult:
        return True, ""


# ── Rate Limiting ──────────────────────────────────────────────


@dataclass
class RateLimit:
    max_requests: int
    window_seconds: float


@dataclass
class RateLimiter:
    """Sliding-window rate limiter per client token."""

    limits: Dict[str, RateLimit] = field(default_factory=dict)
    _history: Dict[str, List[float]] = field(default_factory=dict)

    def set_limit(self, client_id: str, max_requests: int, window_seconds: float):
        self.limits[client_id] = RateLimit(max_requests, window_seconds)

    def check(self, client_id: str) -> Tuple[bool, str]:
        """Check if client_id can make a request. Returns (allowed, reason)."""
        limit = self.limits.get(client_id)
        if limit is None:
            return True, ""  # no limit set

        now = time.time()
        history = self._history.setdefault(client_id, [])
        # Prune old entries
        cutoff = now - limit.window_seconds
        self._history[client_id] = [t for t in history if t > cutoff]

        if limit.max_requests <= 0:
            return False, "Rate limit exceeded (no requests allowed)"

        if len(self._history[client_id]) >= limit.max_requests:
            retry_after = int(self._history[client_id][0] + limit.window_seconds - now)
            return False, f"Rate limit exceeded. Retry after {retry_after}s"

        self._history[client_id].append(now)
        return True, ""


# ── Audit Log ──────────────────────────────────────────────────


@dataclass
class AuditEntry:
    timestamp: float
    client_id: str
    tool_name: str
    params: Dict[str, Any]
    allowed: bool
    reason: str = ""


@dataclass
class AuditLog:
    """Append-only audit log."""

    entries: List[AuditEntry] = field(default_factory=list)
    max_entries: int = 10_000

    def record(self, client_id: str, tool_name: str,
               params: Dict[str, Any], allowed: bool, reason: str = ""):
        entry = AuditEntry(
            timestamp=time.time(),
            client_id=client_id,
            tool_name=tool_name,
            params=params,
            allowed=allowed,
            reason=reason,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def recent(self, n: int = 10) -> List[AuditEntry]:
        return self.entries[-n:]

    def denied(self) -> List[AuditEntry]:
        return [e for e in self.entries if not e.allowed]

    @property
    def total_requests(self) -> int:
        return len(self.entries)

    @property
    def total_denied(self) -> int:
        return len(self.denied())


# ── Request Validation ─────────────────────────────────────────

ToolHandler = Callable[..., Any]


@dataclass
class ToolSpec:
    name: str
    required_params: Set[str] = field(default_factory=set)
    param_types: Dict[str, type] = field(default_factory=dict)
    max_param_length: int = 10_000


@dataclass
class RequestValidator:
    """Validates tool requests against specs."""

    specs: Dict[str, ToolSpec] = field(default_factory=dict)

    def add_spec(self, spec: ToolSpec):
        self.specs[spec.name] = spec

    def validate(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        spec = self.specs.get(tool_name)
        if spec is None:
            return True, ""  # unknown tools pass through

        for required in spec.required_params:
            if required not in params:
                return False, f"Missing required parameter '{required}'"

        for key, value in params.items():
            if key in spec.param_types:
                if not isinstance(value, spec.param_types[key]):
                    return False, (
                        f"Parameter '{key}' should be {spec.param_types[key].__name__}"
                    )
            if isinstance(value, str) and len(value) > spec.max_param_length:
                return False, f"Parameter '{key}' exceeds max length {spec.max_param_length}"

        return True, ""


# ── Gateway ────────────────────────────────────────────────────


@dataclass
class GatewayConfig:
    auth_provider: AuthProvider = field(default_factory=AllowAllAuthProvider)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    audit_log: AuditLog = field(default_factory=AuditLog)
    request_validator: RequestValidator = field(default_factory=RequestValidator)
    require_auth: bool = True


@dataclass
class GatewayResult:
    allowed: bool
    reason: str = ""
    status_code: int = 200


class EnterpriseGateway:
    """Middleware chain for MCP tools.

    On each tool request:
      1. Auth — validate token against AuthProvider
      2. Rate limit — check RateLimiter
      3. Validate — check params against RequestValidator
      4. Audit — record result in AuditLog
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()

    def check_request(
        self,
        tool_name: str,
        params: Dict[str, Any],
        client_id: str = "anonymous",
        token: str = "",
    ) -> GatewayResult:
        """Run all checks. Returns GatewayResult with allowed bool."""

        # 1. Auth
        #
        # A missing token must be denied, not waved through. Gating this block
        # on `and token` meant an unauthenticated caller bypassed auth entirely
        # while a caller presenting a BAD token was correctly rejected - i.e.
        # supplying no credential was safer than supplying a wrong one.
        if self.config.require_auth:
            if not token:
                reason = "Missing authentication token"
                self.config.audit_log.record(
                    client_id, tool_name, params, False, reason
                )
                return GatewayResult(allowed=False, reason=reason, status_code=401)
            allowed, reason = self.config.auth_provider.authenticate(token, tool_name)
            if not allowed:
                self.config.audit_log.record(
                    client_id, tool_name, params, False, reason
                )
                return GatewayResult(allowed=False, reason=reason, status_code=401)

        # 2. Rate limit
        allowed, reason = self.config.rate_limiter.check(client_id)
        if not allowed:
            self.config.audit_log.record(
                client_id, tool_name, params, False, reason
            )
            return GatewayResult(allowed=False, reason=reason, status_code=429)

        # 3. Validate
        allowed, reason = self.config.request_validator.validate(tool_name, params)
        if not allowed:
            self.config.audit_log.record(
                client_id, tool_name, params, False, reason
            )
            return GatewayResult(allowed=False, reason=reason, status_code=400)

        # 4. All passed
        self.config.audit_log.record(
            client_id, tool_name, params, True, "OK"
        )
        return GatewayResult(allowed=True)

    def wrap_handler(
        self, handler: ToolHandler, tool_name: str
    ) -> ToolHandler:
        """Wrap a tool handler with gateway checks."""
        def wrapped(**kwargs: Any) -> str:
            result = self.check_request(
                tool_name=tool_name,
                params=kwargs,
                client_id=kwargs.pop("_client_id", "anonymous"),
                token=kwargs.pop("_token", ""),
            )
            if not result.allowed:
                return f"Error {result.status_code}: {result.reason}"
            return handler(**kwargs)
        wrapped.__name__ = handler.__name__
        return wrapped

    def audit_summary(self) -> str:
        log = self.config.audit_log
        lines = [
            "## Gateway Audit Summary",
            f"- Total requests: {log.total_requests}",
            f"- Total denied: {log.total_denied}",
            f"- Denial rate: {(log.total_denied / max(log.total_requests, 1)) * 100:.1f}%",
            "",
        ]
        denied = log.denied()
        if denied:
            lines.append("### Recent Denials")
            for entry in denied[-5:]:
                lines.append(
                    f"- [{entry.tool_name}] {entry.client_id}: {entry.reason}"
                )
        return "\n".join(lines)
