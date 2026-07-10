# 6.4 Enterprise MCP Gateway

A middleware layer for MCP servers with authentication, rate-limiting, request validation, and audit logging.

## Architecture

```
  Request ──▶ Auth ──▶ Rate Limit ──▶ Validate ──▶ Handler
                 │          │             │
                 ▼          ▼             ▼
              AuditLog ──────────────────────▶ Summary Report
```

**Components:**

| Component | Description |
|-----------|-------------|
| `ApiKeyAuthProvider` | Token-based auth with per-tool permissions |
| `RateLimiter` | Sliding-window rate limiter per client |
| `RequestValidator` | Checks required params, types, and lengths |
| `AuditLog` | Append-only log with denial tracking |
| `EnterpriseGateway` | Chains all checks, wraps handlers |

## Usage

```python
from gateway.core import (
    ApiKeyAuthProvider, EnterpriseGateway, GatewayConfig,
    RateLimiter, RequestValidator, ToolSpec,
)

auth = ApiKeyAuthProvider()
auth.add_key("sk-admin", allowed_tasks={"read", "write"})
auth.add_key("sk-readonly", allowed_tasks={"read"})

limiter = RateLimiter()
limiter.set_limit("alice", max_requests=100, window_seconds=60)

validator = RequestValidator()
validator.add_spec(ToolSpec(name="write_file", required_params={"path", "content"}))

gw = EnterpriseGateway(GatewayConfig(
    auth_provider=auth,
    rate_limiter=limiter,
    request_validator=validator,
))

# Check a request
result = gw.check_request("write_file", {"path": "/tmp", "content": "data"},
                          client_id="alice", token="sk-admin")
print(result.allowed)  # True

# Or wrap an existing handler
@mcp.tool()
def read_file(path: str) -> str:
    result = gw.check_request("read_file", {"path": path},
                               client_id="anonymous", token="sk-readonly")
    if not result.allowed:
        return f"Error {result.status_code}: {result.reason}"
    # ... actual logic ...
```

## Files

| File | Purpose |
|------|---------|
| `gateway/core.py` | All components: auth, rate-limiter, audit-log, validator, gateway |
| `tests/` | 26 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
