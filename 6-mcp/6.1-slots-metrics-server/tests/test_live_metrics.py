"""Live integration tests for the metrics MCP server.

Two distinct things are covered here, and it is worth being explicit about the
difference:

1. `MetricsCollector` with psutil against the REAL machine. The offline tests
   run it with `use_simulated=True`, which returns `random.uniform()` values,
   so the psutil branch of `collect()` is barely exercised there.

2. The llama-server observability endpoints this project is named after
   (`/slots`, `/metrics`, `/props`). NOTE: `metrics_server` does not call
   llama-server at all — it reports host CPU/memory/disk via psutil. These
   tests therefore assert the real endpoint contract rather than project code,
   so that the capability gap is recorded rather than assumed. See
   drift-graph.md entries 7-8.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from metrics_server.collector import (
    CpuMetrics,
    DiskMetrics,
    MemoryMetrics,
    MetricsCollector,
    SystemMetrics,
    format_metrics,
)
from metrics_server.server import (
    get_cpu_usage,
    get_disk_usage,
    get_memory_usage,
    get_system_metrics,
)

psutil = pytest.importorskip("psutil", reason="live collector tests need psutil")


def _get(url: str, timeout: float = 10.0):
    """GET returning (status, decoded_body). HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except (json.JSONDecodeError, OSError):
            return exc.code, None


# ---------------------------------------------------------------------------
# 1. Real host metrics through psutil
# ---------------------------------------------------------------------------


@pytest.fixture
def real_collector():
    collector = MetricsCollector(use_simulated=False)
    if collector._psutil is None:
        pytest.skip("psutil not importable inside the collector")
    return collector


@pytest.mark.live
def test_real_cpu_metrics(real_collector):
    cpu = real_collector.get_cpu()
    assert isinstance(cpu, CpuMetrics)
    assert 0.0 <= cpu.percent <= 100.0
    assert cpu.cores >= 1
    assert len(cpu.load_avg) == 3
    assert all(x >= 0 for x in cpu.load_avg)


@pytest.mark.live
def test_real_memory_metrics(real_collector):
    mem = real_collector.get_memory()
    assert isinstance(mem, MemoryMetrics)
    assert mem.total_gb > 0
    assert 0 <= mem.used_gb <= mem.total_gb
    assert 0.0 <= mem.percent <= 100.0


@pytest.mark.live
def test_real_disk_metrics(real_collector):
    disk = real_collector.get_disk()
    assert isinstance(disk, DiskMetrics)
    assert disk.total_gb > 0
    assert 0.0 <= disk.percent <= 100.0
    # used + free need not equal total (reserved blocks), so bound loosely.
    assert disk.used_gb <= disk.total_gb
    assert disk.free_gb <= disk.total_gb


@pytest.mark.live
def test_real_collect_and_format(real_collector):
    metrics = real_collector.collect()
    assert isinstance(metrics, SystemMetrics)
    assert metrics.timestamp > 0

    out = format_metrics(metrics)
    assert "## System Metrics" in out
    assert "### CPU" in out
    assert "### Memory" in out
    assert "### Disk" in out


@pytest.mark.live
def test_mcp_tools_against_real_host():
    """The four exported MCP tools, backed by the real module-level collector."""
    full = get_system_metrics()
    assert "### CPU" in full and "### Memory" in full and "### Disk" in full

    assert "CPU:" in get_cpu_usage()
    assert "Memory:" in get_memory_usage()
    assert "GB" in get_disk_usage("/")


# ---------------------------------------------------------------------------
# 2. llama-server observability endpoints
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_llamacpp_slots_endpoint_shape(chat_base_url):
    """/slots returns a LIST of slot objects, not a dict.

    Anything keying into a dict (e.g. body["slots"]) would break against the
    real server. Recorded as drift-graph.md entry 7.
    """
    status, body = _get(f"{chat_base_url}/slots")
    assert status == 200
    assert isinstance(body, list), f"/slots must be a list, got {type(body).__name__}"
    assert body, "at least one slot must be reported"

    slot = body[0]
    assert {"id", "n_ctx", "speculative", "is_processing"} <= set(slot)
    assert slot["id"] == 0
    assert slot["n_ctx"] == 2048
    assert isinstance(slot["is_processing"], bool)
    assert isinstance(slot["speculative"], bool)


@pytest.mark.live
def test_llamacpp_props_reports_capabilities(chat_base_url):
    """/props is the authoritative source for which endpoints are enabled."""
    status, props = _get(f"{chat_base_url}/props")
    assert status == 200

    assert props["endpoint_slots"] is True
    assert props["endpoint_metrics"] is False, "server started without --metrics"
    assert props["total_slots"] == 3
    assert props["build_info"] == "b11046-60081bb2b"


@pytest.mark.live
def test_llamacpp_slot_count_matches_props(chat_base_url):
    status, props = _get(f"{chat_base_url}/props")
    _, slots = _get(f"{chat_base_url}/slots")
    assert len(slots) == props["total_slots"]


@pytest.mark.live
@pytest.mark.xfail(
    reason=(
        "unsupported: llama-server was started without --metrics, so /metrics "
        "returns 501 not_supported_error and /props reports "
        "endpoint_metrics=false. Enabling it requires restarting the server "
        "with --metrics; nothing in this project can work around it."
    ),
    strict=True,
)
def test_llamacpp_metrics_endpoint(chat_base_url):
    """Prometheus /metrics is not available on this server.

    strict=True so this starts failing (XPASS) the moment someone restarts
    llama-server with --metrics, prompting a real assertion here.
    """
    status, _ = _get(f"{chat_base_url}/metrics")
    assert status == 200


@pytest.mark.live
def test_llamacpp_metrics_really_is_501(chat_base_url):
    """The positive assertion of the gap above: a 501 with the real error shape."""
    status, body = _get(f"{chat_base_url}/metrics")
    assert status == 501
    assert body["error"]["code"] == 501
    assert body["error"]["type"] == "not_supported_error"
    assert "--metrics" in body["error"]["message"]
