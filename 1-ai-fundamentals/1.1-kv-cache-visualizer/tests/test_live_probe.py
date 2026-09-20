"""Live integration tests for ServerProbe against a real llama-server.

The offline suite never exercises probe.py at all, so every field it reads
out of /slots, /props and /metrics is unverified until it meets a real
server. Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

# The package directory is `server-probe`; the hyphen makes it unimportable
# by name, so load the module straight off disk.
_PROBE_PATH = Path(__file__).resolve().parent.parent / "server-probe" / "probe.py"
_spec = importlib.util.spec_from_file_location("_live_probe", _PROBE_PATH)
probe_mod = importlib.util.module_from_spec(_spec)
# @dataclass resolves annotations via sys.modules[cls.__module__], so the
# module has to be registered before it is executed.
sys.modules[_spec.name] = probe_mod
_spec.loader.exec_module(probe_mod)

ServerProbe = probe_mod.ServerProbe
ServerMetrics = probe_mod.ServerMetrics
SlotInfo = probe_mod.SlotInfo


@pytest.fixture
def probe(chat_base_url):
    p = ServerProbe(chat_base_url)
    yield p
    p.close()


@pytest.mark.live
def test_get_slots_parses_real_payload(probe):
    """Every slot the real server reports round-trips through SlotInfo."""
    slots = probe.get_slots()
    assert slots, "llama-server should expose at least one slot"
    for s in slots:
        assert isinstance(s, SlotInfo)
        assert s.id >= 0
        # The bug this covers: `state` used to be read from a key the
        # current server never sends, leaving every slot "unknown".
        assert s.state in {"idle", "processing"}
        assert isinstance(s.n_prompt_tokens, int)
        assert isinstance(s.n_decoded, int)


@pytest.mark.live
def test_slot_state_partitions_cleanly(probe):
    """active + idle must account for every slot (regression guard)."""
    slots = probe.get_slots()
    active = probe.get_active_slots()
    idle = probe.get_idle_slots()
    assert len(active) + len(idle) == len(slots)


@pytest.mark.live
def test_slot_n_ctx_matches_server(probe, live_server_meta):
    """Slot context size agrees with the model metadata."""
    raw = probe._client.get(f"{probe.base_url}/slots").json()
    n_ctx_meta = live_server_meta["chat"]["data"][0]["meta"]["n_ctx"]
    for slot in raw:
        assert slot["n_ctx"] == n_ctx_meta


@pytest.mark.live
def test_get_props_returns_generation_settings(probe):
    """/props carries the fields the visualiser reads."""
    props = probe.get_props()
    assert "default_generation_settings" in props
    assert "total_slots" in props
    params = props["default_generation_settings"]["params"]
    assert "temperature" in params
    assert "top_k" in params


@pytest.mark.live
def test_metrics_endpoint_matches_deployment(probe):
    """/metrics is gated on the --metrics start flag; assert whichever is true.

    This originally pinned the 501 because the server ran without --metrics.
    That is a deployment choice, so when the flag is on we exercise the
    Prometheus parser for real instead, and when it is off we still assert the
    documented 501.
    """
    if probe.get_props()["endpoint_metrics"]:
        metrics = probe.get_metrics()
        # ServerMetrics.from_prometheus parsed a real exposition payload.
        assert metrics.kv_cache_usage_ratio >= 0.0
        assert metrics.tokens_per_second >= 0.0
    else:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            probe.get_metrics()
        assert exc.value.response.status_code == 501
        assert "--metrics" in exc.value.response.text


@pytest.mark.live
def test_report_survives_either_metrics_deployment(probe):
    """report() returns slots+props whether or not /metrics is enabled."""
    report = probe.report()
    assert report["slots"]
    assert report["server"] == probe.base_url
    # Slot counts come from /slots and must still partition correctly.
    assert report["total_slots"] == len(report["slots"])
    assert report["active_slots"] + report["idle_slots"] == report["total_slots"]
    # The Prometheus-derived block is present either way: parsed values when
    # --metrics is on, the zero-valued default when the 501 was swallowed.
    assert report["metrics"]["kv_cache_usage_ratio"] >= 0.0
    assert report["metrics"]["tokens_per_second"] >= 0.0
    # /props still succeeded, so it is populated.
    assert report["props"]["total_slots"] == report["total_slots"]
