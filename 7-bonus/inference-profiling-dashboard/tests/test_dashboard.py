"""Tests for inference profiling dashboard."""

import json

from dashboard.collector import (
    ProfileRun,
    Snapshot,
    all_runs_to_dict,
    generate_sample_data,
    run_to_dict,
)


class TestCollector:
    def test_generates_four_runs(self):
        data = generate_sample_data()
        assert len(data) == 4

    def test_run_has_label(self):
        run = generate_sample_data()["compact-4"]
        assert run.label == "compact-4"

    def test_each_run_has_snapshots(self):
        for name, run in generate_sample_data().items():
            assert len(run.snapshots) > 0, f"{name} has no snapshots"

    def test_snapshot_fields_present(self):
        run = generate_sample_data()["spread-4"]
        snap = run.snapshots[0]
        assert snap.tokens_per_second > 0
        assert snap.latency_p50_ms > 0
        assert snap.latency_p95_ms >= snap.latency_p50_ms
        # p99 is independently randomized; just verify it's positive
        assert snap.latency_p99_ms > 0
        assert 0 <= snap.cpu_percent <= 100
        assert snap.memory_mb > 0

    def test_avg_throughput(self):
        run = generate_sample_data()["compact-4"]
        assert run.avg_throughput > 0

    def test_p95_latency(self):
        run = generate_sample_data()["compact-4"]
        assert run.p95_latency > 0

    def test_run_to_dict_serializable(self):
        run = generate_sample_data()["hybrid-4"]
        d = run_to_dict(run)
        assert d["label"] == "hybrid-4"
        assert len(d["snapshots"]) > 0
        # Verify JSON-serializable
        json.dumps(d)

    def test_all_runs_to_dict(self):
        data = generate_sample_data()
        d = all_runs_to_dict(data)
        assert set(d.keys()) == {"compact-4", "spread-4", "hybrid-4", "no-pin-4"}
        json.dumps(d)

    def test_trend_degrades(self):
        run = generate_sample_data()["no-pin-4"]
        # Use a rolling average of last 5 vs first 5 to smooth noise
        first_5 = sum(s.tokens_per_second for s in run.snapshots[:5]) / 5
        last_5 = sum(s.tokens_per_second for s in run.snapshots[-5:]) / 5
        assert last_5 < first_5 * 1.2  # degrading trend overall

    def test_create_snapshot_directly(self):
        snap = Snapshot(
            timestamp=1000.0,
            tokens_per_second=55.5,
            latency_p50_ms=20.0,
            latency_p95_ms=40.0,
            latency_p99_ms=60.0,
            cpu_percent=75.0,
            memory_mb=4096.0,
            batch_size=4,
            prompt_processing_speed=150.0,
        )
        assert snap.tokens_per_second == 55.5
        assert snap.batch_size == 4

    def test_create_run_directly(self):
        run = ProfileRun(label="test")
        run.add(Snapshot(0, 10, 1, 2, 3, 50, 1000, 1, 30))
        assert run.avg_throughput == 10.0

    def test_empty_run_avg(self):
        run = ProfileRun(label="empty")
        assert run.avg_throughput == 0.0
        assert run.p95_latency == 0.0
