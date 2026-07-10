"""Tests for cpu-affinity-tuning."""

import json
import platform

import pytest

from cpu_affinity_tuning.benchmark import (
    BenchmarkSuite,
    _coefficient_of_variation,
    benchmark_plans,
)
from cpu_affinity_tuning.strategies import (
    PinningPlan,
    STRATEGIES,
    generate_plans,
    strategy_compact,
    strategy_hybrid,
    strategy_spread,
)
from cpu_affinity_tuning.topology import (
    CoreInfo,
    CpuTopology,
    detect_topology,
    topology_to_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dual_package_topology():
    """Simulate a 2-package, 16-logical-CPU machine (8 per package, SMT=2)."""
    cores = []
    for pkg in range(2):
        for phys in range(4):  # 4 physical cores per package
            for smt in range(2):  # SMT-2
                cores.append(CoreInfo(
                    core_id=len(cores),
                    package_id=pkg,
                    numa_node=0,
                    smt_id=phys * 2 + smt,
                ))
    return CpuTopology(
        logical_cpus=16,
        physical_cores=8,
        packages=2,
        numa_nodes=1,
        cores=cores,
    )


@pytest.fixture
def single_package_topology():
    """Simulate a 1-package, 8-logical-CPU machine."""
    cores = [
        CoreInfo(core_id=i, package_id=0, numa_node=0, smt_id=i % 2)
        for i in range(8)
    ]
    return CpuTopology(
        logical_cpus=8,
        physical_cores=4,
        packages=1,
        numa_nodes=1,
        cores=cores,
    )


@pytest.fixture
def numa_topology():
    """Simulate a 2-NUMA-node, 16-logical-CPU machine."""
    cores = []
    for numa in range(2):
        for core in range(4):
            cores.append(CoreInfo(
                core_id=len(cores),
                package_id=numa,
                numa_node=numa,
                smt_id=0,
            ))
            cores.append(CoreInfo(
                core_id=len(cores),
                package_id=numa,
                numa_node=numa,
                smt_id=1,
            ))
    return CpuTopology(
        logical_cpus=16,
        physical_cores=8,
        packages=2,
        numa_nodes=2,
        cores=cores,
    )


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------


class TestDetectTopology:
    def test_detects_positive_counts(self):
        topo = detect_topology()
        assert topo.logical_cpus >= 1
        assert topo.physical_cores >= 1
        assert topo.packages >= 1
        assert topo.numa_nodes >= 1

    def test_cores_list_populated(self):
        topo = detect_topology()
        assert len(topo.cores) == topo.logical_cpus

    def test_cores_by_package(self):
        topo = detect_topology()
        by_pkg = topo.cores_by_package()
        assert len(by_pkg) == topo.packages
        total = sum(len(v) for v in by_pkg.values())
        assert total == topo.logical_cpus

    def test_json_roundtrip(self):
        topo = detect_topology()
        j = topology_to_json(topo)
        data = json.loads(j)
        assert data["logical_cpus"] == topo.logical_cpus
        assert len(data["cores"]) == topo.logical_cpus

    def test_dual_package(self, dual_package_topology):
        t = dual_package_topology
        assert t.packages == 2
        assert t.logical_cpus == 16

    def test_numa_topology(self, numa_topology):
        t = numa_topology
        assert t.numa_nodes == 2
        assert len(t.cores_by_numa()) == 2


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_compact_cpu_count(self, dual_package_topology):
        plan = strategy_compact(dual_package_topology, 4)
        assert plan.worker_count == 4
        assert len(plan.logical_cpus) == 4
        total = sum(len(c) for c in plan.logical_cpus)
        assert total == 16

    def test_spread_uses_both_packages(self, dual_package_topology):
        plan = strategy_spread(dual_package_topology, 4)
        # Workers are spread across packages (each worker gets one pkg)
        all_packages = set()
        for cpus in plan.logical_cpus:
            for c in cpus:
                all_packages.add(c // 8)  # 0-7 = pkg0, 8-15 = pkg1
        # Both packages should be represented across all workers
        assert all_packages == {0, 1}

    def test_hybrid_prompt_pkg(self, dual_package_topology):
        plan = strategy_hybrid(dual_package_topology, 2)
        # Worker 0 should get package-0 CPUs
        assert all(c < 8 for c in plan.logical_cpus[0])
        # Worker 1 should get some from package 1
        assert any(c >= 8 for c in plan.logical_cpus[1])

    def test_hybrid_fallback_to_compact(self, single_package_topology):
        plan = strategy_hybrid(single_package_topology, 2)
        # Falls back to compact when < 2 packages
        assert plan.strategy_name == "compact"
        total = sum(len(c) for c in plan.logical_cpus)
        assert total == 8

    def test_generate_plans_default(self, dual_package_topology):
        plans = generate_plans(dual_package_topology, 4)
        assert set(plans.keys()) == {"compact", "spread", "hybrid"}
        for name, plan in plans.items():
            assert plan.strategy_name == name

    def test_generate_plans_subset(self, dual_package_topology):
        plans = generate_plans(dual_package_topology, 4, strategies=["compact"])
        assert list(plans.keys()) == ["compact"]

    def test_strategies_are_callable(self):
        for name, fn in STRATEGIES.items():
            assert callable(fn)

    def test_taskset_cmds_linux(self, dual_package_topology, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        plan = strategy_compact(dual_package_topology, 2)
        cmds = plan.to_taskset_cmds(binary="./llama-cli", args="--temp 0.7")
        assert len(cmds) == 2
        assert all("taskset" in c for c in cmds)

    def test_taskset_cmds_non_linux(self, dual_package_topology, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        plan = strategy_compact(dual_package_topology, 2)
        cmds = plan.to_taskset_cmds()
        assert len(cmds) == 4  # 2 comment + 2 run lines


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_suite_runs(self, dual_package_topology):
        plans = generate_plans(dual_package_topology, 4)
        suite = benchmark_plans(plans, dual_package_topology)
        assert len(suite.results) == 3

    def test_best_strategy_present(self, dual_package_topology):
        plans = generate_plans(dual_package_topology, 4)
        suite = benchmark_plans(plans, dual_package_topology)
        best = suite.best()
        assert best.strategy in ("compact", "spread", "hybrid")
        assert best.throughput > 0

    def test_report_output(self, dual_package_topology):
        plans = generate_plans(dual_package_topology, 4)
        suite = benchmark_plans(plans, dual_package_topology)
        report = suite.report()
        assert "Benchmark Report" in report
        assert "compact" in report

    def test_cv_computation(self):
        assert _coefficient_of_variation([10, 10, 10]) == 0.0
        cv = _coefficient_of_variation([10, 20, 30])
        assert cv > 0
        assert _coefficient_of_variation([]) == 0.0

    def test_results_have_positive_values(self, dual_package_topology):
        plans = generate_plans(dual_package_topology, 4)
        suite = benchmark_plans(plans, dual_package_topology)
        for r in suite.results:
            assert r.duration > 0
            assert r.throughput > 0
            assert r.cv >= 0
