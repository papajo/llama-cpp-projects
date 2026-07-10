"""Tests for NUMA deployment guide."""

import platform

import pytest

from numa_guide.deployment import (
    MemoryPolicy,
    NumaDeploymentPlan,
    recommend_plan,
)
from numa_guide.topology import NumaNode, NumaTopology, detect_numa_topology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_node_topology():
    """Single NUMA node, 8 CPUs."""
    return NumaTopology(
        nodes=[NumaNode(node_id=0, cpu_ids=list(range(8)), total_memory_kb=16 * 1024 * 1024)],
        has_numa=False,
    )


@pytest.fixture
def dual_node_topology():
    """Two NUMA nodes, 8 CPUs each."""
    return NumaTopology(
        nodes=[
            NumaNode(node_id=0, cpu_ids=list(range(0, 8)), total_memory_kb=8 * 1024 * 1024),
            NumaNode(node_id=1, cpu_ids=list(range(8, 16)), total_memory_kb=8 * 1024 * 1024),
        ],
        has_numa=True,
    )


@pytest.fixture
def quad_node_topology():
    """Four NUMA nodes, 4 CPUs each."""
    nodes = []
    for nid in range(4):
        nodes.append(NumaNode(
            node_id=nid,
            cpu_ids=list(range(nid * 4, nid * 4 + 4)),
            total_memory_kb=4 * 1024 * 1024,
            distance_map={i: 10 if i == nid else 20 + abs(i - nid) * 3 for i in range(4)},
        ))
    return NumaTopology(nodes=nodes, has_numa=True)


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------


class TestTopology:
    def test_detect_returns_something(self):
        topo = detect_numa_topology()
        assert topo.node_count >= 1
        for node in topo.nodes:
            assert len(node.cpu_ids) > 0

    def test_single_node(self, single_node_topology):
        assert single_node_topology.node_count == 1
        assert single_node_topology.has_numa is False
        assert single_node_topology.nodes[0].cpu_ids == list(range(8))

    def test_dual_node(self, dual_node_topology):
        assert dual_node_topology.node_count == 2
        assert dual_node_topology.has_numa is True

    def test_quad_node_distances(self, quad_node_topology):
        n0 = quad_node_topology.nodes[0]
        assert n0.distance_map[0] == 10
        assert n0.distance_map[1] > 10

    def test_summary_output(self, dual_node_topology):
        s = dual_node_topology.summary()
        assert "NUMA nodes: 2" in s
        assert "Node 0" in s
        assert "Node 1" in s


# ---------------------------------------------------------------------------
# Deployment plan tests
# ---------------------------------------------------------------------------


class TestDeployment:
    def test_single_node_plan(self, single_node_topology):
        plan = recommend_plan(single_node_topology, 2)
        assert plan.memory_policy == MemoryPolicy.LOCAL
        assert plan.worker_count == 2

    def test_dual_node_distributes(self, dual_node_topology):
        plan = recommend_plan(dual_node_topology, 4)
        # 4 workers across 2 nodes
        assert 0 in plan.node_assignments
        assert 1 in plan.node_assignments
        total = sum(len(v) for v in plan.node_assignments.values())
        assert total == 4

    def test_interleave_policy(self, dual_node_topology):
        plan = recommend_plan(dual_node_topology, 2, policy=MemoryPolicy.INTERLEAVE)
        assert plan.memory_policy == MemoryPolicy.INTERLEAVE
        assert len(plan.policy_args) > 0

    def test_quad_node_even_distribution(self, quad_node_topology):
        plan = recommend_plan(quad_node_topology, 8)
        for nid in range(4):
            assert nid in plan.node_assignments

    def test_numactl_cmds_linux(self, dual_node_topology, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        plan = recommend_plan(dual_node_topology, 2)
        cmds = plan.to_numactl_cmds()
        assert len(cmds) == 2
        assert all("numactl" in c for c in cmds)

    def test_numactl_cmds_macos(self, dual_node_topology, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        plan = recommend_plan(dual_node_topology, 2)
        cmds = plan.to_numactl_cmds()
        assert len(cmds) == 2
        assert all("macOS" in c for c in cmds)

    def test_summary_format(self, dual_node_topology):
        plan = recommend_plan(dual_node_topology, 2)
        s = plan.summary()
        assert "NUMA" in s
        assert "Worker" in s or "worker" in s

    def test_custom_binary(self, dual_node_topology):
        plan = recommend_plan(dual_node_topology, 1)
        cmds = plan.to_numactl_cmds(binary="./my-binary", extra_args="--temp 0.8")
        assert all("my-binary" in c for c in cmds)
        assert all("temp" in c for c in cmds)

    # --- Regression: binding to wrong node ---
    def test_worker_cpus_match_node(self, dual_node_topology):
        plan = recommend_plan(dual_node_topology, 2, policy=MemoryPolicy.BIND)
        for wid in range(2):
            cpus = plan.cpu_assignments[wid]
            # Worker 0 → node 0 CPUs (0-7), Worker 1 → node 1 CPUs (8-15)
            if wid == 0:
                assert all(c < 8 for c in cpus)
            else:
                assert all(c >= 8 for c in cpus)

    def test_policy_args_empty_for_local(self, single_node_topology):
        plan = recommend_plan(single_node_topology, 1, policy=MemoryPolicy.LOCAL)
        assert plan.policy_args == []
