"""Tests for checkpoint store."""

import pytest

from checkpoint_rollback_agent.store import Checkpoint, CheckpointStore


class TestCheckpoint:
    def test_immutable(self):
        with pytest.raises(Exception):
            Checkpoint(step_index=0, step_name="test", state={"x": 1}).state = {}  # type: ignore

    def test_label(self):
        cp = Checkpoint(step_index=3, step_name="verify")
        assert cp.label == "[3] verify"

    def test_timestamp_auto(self):
        cp = Checkpoint(step_index=0, step_name="t")
        assert cp.timestamp > 0


class TestCheckpointStore:
    def test_save_and_retrieve(self):
        store = CheckpointStore()
        cp = store.save("step1", {"x": 1})
        assert cp.step_index == 0
        assert cp.step_name == "step1"
        assert store.get(0) is cp
        assert store.count == 1

    def test_latest(self):
        store = CheckpointStore()
        assert store.latest is None
        store.save("a", {"x": 1})
        store.save("b", {"y": 2})
        assert store.latest is not None
        assert store.latest.step_name == "b"

    def test_rollback_discards_later(self):
        store = CheckpointStore()
        store.save("a", {"x": 1})
        store.save("b", {"x": 2})
        store.save("c", {"x": 3})

        cp = store.rollback(1)  # back to step 1
        assert cp is not None
        assert cp.step_index == 1
        assert cp.state["x"] == 2
        assert store.count == 2  # step 2 discarded
        assert store.get(2) is None

    def test_rollback_invalid(self):
        store = CheckpointStore()
        store.save("a", {"x": 1})
        assert store.rollback(99) is None

    def test_list_checkpoints(self):
        store = CheckpointStore()
        store.save("a", {"x": 1})
        store.save("b", {"y": 2})
        cps = store.list_checkpoints()
        assert len(cps) == 2
        assert [cp.step_name for cp in cps] == ["a", "b"]

    def test_clear(self):
        store = CheckpointStore()
        store.save("a", {"x": 1})
        store.clear()
        assert store.count == 0
        assert store.latest is None

    def test_save_with_metadata(self):
        store = CheckpointStore()
        cp = store.save("step", {"x": 1}, metadata={"version": 2})
        assert cp.metadata["version"] == 2
