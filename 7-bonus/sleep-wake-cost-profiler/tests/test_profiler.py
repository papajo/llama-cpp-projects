"""Tests for sleep/wake cost profiler."""

import pytest

from sleep_wake_profiler.profiler import (
    MACOS_GPU_STATES,
    LINUX_GPU_STATES,
    PowerState,
    ProfileResult,
    SleepWakeProfiler,
    TransitionSample,
)


class TestPowerState:
    def test_all_states_defined_macos(self):
        assert PowerState.P0 in MACOS_GPU_STATES
        assert PowerState.P8 in MACOS_GPU_STATES
        assert PowerState.P12 in MACOS_GPU_STATES

    def test_all_states_defined_linux(self):
        assert PowerState.P0 in LINUX_GPU_STATES
        assert PowerState.P8 in LINUX_GPU_STATES
        assert PowerState.P12 in LINUX_GPU_STATES

    def test_power_watts_positive(self):
        for info in MACOS_GPU_STATES.values():
            assert info.power_watts > 0
        for info in LINUX_GPU_STATES.values():
            assert info.power_watts > 0


class TestProfiler:
    def test_measure_transition(self):
        profiler = SleepWakeProfiler()
        sample = profiler.measure_transition(PowerState.P0, PowerState.P8)
        assert sample.duration_ms > 0
        assert sample.energy_estimate_joules > 0
        assert sample.from_state == PowerState.P0
        assert sample.to_state == PowerState.P8

    def test_measure_reverse(self):
        profiler = SleepWakeProfiler()
        sample = profiler.measure_transition(PowerState.P8, PowerState.P0)
        assert sample.duration_ms > 0

    def test_deep_sleep_slower(self):
        profiler = SleepWakeProfiler()
        to_p8 = profiler.measure_transition(PowerState.P0, PowerState.P8)
        to_p12 = profiler.measure_transition(PowerState.P0, PowerState.P12)
        assert to_p12.duration_ms > to_p8.duration_ms

    def test_linux_model(self):
        profiler = SleepWakeProfiler(gpu_model="nvidia-a100", state_model=LINUX_GPU_STATES)
        sample = profiler.measure_transition(PowerState.P0, PowerState.P8)
        assert sample.duration_ms > 0
        assert profiler.gpu_model == "nvidia-a100"

    def test_profile_workload_adds_samples(self):
        profiler = SleepWakeProfiler()
        gaps = [0.05, 0.5, 2.0, 0.1, 3.0]
        profiler.profile_workload(gaps, sleep_state=PowerState.P8)
        assert len(profiler.result.samples) > 0

    def test_short_gaps_keep_awake(self):
        profiler = SleepWakeProfiler()
        gaps = [0.01, 0.01, 0.01]
        profiler.profile_workload(gaps, sleep_state=PowerState.P8)
        # Gaps too short for sleep → no transitions
        assert len(profiler.result.samples) == 0

    def test_long_gaps_trigger_sleep_wake(self):
        profiler = SleepWakeProfiler()
        gaps = [5.0, 5.0, 5.0]
        profiler.profile_workload(gaps, sleep_state=PowerState.P8)
        # Each gap > threshold: sleep → wake → sleep → wake → sleep → (final wake)
        assert len(profiler.result.samples) >= 4

    def test_recommendation_with_data(self):
        profiler = SleepWakeProfiler()
        profiler.measure_transition(PowerState.P0, PowerState.P8)
        profiler.measure_transition(PowerState.P8, PowerState.P0)
        rec = profiler.recommendation()
        assert len(rec) > 0
        assert "GPU" in rec

    def test_recommendation_no_data(self):
        profiler = SleepWakeProfiler()
        rec = profiler.recommendation()
        assert "No data" in rec


class TestProfileResult:
    def test_avg_wake(self):
        result = ProfileResult()
        result.add(TransitionSample(PowerState.P0, PowerState.P8, 10.0, 0.5))
        result.add(TransitionSample(PowerState.P0, PowerState.P8, 20.0, 1.0))
        assert result.avg_wake_duration_ms == 15.0

    def test_avg_wake_from_deep(self):
        result = ProfileResult()
        result.add(TransitionSample(PowerState.P12, PowerState.P0, 300.0, 15.0))
        result.add(TransitionSample(PowerState.P12, PowerState.P0, 500.0, 25.0))
        assert result.avg_wake_from_deep_ms == 400.0

    def test_summary_output(self):
        result = ProfileResult()
        result.add(TransitionSample(PowerState.P0, PowerState.P8, 10.0, 0.5))
        result.add(TransitionSample(PowerState.P8, PowerState.P0, 15.0, 0.3))
        s = result.summary()
        assert "GPU Sleep/Wake Cost Profile" in s
        assert "Break-even" in s

    def test_break_even_positive(self):
        result = ProfileResult()
        be = result._break_even_seconds(25.0, 0.015)
        assert be > 0
        assert be < 10  # reasonable break-even

    def test_empty_result(self):
        result = ProfileResult()
        assert result.avg_sleep_duration_ms == 0.0
        assert result.avg_wake_duration_ms == 0.0
        assert result.total_energy_joules == 0.0

    def test_energy_total(self):
        result = ProfileResult()
        result.add(TransitionSample(PowerState.P0, PowerState.P8, 10, 0.5))
        result.add(TransitionSample(PowerState.P8, PowerState.P0, 10, 0.3))
        assert result.total_energy_joules == 0.8
