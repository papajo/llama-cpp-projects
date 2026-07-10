"""GPU sleep/wake transition cost profiler for inference servers.

Measures the time and estimated energy cost of GPU power-state transitions
between inference requests.  Helps answer: *Should we keep the GPU awake
between requests, or let it sleep and wake?*
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class PowerState(Enum):
    """GPU power states (naming follows NVIDIA conventions)."""
    P0 = "P0"       # Max performance, fully awake
    P2 = "P2"       # Medium performance
    P8 = "P8"       # Low power / idle
    P12 = "P12"     # Deep sleep
    D3COLD = "D3"   # Off / D3cold


@dataclass
class PowerStateInfo:
    """Information about a power state."""
    state: PowerState
    power_watts: float  # power draw in this state
    transition_to: Dict[PowerState, float] = field(default_factory=dict)
    # ↑ transition time in seconds to reach each other state

    def __hash__(self):
        return hash(self.state.value)


@dataclass
class TransitionSample:
    """A single measured sleep/wake transition."""
    from_state: PowerState
    to_state: PowerState
    duration_ms: float
    energy_estimate_joules: float
    timestamp: float = 0.0


@dataclass
class ProfileResult:
    """Aggregated results from a profiling session."""
    samples: List[TransitionSample] = field(default_factory=list)

    def add(self, sample: TransitionSample) -> None:
        self.samples.append(sample)

    @property
    def avg_sleep_duration_ms(self) -> float:
        sleep = [s for s in self.samples if s.from_state != PowerState.P0 and s.to_state != PowerState.P0]
        if not sleep:
            return 0.0
        return sum(s.duration_ms for s in sleep) / len(sleep)

    @property
    def avg_wake_duration_ms(self) -> float:
        wake = [s for s in self.samples if s.from_state == PowerState.P0 and s.to_state != PowerState.P0]
        return sum(s.duration_ms for s in wake) / len(wake) if wake else 0.0

    @property
    def avg_wake_from_deep_ms(self) -> float:
        deep = [s for s in self.samples if s.from_state == PowerState.P12 or s.from_state == PowerState.D3COLD]
        return sum(s.duration_ms for s in deep) / len(deep) if deep else 0.0

    @property
    def total_energy_joules(self) -> float:
        return sum(s.energy_estimate_joules for s in self.samples)

    def summary(self) -> str:
        awake = [s for s in self.samples if s.to_state == PowerState.P0]
        lines = [
            "=" * 60,
            "GPU Sleep/Wake Cost Profile",
            "=" * 60,
            f"Samples collected: {len(self.samples)}",
            "",
            f"Avg sleep transition:     {self.avg_sleep_duration_ms:8.2f} ms",
            f"Avg wake transition:      {self.avg_wake_duration_ms:8.2f} ms",
            f"Avg wake from deep sleep: {self.avg_wake_from_deep_ms:8.2f} ms",
            f"Total energy (all):       {self.total_energy_joules:8.2f} J",
            "",
            "--- Break-even Analysis ---",
        ]
        if awake:
            idle_power = 25.0  # typical idle watts
            avg_wake = self.avg_wake_duration_ms / 1000.0
            break_even = self._break_even_seconds(idle_power, avg_wake)
            lines.append(f"Keep-awake vs sleep: {break_even:.1f}s between requests")
            if break_even < 5:
                lines.append("→ Short idle: keep GPU awake (wake cost exceeds idle cost)")
            else:
                lines.append("→ Long idle: let GPU sleep (wake cost < idle cost)")
        return "\n".join(lines)

    def _break_even_seconds(self, idle_power_watts: float, wake_time_s: float) -> float:
        """Calculate the idle duration where sleeping == staying awake."""
        wake_energy = 30.0  # Joules consumed during wake transition (estimate)
        sleep_energy = 2.0  # Joules consumed during sleep transition (estimate)
        if idle_power_watts <= 0:
            return float('inf')
        # awake_cost = idle_power * t
        # sleep_cost = sleep_energy + wake_energy + 0 * t (sleeping uses ~0W)
        # Break-even: idle_power * t = sleep_energy + wake_energy
        # t = (sleep_energy + wake_energy) / idle_power
        return (sleep_energy + wake_energy) / idle_power_watts


# ---------------------------------------------------------------------------
# Platform-specific state models
# ---------------------------------------------------------------------------

MACOS_GPU_STATES: Dict[PowerState, PowerStateInfo] = {
    PowerState.P0: PowerStateInfo(
        state=PowerState.P0,
        power_watts=45.0,
        transition_to={
            PowerState.P8: 8.0 / 1000,    # 8 ms to idle
            PowerState.P12: 150.0 / 1000,  # 150 ms to deep sleep
        },
    ),
    PowerState.P8: PowerStateInfo(
        state=PowerState.P8,
        power_watts=8.0,
        transition_to={
            PowerState.P0: 12.0 / 1000,    # 12 ms to full perf
            PowerState.P12: 100.0 / 1000,  # 100 ms to deep sleep
        },
    ),
    PowerState.P12: PowerStateInfo(
        state=PowerState.P12,
        power_watts=0.5,
        transition_to={
            PowerState.P0: 300.0 / 1000,   # 300 ms to full perf
            PowerState.P8: 80.0 / 1000,    # 80 ms to idle
        },
    ),
}

LINUX_GPU_STATES: Dict[PowerState, PowerStateInfo] = {
    PowerState.P0: PowerStateInfo(
        state=PowerState.P0,
        power_watts=300.0,
        transition_to={PowerState.P8: 10.0 / 1000, PowerState.P12: 200.0 / 1000},
    ),
    PowerState.P8: PowerStateInfo(
        state=PowerState.P8,
        power_watts=50.0,
        transition_to={PowerState.P0: 15.0 / 1000, PowerState.P12: 120.0 / 1000},
    ),
    PowerState.P12: PowerStateInfo(
        state=PowerState.P12,
        power_watts=3.0,
        transition_to={PowerState.P0: 500.0 / 1000, PowerState.P8: 100.0 / 1000},
    ),
}


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class SleepWakeProfiler:
    """Measures GPU sleep/wake transition costs.

    Simulates realistic measurements based on platform power-state models.
    On real hardware this would call platform APIs (nvidia-smi, ioreg, etc.).
    """

    def __init__(self, gpu_model: str = "apple-m4-max", state_model: Optional[Dict[PowerState, PowerStateInfo]] = None):
        self.gpu_model = gpu_model
        self.state_model = state_model or MACOS_GPU_STATES
        self.result = ProfileResult()

    def measure_transition(self, from_state: PowerState, to_state: PowerState) -> TransitionSample:
        """Measure a single sleep/wake transition (simulated)."""
        base_duration = self._lookup_transition_time(from_state, to_state)
        # Add realistic noise (±10%)
        duration_ms = base_duration * 1000 * random.uniform(0.9, 1.1)

        # Energy: average power during transition × time
        from_power = self.state_model[from_state].power_watts if from_state in self.state_model else 0
        to_power = self.state_model[to_state].power_watts if to_state in self.state_model else 0
        avg_power = (from_power + to_power) / 2
        energy = avg_power * (duration_ms / 1000.0)

        sample = TransitionSample(
            from_state=from_state,
            to_state=to_state,
            duration_ms=round(duration_ms, 2),
            energy_estimate_joules=round(energy, 3),
            timestamp=time.time(),
        )
        self.result.add(sample)
        return sample

    def _lookup_transition_time(self, from_state: PowerState, to_state: PowerState) -> float:
        """Look up transition time in seconds from the state model."""
        info = self.state_model.get(from_state)
        if info and to_state in info.transition_to:
            return info.transition_to[to_state]
        # Reverse: check if the to-state knows how to get to from-state
        info_to = self.state_model.get(to_state)
        if info_to and from_state in info_to.transition_to:
            return info_to.transition_to[from_state]
        # Fallback: estimate based on power state difference
        state_order = [PowerState.P0, PowerState.P2, PowerState.P8, PowerState.P12]
        try:
            diff = abs(state_order.index(from_state) - state_order.index(to_state))
            return diff * 0.05  # 50 ms per level
        except ValueError:
            return 0.1  # 100 ms default

    def profile_workload(
        self,
        idle_gaps: List[float],
        sleep_state: PowerState = PowerState.P8,
    ) -> ProfileResult:
        """Profile sleep/wake costs for a sequence of idle gaps between requests.

        For each idle gap between requests, decides whether the transition
        cost is worth it.  If the gap exceeds the sleep threshold, the GPU
        sleeps then wakes — both transitions are recorded.

        Args:
            idle_gaps: List of idle durations (seconds) between inference requests.
            sleep_state: The power state to enter during idle.

        Returns:
            ProfileResult with all measured transitions.
        """
        for gap in idle_gaps:
            sleep_threshold = self._sleep_threshold(sleep_state, PowerState.P0)
            if gap > sleep_threshold:
                # Deep enough idle: sleep at start, wake at end
                self.measure_transition(PowerState.P0, sleep_state)
                self.measure_transition(sleep_state, PowerState.P0)

        return self.result

    def _sleep_threshold(self, sleep_state: PowerState, current_state: PowerState) -> float:
        """Calculate the minimum idle time where sleeping is worthwhile."""
        sleep_time = self._lookup_transition_time(PowerState.P0, sleep_state)
        wake_time = self._lookup_transition_time(sleep_state, PowerState.P0)
        total_transition = sleep_time + wake_time
        # Arbitrary: sleep if idle gap > 2× transition overhead
        return total_transition * 2

    def recommendation(self) -> str:
        if not self.result.samples:
            return "No data collected. Run a profiling workload first."

        lines = [
            f"GPU: {self.gpu_model}",
            f"Samples: {len(self.result.samples)}",
            f"Avg wake: {self.result.avg_wake_duration_ms:.1f} ms",
            f"Avg sleep: {self.result.avg_sleep_duration_ms:.1f} ms",
            "",
        ]

        avg_wake = self.result.avg_wake_duration_ms
        if avg_wake < 50:
            lines.append("✅ Fast wake: aggressive power saving is safe.")
            lines.append("   Let GPU sleep between requests (threshold: ~100ms idle).")
        elif avg_wake < 200:
            lines.append("⚡ Moderate wake: balance idle power vs wake cost.")
            lines.append("   Sleep after idle gaps > 500ms to break even.")
        else:
            lines.append("🐌 Slow wake: keep GPU awake during typical inter-request gaps.")
            lines.append("   Only sleep when idle exceeds 2-5 seconds.")

        return "\n".join(lines)
