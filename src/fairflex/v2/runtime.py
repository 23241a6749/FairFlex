"""V2-only decision-level runtime telemetry for an unchanged Fair MPC core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter

from fairflex.domain import EVSession
from fairflex.safety import ACRepairedMPCPolicy


@dataclass(frozen=True)
class V2DecisionRuntime:
    """One controller invocation, recorded independently of outcome metrics."""

    step: int
    active_evs: int
    idle_step_skipped: bool
    total_seconds: float
    mpc_stage_one_seconds: float
    mpc_stage_two_seconds: float
    mpc_stage_three_seconds: float
    mpc_solves: int
    ac_repair_activated: bool
    economic_tiebreak_failures_this_call: int
    soft_latency_exceeded: bool


class ProfiledACRepairedMPCPolicy:
    """Profile a V1-safe AC-repaired MPC policy without changing its logic.

    The wrapper is intentionally telemetry-only: a soft latency excess is
    recorded but does not discard an otherwise verified MPC/AC-safe result.
    A future V2 hard-timeout implementation must use solver-level time limits
    and separately test its deterministic fallback; it must not pretend that a
    Python timer can interrupt an already-running solver safely.
    """

    def __init__(
        self,
        delegate: ACRepairedMPCPolicy,
        *,
        soft_latency_seconds: float = 1.0,
    ) -> None:
        if soft_latency_seconds <= 0:
            raise ValueError("soft_latency_seconds must be positive")
        self.delegate = delegate
        self.soft_latency_seconds = soft_latency_seconds
        self.decision_runtimes: list[V2DecisionRuntime] = []
        self.last_applied_grid_validation = None
        self.last_applied_station_powers_kw = None

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        active_evs = sum(len(evs) for evs in station_evs.values())
        started = perf_counter()
        # An idle replay step has exactly one safe, useful action: allocate no
        # power.  Avoiding the solver and AC power-flow call here is an exact
        # behavioural equivalence, not a heuristic event trigger.
        if active_evs == 0:
            elapsed = perf_counter() - started
            self.decision_runtimes.append(
                V2DecisionRuntime(
                    step=current_step,
                    active_evs=0,
                    idle_step_skipped=True,
                    total_seconds=elapsed,
                    mpc_stage_one_seconds=0.0,
                    mpc_stage_two_seconds=0.0,
                    mpc_stage_three_seconds=0.0,
                    mpc_solves=0,
                    ac_repair_activated=False,
                    economic_tiebreak_failures_this_call=0,
                    soft_latency_exceeded=False,
                )
            )
            return {}

        history_start = len(self.delegate.controller.solve_timing_history)
        repair_start = len(self.delegate.repair_history)
        tiebreak_start = self.delegate.economic_tiebreak_failures
        allocations = self.delegate(
            station_evs,
            station_capacities_kw,
            step_hours,
            current_step,
        )
        elapsed = perf_counter() - started
        timings = self.delegate.controller.solve_timing_history[history_start:]
        repairs = self.delegate.repair_history[repair_start:]
        self.decision_runtimes.append(
            V2DecisionRuntime(
                step=current_step,
                active_evs=active_evs,
                idle_step_skipped=False,
                total_seconds=elapsed,
                mpc_stage_one_seconds=float(sum(item["stage_one_seconds"] for item in timings)),
                mpc_stage_two_seconds=float(sum(item["stage_two_seconds"] for item in timings)),
                mpc_stage_three_seconds=float(sum(item["stage_three_seconds"] for item in timings)),
                mpc_solves=len(timings),
                ac_repair_activated=any(repair.attempts > 0 for repair in repairs),
                economic_tiebreak_failures_this_call=(
                    self.delegate.economic_tiebreak_failures - tiebreak_start
                ),
                soft_latency_exceeded=elapsed > self.soft_latency_seconds,
            )
        )
        # Preserve the cached AC validation contract understood by
        # ChargingSimulation; the wrapped policy remains the one that supplied
        # the physical action.
        last_repair = self.delegate.last_repair
        if last_repair is not None:
            self.last_applied_grid_validation = last_repair.final_grid
            self.last_applied_station_powers_kw = last_repair.applied_powers_kw
        return allocations

    def runtime_records(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.decision_runtimes]

    def runtime_summary(self) -> dict[str, float | int]:
        if not self.decision_runtimes:
            return {
                "decisions": 0,
                "controller_calls": 0,
                "idle_steps_skipped": 0,
                "median_seconds": float("nan"),
                "p95_seconds": float("nan"),
                "soft_latency_exceeded_decisions": 0,
                "mean_mpc_solves_per_decision": float("nan"),
            }
        import numpy as np

        controller_records = [item for item in self.decision_runtimes if not item.idle_step_skipped]
        if not controller_records:
            return {
                "decisions": len(self.decision_runtimes),
                "controller_calls": 0,
                "idle_steps_skipped": len(self.decision_runtimes),
                "median_seconds": float("nan"),
                "p95_seconds": float("nan"),
                "soft_latency_exceeded_decisions": 0,
                "mean_mpc_solves_per_decision": float("nan"),
            }
        values = np.asarray([item.total_seconds for item in controller_records], dtype=float)
        solves = np.asarray([item.mpc_solves for item in controller_records], dtype=float)
        return {
            "decisions": len(self.decision_runtimes),
            "controller_calls": len(controller_records),
            "idle_steps_skipped": len(self.decision_runtimes) - len(controller_records),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, 0.95)),
            "soft_latency_exceeded_decisions": sum(
                item.soft_latency_exceeded for item in controller_records
            ),
            "mean_mpc_solves_per_decision": float(np.mean(solves)),
        }
