"""Common executed-action safety wrapper for FairFlex V2 baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from fairflex.baselines import Policy
from fairflex.domain import EVSession
from fairflex.grid import IEEE33Grid
from fairflex.safety import ACFeasibilityRepair, ACRepairResult, ACRepairedMPCPolicy


CurrentStepFeederCap = float | Callable[[int], float]


class ACRepairedBaselinePolicyV2:
    """Apply the V2 common feeder and AC execution repair to a baseline.

    Baselines retain their own priority rule, but they cannot obtain extra
    delivered energy by proposing an infeasible feeder or AC-grid action.  If
    AC repair curtails a station, all baseline allocations at that station are
    scaled proportionally; unlike MPC, a baseline does not get an undisclosed
    second optimisation pass.
    """

    def __init__(
        self,
        baseline: Policy,
        grid: IEEE33Grid,
        *,
        feeder_capacity_kw: CurrentStepFeederCap,
    ) -> None:
        self.baseline = baseline
        self.grid = grid
        self.feeder_capacity_kw = feeder_capacity_kw
        self.repairer = ACFeasibilityRepair(grid)
        self.last_repair: ACRepairResult | None = None
        self.repair_history: list[ACRepairResult] = []
        self.last_applied_grid_validation = None
        self.last_applied_station_powers_kw = None

    def _feeder_cap(self, current_step: int) -> float:
        value = (
            self.feeder_capacity_kw(current_step)
            if callable(self.feeder_capacity_kw)
            else self.feeder_capacity_kw
        )
        if float(value) < 0:
            raise ValueError("V2 feeder capacity must be non-negative")
        return float(value)

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        if set(station_capacities_kw) != set(self.grid.station_load_indices):
            raise ValueError("baseline station capacities must exactly match the V2 AC grid stations")
        proposed = {
            ev_id: max(0.0, float(power))
            for ev_id, power in self.baseline(
                station_evs,
                station_capacities_kw,
                step_hours,
                current_step,
            ).items()
        }
        total = sum(proposed.values())
        feeder_cap = self._feeder_cap(current_step)
        if total > feeder_cap and total > 0:
            factor = feeder_cap / total
            proposed = {ev_id: value * factor for ev_id, value in proposed.items()}
        requested = {
            station_id: sum(proposed.get(ev.ev_id, 0.0) for ev in evs)
            for station_id, evs in station_evs.items()
        }
        repair = self.repairer.repair(
            requested,
            flexibility_scores=ACRepairedMPCPolicy._station_flexibility(
                station_evs,
                current_step=current_step,
                step_hours=step_hours,
            ),
        )
        if not repair.safe:
            raise RuntimeError("V2 AC repair could not recover a safe baseline action")
        applied: dict[str, float] = {}
        for station_id, evs in station_evs.items():
            requested_power = requested[station_id]
            factor = (
                repair.applied_powers_kw[station_id] / requested_power
                if requested_power > 0
                else 1.0
            )
            for ev in evs:
                applied[ev.ev_id] = proposed.get(ev.ev_id, 0.0) * factor
        self.last_repair = repair
        self.repair_history.append(repair)
        self.last_applied_grid_validation = repair.final_grid
        self.last_applied_station_powers_kw = repair.applied_powers_kw
        return applied
