"""P10-aligned lower-tail MPC for the separately versioned FairFlex V3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import cvxpy as cp
import numpy as np

from ..domain import EVSession
from ..fair_mpc import CentralizedFairMPC, MPCResult


class LowerTailFairMPC(CentralizedFairMPC):
    """Fair MPC that minimizes expected shortfall of the largest deficit tail.

    The V1 controller minimizes the single maximum service deficit.  This V3
    controller instead minimizes CVaR of deficits at ``alpha=1-tail_fraction``.
    This is a convex lower-tail surrogate for the *reported* raw P10 service;
    it is not claimed to optimize finite-sample P10 exactly.  Stages two and
    three retain the V1 energy and economy tie-breaks.
    """

    def __init__(self, *, tail_fraction: float = 0.10, **kwargs: float) -> None:
        super().__init__(**kwargs)
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must lie in (0, 1]")
        self.tail_fraction = float(tail_fraction)
        self.lower_tail_objective_history: list[float] = []

    def solve(
        self,
        evs: Sequence[EVSession],
        station_capacities_kw: Mapping[str, float | Sequence[float]],
        *,
        current_step: int,
        horizon_steps: int,
        step_hours: float,
        feeder_capacity_kw: float | Sequence[float] | None = None,
        prices_per_kwh: Sequence[float] | None = None,
    ) -> MPCResult:
        if not evs:
            return MPCResult((), np.zeros((0, horizon_steps)), 0.0, 0.0, True, {})
        if horizon_steps <= 0 or step_hours <= 0:
            raise ValueError("horizon_steps and step_hours must be positive")
        if current_step < 0:
            raise ValueError("current_step must be non-negative")
        unknown_stations = {ev.station_id for ev in evs} - set(station_capacities_kw)
        if unknown_stations:
            raise ValueError(f"EVs reference unknown stations: {sorted(unknown_stations)}")

        n_evs = len(evs)
        feeder_cap = self._profile(feeder_capacity_kw, horizon_steps, name="feeder_capacity_kw")
        prices = (
            np.zeros(horizon_steps)
            if prices_per_kwh is None
            else self._profile(prices_per_kwh, horizon_steps, name="prices_per_kwh")
        )
        assert prices is not None
        station_profiles: dict[str, np.ndarray] = {}
        for station_id, capacity in station_capacities_kw.items():
            profile = self._profile(capacity, horizon_steps, name=f"capacity for {station_id}")
            assert profile is not None
            station_profiles[station_id] = profile

        power = cp.Variable((n_evs, horizon_steps), nonneg=True)
        deficits = cp.Variable(n_evs, nonneg=True)
        eta = cp.Variable(nonneg=True)
        excess = cp.Variable(n_evs, nonneg=True)
        delivered = np.array([ev.delivered_energy_kwh for ev in evs], dtype=float) + step_hours * cp.sum(power, axis=1)
        tail_risk = eta + cp.sum(excess) / (self.tail_fraction * n_evs)
        constraints: list[cp.Constraint] = [eta <= 1.0, deficits <= 1.0, excess >= deficits - eta]
        deadline_guards_kw = np.zeros(n_evs)
        for ev_index, ev in enumerate(evs):
            effective_deadline = ev.effective_planning_deadline_step(current_step)
            constraints.extend((
                delivered[ev_index] <= ev.requested_energy_kwh,
                deficits[ev_index] >= (ev.requested_energy_kwh - delivered[ev_index]) / ev.requested_energy_kwh,
            ))
            for offset in range(horizon_steps):
                if current_step + offset < effective_deadline:
                    constraints.append(power[ev_index, offset] <= ev.max_power_kw)
                else:
                    constraints.append(power[ev_index, offset] == 0.0)
            available_slots = max(0, effective_deadline - current_step)
            future_slots = max(0, available_slots - 1)
            individually_feasible = ev.remaining_energy_kwh <= available_slots * ev.max_power_kw * step_hours + self.delivery_tolerance_kwh
            required_now_energy = max(0.0, ev.remaining_energy_kwh - future_slots * ev.max_power_kw * step_hours) if individually_feasible else 0.0
            deadline_guards_kw[ev_index] = required_now_energy / step_hours

        station_rows_by_id: dict[str, list[int]] = {}
        for station_id, capacity_profile in station_profiles.items():
            rows = [index for index, ev in enumerate(evs) if ev.station_id == station_id]
            station_rows_by_id[station_id] = rows
            if rows:
                constraints.append(cp.sum(power[rows, :], axis=0) <= capacity_profile)
        if feeder_cap is not None:
            constraints.append(cp.sum(power, axis=0) <= feeder_cap)
        guards_fit_feeder = feeder_cap is None or float(deadline_guards_kw.sum()) <= feeder_cap[0] + self.delivery_tolerance_kwh
        if guards_fit_feeder:
            for station_id, rows in station_rows_by_id.items():
                if rows and float(deadline_guards_kw[rows].sum()) <= station_profiles[station_id][0] + self.delivery_tolerance_kwh:
                    for ev_index in rows:
                        if deadline_guards_kw[ev_index] > 0:
                            constraints.append(power[ev_index, 0] >= deadline_guards_kw[ev_index])

        total_start = perf_counter()
        stage_one_start = perf_counter()
        self._solve(cp.Problem(cp.Minimize(tail_risk), constraints))
        stage_one_seconds = perf_counter() - stage_one_start
        best_tail_risk = float(tail_risk.value)
        tail_constraints = constraints + [tail_risk <= best_tail_risk + self.fairness_tolerance]
        total_delivered = cp.sum(delivered)
        stage_two_start = perf_counter()
        self._solve(cp.Problem(cp.Maximize(total_delivered), tail_constraints))
        stage_two_seconds = perf_counter() - stage_two_start
        best_total_energy = float(total_delivered.value)
        stage_two_schedule = np.asarray(power.value, dtype=float).copy()
        cost = step_hours * cp.sum(cp.multiply(power, prices.reshape(1, -1)))
        if horizon_steps > 1 and self.ramp_weight:
            cost += self.ramp_weight * cp.sum_squares(power[:, 1:] - power[:, :-1])
        stage_three_start = perf_counter()
        try:
            self._solve(cp.Problem(cp.Minimize(cost), tail_constraints + [total_delivered >= best_total_energy - self.delivery_tolerance_kwh]))
            stage_three_seconds = perf_counter() - stage_three_start
            raw_schedule = np.asarray(power.value, dtype=float)
            economic_tiebreak_solved = True
        except RuntimeError:
            stage_three_seconds = perf_counter() - stage_three_start
            raw_schedule = stage_two_schedule
            economic_tiebreak_solved = False
        schedule = self._sanitize_schedule(raw_schedule, evs, station_profiles, feeder_cap, current_step=current_step, step_hours=step_hours)
        delivered_values = np.array([ev.delivered_energy_kwh for ev in evs], dtype=float) + step_hours * schedule.sum(axis=1)
        service_ratios = {ev.ev_id: float(delivered_values[index] / ev.requested_energy_kwh) for index, ev in enumerate(evs)}
        self.lower_tail_objective_history.append(best_tail_risk)
        self.solve_timing_history.append({
            "stage_one_seconds": stage_one_seconds,
            "stage_two_seconds": stage_two_seconds,
            "stage_three_seconds": stage_three_seconds,
            "total_seconds": perf_counter() - total_start,
            "expected_shortfall_of_worst_tail": best_tail_risk,
        })
        return MPCResult(
            tuple(ev.ev_id for ev in evs),
            schedule,
            max(0.0, float(np.max(1.0 - np.asarray(list(service_ratios.values()))))),
            float(delivered_values.sum()),
            economic_tiebreak_solved,
            service_ratios,
        )
