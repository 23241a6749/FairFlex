"""Lexicographic, deadline-aware centralized model-predictive control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np

from .domain import EVSession


@dataclass(frozen=True)
class MPCResult:
    """A complete planned schedule; the simulator executes only its first column."""

    ev_ids: tuple[str, ...]
    schedule_kw: np.ndarray
    worst_shortfall: float
    total_delivered_kwh: float
    economic_tiebreak_solved: bool
    service_ratios: Mapping[str, float]

    def first_step_allocations(self) -> dict[str, float]:
        return {
            ev_id: max(0.0, float(self.schedule_kw[index, 0]))
            for index, ev_id in enumerate(self.ev_ids)
        }


class CentralizedFairMPC:
    """Solve a transparent three-stage fairness-first charging problem.

    Stage 1 minimises the maximum normalized requested-energy shortfall. Stage 2
    maximises total delivered energy while preserving that fairness result. Stage
    3 minimises time-of-use cost and a small ramp penalty without sacrificing
    either preceding result. This is intentionally preferable to a single
    weighted reward, whose fairness/cost weights would be arbitrary.
    """

    def __init__(
        self,
        *,
        fairness_tolerance: float = 1e-5,
        delivery_tolerance_kwh: float = 1e-5,
        ramp_weight: float = 1e-4,
    ) -> None:
        if fairness_tolerance <= 0 or delivery_tolerance_kwh <= 0:
            raise ValueError("optimization tolerances must be positive")
        if ramp_weight < 0:
            raise ValueError("ramp_weight must be non-negative")
        self.fairness_tolerance = fairness_tolerance
        self.delivery_tolerance_kwh = delivery_tolerance_kwh
        self.ramp_weight = ramp_weight
        self.solve_timing_history: list[dict[str, float]] = []

    @staticmethod
    def _profile(
        value: float | Sequence[float] | None,
        horizon_steps: int,
        *,
        name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return np.full(horizon_steps, float(value))
        profile = np.asarray(value, dtype=float)
        if profile.shape != (horizon_steps,) or np.any(profile < 0):
            raise ValueError(f"{name} must contain {horizon_steps} non-negative values")
        return profile

    @staticmethod
    def _solve(problem: cp.Problem) -> None:
        """Solve a verified MPC stage using the numerically clean solver order."""
        # CLARABEL is the normal solver for these small convex programs. A
        # status-based OSQP fallback handles rare iteration-limit outcomes;
        # HiGHS is retained only as a final alternative because its CVXPY LP
        # reduction path emits runtime warnings on some congested instances.
        # Inspect *status* after every attempt rather than trusting exceptions.
        attempts = (
            (cp.CLARABEL, {}),
            (cp.OSQP, {"max_iter": 100_000}),
            (cp.HIGHS, {}),
        )
        statuses: list[str] = []
        for solver, options in attempts:
            try:
                problem.solve(solver=solver, **options)
            except cp.error.SolverError:
                statuses.append(f"{solver}:solver_error")
                continue
            statuses.append(f"{solver}:{problem.status}")
            # The result of each lexicographic stage becomes a hard constraint
            # in the next stage. An ``optimal_inaccurate`` value can therefore
            # make a feasible next stage look infeasible, so seek a verified
            # optimum from the next solver instead of accepting it immediately.
            if problem.status == cp.OPTIMAL:
                return
        raise RuntimeError(f"MPC did not solve optimally ({'; '.join(statuses)})")

    @staticmethod
    def _sanitize_schedule(
        raw_schedule: np.ndarray,
        evs: Sequence[EVSession],
        station_profiles: Mapping[str, np.ndarray],
        feeder_cap: np.ndarray | None,
        *,
        current_step: int,
        step_hours: float,
    ) -> np.ndarray:
        """Remove solver-scale numerical violations before executing an action."""
        raw = np.asarray(raw_schedule, dtype=float)
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("MPC solver returned a non-finite schedule")
        schedule = np.maximum(raw, 0.0).copy()
        for row, ev in enumerate(evs):
            effective_deadline = ev.effective_planning_deadline_step(current_step)
            for offset in range(schedule.shape[1]):
                if current_step + offset >= effective_deadline:
                    schedule[row, offset] = 0.0
                else:
                    schedule[row, offset] = min(schedule[row, offset], ev.max_power_kw)
            planned_energy = float(schedule[row].sum() * step_hours)
            if planned_energy > ev.remaining_energy_kwh and planned_energy > 0:
                schedule[row] *= ev.remaining_energy_kwh / planned_energy

        for offset in range(schedule.shape[1]):
            for station_id, capacity_profile in station_profiles.items():
                rows = [row for row, ev in enumerate(evs) if ev.station_id == station_id]
                total = float(schedule[rows, offset].sum()) if rows else 0.0
                if total > capacity_profile[offset] and total > 0:
                    schedule[rows, offset] *= capacity_profile[offset] / total
            total_feeder_power = float(schedule[:, offset].sum())
            if feeder_cap is not None and total_feeder_power > feeder_cap[offset] and total_feeder_power > 0:
                schedule[:, offset] *= feeder_cap[offset] / total_feeder_power
        return schedule

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
        """Plan charging for currently active EVs over a finite MPC horizon.

        The optimizer knows declared deadlines for active EVs, but it does not
        assume knowledge of future EV arrivals. Receding-horizon replanning
        incorporates arrivals at later simulation steps.
        """
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
        if prices_per_kwh is None:
            prices = np.zeros(horizon_steps)
        else:
            prices = self._profile(prices_per_kwh, horizon_steps, name="prices_per_kwh")
            assert prices is not None

        station_profiles: dict[str, np.ndarray] = {}
        for station_id, capacity in station_capacities_kw.items():
            profile = self._profile(capacity, horizon_steps, name=f"capacity for {station_id}")
            assert profile is not None
            station_profiles[station_id] = profile

        power = cp.Variable((n_evs, horizon_steps), nonneg=True)
        shortfall = cp.Variable(nonneg=True)
        delivered = np.array([ev.delivered_energy_kwh for ev in evs], dtype=float) + (
            step_hours * cp.sum(power, axis=1)
        )

        constraints: list[cp.Constraint] = [shortfall <= 1.0]
        deadline_guards_kw = np.zeros(n_evs)
        for ev_index, ev in enumerate(evs):
            effective_deadline = ev.effective_planning_deadline_step(current_step)
            constraints.append(delivered[ev_index] <= ev.requested_energy_kwh)
            constraints.append(
                (ev.requested_energy_kwh - delivered[ev_index])
                / ev.requested_energy_kwh
                <= shortfall
            )
            for offset in range(horizon_steps):
                absolute_step = current_step + offset
                if absolute_step < effective_deadline:
                    constraints.append(power[ev_index, offset] <= ev.max_power_kw)
                else:
                    constraints.append(power[ev_index, offset] == 0.0)

            # This lower bound is the power required *now* to keep a currently
            # feasible EV feasible by its deadline if all later slots run at max.
            available_slots = max(0, effective_deadline - current_step)
            future_slots = max(0, available_slots - 1)
            individually_feasible = (
                ev.remaining_energy_kwh
                <= available_slots * ev.max_power_kw * step_hours + self.delivery_tolerance_kwh
            )
            required_now_energy = (
                max(
                    0.0,
                    ev.remaining_energy_kwh
                    - future_slots * ev.max_power_kw * step_hours,
                )
                if individually_feasible
                else 0.0
            )
            deadline_guards_kw[ev_index] = required_now_energy / step_hours

        station_rows_by_id: dict[str, list[int]] = {}
        for station_id, capacity_profile in station_profiles.items():
            station_rows = [index for index, ev in enumerate(evs) if ev.station_id == station_id]
            station_rows_by_id[station_id] = station_rows
            if station_rows:
                constraints.append(cp.sum(power[station_rows, :], axis=0) <= capacity_profile)

        if feeder_cap is not None:
            constraints.append(cp.sum(power, axis=0) <= feeder_cap)

        # A deadline guard must never make the optimization infeasible. For
        # example, AC repair may temporarily cap a station below the power an
        # individual EV would need to finish. Apply the guards only when the
        # shared station and feeder resources can honour all of them; otherwise
        # the lexicographic fairness objective decides the unavoidable deficit.
        guards_fit_feeder = feeder_cap is None or (
            float(deadline_guards_kw.sum()) <= feeder_cap[0] + self.delivery_tolerance_kwh
        )
        if guards_fit_feeder:
            for station_id, station_rows in station_rows_by_id.items():
                if not station_rows:
                    continue
                station_guard_kw = float(deadline_guards_kw[station_rows].sum())
                if station_guard_kw <= station_profiles[station_id][0] + self.delivery_tolerance_kwh:
                    for ev_index in station_rows:
                        if deadline_guards_kw[ev_index] > 0:
                            constraints.append(power[ev_index, 0] >= deadline_guards_kw[ev_index])

        # 1. Equity: make the worst service ratio as high as possible.
        total_start = perf_counter()
        stage_one = cp.Problem(cp.Minimize(shortfall), constraints)
        stage_one_start = perf_counter()
        self._solve(stage_one)
        stage_one_seconds = perf_counter() - stage_one_start
        best_shortfall = float(shortfall.value)

        # 2. Efficiency: retain fairness, then deliver the maximum possible energy.
        fair_constraints = constraints + [shortfall <= best_shortfall + self.fairness_tolerance]
        total_delivered = cp.sum(delivered)
        stage_two = cp.Problem(cp.Maximize(total_delivered), fair_constraints)
        stage_two_start = perf_counter()
        self._solve(stage_two)
        stage_two_seconds = perf_counter() - stage_two_start
        best_total_energy = float(total_delivered.value)
        stage_two_schedule = np.asarray(power.value, dtype=float).copy()

        # 3. Economy/stability: retain both earlier results, then choose the
        # cheapest and smoothest of the equally fair, equally productive plans.
        cost = step_hours * cp.sum(cp.multiply(power, prices.reshape(1, -1)))
        if horizon_steps > 1 and self.ramp_weight:
            cost += self.ramp_weight * cp.sum_squares(power[:, 1:] - power[:, :-1])
        final_constraints = fair_constraints + [
            total_delivered >= best_total_energy - self.delivery_tolerance_kwh
        ]
        stage_three = cp.Problem(cp.Minimize(cost), final_constraints)
        try:
            stage_three_start = perf_counter()
            self._solve(stage_three)
            stage_three_seconds = perf_counter() - stage_three_start
            raw_schedule = np.asarray(power.value, dtype=float)
            economic_tiebreak_solved = True
        except RuntimeError:
            stage_three_seconds = perf_counter() - stage_three_start
            # Stage 3 is only an economy/stability tie-breaker. Its numerical
            # failure must not erase a verified solution to the safety, equity,
            # and delivery objectives established in stages 1–2.
            raw_schedule = stage_two_schedule
            economic_tiebreak_solved = False

        schedule = self._sanitize_schedule(
            raw_schedule,
            evs,
            station_profiles,
            feeder_cap,
            current_step=current_step,
            step_hours=step_hours,
        )
        delivered_values = np.array([ev.delivered_energy_kwh for ev in evs], dtype=float) + (
            step_hours * schedule.sum(axis=1)
        )
        service_ratios = {
            ev.ev_id: float(delivered_values[index] / ev.requested_energy_kwh)
            for index, ev in enumerate(evs)
        }
        self.solve_timing_history.append(
            {
                "stage_one_seconds": stage_one_seconds,
                "stage_two_seconds": stage_two_seconds,
                "stage_three_seconds": stage_three_seconds,
                "total_seconds": perf_counter() - total_start,
            }
        )
        return MPCResult(
            tuple(ev.ev_id for ev in evs),
            schedule,
            max(0.0, float(np.max(1.0 - np.asarray(list(service_ratios.values()))))),
            float(delivered_values.sum()),
            economic_tiebreak_solved,
            service_ratios,
        )

    def policy(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        """Policy adapter for :class:`ChargingSimulation`.

        The default four-step horizon keeps the initial demonstration compact.
        Experiments will configure longer horizons and forecast inputs.
        """
        active_evs = [ev for evs in station_evs.values() for ev in evs]
        result = self.solve(
            active_evs,
            station_capacities_kw,
            current_step=current_step,
            horizon_steps=4,
            step_hours=step_hours,
        )
        return result.first_step_allocations()
