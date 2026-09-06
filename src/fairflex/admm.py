"""Privacy-preserving feeder negotiation using consensus ADMM."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .domain import EVSession
from .fair_mpc import CentralizedFairMPC, MPCResult


FeederCapacitySource = (
    float
    | Sequence[float]
    | Callable[[int, int], float | Sequence[float]]
)


@dataclass(frozen=True)
class ADMMResult:
    """Station import caps agreed without sharing individual EV sessions."""

    station_profiles_kw: Mapping[str, np.ndarray]
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool


def _nonnegative_profile(
    value: float | Sequence[float], horizon_steps: int, name: str
) -> np.ndarray:
    if isinstance(value, (float, int)):
        result = np.full(horizon_steps, float(value))
    else:
        result = np.asarray(value, dtype=float)
        if result.shape != (horizon_steps,):
            raise ValueError(f"{name} must contain exactly {horizon_steps} values")
    if np.any(result < 0):
        raise ValueError(f"{name} must be non-negative")
    return result


class FeederADMMNegotiator:
    """Allocate feeder capacity from station-level desired import profiles.

    Each station keeps its EV sessions and local MPC objective private. It sends
    only a desired aggregate profile and an auditable scalar priority. ADMM then
    solves a convex consensus problem whose solution is equivalent to minimizing
    weighted deviation from those profiles under feeder and station caps.
    """

    def __init__(
        self,
        *,
        rho: float = 2.0,
        max_iterations: int = 250,
        tolerance: float = 1e-4,
    ) -> None:
        if rho <= 0 or max_iterations <= 0 or tolerance <= 0:
            raise ValueError("rho, max_iterations, and tolerance must be positive")
        self.rho = rho
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    @staticmethod
    def _project_capped_profile(
        values: np.ndarray, upper_bounds: np.ndarray, feeder_cap: float
    ) -> np.ndarray:
        """Euclidean projection onto ``0 <= x <= upper_bounds, sum(x) <= cap``."""
        clipped = np.clip(values, 0.0, upper_bounds)
        if clipped.sum() <= feeder_cap:
            return clipped

        # x(lambda) = clip(values - lambda, 0, upper). Its sum is monotone,
        # so bisection obtains the unique capacity-binding projection.
        low = 0.0
        high = max(float(np.max(values)), 0.0)
        for _ in range(60):
            midpoint = (low + high) / 2
            candidate = np.clip(values - midpoint, 0.0, upper_bounds)
            if candidate.sum() > feeder_cap:
                low = midpoint
            else:
                high = midpoint
        return np.clip(values - high, 0.0, upper_bounds)

    def negotiate(
        self,
        desired_profiles_kw: Mapping[str, Sequence[float]],
        station_capacities_kw: Mapping[str, float | Sequence[float]],
        feeder_capacity_kw: float | Sequence[float],
        *,
        priorities: Mapping[str, float] | None = None,
    ) -> ADMMResult:
        """Return capacity profiles; no individual EV identifier is an input."""
        station_ids = tuple(sorted(desired_profiles_kw))
        if not station_ids:
            return ADMMResult({}, 0, 0.0, 0.0, True)
        if set(station_ids) != set(station_capacities_kw):
            raise ValueError("desired profiles and station capacities must share station ids")

        first_profile = np.asarray(desired_profiles_kw[station_ids[0]], dtype=float)
        if first_profile.ndim != 1 or first_profile.size == 0:
            raise ValueError("desired profiles must be non-empty one-dimensional arrays")
        horizon_steps = first_profile.size
        desired = np.vstack(
            [
                _nonnegative_profile(desired_profiles_kw[station_id], horizon_steps, station_id)
                for station_id in station_ids
            ]
        )
        upper_bounds = np.vstack(
            [
                _nonnegative_profile(
                    station_capacities_kw[station_id], horizon_steps, station_id
                )
                for station_id in station_ids
            ]
        )
        feeder_cap = _nonnegative_profile(feeder_capacity_kw, horizon_steps, "feeder capacity")
        priority_values = np.array(
            [(priorities or {}).get(station_id, 1.0) for station_id in station_ids],
            dtype=float,
        ).reshape(-1, 1)
        if np.any(priority_values <= 0):
            raise ValueError("station priorities must be positive")

        # x is held locally; z is the coordinator's feasible consensus action;
        # u contains the dual congestion signal sent back to each station.
        x = np.minimum(desired, upper_bounds)
        z = np.column_stack(
            [
                self._project_capped_profile(x[:, step], upper_bounds[:, step], feeder_cap[step])
                for step in range(horizon_steps)
            ]
        )
        dual = np.zeros_like(x)
        primal_residual = dual_residual = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            x = np.clip(
                (priority_values * desired + self.rho * (z - dual))
                / (priority_values + self.rho),
                0.0,
                upper_bounds,
            )
            previous_z = z
            z = np.column_stack(
                [
                    self._project_capped_profile(
                        x[:, step] + dual[:, step],
                        upper_bounds[:, step],
                        feeder_cap[step],
                    )
                    for step in range(horizon_steps)
                ]
            )
            dual += x - z
            primal_residual = float(np.linalg.norm(x - z))
            dual_residual = float(self.rho * np.linalg.norm(z - previous_z))
            if primal_residual <= self.tolerance and dual_residual <= self.tolerance:
                return ADMMResult(
                    {station_id: z[index].copy() for index, station_id in enumerate(station_ids)},
                    iteration,
                    primal_residual,
                    dual_residual,
                    True,
                )

        return ADMMResult(
            {station_id: z[index].copy() for index, station_id in enumerate(station_ids)},
            self.max_iterations,
            primal_residual,
            dual_residual,
            False,
        )


class DistributedFairMPCPolicy:
    """Local fair MPC plus ADMM feeder negotiation.

    Stations first calculate their own fair desired charging profile. The
    coordinator sees only those aggregate profiles, allocates import caps with
    ADMM, and each station solves its fair local problem again under its cap.
    ``equity_debt`` optionally carries a station's past service deficit across
    rolling windows, preventing a repeatedly congested station from being
    systematically disadvantaged.
    """

    def __init__(
        self,
        controller: CentralizedFairMPC,
        negotiator: FeederADMMNegotiator,
        *,
        horizon_steps: int = 4,
        feeder_capacity_kw: FeederCapacitySource,
        prices_per_kwh: Sequence[float] | None = None,
        equity_debt: Mapping[str, float] | None = None,
        equity_debt_decay: float = 0.95,
        equity_debt_gain: float = 1.0,
    ) -> None:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if not 0 <= equity_debt_decay <= 1 or equity_debt_gain < 0:
            raise ValueError("equity debt decay must be in [0, 1] and gain non-negative")
        self.controller = controller
        self.negotiator = negotiator
        self.horizon_steps = horizon_steps
        self.feeder_capacity_kw = feeder_capacity_kw
        self.prices_per_kwh = prices_per_kwh
        self.equity_debt = dict(equity_debt or {})
        if any(value < 0 for value in self.equity_debt.values()):
            raise ValueError("equity debt values must be non-negative")
        self.equity_debt_decay = equity_debt_decay
        self.equity_debt_gain = equity_debt_gain
        self._settled_ev_ids: set[str] = set()
        self.debt_history: list[tuple[int, Mapping[str, float]]] = []
        self.last_negotiation: ADMMResult | None = None
        self.last_station_results: Mapping[str, MPCResult] = {}
        self.negotiation_history: list[ADMMResult] = []

    def _current_feeder_capacity(self, current_step: int) -> float | Sequence[float]:
        if callable(self.feeder_capacity_kw):
            return self.feeder_capacity_kw(current_step, self.horizon_steps)
        return self.feeder_capacity_kw

    @staticmethod
    def _priority(evs: Sequence[EVSession], current_step: int, step_hours: float) -> float:
        if not evs:
            return 1.0
        criticality = max(
            min(
                1.0,
                ev.remaining_energy_kwh
                / max(
                    (ev.effective_planning_deadline_step(current_step) - current_step)
                    * ev.max_power_kw
                    * step_hours,
                    1e-9,
                ),
            )
            for ev in evs
        )
        return 1.0 + criticality

    def plan(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> Mapping[str, MPCResult]:
        if set(station_evs) != set(station_capacities_kw):
            raise ValueError("station EVs and capacities must share station ids")

        desired: dict[str, np.ndarray] = {}
        priorities: dict[str, float] = {}
        for station_id, evs in station_evs.items():
            local = self.controller.solve(
                list(evs),
                {station_id: station_capacities_kw[station_id]},
                current_step=current_step,
                horizon_steps=self.horizon_steps,
                step_hours=step_hours,
                prices_per_kwh=self.prices_per_kwh,
            )
            desired[station_id] = local.schedule_kw.sum(axis=0)
            priorities[station_id] = self._priority(evs, current_step, step_hours) + self.equity_debt.get(
                station_id, 0.0
            )

        negotiation = self.negotiator.negotiate(
            desired,
            station_capacities_kw,
            self._current_feeder_capacity(current_step),
            priorities=priorities,
        )
        if not negotiation.converged:
            raise RuntimeError("ADMM negotiation did not converge within the configured iteration limit")

        station_results = {
            station_id: self.controller.solve(
                list(evs),
                {station_id: negotiation.station_profiles_kw[station_id]},
                current_step=current_step,
                horizon_steps=self.horizon_steps,
                step_hours=step_hours,
                prices_per_kwh=self.prices_per_kwh,
            )
            for station_id, evs in station_evs.items()
        }
        self.last_negotiation = negotiation
        self.last_station_results = station_results
        self.negotiation_history.append(negotiation)
        return station_results

    def observe_step(
        self, all_evs: Sequence[EVSession], completed_through_step: int
    ) -> None:
        """Update private station debt once for each EV that has departed.

        A station's debt is a decayed memory of normalized service shortfall.
        It affects a later ADMM priority but no EV ID, deadline, or requested
        energy leaves the station. Calling this twice for the same EV is
        intentionally idempotent.
        """
        if completed_through_step < 0:
            raise ValueError("completed_through_step must be non-negative")
        updated = False
        for ev in all_evs:
            if ev.departure_step > completed_through_step or ev.ev_id in self._settled_ev_ids:
                continue
            self._settled_ev_ids.add(ev.ev_id)
            previous = self.equity_debt.get(ev.station_id, 0.0)
            shortfall = max(0.0, 1.0 - ev.service_ratio())
            self.equity_debt[ev.station_id] = (
                self.equity_debt_decay * previous + self.equity_debt_gain * shortfall
            )
            updated = True
        if updated:
            self.debt_history.append((completed_through_step, dict(self.equity_debt)))

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        station_results = self.plan(
            station_evs, station_capacities_kw, step_hours, current_step
        )
        return {
            ev_id: allocation
            for result in station_results.values()
            for ev_id, allocation in result.first_step_allocations().items()
        }
