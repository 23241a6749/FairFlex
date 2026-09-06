"""AC feasibility repair for schedules proposed by an optimization policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .admm import DistributedFairMPCPolicy
from .domain import EVSession
from .fair_mpc import CentralizedFairMPC, MPCResult
from .grid import GridValidation, IEEE33Grid


FeederCapacitySource = (
    float
    | Sequence[float]
    | Callable[[int, int], float | Sequence[float]]
)


@dataclass(frozen=True)
class ACRepairResult:
    requested_powers_kw: Mapping[str, float]
    applied_powers_kw: Mapping[str, float]
    initial_grid: GridValidation
    final_grid: GridValidation
    attempts: int

    @property
    def safe(self) -> bool:
        return self.final_grid.safe

    @property
    def curtailed_energy_rate_kw(self) -> float:
        return sum(self.requested_powers_kw.values()) - sum(self.applied_powers_kw.values())


class ACFeasibilityRepair:
    """Curtail the most flexible stations until pandapower confirms feasibility.

    The fast MPC model will later use linearized network constraints. This final
    repair stays deliberately model-faithful: every candidate is checked by AC
    power flow, and bisection retains the largest safe import at each curtailed
    station. The station-level MPC is then re-solved under the accepted caps.
    """

    def __init__(self, grid: IEEE33Grid, *, bisection_steps: int = 14) -> None:
        if bisection_steps <= 0:
            raise ValueError("bisection_steps must be positive")
        self.grid = grid
        self.bisection_steps = bisection_steps

    def repair(
        self,
        station_powers_kw: Mapping[str, float],
        *,
        flexibility_scores: Mapping[str, float] | None = None,
    ) -> ACRepairResult:
        requested = {
            station_id: float(station_powers_kw.get(station_id, 0.0))
            for station_id in self.grid.station_load_indices
        }
        if any(power < 0 for power in requested.values()):
            raise ValueError("station powers must be non-negative")
        initial = self.grid.validate(requested)
        if initial.safe:
            return ACRepairResult(requested, requested.copy(), initial, initial, 0)

        flexibility_scores = flexibility_scores or {}
        order = sorted(
            (station_id for station_id, power in requested.items() if power > 0),
            key=lambda station_id: (-flexibility_scores.get(station_id, 0.0), station_id),
        )
        applied = requested.copy()
        attempts = 1
        validation = initial
        for station_id in order:
            high = applied[station_id]
            candidate = applied.copy()
            candidate[station_id] = 0.0
            zero_validation = self.grid.validate(candidate)
            attempts += 1
            if not zero_validation.safe:
                applied = candidate
                validation = zero_validation
                continue

            low = 0.0
            for _ in range(self.bisection_steps):
                midpoint = (low + high) / 2
                candidate[station_id] = midpoint
                midpoint_validation = self.grid.validate(candidate)
                attempts += 1
                if midpoint_validation.safe:
                    low = midpoint
                    validation = midpoint_validation
                else:
                    high = midpoint
            applied[station_id] = low
            validation = self.grid.validate(applied)
            attempts += 1
            if validation.safe:
                break

        return ACRepairResult(requested, applied, initial, validation, attempts)


class ACRepairedMPCPolicy:
    """Run fair MPC, validate its executed action, then re-plan under AC caps.

    Only the first MPC action is executed before the next observation. It is
    therefore both sufficient and less conservative to perform an exact AC
    check at that step, turn the accepted imports into first-step station caps,
    and re-solve the fair allocation. This retains EV-level fairness after a
    station must curtail; merely scaling individual EV powers would not.
    """

    def __init__(
        self,
        controller: CentralizedFairMPC,
        grid: IEEE33Grid,
        *,
        horizon_steps: int = 4,
        feeder_capacity_kw: FeederCapacitySource | None = None,
        prices_per_kwh: Sequence[float] | None = None,
    ) -> None:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        self.controller = controller
        self.grid = grid
        self.horizon_steps = horizon_steps
        self.feeder_capacity_kw = feeder_capacity_kw
        self.prices_per_kwh = prices_per_kwh
        self.repairer = ACFeasibilityRepair(grid)
        self.last_repair: ACRepairResult | None = None
        self.last_result: MPCResult | None = None
        self.last_feeder_capacity_kw: float | Sequence[float] | None = None
        self.repair_history: list[ACRepairResult] = []
        self.economic_tiebreak_failures = 0

    def _current_feeder_capacity(self, current_step: int) -> float | Sequence[float] | None:
        """Resolve a static or causal time-varying feeder profile for this MPC call."""
        if callable(self.feeder_capacity_kw):
            return self.feeder_capacity_kw(current_step, self.horizon_steps)
        return self.feeder_capacity_kw

    @staticmethod
    def _station_flexibility(
        station_evs: Mapping[str, Sequence[EVSession]],
        *,
        current_step: int,
        step_hours: float,
    ) -> dict[str, float]:
        """Energy slack: larger values identify less urgent station demand."""
        return {
            station_id: sum(
                max(
                    0.0,
                    (ev.effective_planning_deadline_step(current_step) - current_step)
                    * ev.max_power_kw
                    * step_hours
                    - ev.remaining_energy_kwh,
                )
                for ev in evs
            )
            for station_id, evs in station_evs.items()
        }

    def plan(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> tuple[MPCResult, ACRepairResult]:
        if set(station_capacities_kw) != set(self.grid.station_load_indices):
            raise ValueError("policy station capacities must exactly match the AC grid stations")
        active_evs = [ev for evs in station_evs.values() for ev in evs]
        ev_station = {ev.ev_id: ev.station_id for ev in active_evs}
        feeder_capacity = self._current_feeder_capacity(current_step)
        initial = self.controller.solve(
            active_evs,
            station_capacities_kw,
            current_step=current_step,
            horizon_steps=self.horizon_steps,
            step_hours=step_hours,
            feeder_capacity_kw=feeder_capacity,
            prices_per_kwh=self.prices_per_kwh,
        )
        requested = {
            station_id: sum(
                allocation
                for ev_id, allocation in initial.first_step_allocations().items()
                if ev_station[ev_id] == station_id
            )
            for station_id in station_capacities_kw
        }
        repair = self.repairer.repair(
            requested,
            flexibility_scores=self._station_flexibility(
                station_evs, current_step=current_step, step_hours=step_hours
            ),
        )

        if repair.attempts == 0:
            # The first action has already passed exact AC validation, so the
            # initial fair plan is both safe and optimal for its original
            # constraints. Re-solving the identical problem only adds latency.
            result = initial
        else:
            # Every unchanged station is capped at its originally requested
            # import. Hence re-solving cannot introduce a new AC violation at
            # another bus.
            repaired_profiles = {
                station_id: [
                    min(station_capacities_kw[station_id], repair.applied_powers_kw[station_id])
                ]
                + [station_capacities_kw[station_id]] * (self.horizon_steps - 1)
                for station_id in station_capacities_kw
            }
            result = self.controller.solve(
                active_evs,
                repaired_profiles,
                current_step=current_step,
                horizon_steps=self.horizon_steps,
                step_hours=step_hours,
                feeder_capacity_kw=feeder_capacity,
                prices_per_kwh=self.prices_per_kwh,
            )
        applied = {
            station_id: sum(
                allocation
                for ev_id, allocation in result.first_step_allocations().items()
                if ev_station[ev_id] == station_id
            )
            for station_id in station_capacities_kw
        }
        final_grid = self.grid.validate(applied)
        if not final_grid.safe:
            raise RuntimeError("AC repair caps did not produce a safe re-planned action")
        self.last_repair = repair
        self.last_result = result
        self.last_feeder_capacity_kw = feeder_capacity
        self.repair_history.append(repair)
        if not result.economic_tiebreak_solved:
            self.economic_tiebreak_failures += 1
        return result, repair

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        result, _ = self.plan(
            station_evs, station_capacities_kw, step_hours, current_step
        )
        return result.first_step_allocations()


class ACRepairedDistributedPolicy:
    """Make a distributed ADMM action AC-safe without exposing EV-level data.

    ADMM enforces its negotiated feeder-import profile, but a scalar feeder cap
    cannot by itself capture every voltage or line constraint. This wrapper
    applies the same exact AC repair to the aggregate station actions, then each
    station re-solves only its own local fair MPC under the repaired profile.
    """

    def __init__(self, distributed_policy: DistributedFairMPCPolicy, grid: IEEE33Grid) -> None:
        self.distributed_policy = distributed_policy
        self.grid = grid
        self.repairer = ACFeasibilityRepair(grid)
        self.last_repair: ACRepairResult | None = None
        self.last_station_results: Mapping[str, MPCResult] = {}
        self.repair_history: list[ACRepairResult] = []

    def plan(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> Mapping[str, MPCResult]:
        if set(station_capacities_kw) != set(self.grid.station_load_indices):
            raise ValueError("policy station capacities must exactly match the AC grid stations")
        initial_results = self.distributed_policy.plan(
            station_evs, station_capacities_kw, step_hours, current_step
        )
        requested = {
            station_id: float(result.schedule_kw[:, 0].sum())
            for station_id, result in initial_results.items()
        }
        repair = self.repairer.repair(
            requested,
            flexibility_scores=ACRepairedMPCPolicy._station_flexibility(
                station_evs, current_step=current_step, step_hours=step_hours
            ),
        )
        negotiation = self.distributed_policy.last_negotiation
        if negotiation is None:  # Defensive: plan above must populate this state.
            raise RuntimeError("distributed policy did not retain its ADMM result")

        if repair.attempts == 0:
            station_results = dict(initial_results)
        else:
            station_results = {}
            for station_id, evs in station_evs.items():
                capacity_profile = negotiation.station_profiles_kw[station_id].copy()
                capacity_profile[0] = min(capacity_profile[0], repair.applied_powers_kw[station_id])
                station_results[station_id] = self.distributed_policy.controller.solve(
                    list(evs),
                    {station_id: capacity_profile},
                    current_step=current_step,
                    horizon_steps=self.distributed_policy.horizon_steps,
                    step_hours=step_hours,
                    prices_per_kwh=self.distributed_policy.prices_per_kwh,
                )
        applied = {
            station_id: float(result.schedule_kw[:, 0].sum())
            for station_id, result in station_results.items()
        }
        if not self.grid.validate(applied).safe:
            raise RuntimeError("AC repair caps did not produce a safe distributed action")
        self.last_repair = repair
        self.last_station_results = station_results
        self.repair_history.append(repair)
        return station_results

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        return {
            ev_id: allocation
            for result in self.plan(
                station_evs, station_capacities_kw, step_hours, current_step
            ).values()
            for ev_id, allocation in result.first_step_allocations().items()
        }

    def observe_step(
        self, all_evs: Sequence[EVSession], completed_through_step: int
    ) -> None:
        """Forward local completion outcomes to the distributed debt updater."""
        self.distributed_policy.observe_step(all_evs, completed_through_step)
