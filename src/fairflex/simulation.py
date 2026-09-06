"""Deterministic 15-minute EV-charging simulation runner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .domain import ChargingStation, EVSession
from .grid import GridValidation, IEEE33Grid

Policy = Callable[
    [Mapping[str, Sequence[EVSession]], Mapping[str, float], float, int],
    Mapping[str, float],
]


@dataclass(frozen=True)
class StepResult:
    step: int
    allocations_kw: Mapping[str, float]
    station_powers_kw: Mapping[str, float]
    grid: GridValidation


@dataclass(frozen=True)
class SimulationResult:
    """Complete replay trace and final session outcomes for one policy run."""

    steps: tuple[StepResult, ...]
    service_ratios: Mapping[str, float]
    delivered_energy_kwh: float
    requested_energy_kwh: float
    individually_feasible_sessions: int
    unavoidable_individual_shortfall_kwh: float

    @property
    def unsafe_steps(self) -> int:
        return sum(not result.grid.safe for result in self.steps)

class ChargingSimulation:
    """Replay EV sessions while keeping scheduling and grid validation separate."""

    def __init__(
        self,
        stations: Sequence[ChargingStation],
        ev_sessions: Sequence[EVSession],
        grid: IEEE33Grid,
        *,
        step_hours: float = 0.25,
    ) -> None:
        if step_hours <= 0:
            raise ValueError("step_hours must be positive")
        self.stations = {station.station_id: station for station in stations}
        if len(self.stations) != len(stations):
            raise ValueError("station IDs must be unique")
        if set(self.stations) != set(grid.station_load_indices):
            raise ValueError("stations and grid station loads must have identical IDs")
        # EV sessions carry delivered energy and are therefore mutable replay
        # state. Copy them at the simulation boundary so a scenario factory
        # cannot accidentally let a later policy inherit an earlier policy's
        # charging progress.
        self.evs = {
            ev.ev_id: EVSession(
                ev.ev_id,
                ev.station_id,
                ev.arrival_step,
                ev.departure_step,
                ev.requested_energy_kwh,
                ev.max_power_kw,
                ev.delivered_energy_kwh,
                ev.planning_departure_step,
                ev.declared_departure_step,
            )
            for ev in ev_sessions
        }
        if len(self.evs) != len(ev_sessions):
            raise ValueError("EV IDs must be unique")
        unknown_stations = {ev.station_id for ev in ev_sessions} - set(self.stations)
        if unknown_stations:
            raise ValueError(f"EVs reference unknown stations: {sorted(unknown_stations)}")
        self.grid = grid
        self.step_hours = step_hours
        self.step = 0

    @property
    def station_capacities_kw(self) -> dict[str, float]:
        return {station_id: station.capacity_kw for station_id, station in self.stations.items()}

    def active_evs_by_station(self) -> dict[str, list[EVSession]]:
        active: dict[str, list[EVSession]] = defaultdict(list)
        for ev in self.evs.values():
            if ev.is_active(self.step):
                active[ev.station_id].append(ev)
        return {station_id: active[station_id] for station_id in self.stations}

    def _validate_allocations(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        allocations_kw: Mapping[str, float],
    ) -> dict[str, float]:
        active = {ev.ev_id: ev for evs in station_evs.values() for ev in evs}
        unknown = set(allocations_kw) - set(active)
        if unknown:
            raise ValueError(f"policy allocated power to inactive or unknown EVs: {sorted(unknown)}")

        normalized = {ev_id: float(allocations_kw.get(ev_id, 0.0)) for ev_id in active}
        for ev_id, power_kw in normalized.items():
            if power_kw < -1e-9:
                raise ValueError("charging powers must be non-negative")
            if power_kw > active[ev_id].feasible_power_limit_kw(self.step_hours) + 1e-7:
                raise ValueError(f"allocation exceeds useful EV limit for {ev_id}")

        for station_id, evs in station_evs.items():
            total_power = sum(normalized[ev.ev_id] for ev in evs)
            if total_power > self.stations[station_id].capacity_kw + 1e-7:
                raise ValueError(f"allocation exceeds physical capacity at {station_id}")
        return normalized

    def advance(self, policy: Policy) -> StepResult:
        station_evs = self.active_evs_by_station()
        proposed = policy(station_evs, self.station_capacities_kw, self.step_hours, self.step)
        allocations = self._validate_allocations(station_evs, proposed)
        station_powers = {
            station_id: sum(allocations[ev.ev_id] for ev in evs)
            for station_id, evs in station_evs.items()
        }
        cached_grid = getattr(policy, "last_applied_grid_validation", None)
        cached_powers = getattr(policy, "last_applied_station_powers_kw", None)
        if cached_grid is not None and cached_powers is not None and all(
            abs(float(cached_powers.get(station_id, 0.0)) - power_kw) <= 1e-9
            for station_id, power_kw in station_powers.items()
        ):
            grid_result = cached_grid
        else:
            grid_result = self.grid.validate(station_powers)

        for ev_id, power_kw in allocations.items():
            ev = self.evs[ev_id]
            ev.delivered_energy_kwh = min(
                ev.requested_energy_kwh,
                ev.delivered_energy_kwh + power_kw * self.step_hours,
            )

        # Some stateful policies (for example, distributed equity debt) need
        # the *actual* outcome after energy is committed, not their proposed
        # schedule. The optional hook keeps ordinary stateless baselines fully
        # compatible and does not expose this local state to a coordinator.
        observe_step = getattr(policy, "observe_step", None)
        if callable(observe_step):
            observe_step(tuple(self.evs.values()), self.step + 1)

        result = StepResult(self.step, allocations, station_powers, grid_result)
        self.step += 1
        return result

    def run(self, policy: Policy, *, through_step: int | None = None) -> SimulationResult:
        """Replay from the current step through a finite, deterministic horizon."""
        if through_step is None:
            through_step = max((ev.departure_step for ev in self.evs.values()), default=self.step)
        if through_step < self.step:
            raise ValueError("through_step cannot precede the current simulation step")
        trace = tuple(self.advance(policy) for _ in range(self.step, through_step))
        return SimulationResult(
            trace,
            {ev_id: ev.service_ratio() for ev_id, ev in self.evs.items()},
            sum(ev.delivered_energy_kwh for ev in self.evs.values()),
            sum(ev.requested_energy_kwh for ev in self.evs.values()),
            sum(
                ev.requested_energy_kwh
                <= (ev.departure_step - ev.arrival_step) * ev.max_power_kw * self.step_hours
                + 1e-9
                for ev in self.evs.values()
            ),
            sum(
                max(
                    0.0,
                    ev.requested_energy_kwh
                    - (ev.departure_step - ev.arrival_step) * ev.max_power_kw * self.step_hours,
                )
                for ev in self.evs.values()
            ),
        )
