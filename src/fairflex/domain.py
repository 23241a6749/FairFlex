"""Domain objects and invariants for EV charging experiments."""

from __future__ import annotations

from dataclasses import dataclass


EPSILON = 1e-9


@dataclass
class EVSession:
    """A single charging request replayed by the simulator.

    ``departure_step`` is the *realized physical* unplug time and is exclusive:
    an EV can receive power at steps before it, but never at or after it.
    ``planning_departure_step`` is the deadline currently supplied to a
    controller. ``declared_departure_step`` preserves the driver's original
    declaration when a risk guard makes the planning deadline earlier. Keeping
    both values separate prevents a trace-replay controller from accidentally
    using a future observed unplug time as if it were available when the EV
    connected.

    Energy is tracked in kWh and power in kW.
    """

    ev_id: str
    station_id: str
    arrival_step: int
    departure_step: int
    requested_energy_kwh: float
    max_power_kw: float
    delivered_energy_kwh: float = 0.0
    planning_departure_step: int | None = None
    declared_departure_step: int | None = None

    def __post_init__(self) -> None:
        if not self.ev_id:
            raise ValueError("ev_id must not be empty")
        if not self.station_id:
            raise ValueError("station_id must not be empty")
        if self.arrival_step < 0 or self.departure_step <= self.arrival_step:
            raise ValueError("departure_step must be after a non-negative arrival_step")
        if (
            self.planning_departure_step is not None
            and self.planning_departure_step <= self.arrival_step
        ):
            raise ValueError("planning departure must be after arrival_step")
        if (
            self.declared_departure_step is not None
            and self.declared_departure_step <= self.arrival_step
        ):
            raise ValueError("declared departure must be after arrival_step")
        if self.requested_energy_kwh <= 0:
            raise ValueError("requested_energy_kwh must be positive")
        if self.max_power_kw <= 0:
            raise ValueError("max_power_kw must be positive")
        if not 0 <= self.delivered_energy_kwh <= self.requested_energy_kwh:
            raise ValueError("delivered energy must lie between zero and requested energy")

    @property
    def remaining_energy_kwh(self) -> float:
        return max(0.0, self.requested_energy_kwh - self.delivered_energy_kwh)

    @property
    def planning_deadline_step(self) -> int:
        """Return the only departure deadline a controller may use for planning.

        Legacy synthetic sessions do not have a separate declared deadline, so
        they fall back to their physical departure.  Causal ACN replays set
        ``planning_departure_step`` explicitly and must use this property in
        every scheduler.
        """
        return self.planning_departure_step or self.departure_step

    def effective_planning_deadline_step(self, current_step: int) -> int:
        """Return the causal deadline for the current MPC re-plan.

        A guarded deadline protects an EV before its risky early-departure
        period. If that deadline has passed and the EV is *observably still
        connected*, retaining it as a permanent hard stop would waste known
        availability. In that case the controller treats the current action as
        the only guaranteed remaining slot and replans after observing the
        next state. This uses no future unplug time.

        Legacy sessions have no separate declaration and retain their physical
        deadline; they cannot be active once it has passed.
        """
        if current_step < 0:
            raise ValueError("current_step must be non-negative")
        deadline = self.planning_deadline_step
        if current_step < deadline or self.declared_departure_step is None:
            return deadline
        return current_step + 1

    def is_active(self, step: int) -> bool:
        return (
            self.arrival_step <= step < self.departure_step
            and self.remaining_energy_kwh > EPSILON
        )

    def feasible_power_limit_kw(self, step_hours: float) -> float:
        """Return the largest useful power without over-delivering requested energy."""
        if step_hours <= 0:
            raise ValueError("step_hours must be positive")
        return min(self.max_power_kw, self.remaining_energy_kwh / step_hours)

    def service_ratio(self) -> float:
        return self.delivered_energy_kwh / self.requested_energy_kwh


@dataclass(frozen=True)
class ChargingStation:
    """A station represented by its feeder bus and physical aggregate capacity."""

    station_id: str
    bus: int
    capacity_kw: float

    def __post_init__(self) -> None:
        if not self.station_id:
            raise ValueError("station_id must not be empty")
        if self.bus < 0:
            raise ValueError("bus must be non-negative")
        if self.capacity_kw <= 0:
            raise ValueError("capacity_kw must be positive")
