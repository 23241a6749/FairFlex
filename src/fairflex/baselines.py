"""Transparent baseline policies used before any optimization is introduced."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .domain import EVSession

Allocation = dict[str, float]
Policy = Callable[
    [Mapping[str, Sequence[EVSession]], Mapping[str, float], float, int], Allocation
]


def uncontrolled(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    step_hours: float,
    current_step: int = 0,
) -> Allocation:
    """Charge every active EV at its useful maximum power.

    This represents uncoordinated charging. It intentionally ignores feeder
    state, price, urgency, and fairness. The site hardware limit still remains
    physical: when simultaneous plug-in demand exceeds it, the model applies a
    neutral proportional breaker-share rather than creating an impossible load.
    """
    del current_step
    allocations: Allocation = {}
    for station_id, evs in station_evs.items():
        requested = {ev.ev_id: ev.feasible_power_limit_kw(step_hours) for ev in evs}
        total_requested = sum(requested.values())
        scale = min(1.0, station_capacities_kw[station_id] / total_requested) if total_requested else 1.0
        allocations.update({ev_id: power * scale for ev_id, power in requested.items()})
    return allocations


def first_come_first_served(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    step_hours: float,
    current_step: int = 0,
) -> Allocation:
    """Allocate each station's capacity in arrival order.

    FCFS is a useful operational baseline: easy to deploy, but it can be unfair
    when an urgent EV arrives after flexible EVs have already occupied capacity.
    """
    del current_step
    allocations: Allocation = {}
    for station_id, evs in station_evs.items():
        capacity_left = station_capacities_kw[station_id]
        for ev in sorted(evs, key=lambda item: (item.arrival_step, item.ev_id)):
            power = min(capacity_left, ev.feasible_power_limit_kw(step_hours))
            allocations[ev.ev_id] = max(0.0, power)
            capacity_left -= power
    return allocations


def earliest_deadline_first(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    step_hours: float,
    current_step: int = 0,
) -> Allocation:
    """Allocate each station's capacity to the earliest planning deadline.

    EDF is a standard online charging/deadline-scheduling baseline.  It uses
    the controller-visible effective planning deadline, so a commitment guard
    can make an EV urgent without exposing its future physical unplug time.
    EDF is deliberately not a fairness optimizer: two equally urgent EVs are
    ordered deterministically by arrival and identifier.
    """
    allocations: Allocation = {}
    for station_id, evs in station_evs.items():
        capacity_left = station_capacities_kw[station_id]
        ordered = sorted(
            evs,
            key=lambda item: (
                item.effective_planning_deadline_step(current_step),
                item.arrival_step,
                item.ev_id,
            ),
        )
        for ev in ordered:
            power = min(capacity_left, ev.feasible_power_limit_kw(step_hours))
            allocations[ev.ev_id] = max(0.0, power)
            capacity_left -= power
    return allocations


def equal_share(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    step_hours: float,
    current_step: int = 0,
) -> Allocation:
    """Water-fill each station capacity equally across active EVs.

    Equal sharing is more equitable than FCFS, but it is not deadline-aware.
    An EV with very little time remaining can still receive too little power.
    """
    del current_step
    allocations: Allocation = {}
    for station_id, evs in station_evs.items():
        unfilled = list(evs)
        capacity_left = station_capacities_kw[station_id]
        while unfilled and capacity_left > 1e-9:
            equal_power = capacity_left / len(unfilled)
            bounded = [
                ev
                for ev in unfilled
                if ev.feasible_power_limit_kw(step_hours) <= equal_power + 1e-9
            ]
            if not bounded:
                for ev in unfilled:
                    allocations[ev.ev_id] = allocations.get(ev.ev_id, 0.0) + equal_power
                capacity_left = 0.0
                break

            for ev in bounded:
                power = ev.feasible_power_limit_kw(step_hours)
                allocations[ev.ev_id] = allocations.get(ev.ev_id, 0.0) + power
                capacity_left -= power
            unfilled = [ev for ev in unfilled if ev not in bounded]

        for ev in evs:
            allocations.setdefault(ev.ev_id, 0.0)
    return allocations


def round_robin(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    step_hours: float,
    current_step: int = 0,
) -> Allocation:
    """Ideal continuous-rate Round Robin benchmark.

    ACN-Sim's Round Robin repeatedly allocates a *discrete* pilot-current
    increment to each active EV.  FairFlex models continuous power and has no
    EVSE pilot-current quantum.  In that continuous limit, the completed
    Round-Robin cycle is exactly the water-filled equal-share allocation.
    Delegating to :func:`equal_share` preserves this mathematical equivalence
    and avoids fabricating an arbitrary hardware-specific increment.  We keep
    the separate name so paper tables can state the standard benchmark and
    explicitly note that it is equivalent here.
    """
    return equal_share(station_evs, station_capacities_kw, step_hours, current_step)


def least_laxity_first(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    step_hours: float,
    current_step: int = 0,
) -> Allocation:
    """Allocate each station's capacity to the least-flexible EVs first.

    Laxity is the remaining energy that an EV could still receive at maximum
    charging rate before departure minus its unmet request.  Lower laxity means
    less schedule flexibility, so this online rule prioritizes the EV most at
    risk of missing its requested energy.  It uses no future arrivals, forecast
    data, or optimization solver.
    """
    allocations: Allocation = {}
    for station_id, evs in station_evs.items():
        capacity_left = station_capacities_kw[station_id]

        def laxity(ev: EVSession) -> tuple[float, int, str]:
            effective_deadline = ev.effective_planning_deadline_step(current_step)
            remaining_slots = max(0, effective_deadline - current_step)
            slack_kwh = remaining_slots * ev.max_power_kw * step_hours - ev.remaining_energy_kwh
            return (slack_kwh, effective_deadline, ev.ev_id)

        for ev in sorted(evs, key=laxity):
            power = min(capacity_left, ev.feasible_power_limit_kw(step_hours))
            allocations[ev.ev_id] = max(0.0, power)
            capacity_left -= power
        for ev in evs:
            allocations.setdefault(ev.ev_id, 0.0)
    return allocations


def feeder_capped(
    policy: Policy,
    feeder_capacity_kw: float | Callable[[int], float],
) -> Policy:
    """Apply a common current-step feeder cap to a transparent baseline.

    The wrapper preserves the baseline's proposed relative allocations and
    scales them only when their aggregate import exceeds the common limit. It
    is intentionally *not* a fairness optimizer: this keeps FCFS and equal
    sharing meaningful baselines while preventing an infeasible method from
    gaining energy simply by violating the feeder constraint.
    """

    def capped_policy(
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> Allocation:
        proposed = policy(station_evs, station_capacities_kw, step_hours, current_step)
        cap = (
            float(feeder_capacity_kw(current_step))
            if callable(feeder_capacity_kw)
            else float(feeder_capacity_kw)
        )
        if cap < 0:
            raise ValueError("feeder capacity must be non-negative")
        requested = sum(proposed.values())
        if requested <= cap or requested <= 0:
            return proposed
        scale = cap / requested
        return {ev_id: power * scale for ev_id, power in proposed.items()}

    return capped_policy
