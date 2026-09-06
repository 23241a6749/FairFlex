"""Small deterministic scenarios for manual exploration and automated tests."""

from __future__ import annotations

from .domain import ChargingStation, EVSession
from .grid import IEEE33Grid
from .simulation import ChargingSimulation


def make_demo_simulation() -> ChargingSimulation:
    """Return a three-station scenario with different arrival/departure patterns."""
    stations = [
        ChargingStation("north", bus=6, capacity_kw=28.8),
        ChargingStation("central", bus=18, capacity_kw=28.8),
        ChargingStation("south", bus=30, capacity_kw=28.8),
    ]
    sessions = [
        EVSession("n-1", "north", 0, 8, 10.0, 7.2),
        EVSession("n-2", "north", 1, 5, 6.0, 7.2),
        EVSession("c-1", "central", 0, 12, 18.0, 7.2),
        EVSession("s-1", "south", 2, 6, 7.0, 7.2),
        EVSession("s-2", "south", 2, 10, 12.0, 7.2),
    ]
    grid = IEEE33Grid({station.station_id: station.bus for station in stations})
    return ChargingSimulation(stations, sessions, grid)


def make_deadline_stress_simulation() -> ChargingSimulation:
    """A feasible, congested case where non-deadline-aware policies fail.

    There is exactly enough station energy to satisfy both EVs. The first EV is
    flexible, while the second is already connected but must receive a full
    interval of charge immediately. A good fairness/deadline controller serves
    both; FCFS and equal sharing do not. This isolates deadline awareness from
    future-arrival forecasting, which is a separate uncertainty problem. It is
    a diagnostic scenario, not a claim about average field performance.
    """
    station = ChargingStation("north", bus=6, capacity_kw=7.2)
    sessions = [
        EVSession("flexible", "north", 0, 4, 5.4, 7.2),
        EVSession("urgent", "north", 0, 1, 1.8, 7.2),
    ]
    grid = IEEE33Grid({station.station_id: station.bus})
    return ChargingSimulation([station], sessions, grid)
