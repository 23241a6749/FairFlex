import pytest

from fairflex.baselines import equal_share, uncontrolled
from fairflex.domain import ChargingStation, EVSession
from fairflex.grid import IEEE33Grid
from fairflex.simulation import ChargingSimulation


def make_one_station_simulation(requested_energy_kwh: float = 1.0) -> ChargingSimulation:
    stations = [ChargingStation("station", 6, 7.2)]
    evs = [EVSession("ev", "station", 0, 2, requested_energy_kwh, 7.2)]
    grid = IEEE33Grid({"station": 6})
    return ChargingSimulation(stations, evs, grid)


def test_simulation_caps_energy_at_the_requested_amount():
    simulation = make_one_station_simulation(requested_energy_kwh=1.0)
    result = simulation.advance(uncontrolled)
    assert result.allocations_kw["ev"] == 4.0
    assert simulation.evs["ev"].delivered_energy_kwh == 1.0


def test_simulation_rejects_power_to_departed_ev():
    simulation = make_one_station_simulation()
    simulation.step = 2
    with pytest.raises(ValueError, match="inactive or unknown"):
        simulation.advance(lambda *_: {"ev": 1.0})


def test_actual_unplug_time_remains_a_hard_simulation_limit_when_planning_deadline_is_later():
    station = ChargingStation("station", 6, 7.2)
    ev = EVSession("ev", "station", 0, 1, 3.6, 7.2, planning_departure_step=4)
    simulation = ChargingSimulation([station], [ev], IEEE33Grid({"station": 6}))

    simulation.advance(uncontrolled)
    assert not simulation.evs["ev"].is_active(1)


def test_grid_reports_unsafe_heavy_station_load():
    grid = IEEE33Grid({"south": 30})
    result = grid.validate({"south": 1_000.0})
    assert not result.safe


def test_equal_share_policy_can_run_through_the_simulator():
    simulation = make_one_station_simulation(requested_energy_kwh=3.0)
    result = simulation.advance(equal_share)
    assert result.grid.converged
    assert simulation.evs["ev"].delivered_energy_kwh == 1.8


def test_run_returns_a_complete_trace_and_final_service_metrics():
    simulation = make_one_station_simulation(requested_energy_kwh=3.0)
    result = simulation.run(equal_share)

    assert len(result.steps) == 2
    assert result.unsafe_steps == 0
    assert result.delivered_energy_kwh == 3.0
    assert result.service_ratios["ev"] == 1.0
