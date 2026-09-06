import pytest

from fairflex.baselines import first_come_first_served
from fairflex.domain import EVSession
from fairflex.fair_mpc import CentralizedFairMPC


def test_deadline_guard_reserves_current_power_for_a_feasible_urgent_ev():
    flexible = EVSession("a-flexible", "station", 0, 2, 5.4, 7.2)
    urgent = EVSession("z-urgent", "station", 0, 1, 1.8, 7.2)
    capacities = {"station": 7.2}

    fcfs = first_come_first_served({"station": [flexible, urgent]}, capacities, 0.25, 0)
    mpc = CentralizedFairMPC().solve(
        [flexible, urgent],
        capacities,
        current_step=0,
        horizon_steps=2,
        step_hours=0.25,
    )

    assert fcfs["z-urgent"] == 0.0
    assert mpc.first_step_allocations()["z-urgent"] == pytest.approx(7.2)


def test_mpc_respects_station_and_feeder_caps_for_every_planned_step():
    evs = [
        EVSession("north-ev", "north", 0, 2, 5.0, 7.2),
        EVSession("south-ev", "south", 0, 2, 5.0, 7.2),
    ]
    result = CentralizedFairMPC().solve(
        evs,
        {"north": 7.2, "south": 7.2},
        current_step=0,
        horizon_steps=2,
        step_hours=0.25,
        feeder_capacity_kw=7.2,
    )
    assert result.schedule_kw.sum(axis=0).max() <= 7.2 + 1e-6


def test_mpc_remains_feasible_when_a_shared_cap_blocks_a_deadline_guard():
    urgent = EVSession("urgent", "station", 0, 1, 1.8, 7.2)
    result = CentralizedFairMPC().solve(
        [urgent],
        {"station": [3.6, 7.2]},
        current_step=0,
        horizon_steps=2,
        step_hours=0.25,
    )

    # The EV would need 7.2 kW now, but a safety layer has imposed a 3.6 kW
    # first-step cap. Fair MPC should record the unavoidable deficit, not fail.
    assert result.schedule_kw[0, 0] <= 3.6 + 1e-6
    assert result.service_ratios["urgent"] == pytest.approx(0.5, abs=1e-4)


def test_mpc_uses_later_low_price_when_the_deadline_allows_it():
    ev = EVSession("flexible", "station", 0, 2, 1.8, 7.2)
    result = CentralizedFairMPC(ramp_weight=0.0).solve(
        [ev],
        {"station": 7.2},
        current_step=0,
        horizon_steps=2,
        step_hours=0.25,
        prices_per_kwh=[2.0, 1.0],
    )
    assert result.schedule_kw[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert result.schedule_kw[0, 1] == pytest.approx(7.2, abs=1e-4)


def test_mpc_uses_declared_planning_deadline_not_future_realized_unplug_time():
    ev = EVSession(
        "commitment-aware",
        "station",
        0,
        4,  # physical unplug time: visible only to the simulator
        3.6,
        7.2,
        planning_departure_step=2,  # controller-visible declared deadline
    )

    result = CentralizedFairMPC().solve(
        [ev],
        {"station": 7.2},
        current_step=0,
        horizon_steps=4,
        step_hours=0.25,
    )

    assert result.schedule_kw[0, 2:].sum() == pytest.approx(0.0)


def test_mpc_reopens_only_the_observed_current_slot_after_a_guarded_deadline():
    ev = EVSession(
        "survived-guard",
        "station",
        0,
        6,  # physical future unplug remains hidden from the controller
        3.6,
        7.2,
        planning_departure_step=2,  # calibrated robust deadline
        declared_departure_step=5,  # original driver declaration
    )

    result = CentralizedFairMPC().solve(
        [ev],
        {"station": 7.2},
        current_step=2,
        horizon_steps=3,
        step_hours=0.25,
    )

    # At step 2 the observed connection is real information, so one immediate
    # slot is safe to use. No later slot is assumed available until the next
    # MPC re-plan observes the EV again.
    assert result.schedule_kw[0, 0] == pytest.approx(7.2, abs=1e-4)
    assert result.schedule_kw[0, 1:].sum() == pytest.approx(0.0)
