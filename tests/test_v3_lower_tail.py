import pytest

from fairflex.domain import EVSession
from fairflex.v3.lower_tail_mpc import LowerTailFairMPC


def test_lower_tail_mpc_respects_station_and_feeder_limits_and_records_tail_objective():
    evs = (
        EVSession("a", "north", 0, 2, 4.0, 7.2, planning_departure_step=2),
        EVSession("b", "south", 0, 2, 4.0, 7.2, planning_departure_step=2),
    )
    controller = LowerTailFairMPC(tail_fraction=0.10)
    result = controller.solve(
        evs,
        {"north": 7.2, "south": 7.2},
        current_step=0,
        horizon_steps=2,
        step_hours=0.25,
        feeder_capacity_kw=7.2,
    )

    assert result.schedule_kw.sum(axis=0).max() <= 7.2 + 1e-6
    assert all(0.0 <= ratio <= 1.0 + 1e-6 for ratio in result.service_ratios.values())
    assert controller.lower_tail_objective_history[-1] >= 0.0
    assert controller.solve_timing_history[-1]["expected_shortfall_of_worst_tail"] >= 0.0


def test_lower_tail_mpc_keeps_current_deadline_guard_behavior():
    urgent = EVSession("urgent", "station", 0, 1, 1.8, 7.2, planning_departure_step=1)
    result = LowerTailFairMPC().solve(
        (urgent,),
        {"station": 7.2},
        current_step=0,
        horizon_steps=2,
        step_hours=0.25,
    )
    assert result.first_step_allocations()["urgent"] == pytest.approx(7.2, abs=1e-4)
