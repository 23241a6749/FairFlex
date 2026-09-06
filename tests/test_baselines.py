from fairflex.baselines import (
    earliest_deadline_first,
    equal_share,
    feeder_capped,
    first_come_first_served,
    least_laxity_first,
    round_robin,
    uncontrolled,
)
from fairflex.domain import EVSession


def make_evs():
    return [
        EVSession("early", "station", 0, 8, 10.0, 7.0),
        EVSession("late", "station", 1, 8, 10.0, 7.0),
    ]


def test_fcfs_gives_capacity_to_earliest_arrival_first():
    early, late = make_evs()
    allocation = first_come_first_served({"station": [late, early]}, {"station": 7.0}, 0.25)
    assert allocation == {"early": 7.0, "late": 0.0}


def test_uncontrolled_baseline_keeps_the_physical_station_limit():
    early, late = make_evs()
    allocation = uncontrolled({"station": [early, late]}, {"station": 7.0}, 0.25)
    assert allocation == {"early": 3.5, "late": 3.5}


def test_equal_share_splits_capacity_between_equal_evs():
    early, late = make_evs()
    allocation = equal_share({"station": [early, late]}, {"station": 7.0}, 0.25)
    assert allocation == {"early": 3.5, "late": 3.5}


def test_equal_share_redistributes_when_one_ev_needs_less_energy():
    small = EVSession("small", "station", 0, 2, 0.5, 7.0)
    large = EVSession("large", "station", 0, 2, 10.0, 7.0)
    allocation = equal_share({"station": [small, large]}, {"station": 7.0}, 0.25)
    assert allocation["small"] == 2.0
    assert allocation["large"] == 5.0


def test_continuous_round_robin_matches_equal_share_without_pilot_quantization():
    small = EVSession("small", "station", 0, 2, 0.5, 7.0)
    large = EVSession("large", "station", 0, 2, 10.0, 7.0)

    allocation = round_robin({"station": [small, large]}, {"station": 7.0}, 0.25)

    assert allocation == equal_share({"station": [small, large]}, {"station": 7.0}, 0.25)


def test_earliest_deadline_first_uses_visible_planning_deadline_not_future_unplug():
    flexible = EVSession(
        "flexible",
        "station",
        0,
        1,  # physical unplug is hidden from the controller
        10.0,
        7.0,
        planning_departure_step=8,
    )
    urgent = EVSession(
        "urgent",
        "station",
        0,
        8,
        10.0,
        7.0,
        planning_departure_step=2,
    )

    allocation = earliest_deadline_first(
        {"station": [flexible, urgent]}, {"station": 7.0}, 0.25, current_step=0
    )

    assert allocation == {"urgent": 7.0, "flexible": 0.0}


def test_least_laxity_first_prioritizes_the_tightest_deadline():
    flexible = EVSession("flexible", "station", 0, 8, 10.0, 7.0)
    urgent = EVSession("urgent", "station", 0, 2, 3.5, 7.0)

    allocation = least_laxity_first({"station": [flexible, urgent]}, {"station": 7.0}, 0.25, 0)

    assert allocation == {"urgent": 7.0, "flexible": 0.0}


def test_feeder_cap_makes_a_baseline_feasible_without_changing_its_ratio():
    early, late = make_evs()
    policy = feeder_capped(uncontrolled, lambda step: 4.0 + step)

    allocation = policy({"station": [early, late]}, {"station": 7.0}, 0.25, 0)

    assert sum(allocation.values()) == 4.0
    assert allocation == {"early": 2.0, "late": 2.0}
