import numpy as np

from fairflex.admm import DistributedFairMPCPolicy, FeederADMMNegotiator
from fairflex.domain import EVSession
from fairflex.fair_mpc import CentralizedFairMPC


def test_admm_enforces_feeder_capacity_and_converges():
    result = FeederADMMNegotiator().negotiate(
        {"north": [5.0, 5.0], "south": [5.0, 5.0]},
        {"north": 10.0, "south": 10.0},
        [6.0, 6.0],
    )

    assert result.converged
    allocation = np.vstack([result.station_profiles_kw["north"], result.station_profiles_kw["south"]])
    assert np.all(allocation.sum(axis=0) <= 6.0 + 1e-6)
    assert np.allclose(allocation[0], allocation[1], atol=1e-3)


def test_admm_priority_preserves_more_of_an_urgent_station_request():
    result = FeederADMMNegotiator().negotiate(
        {"flexible": [5.0], "urgent": [5.0]},
        {"flexible": 10.0, "urgent": 10.0},
        6.0,
        priorities={"flexible": 1.0, "urgent": 5.0},
    )

    assert result.converged
    assert result.station_profiles_kw["urgent"][0] > result.station_profiles_kw["flexible"][0]


def test_distributed_policy_never_exceeds_the_negotiated_feeder_profile():
    policy = DistributedFairMPCPolicy(
        CentralizedFairMPC(),
        FeederADMMNegotiator(),
        horizon_steps=2,
        feeder_capacity_kw=7.2,
    )
    sessions = {
        "north": [EVSession("north-1", "north", 0, 2, 3.6, 7.2)],
        "south": [EVSession("south-1", "south", 0, 2, 3.6, 7.2)],
    }
    results = policy.plan(sessions, {"north": 7.2, "south": 7.2}, 0.25, 0)

    assert policy.last_negotiation is not None
    negotiated = policy.last_negotiation.station_profiles_kw
    assert sum(profile[0] for profile in negotiated.values()) <= 7.2 + 1e-6
    assert sum(result.schedule_kw[:, 0].sum() for result in results.values()) <= 7.2 + 1e-6


def test_equity_debt_updates_once_when_an_under_served_ev_departs():
    policy = DistributedFairMPCPolicy(
        CentralizedFairMPC(), FeederADMMNegotiator(), feeder_capacity_kw=7.2
    )
    departed = EVSession("departed", "north", 0, 1, 3.6, 7.2)

    policy.observe_step([departed], completed_through_step=1)
    first_debt = policy.equity_debt["north"]
    policy.observe_step([departed], completed_through_step=1)

    assert first_debt == 1.0
    assert policy.equity_debt["north"] == first_debt
    assert len(policy.debt_history) == 1
