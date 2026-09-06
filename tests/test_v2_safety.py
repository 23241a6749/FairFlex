from __future__ import annotations

from fairflex.baselines import uncontrolled
from fairflex.domain import EVSession
from fairflex.grid import IEEE33Grid
from fairflex.v2.safety import ACRepairedBaselinePolicyV2


def test_v2_baseline_wrapper_applies_common_feeder_and_ac_repair() -> None:
    grid = IEEE33Grid({"south": 30})
    policy = ACRepairedBaselinePolicyV2(
        uncontrolled,
        grid,
        feeder_capacity_kw=1_000.0,
    )
    session = EVSession("large", "south", 0, 1, 250.0, 1_000.0)

    action = policy({"south": [session]}, {"south": 1_000.0}, 0.25, 0)

    assert policy.last_repair is not None
    assert not policy.last_repair.initial_grid.safe
    assert policy.last_repair.safe
    assert sum(action.values()) <= policy.last_repair.applied_powers_kw["south"] + 1e-8
    assert grid.validate({"south": sum(action.values())}).safe
