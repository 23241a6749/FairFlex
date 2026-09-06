import pytest

from fairflex.domain import EVSession
from fairflex.admm import DistributedFairMPCPolicy, FeederADMMNegotiator
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.grid import IEEE33Grid
from fairflex.safety import (
    ACFeasibilityRepair,
    ACRepairedDistributedPolicy,
    ACRepairedMPCPolicy,
)


def test_ac_repair_keeps_safe_schedule_unchanged():
    grid = IEEE33Grid({"south": 30})
    repair = ACFeasibilityRepair(grid).repair({"south": 10.0})
    assert repair.safe
    assert repair.attempts == 0
    assert repair.applied_powers_kw == repair.requested_powers_kw


def test_ac_repair_curtails_an_unsafe_station_power():
    grid = IEEE33Grid({"south": 30})
    repair = ACFeasibilityRepair(grid).repair({"south": 1_000.0})
    assert not repair.initial_grid.safe
    assert repair.safe
    assert repair.applied_powers_kw["south"] < repair.requested_powers_kw["south"]
    assert repair.curtailed_energy_rate_kw > 0


def test_repaired_mpc_reallocates_under_an_ac_safe_station_cap():
    grid = IEEE33Grid({"south": 30})
    policy = ACRepairedMPCPolicy(CentralizedFairMPC(), grid, horizon_steps=1)
    ev = EVSession("large-request", "south", 0, 1, 250.0, 1_000.0)

    result, repair = policy.plan({"south": [ev]}, {"south": 1_000.0}, 0.25, 0)
    applied_kw = sum(result.first_step_allocations().values())

    assert not repair.initial_grid.safe
    assert repair.safe
    assert applied_kw <= repair.applied_powers_kw["south"] + 1e-5
    assert grid.validate({"south": applied_kw}).safe
    assert result.service_ratios["large-request"] < 1.0


def test_repaired_mpc_resolves_a_causal_feeder_capacity_source_once_per_plan():
    grid = IEEE33Grid({"south": 30})
    calls = []

    def capacity_source(current_step, horizon_steps):
        calls.append((current_step, horizon_steps))
        return [7.2] * horizon_steps

    policy = ACRepairedMPCPolicy(
        CentralizedFairMPC(), grid, horizon_steps=2, feeder_capacity_kw=capacity_source
    )
    ev = EVSession("request", "south", 0, 2, 3.6, 7.2)

    policy.plan({"south": [ev]}, {"south": 7.2}, 0.25, 0)

    assert calls == [(0, 2)]
    assert policy.last_feeder_capacity_kw == [7.2, 7.2]


def test_distributed_action_is_repaired_when_a_feeder_cap_misses_a_voltage_limit():
    grid = IEEE33Grid({"south": 30})
    distributed = DistributedFairMPCPolicy(
        CentralizedFairMPC(),
        FeederADMMNegotiator(),
        horizon_steps=1,
        feeder_capacity_kw=1_000.0,
    )
    policy = ACRepairedDistributedPolicy(distributed, grid)
    ev = EVSession("large-request", "south", 0, 1, 250.0, 1_000.0)

    results = policy.plan({"south": [ev]}, {"south": 1_000.0}, 0.25, 0)
    applied_kw = results["south"].schedule_kw[:, 0].sum()

    assert policy.last_repair is not None
    assert not policy.last_repair.initial_grid.safe
    assert policy.last_repair.safe
    assert grid.validate({"south": applied_kw}).safe
