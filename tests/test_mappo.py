import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fairflex.domain import ChargingStation, EVSession
from fairflex.grid import IEEE33Grid
from fairflex.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    StationMARLEnvironment,
    load_station_marl_actor,
)
from fairflex.simulation import ChargingSimulation
from fairflex.v3.lower_tail_mpc import LowerTailFairMPC


def _scenario():
    stations = (
        ChargingStation("north", 6, 7.2),
        ChargingStation("south", 30, 7.2),
    )
    grid = IEEE33Grid({station.station_id: station.bus for station in stations})
    sessions = (
        EVSession("north-1", "north", 0, 3, 4.0, 7.2),
        EVSession("south-1", "south", 0, 3, 4.0, 7.2),
    )
    return ChargingSimulation(stations, sessions, grid)


def _environment():
    return StationMARLEnvironment(lambda: _scenario(), lambda simulation: 7.2, horizon_steps=2)


def test_station_marl_environment_projects_an_unsafe_raw_request_safely():
    environment = _environment()
    observations = environment.reset()

    _, _, _, info = environment.step({agent_id: 1.0 for agent_id in observations})

    assert info["raw_feeder_violation"]
    assert info["raw_feeder_excess_kw"] > 0
    assert info["raw_feeder_excess_fraction"] > 0
    assert info["actual_grid_safe"]
    assert observations["north"].shape == (6,)
    assert np.all(np.isfinite(observations["north"]))
    assert environment.raw_feeder_violation_penalty > 0


def test_feasible_cap_transform_makes_priority_actions_feeder_safe_by_construction():
    environment = StationMARLEnvironment(
        lambda: _scenario(),
        lambda simulation: 7.2,
        horizon_steps=2,
        feasible_cap_transform=True,
    )
    observations = environment.reset()

    _, _, _, info = environment.step({agent_id: 1.0 for agent_id in observations})

    assert not info["raw_feeder_violation"]
    assert info["raw_feeder_excess_kw"] == 0.0
    assert info["actual_grid_safe"]


def test_station_marl_can_use_the_same_v3_lower_tail_allocator():
    environment = StationMARLEnvironment(
        lambda: _scenario(),
        lambda simulation: 7.2,
        controller_factory=lambda: LowerTailFairMPC(tail_fraction=0.10),
        horizon_steps=2,
        feasible_cap_transform=True,
    )

    observations = environment.reset()
    _, _, _, info = environment.step({agent_id: 0.5 for agent_id in observations})

    assert isinstance(environment.policy.controller, LowerTailFairMPC)
    assert info["actual_grid_safe"]


def test_mappo_trainer_updates_and_writes_a_loadable_actor(tmp_path):
    trainer = MAPPOTrainer(6, 2, config=MAPPOConfig(hidden_dim=16, update_epochs=1), seed=7)

    history = trainer.train(lambda episode: _environment(), episodes=2)
    checkpoint = tmp_path / "safe_mappo.pt"
    trainer.save_checkpoint(checkpoint)

    assert len(history) == 2
    assert all(item["steps"] == 3 for item in history)
    assert checkpoint.is_file()


def test_parameter_shared_ippo_uses_only_local_values_and_writes_metadata(tmp_path):
    trainer = MAPPOTrainer(
        6,
        2,
        config=MAPPOConfig(hidden_dim=16, update_epochs=1),
        seed=8,
        critic_mode="local",
        action_parameterization="feasible_cap_transform",
    )

    history = trainer.train(lambda episode: _environment(), episodes=2)
    checkpoint = tmp_path / "safe_ippo.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor = load_station_marl_actor(checkpoint)

    assert len(history) == 2
    assert trainer.algorithm == "ippo_parameter_shared"
    assert payload["format"] == "fairflex_safe_marl_v2"
    assert payload["critic_mode"] == "local"
    assert payload["algorithm"] == "ippo_parameter_shared"
    assert actor.fairflex_algorithm == "ippo_parameter_shared"
    assert actor.fairflex_action_parameterization == "feasible_cap_transform"
