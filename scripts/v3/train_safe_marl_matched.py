"""Train predeclared Safe-MAPPO/IPPO seeds under the FairFlex-UC V3 protocol.

This is intentionally a *pre-test* command.  It rejects a present final test
file, trains on the registered historical training days only, validates without
selecting a best test seed, and writes one immutable directory per seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from fairflex.evaluation import summarize_outcome
from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.forecasting import RobustForecastCapacitySource
from fairflex.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    SafeMAPPOStationPolicy,
    SafeStationCapPolicy,
    StationMARLEnvironment,
)
from fairflex.scenarios import (
    ReplayWindow,
    load_pv_proxy,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
    with_replay_window,
)
from fairflex.v3.lower_tail_mpc import LowerTailFairMPC
from fairflex.v3.marl_protocol import fit_v3_hybrid_deadline_preprocessor


def _project_root(config_path: Path) -> Path:
    return next(parent for parent in (config_path.parent, *config_path.parents) if (parent / "pyproject.toml").is_file())


def _whole_days(window: ReplayWindow) -> list[ReplayWindow]:
    return [
        ReplayWindow(start, start + pd.Timedelta(days=1))
        for start in pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left")
    ]


def _environment(
    config: dict,
    prepared,
    *,
    pv_proxy: pd.Series,
    forecaster,
    reward: dict,
) -> StationMARLEnvironment:
    feeder = config["feeder_sensitivity"]

    def scenario_factory():
        return make_trace_simulation(config, prepared)

    def feeder_capacity_factory(simulation):
        return RobustForecastCapacitySource(
            forecaster,
            pv_proxy,
            replay_origin=prepared.window.start,
            grid_import_limit_kw=float(feeder["base_import_limit_kw"]),
            step_minutes=int(config["time_step_minutes"]),
            metered_first_step=True,
        )

    return StationMARLEnvironment(
        scenario_factory,
        feeder_capacity_factory,
        controller_factory=lambda: LowerTailFairMPC(tail_fraction=float(config["v3"]["tail_fraction"])),
        horizon_steps=int(feeder["forecast_horizon_steps"]),
        fairness_reward_weight=float(reward["fairness_reward_weight"]),
        raw_feeder_violation_penalty=float(reward["raw_feeder_violation_penalty"]),
        raw_ac_violation_penalty=float(reward["raw_ac_violation_penalty"]),
        feasible_cap_transform=True,
    )


def _evaluate_actor(actor, environment: StationMARLEnvironment, policy_name: str) -> dict[str, object]:
    simulation = environment.scenario_factory()
    safe_policy = SafeStationCapPolicy(
        environment.controller_factory(),
        simulation.grid,
        horizon_steps=environment.horizon_steps,
        feeder_capacity_kw=environment.feeder_capacity_factory(simulation),
        feasible_cap_transform=True,
    )
    start = perf_counter()
    result = simulation.run(SafeMAPPOStationPolicy(actor, safe_policy))
    return {
        **asdict(summarize_outcome(policy_name, result, runtime_seconds=perf_counter() - start)),
        "raw_feeder_violation_steps": sum(audit.raw_feeder_violation for audit in safe_policy.action_history),
        "raw_ac_unsafe_steps": sum(audit.raw_ac_unsafe for audit in safe_policy.action_history),
        "ac_repair_activations": sum(audit.ac_repair.attempts > 0 for audit in safe_policy.action_history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v3/caltech_2019_safe_marl_matched_protocol.json"),
    )
    parser.add_argument("--algorithm", choices=("mappo", "ippo"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, help="Override only to resume a predeclared seed list")
    parser.add_argument("--episodes", type=int, help="Override only to resume the predeclared episode budget")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3/safe_marl_matched_training"))
    args = parser.parse_args()

    config = load_study_config(args.config)
    declaration = config.get("safe_marl")
    if not isinstance(declaration, dict):
        parser.error("config requires a safe_marl protocol object")
    training = declaration.get("training")
    reward = declaration.get("reward")
    if not isinstance(training, dict) or not isinstance(reward, dict):
        parser.error("safe_marl.training and safe_marl.reward must be objects")
    configured_seeds = [int(seed) for seed in training["seeds"]]
    seeds = args.seeds or configured_seeds
    if set(seeds) - set(configured_seeds):
        parser.error("--seeds may contain only seeds predeclared in the configuration")
    episodes = int(args.episodes if args.episodes is not None else training["episodes_per_seed"])
    if episodes != int(training["episodes_per_seed"]):
        parser.error("--episodes must equal the predeclared episodes_per_seed")
    if not seeds or episodes <= 0:
        parser.error("at least one seed and a positive episode count are required")

    root = _project_root(args.config.resolve())
    final_test_raw = root / str(config["raw_files"]["acn"]["test"])
    if final_test_raw.exists():
        parser.error(
            "the registered final ACN test file is already present; do not train or retune after its acquisition"
        )

    _, _, preprocessor = fit_v3_hybrid_deadline_preprocessor(config, args.config)
    train_window = ReplayWindow.from_config(config["splits"]["train"])
    validation = preprocessor.apply(prepare_acn_split(config, args.config, "validation"))[0]
    pv_proxy = load_pv_proxy(
        config,
        args.config,
        dc_capacity_kw=float(config["pv_proxy_sensitivity"]["dc_capacity_kw"]),
        ac_capacity_kw=float(config["pv_proxy_sensitivity"]["ac_capacity_kw"]),
    )
    forecast = fit_chronological_pv_forecaster(
        pv_proxy,
        train_window=train_window,
        calibration_window=ReplayWindow.from_config(config["splits"]["calibration"]),
        test_window=ReplayWindow.from_config(config["splits"]["validation"]),
    ).forecaster

    train_environments: list[StationMARLEnvironment] = []
    for day in _whole_days(train_window):
        daily_config = with_replay_window(config, "train", day.start, day.end)
        prepared = prepare_acn_split(daily_config, args.config, "train")
        if prepared.sessions:
            guarded, _ = preprocessor.apply(prepared)
            train_environments.append(_environment(config, guarded, pv_proxy=pv_proxy, forecaster=forecast, reward=reward))
    if not train_environments:
        parser.error("no usable registered training days")
    validation_environment = _environment(config, validation, pv_proxy=pv_proxy, forecaster=forecast, reward=reward)
    first_observations = train_environments[0].reset()
    observation_dim = next(iter(first_observations.values())).size

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_dir = args.output_dir / f"{args.algorithm}_seed{seed}"
        if seed_dir.exists():
            parser.error(f"refusing to overwrite existing seed artifact: {seed_dir}")
        seed_dir.mkdir()
        trainer = MAPPOTrainer(
            observation_dim,
            len(first_observations),
            config=MAPPOConfig(
                hidden_dim=int(training["hidden_dim"]),
                update_epochs=int(training["update_epochs"]),
            ),
            seed=seed,
            action_parameterization=str(training["action_parameterization"]),
            torch_threads=int(training["torch_threads"]),
            critic_mode="centralized" if args.algorithm == "mappo" else "local",
        )
        random = np.random.default_rng(seed)

        def factory(_: int) -> StationMARLEnvironment:
            return train_environments[int(random.integers(len(train_environments)))]

        def progress(record: dict[str, float]) -> None:
            print(
                f"{args.algorithm} seed={seed} episode={int(record['episode']) + 1}/{episodes} "
                f"reward={record['episode_reward']:.3f} steps={int(record['steps'])} "
                f"elapsed={record['episode_seconds']:.1f}s",
                flush=True,
            )

        history = trainer.train(factory, episodes=episodes, progress_callback=progress)
        checkpoint = seed_dir / f"safe_{args.algorithm}_actor.pt"
        trainer.save_checkpoint(checkpoint)
        pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
        validation_row = _evaluate_actor(trainer.actor, validation_environment, f"safe_{args.algorithm}_station")
        pd.DataFrame([validation_row]).to_csv(seed_dir / "validation_metrics.csv", index=False)
        (seed_dir / "training_manifest.json").write_text(json.dumps({
            "protocol_config": str(args.config),
            "algorithm": args.algorithm,
            "seed": seed,
            "checkpoint_rule": training["checkpoint_rule"],
            "train_split": config["splits"]["train"],
            "calibration_split": config["splits"]["calibration"],
            "validation_split": config["splits"]["validation"],
            "final_test_split": config["splits"]["test"],
            "final_test_file_absent_at_training": True,
            "deadline_guard": declaration["deadline_guard"],
            "downstream_execution": declaration["downstream_execution"],
            "reward": reward,
            "checkpoint": str(checkpoint),
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Completed pre-test {args.algorithm.upper()} seed {seed}: {seed_dir}")


if __name__ == "__main__":
    main()
