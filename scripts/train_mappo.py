"""Train a safe station-level MAPPO benchmark without touching test cohorts.

Training uses only the configured chronological ``train`` split.  Validation
uses the separately labelled April validation-only configuration; May--December
cohorts must be reserved for the final frozen-policy comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

# Import the learning runtime before the heavier tabular stack.  On memory
# constrained Windows machines this avoids a transient peak while PyTorch
# loads its native CPU libraries.
from fairflex.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    SafeMAPPOStationPolicy,
    SafeStationCapPolicy,
    StationMARLEnvironment,
)

import pandas as pd

from fairflex.evaluation import summarize_outcome
from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.forecasting import RobustForecastCapacitySource
from fairflex.scenarios import (
    ReplayWindow,
    load_pv_proxy,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
    with_replay_window,
)


def _day_windows(window: ReplayWindow, requested_days: list[str] | None) -> list[ReplayWindow]:
    if requested_days:
        starts = [pd.Timestamp(day, tz="UTC").normalize() for day in requested_days]
    else:
        starts = list(pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left"))
    days = [ReplayWindow(start, start + pd.Timedelta(days=1)) for start in starts]
    for day in days:
        if day.start < window.start or day.end > window.end:
            raise ValueError(f"requested day {day.start.date()} is outside {window.start.date()} to {window.end.date()}")
    return days


def _forecast_for_config(config: dict, config_path: Path):
    pv_settings = config["pv_proxy_sensitivity"]
    pv_proxy = load_pv_proxy(
        config,
        config_path,
        dc_capacity_kw=float(pv_settings["dc_capacity_kw"]),
        ac_capacity_kw=float(pv_settings["ac_capacity_kw"]),
    )
    windows = {name: ReplayWindow.from_config(values) for name, values in config["splits"].items()}
    forecast = fit_chronological_pv_forecaster(
        pv_proxy,
        train_window=windows["train"],
        calibration_window=windows["calibration"],
        test_window=windows["test"],
    )
    return pv_proxy, forecast.forecaster


def _environment_factory(
    config: dict,
    config_path: Path,
    *,
    split: str,
    window: ReplayWindow,
    pv_proxy: pd.Series,
    forecaster,
    feasible_cap_transform: bool = False,
) -> StationMARLEnvironment:
    day_config = with_replay_window(config, split, window.start, window.end)
    prepared = prepare_acn_split(day_config, config_path, split)
    feeder = day_config["feeder_sensitivity"]
    base_import_limit_kw = float(feeder["base_import_limit_kw"])
    step_minutes = int(day_config["time_step_minutes"])

    def scenario_factory():
        return make_trace_simulation(day_config, prepared)

    def feeder_capacity_factory(simulation):
        # The first value is metered at execution time. Later robust forecast
        # values are available for audit and parity with the MPC data path, but
        # the station agent only acts on the present interval.
        return RobustForecastCapacitySource(
            forecaster,
            pv_proxy,
            replay_origin=prepared.window.start,
            grid_import_limit_kw=base_import_limit_kw,
            step_minutes=step_minutes,
            metered_first_step=True,
        )

    return StationMARLEnvironment(
        scenario_factory,
        feeder_capacity_factory,
        horizon_steps=int(feeder["forecast_horizon_steps"]),
        feasible_cap_transform=feasible_cap_transform,
    )


def _evaluate_actor(
    actor,
    config: dict,
    config_path: Path,
    *,
    split: str,
    days: list[ReplayWindow],
    pv_proxy: pd.Series,
    forecaster,
    feasible_cap_transform: bool = False,
    policy_name: str = "safe_mappo_station",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in days:
        environment = _environment_factory(
            config,
            config_path,
            split=split,
            window=window,
            pv_proxy=pv_proxy,
            forecaster=forecaster,
            feasible_cap_transform=feasible_cap_transform,
        )
        simulation = environment.scenario_factory()
        safe_policy = SafeStationCapPolicy(
            CentralizedFairMPC(),
            simulation.grid,
            horizon_steps=environment.horizon_steps,
            feeder_capacity_kw=environment.feeder_capacity_factory(simulation),
            feasible_cap_transform=feasible_cap_transform,
        )
        policy = SafeMAPPOStationPolicy(actor, safe_policy)
        start = perf_counter()
        result = simulation.run(policy)
        metrics = summarize_outcome(
            policy_name, result, runtime_seconds=perf_counter() - start
        )
        audits = safe_policy.action_history
        rows.append(
            {
                "replay_start": window.start.isoformat(),
                "replay_end": window.end.isoformat(),
                **asdict(metrics),
                "raw_feeder_violation_steps": sum(audit.raw_feeder_violation for audit in audits),
                "raw_ac_unsafe_steps": sum(audit.raw_ac_unsafe for audit in audits),
                "ac_repair_activations": sum(audit.ac_repair.attempts > 0 for audit in audits),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train safe station-level MAPPO or parameter-shared IPPO on a chronological pre-test FairFlex split."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/caltech_2019_scarce.json"))
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=Path("configs/caltech_2019_distributed_validation.json"),
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument(
        "--algorithm",
        choices=("mappo", "ippo"),
        default="mappo",
        help=(
            "mappo uses a training-only centralized critic; ippo uses a local-observation "
            "critic for every station while retaining parameter sharing"
        ),
    )
    parser.add_argument("--feasible-cap-transform", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--validation-days",
        nargs="+",
        help="optional UTC days for a short validation smoke test",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mappo_training"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")

    train_config = load_study_config(args.config)
    validation_config = load_study_config(args.validation_config)
    if validation_config.get("protocol", {}).get("role") != "validation_only":
        parser.error("--validation-config must be explicitly labelled validation_only")
    train_window = ReplayWindow.from_config(train_config["splits"]["train"])
    train_days = _day_windows(train_window, None)
    if not train_days:
        parser.error("the configured train split contains no whole days")
    validation_window = ReplayWindow.from_config(validation_config["splits"]["test"])
    validation_days = _day_windows(validation_window, args.validation_days)

    train_pv, train_forecaster = _forecast_for_config(train_config, args.config)
    validation_pv, validation_forecaster = _forecast_for_config(validation_config, args.validation_config)
    first_environment = _environment_factory(
        train_config,
        args.config,
        split="train",
        window=train_days[0],
        pv_proxy=train_pv,
        forecaster=train_forecaster,
        feasible_cap_transform=args.feasible_cap_transform,
    )
    first_observations = first_environment.reset()
    observation_dim = next(iter(first_observations.values())).size
    trainer = MAPPOTrainer(
        observation_dim,
        len(first_observations),
        config=MAPPOConfig(hidden_dim=args.hidden_dim, update_epochs=args.update_epochs),
        seed=args.seed,
        action_parameterization=(
            "feasible_cap_transform" if args.feasible_cap_transform else "projected_raw_request"
        ),
        torch_threads=args.torch_threads,
        critic_mode="centralized" if args.algorithm == "mappo" else "local",
    )
    random = np.random.default_rng(args.seed)
    sampled_days: dict[int, str] = {}

    def training_environment(episode: int) -> StationMARLEnvironment:
        # Sampling only from train days creates varied trajectories while the
        # fixed seed makes the learning run reproducible.
        window = train_days[int(random.integers(len(train_days)))]
        sampled_days[episode] = window.start.date().isoformat()
        return _environment_factory(
            train_config,
            args.config,
            split="train",
            window=window,
            pv_proxy=train_pv,
            forecaster=train_forecaster,
            feasible_cap_transform=args.feasible_cap_transform,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training Safe-{args.algorithm.upper()} on March only. Each cohort retains actual ACN "
        "departure times, so an episode may extend past a calendar day."
    )

    def report_progress(record: dict[str, float]) -> None:
        episode = int(record["episode"])
        print(
            f"Episode {episode + 1}/{args.episodes} | cohort={sampled_days[episode]} | "
            f"steps={int(record['steps'])} | reward={record['episode_reward']:.3f} | "
            f"raw feeder excess={record['raw_feeder_excess_kw_sum']:.1f} kW-steps | "
            f"elapsed={record['episode_seconds']:.1f}s",
            flush=True,
        )

    history = trainer.train(
        training_environment,
        episodes=args.episodes,
        progress_callback=report_progress,
    )
    history_path = args.output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    checkpoint_path = args.output_dir / f"safe_{args.algorithm}_actor.pt"
    trainer.save_checkpoint(checkpoint_path)

    validation_path: Path | None = None
    if not args.skip_validation:
        validation = _evaluate_actor(
            trainer.actor,
            validation_config,
            args.validation_config,
            split="test",
            days=validation_days,
            pv_proxy=validation_pv,
            forecaster=validation_forecaster,
            feasible_cap_transform=args.feasible_cap_transform,
            policy_name=f"safe_{args.algorithm}_station",
        )
        validation_path = args.output_dir / "validation_metrics.csv"
        validation.to_csv(validation_path, index=False)

    manifest = {
        "purpose": "Station-MARL training and validation before held-out May--December evaluation.",
        "algorithm": (
            "centralized-training/decentralized-execution MAPPO with a shared Beta actor"
            if args.algorithm == "mappo"
            else "parameter-shared IPPO with a local-observation critic and shared Beta actor"
        ),
        "algorithm_id": args.algorithm,
        "safety": "raw station cap request -> feeder projection -> local fair MPC -> exact AC repair",
        "action_parameterization": "feasible_cap_transform" if args.feasible_cap_transform else "projected_raw_request",
        "reward": {
            "fairness_reward_weight": first_environment.fairness_reward_weight,
            "raw_feeder_violation_penalty": first_environment.raw_feeder_violation_penalty,
            "raw_ac_violation_penalty": first_environment.raw_ac_violation_penalty,
        },
        "training": {
            "config": str(args.config),
            "split": "train",
            "days": [window.start.date().isoformat() for window in train_days],
            "episodes": args.episodes,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
        },
        "validation": {
            "config": str(args.validation_config),
            "role": validation_config["protocol"]["role"],
            "days": [window.start.date().isoformat() for window in validation_days],
            "metrics": str(validation_path) if validation_path else None,
        },
        "forbidden_for_training_or_tuning": "May--December held-out cohorts",
        "checkpoint": str(checkpoint_path),
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Training history: {history_path}")
    print(f"Frozen {args.algorithm.upper()} actor: {checkpoint_path}")
    if validation_path:
        print(f"Validation metrics: {validation_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
