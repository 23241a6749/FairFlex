"""Evaluate frozen Safe-MAPPO/IPPO seeds against FairFlex-UC on one fresh test.

The test split is loaded only here, after the training command has produced
immutable seed artifacts.  MAPPO and IPPO keep the same V3 hybrid deadline,
lower-tail local allocator, feeder projection and AC repair as declared in the
matched protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from fairflex.evaluation import summarize_outcome
from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.forecasting import RobustForecastCapacitySource
from fairflex.mappo import SafeMAPPOStationPolicy, SafeStationCapPolicy, load_station_marl_actor
from fairflex.safety import ACRepairedMPCPolicy
from fairflex.scenarios import (
    ReplayWindow,
    build_data_manifest,
    load_pv_proxy,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
)
from fairflex.v3.evaluation import continuous_replay_session_rows, summarize_early_unplug_subset
from fairflex.v3.lower_tail_mpc import LowerTailFairMPC
from fairflex.v3.marl_protocol import fit_v3_hybrid_deadline_preprocessor


def _project_root(config_path: Path) -> Path:
    return next(parent for parent in (config_path.parent, *config_path.parents) if (parent / "pyproject.toml").is_file())


def _source(config: dict, prepared, pv_proxy: pd.Series, forecaster) -> RobustForecastCapacitySource:
    feeder = config["feeder_sensitivity"]
    return RobustForecastCapacitySource(
        forecaster,
        pv_proxy,
        replay_origin=prepared.window.start,
        grid_import_limit_kw=float(feeder["base_import_limit_kw"]),
        step_minutes=int(config["time_step_minutes"]),
        metered_first_step=True,
    )


def _repair_summary(policy) -> dict[str, int]:
    history = getattr(policy, "repair_history", None)
    if history is None:
        history = [audit.ac_repair for audit in policy.action_history]
    return {
        "repair_activations": sum(item.attempts > 0 for item in history),
        "unsafe_initial_actions": sum(not item.initial_grid.safe for item in history),
        "unsafe_final_actions": sum(not item.final_grid.safe for item in history),
    }


def _run_baseline(config: dict, prepared, pv_proxy: pd.Series, forecaster) -> tuple[dict[str, object], list[dict[str, object]]]:
    simulation = make_trace_simulation(config, prepared)
    policy = ACRepairedMPCPolicy(
        LowerTailFairMPC(tail_fraction=float(config["v3"]["tail_fraction"])),
        simulation.grid,
        horizon_steps=int(config["feeder_sensitivity"]["forecast_horizon_steps"]),
        feeder_capacity_kw=_source(config, prepared, pv_proxy, forecaster),
    )
    start = perf_counter()
    result = simulation.run(policy)
    name = "fairflex_uc_hybrid_lower_tail"
    metrics = summarize_outcome(name, result, runtime_seconds=perf_counter() - start)
    early = summarize_early_unplug_subset(
        result,
        prepared.sessions,
        early_unplug_threshold_steps=int(config["v3"]["early_unplug_threshold_minutes"]) // int(config["time_step_minutes"]),
    )
    return {**asdict(metrics), **early.to_dict(), **_repair_summary(policy)}, continuous_replay_session_rows(
        result,
        prepared.sessions,
        policy=name,
        replay_origin=prepared.window.start,
        step_minutes=int(config["time_step_minutes"]),
        early_unplug_threshold_steps=int(config["v3"]["early_unplug_threshold_minutes"]) // int(config["time_step_minutes"]),
    )


def _run_actor(
    config: dict,
    prepared,
    pv_proxy: pd.Series,
    forecaster,
    *,
    actor_path: Path,
    seed: int,
    algorithm: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    actor = load_station_marl_actor(actor_path)
    expected_algorithm = "mappo" if algorithm == "mappo" else "ippo_parameter_shared"
    if actor.fairflex_algorithm != expected_algorithm:
        raise ValueError(f"checkpoint {actor_path} is {actor.fairflex_algorithm!r}, not {expected_algorithm!r}")
    if actor.fairflex_action_parameterization != "feasible_cap_transform":
        raise ValueError("matched Safe-MARL evaluation requires feasible_cap_transform checkpoints")
    simulation = make_trace_simulation(config, prepared)
    safe_policy = SafeStationCapPolicy(
        LowerTailFairMPC(tail_fraction=float(config["v3"]["tail_fraction"])),
        simulation.grid,
        horizon_steps=int(config["feeder_sensitivity"]["forecast_horizon_steps"]),
        feeder_capacity_kw=_source(config, prepared, pv_proxy, forecaster),
        feasible_cap_transform=True,
    )
    policy = SafeMAPPOStationPolicy(actor, safe_policy)
    start = perf_counter()
    result = simulation.run(policy)
    name = f"safe_{algorithm}_station_seed{seed}"
    metrics = summarize_outcome(name, result, runtime_seconds=perf_counter() - start)
    early = summarize_early_unplug_subset(
        result,
        prepared.sessions,
        early_unplug_threshold_steps=int(config["v3"]["early_unplug_threshold_minutes"]) // int(config["time_step_minutes"]),
    )
    audit = _repair_summary(safe_policy)
    audit.update({
        "raw_feeder_violation_steps": sum(item.raw_feeder_violation for item in safe_policy.action_history),
        "raw_ac_unsafe_steps": sum(item.raw_ac_unsafe for item in safe_policy.action_history),
    })
    return {**asdict(metrics), **early.to_dict(), **audit, "seed": seed, "checkpoint": str(actor_path)}, continuous_replay_session_rows(
        result,
        prepared.sessions,
        policy=name,
        replay_origin=prepared.window.start,
        step_minutes=int(config["time_step_minutes"]),
        early_unplug_threshold_steps=int(config["v3"]["early_unplug_threshold_minutes"]) // int(config["time_step_minutes"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v3/caltech_2019_safe_marl_matched_protocol.json"))
    parser.add_argument("--algorithm", choices=("mappo", "ippo"), required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_study_config(args.config)
    protocol = config.get("safe_marl")
    if not isinstance(protocol, dict):
        parser.error("config requires safe_marl protocol settings")
    configured_seeds = [int(seed) for seed in protocol["training"]["seeds"]]
    root = _project_root(args.config.resolve())
    test_raw = root / str(config["raw_files"]["acn"]["test"])
    if not test_raw.is_file():
        parser.error("registered test file is absent; train and freeze all seeds before acquiring it")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing evaluation artifact: {args.output_dir}")

    checkpoints: list[tuple[int, Path]] = []
    for seed in configured_seeds:
        seed_dir = args.training_dir / f"{args.algorithm}_seed{seed}"
        manifest = seed_dir / "training_manifest.json"
        checkpoint = seed_dir / f"safe_{args.algorithm}_actor.pt"
        if not manifest.is_file() or not checkpoint.is_file():
            parser.error(f"missing frozen training artifact for predeclared seed {seed}: {seed_dir}")
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded.get("seed") != seed or recorded.get("algorithm") != args.algorithm:
            parser.error(f"training manifest does not match requested algorithm/seed: {manifest}")
        checkpoints.append((seed, checkpoint))

    _, _, preprocessor = fit_v3_hybrid_deadline_preprocessor(config, args.config)
    raw_test = prepare_acn_split(config, args.config, "test")
    prepared, guard_audit = preprocessor.apply(raw_test)
    pv_proxy = load_pv_proxy(
        config,
        args.config,
        dc_capacity_kw=float(config["pv_proxy_sensitivity"]["dc_capacity_kw"]),
        ac_capacity_kw=float(config["pv_proxy_sensitivity"]["ac_capacity_kw"]),
    )
    windows = {name: ReplayWindow.from_config(values) for name, values in config["splits"].items()}
    forecaster = fit_chronological_pv_forecaster(
        pv_proxy,
        train_window=windows["train"],
        calibration_window=windows["calibration"],
        test_window=windows["test"],
    ).forecaster

    metrics: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []
    baseline_metrics, baseline_rows = _run_baseline(config, prepared, pv_proxy, forecaster)
    metrics.append(baseline_metrics)
    session_rows.extend(baseline_rows)
    for seed, checkpoint in checkpoints:
        seed_metrics, seed_rows = _run_actor(
            config, prepared, pv_proxy, forecaster,
            actor_path=checkpoint, seed=seed, algorithm=args.algorithm,
        )
        metrics.append(seed_metrics)
        session_rows.extend(seed_rows)

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "test_matched_safe_marl_metrics.csv"
    session_path = args.output_dir / "test_matched_safe_marl_session_metrics.csv"
    provenance_path = args.output_dir / "test_matched_safe_marl_provenance.json"
    pd.DataFrame(metrics).sort_values("policy").to_csv(metrics_path, index=False)
    pd.DataFrame(session_rows).sort_values(["policy", "calendar_day", "ev_id"]).to_csv(session_path, index=False)
    provenance = build_data_manifest(config, args.config, {"test": raw_test})
    provenance.update({
        "algorithm": args.algorithm,
        "seeds": configured_seeds,
        "checkpoint_rule": protocol["training"]["checkpoint_rule"],
        "deadline_guard": protocol["deadline_guard"],
        "downstream_execution": protocol["downstream_execution"],
        "hybrid_guard_audit": guard_audit,
        "continuous_replay": "One continuous test replay per policy/seed; session rows are post-hoc arrival-day blocks.",
        "protocol_amendment": config.get("protocol_amendment"),
    })
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Matched Safe-MARL metrics: {metrics_path}")
    print(f"Continuous session rows: {session_path}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
