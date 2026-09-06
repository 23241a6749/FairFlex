"""Run the separate FairFlex V2 condition-aware-guard development matrix.

This command never writes V1 artifacts and labels all output as development
evidence.  A selected V2 variant must be frozen and evaluated on a new test
window before it can contribute to a paper result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from fairflex.baselines import (
    earliest_deadline_first,
    equal_share,
    first_come_first_served,
    least_laxity_first,
    uncontrolled,
)
from fairflex.evaluation import ExperimentRunner
from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.forecasting import RobustForecastCapacitySource
from fairflex.safety import ACRepairedMPCPolicy
from fairflex.scenarios import (
    ReplayWindow,
    load_pv_proxy,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
    with_replay_window,
)
from fairflex.v2.commitments import ConditionAwareEarlyDepartureGuard
from fairflex.v2.protocol import load_condition_aware_guard_protocol
from fairflex.v2.runtime import ProfiledACRepairedMPCPolicy
from fairflex.v2.safety import ACRepairedBaselinePolicyV2


def _repair_summary(policy) -> dict[str, float | int]:
    repairs = getattr(policy, "repair_history", [])
    return {
        "ac_repair_calls": len(repairs),
        "ac_repair_activations": sum(repair.attempts > 0 for repair in repairs),
        "ac_repair_curtailment_kw_sum": float(
            sum(repair.curtailed_energy_rate_kw for repair in repairs)
        ),
    }


def _guard_group_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby("condition_group", dropna=False)
        .agg(
            decisions=("ev_id", "size"),
            observed_coverage=("covered", "mean"),
            mean_buffer_minutes=("buffer_minutes", "mean"),
            median_buffer_minutes=("buffer_minutes", "median"),
            p90_buffer_minutes=("buffer_minutes", lambda values: values.quantile(0.9)),
            global_fallback_rate=("used_global_fallback", "mean"),
            calibration_sessions=("calibration_sessions", "first"),
        )
        .reset_index()
        .sort_values("condition_group")
    )


def _require_local_midnight_windows(config: dict, split_names: tuple[str, ...]) -> None:
    """Reject a V2 time-of-day grouping with a UTC-shifted replay origin."""
    for name in split_names:
        values = config["splits"][name]
        for boundary in values:
            timestamp = pd.Timestamp(boundary)
            if timestamp.tzinfo is None:
                raise ValueError("V2 replay-window boundaries require explicit UTC offsets")
            if any((timestamp.hour, timestamp.minute, timestamp.second, timestamp.microsecond)):
                raise ValueError(
                    f"V2 split {name!r} must begin/end at local midnight for fixed "
                    "plug-in-time grouping"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the separate FairFlex V2 condition-aware guard development matrix."
    )
    parser.add_argument("--study-config", required=True, type=Path)
    parser.add_argument(
        "--v2-protocol",
        type=Path,
        default=Path("configs/v2/condition_aware_guard_protocol.json"),
    )
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--split", default="test")
    parser.add_argument("--replay-start")
    parser.add_argument("--replay-end")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--soft-latency-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/condition_aware_development"),
    )
    args = parser.parse_args()
    if args.calibration_split == args.split:
        parser.error("V2 calibration and evaluation splits must be different")
    if bool(args.replay_start) != bool(args.replay_end):
        parser.error("--replay-start and --replay-end must be supplied together")
    if args.soft_latency_seconds <= 0:
        parser.error("--soft-latency-seconds must be positive")

    protocol = load_condition_aware_guard_protocol(args.v2_protocol)
    config = load_study_config(args.study_config)
    # V1's reusable data loader predates nested protocol namespaces and
    # resolves paths from ``config_path.parent.parent``. Keep V1 untouched:
    # give it an equivalent project-root-relative facade when V2 study configs
    # live under ``configs/v2``. The facade path is used only for resolving
    # declared inputs; the JSON is already loaded from the real V2 file.
    resolver_config_path = (
        args.study_config.parent.parent / args.study_config.name
        if args.study_config.parent.name == "v2"
        else args.study_config
    )
    if int(config["time_step_minutes"]) != protocol.time_step_minutes:
        parser.error("study config time step does not match the V2 frozen condition protocol")
    if args.calibration_split not in config["splits"] or args.split not in config["splits"]:
        parser.error("V2 calibration and evaluation splits must exist in the study config")
    try:
        _require_local_midnight_windows(config, (args.calibration_split, args.split))
    except ValueError as error:
        parser.error(str(error))
    if args.replay_start:
        try:
            _require_local_midnight_windows(
                {"splits": {args.split: [args.replay_start, args.replay_end]}},
                (args.split,),
            )
            config = with_replay_window(config, args.split, args.replay_start, args.replay_end)
        except ValueError as error:
            parser.error(str(error))

    calibration = prepare_acn_split(config, resolver_config_path, args.calibration_split)
    evaluation = prepare_acn_split(config, resolver_config_path, args.split)
    if not calibration.sessions:
        parser.error("V2 calibration split has no usable sessions")
    if not evaluation.sessions:
        parser.error("V2 evaluation split has no usable sessions")
    if calibration.target_mode != "declared_commitment" or evaluation.target_mode != "declared_commitment":
        parser.error("V2 requires declared_commitment ACN replays to preserve the causal boundary")

    guard = ConditionAwareEarlyDepartureGuard.fit(
        calibration.sessions,
        miscoverage=protocol.miscoverage,
        min_group_calibration_sessions=protocol.minimum_group_calibration_sessions,
        steps_per_day=96,
    )
    guard_decisions = pd.DataFrame(guard.decision_audit(evaluation.sessions))
    guarded_evaluation = replace(evaluation, sessions=guard.apply(evaluation.sessions))

    pv_settings = config["pv_proxy_sensitivity"]
    pv_proxy = load_pv_proxy(
        config,
        resolver_config_path,
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
    feeder = config["feeder_sensitivity"]
    base_import_limit_kw = float(feeder["base_import_limit_kw"])
    horizon_steps = int(feeder["forecast_horizon_steps"])

    policies: dict[str, object] = {}

    def make_source() -> RobustForecastCapacitySource:
        return RobustForecastCapacitySource(
            forecast.forecaster,
            pv_proxy,
            replay_origin=guarded_evaluation.window.start,
            grid_import_limit_kw=base_import_limit_kw,
            step_minutes=protocol.time_step_minutes,
            metered_first_step=True,
        )

    def baseline_builder(name: str, baseline):
        def build(simulation):
            source = make_source()
            policy = ACRepairedBaselinePolicyV2(
                baseline,
                simulation.grid,
                feeder_capacity_kw=lambda step: float(source(step, 1)[0]),
            )
            policies[name] = policy
            return policy

        return build

    def mpc_builder(name: str, *, robust_pv: bool):
        def build(simulation):
            source = make_source() if robust_pv else None
            policy = ProfiledACRepairedMPCPolicy(
                ACRepairedMPCPolicy(
                    CentralizedFairMPC(),
                    simulation.grid,
                    horizon_steps=horizon_steps,
                    feeder_capacity_kw=source if source is not None else base_import_limit_kw,
                ),
                soft_latency_seconds=args.soft_latency_seconds,
            )
            policies[name] = policy
            return policy

        return build

    builders = {
        "v2_uncontrolled_common_safe": baseline_builder(
            "v2_uncontrolled_common_safe", uncontrolled
        ),
        "v2_fcfs_common_safe": baseline_builder("v2_fcfs_common_safe", first_come_first_served),
        "v2_edf_common_safe": baseline_builder("v2_edf_common_safe", earliest_deadline_first),
        "v2_equal_share_common_safe": baseline_builder("v2_equal_share_common_safe", equal_share),
        "v2_llf_common_safe": baseline_builder("v2_llf_common_safe", least_laxity_first),
        "v2_fair_mpc_ac_no_pv": mpc_builder("v2_fair_mpc_ac_no_pv", robust_pv=False),
        "v2_fair_mpc_ac_robust_pv": mpc_builder("v2_fair_mpc_ac_robust_pv", robust_pv=True),
    }
    if args.only:
        unknown = sorted(set(args.only) - set(builders))
        if unknown:
            parser.error(f"unknown V2 policy names: {unknown}")
        builders = {name: builders[name] for name in args.only}

    runner = ExperimentRunner(lambda: make_trace_simulation(config, guarded_evaluation))
    runs = runner.run_with_simulation_policy(builders)
    metrics = ExperimentRunner.metrics_table(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "development_policy_metrics.csv", index=False)
    guard_decisions.to_csv(args.output_dir / "condition_guard_decision_audit.csv", index=False)
    group_summary = _guard_group_summary(guard_decisions)
    group_summary.to_csv(args.output_dir / "condition_guard_group_summary.csv", index=False)

    policy_audits: dict[str, object] = {}
    for name, policy in policies.items():
        if isinstance(policy, ProfiledACRepairedMPCPolicy):
            pd.DataFrame(policy.runtime_records()).to_csv(
                args.output_dir / f"{name}_decision_runtime.csv", index=False
            )
            policy_audits[name] = {
                "ac_repair": _repair_summary(policy.delegate),
                "runtime": policy.runtime_summary(),
            }
        else:
            policy_audits[name] = {"ac_repair": _repair_summary(policy)}
    summary = {
        "status": "development_only_not_a_frozen_test_result",
        "v2_protocol_id": protocol.protocol_id,
        "study_id": config["study_id"],
        "calibration_split": args.calibration_split,
        "evaluation_split": args.split,
        "replay_window": {
            "start": guarded_evaluation.window.start.isoformat(),
            "end": guarded_evaluation.window.end.isoformat(),
        },
        "guard": {
            **guard.describe(),
            "observed_coverage": guard.empirical_coverage(evaluation.sessions),
            "observed_coverage_by_group": guard.empirical_coverage_by_group(
                evaluation.sessions
            ),
            "global_fallback_rate": float(guard_decisions["used_global_fallback"].mean()),
        },
        "forecast_evidence": {
            "train_points": forecast.train_points,
            "calibration_points": forecast.calibration_points,
            "test_points": forecast.test_points,
            "test_one_step_coverage": forecast.test_coverage,
            "test_median_mae_kw": forecast.test_median_mae_kw,
        },
        "policy_metrics": [asdict(run.metrics) for run in runs.values()],
        "policy_audits": policy_audits,
    }
    (args.output_dir / "development_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"V2 development artifacts: {args.output_dir}")
    print("These results are development-only. Do not use them to make a final paper claim.")


if __name__ == "__main__":
    main()
