"""Run the bounded, trace-driven Caltech FairFlex pilot reproducibly.

This is evidence generation, not a paper-result command. The configured PV is
a GHI-derived sensitivity proxy and the real ACN station-to-feeder mapping is
synthetic; both assumptions are written into the resulting summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fairflex.admm import DistributedFairMPCPolicy, FeederADMMNegotiator
from fairflex.baselines import (
    earliest_deadline_first,
    equal_share,
    feeder_capped,
    first_come_first_served,
    least_laxity_first,
    round_robin,
    uncontrolled,
)
from fairflex.commitments import (
    AdaptiveEarlyDepartureGuard,
    AgACIWeightedEarlyDepartureGuard,
    DurationStratifiedEarlyDepartureGuard,
    EarlyDepartureGuard,
    MultiRateAdaptiveEarlyDepartureGuard,
)
from fairflex.evaluation import ExperimentRunner
from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.forecasting import RobustForecastCapacitySource
from fairflex.safety import ACRepairedDistributedPolicy, ACRepairedMPCPolicy
from fairflex.scenarios import (
    ReplayWindow,
    load_pv_proxy,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
    resolve_commitment_uncertainty,
    resolve_distributed_policy_settings,
    with_replay_window,
)


def _forecast_audit(
    *,
    source: RobustForecastCapacitySource,
    result,
    replay_origin: pd.Timestamp,
    actual_pv_kw: pd.Series,
    base_import_limit_kw: float,
    step_minutes: int,
) -> dict[str, float | int]:
    """Audit lower-bound coverage and realized shared-import feasibility offline."""
    lower_bounds: list[float] = []
    actual_values: list[float] = []
    import_violations = 0
    for step_result in result.steps:
        interval = source.intervals_by_step[step_result.step]
        timestamp = replay_origin + pd.Timedelta(minutes=step_minutes * step_result.step)
        actual = float(actual_pv_kw.loc[timestamp])
        lower = float(interval.lower_kw[0])
        lower_bounds.append(lower)
        actual_values.append(actual)
        net_import = sum(step_result.station_powers_kw.values()) - actual
        import_violations += int(net_import > base_import_limit_kw + 1e-7)
    return {
        "actions": len(lower_bounds),
        "first_step_lower_bound_coverage": float(
            np.mean(np.asarray(actual_values) >= np.asarray(lower_bounds))
        ),
        "realized_import_violation_steps": import_violations,
        "metered_safety_clip_steps": sum(
            clip > 1e-9 for clip in source.metered_clips_by_step.values()
        ),
        "metered_safety_clip_kw_sum": float(sum(source.metered_clips_by_step.values())),
    }


def _repair_summary(policy) -> dict[str, float | int]:
    repairs = getattr(policy, "repair_history", [])
    return {
        "ac_repair_calls": len(repairs),
        "ac_repair_activations": sum(repair.attempts > 0 for repair in repairs),
        "ac_repair_curtailment_kw_sum": float(
            sum(repair.curtailed_energy_rate_kw for repair in repairs)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/caltech_2019_pilot.json")
    parser.add_argument("--split", default="test", choices=("train", "calibration", "test"))
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument(
        "--apply-commitment-guard",
        action="store_true",
        help=(
            "fit the locked one-sided early-departure conformal guard on the "
            "configured calibration split and apply it to this replay. "
            "Use only for a held-out test or separately declared ablation."
        ),
    )
    parser.add_argument(
        "--commitment-guard-mode",
        choices=("global", "duration_stratified", "adaptive", "multi_rate_envelope", "agaci_weighted"),
        default="global",
        help=(
            "guard ablation to apply when --apply-commitment-guard is set. "
            "The duration-stratified mode uses predeclared duration groups and a global fallback; "
            "adaptive is a strictly past-only ACI-inspired extension; multi_rate_envelope "
            "uses the earliest causal deadline from fixed-rate adaptive experts; agaci_weighted "
            "uses a bounded AgACI-style online weighting of those experts."
        ),
    )
    parser.add_argument(
        "--adaptive-learning-rate",
        type=float,
        help=(
            "optional explicit override for the adaptive conformal learning rate. "
            "Use only for a separately declared development selection, never a frozen final replay."
        ),
    )
    parser.add_argument(
        "--mappo-checkpoint",
        type=Path,
        help=(
            "optional frozen Safe-MAPPO actor checkpoint selected using only the "
            "separate validation cohort; this adds safe_mappo_station to the policy set"
        ),
    )
    parser.add_argument(
        "--ippo-checkpoint",
        type=Path,
        help=(
            "optional frozen parameter-shared IPPO checkpoint selected using only the "
            "separate validation cohort; this adds safe_ippo_station to the policy set"
        ),
    )
    parser.add_argument(
        "--replay-start",
        help="optional ISO-8601 UTC start inside the configured split; use with --replay-end",
    )
    parser.add_argument(
        "--replay-end",
        help="optional ISO-8601 UTC end inside the configured split; use with --replay-start",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="optional policy names to run; useful for a separately timed ablation",
    )
    parser.add_argument(
        "--distributed-settings",
        type=Path,
        help=(
            "optional credential-free JSON profile with a 'settings' object; "
            "use a validation-locked profile for final seasonal comparisons"
        ),
    )
    parser.add_argument("--distributed-rho", type=float)
    parser.add_argument("--distributed-max-iterations", type=int)
    parser.add_argument("--distributed-tolerance", type=float)
    parser.add_argument("--equity-debt-decay", type=float)
    parser.add_argument("--equity-debt-gain", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/caltech_2019_pilot"))
    args = parser.parse_args()
    if args.commitment_guard_mode != "global" and not args.apply_commitment_guard:
        parser.error("--commitment-guard-mode requires --apply-commitment-guard")
    if args.adaptive_learning_rate is not None and args.commitment_guard_mode != "adaptive":
        parser.error("--adaptive-learning-rate requires --commitment-guard-mode adaptive")

    mappo_actor = ippo_actor = None
    if args.mappo_checkpoint or args.ippo_checkpoint:
        for algorithm, checkpoint in (("MAPPO", args.mappo_checkpoint), ("IPPO", args.ippo_checkpoint)):
            if checkpoint and not checkpoint.is_file():
                parser.error(f"{algorithm} checkpoint does not exist: {checkpoint}")
        try:
            from fairflex.mappo import (
                SafeMAPPOStationPolicy,
                SafeStationCapPolicy,
                load_station_marl_actor,
            )
        except ImportError as error:
            parser.error(
                "Safe station-MARL requires the optional CPU PyTorch dependency; "
                "install it with the command documented in README.md. "
                f"Details: {error}"
            )
        for expected_algorithm, checkpoint in (("mappo", args.mappo_checkpoint), ("ippo_parameter_shared", args.ippo_checkpoint)):
            if checkpoint is None:
                continue
            try:
                actor = load_station_marl_actor(checkpoint)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                parser.error(f"cannot load station-MARL checkpoint: {error}")
            if getattr(actor, "fairflex_algorithm", "mappo") != expected_algorithm:
                parser.error(
                    f"checkpoint {checkpoint} is labelled "
                    f"{getattr(actor, 'fairflex_algorithm', 'unknown')!r}, not {expected_algorithm!r}"
                )
            if expected_algorithm == "mappo":
                mappo_actor = actor
            else:
                ippo_actor = actor

    # Keep the optional learning runtime ahead of the tabular stack when a
    # checkpoint is requested. This lowers the transient memory peak on small
    # Windows machines; helper functions resolve this module global at runtime.
    global pd
    import pandas as pd

    config_path = Path(args.config)
    config = load_study_config(config_path)
    if bool(args.replay_start) != bool(args.replay_end):
        parser.error("--replay-start and --replay-end must be supplied together")
    if args.replay_start:
        try:
            config = with_replay_window(config, args.split, args.replay_start, args.replay_end)
        except ValueError as error:
            parser.error(str(error))
    profile_path: str | None = None
    profile_settings: dict[str, object] = {}
    if args.distributed_settings:
        try:
            profile = json.loads(args.distributed_settings.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read distributed settings profile: {error}")
        if not isinstance(profile, dict) or not isinstance(profile.get("settings"), dict):
            parser.error("distributed settings profile must contain an object named 'settings'")
        profile_path = str(args.distributed_settings)
        profile_settings = profile["settings"]
    command_overrides = {
        "rho": args.distributed_rho,
        "max_iterations": args.distributed_max_iterations,
        "tolerance": args.distributed_tolerance,
        "equity_debt_decay": args.equity_debt_decay,
        "equity_debt_gain": args.equity_debt_gain,
    }
    try:
        distributed_settings = resolve_distributed_policy_settings(
            {
                **config.get("distributed_policy", {}),
                **profile_settings,
                **{name: value for name, value in command_overrides.items() if value is not None},
            }
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    prepared = prepare_acn_split(config, config_path, args.split)
    # An all-empty replay has no defined user-service or fairness metric.  It
    # is therefore a valid *recorded skip*, not a zero-performance result.
    # The matrix runner consumes this manifest and excludes the calendar day
    # from matched-day resampling while retaining a complete audit trail.
    if not prepared.sessions:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        empty_replay = {
            "status": "skipped_empty_replay",
            "reason": "no usable sessions after the declared ACN data-preparation rule",
            "study_id": config["study_id"],
            "split": args.split,
            "replay_window": {
                "start": prepared.window.start.isoformat(),
                "end": prepared.window.end.isoformat(),
            },
            "source_records": prepared.source_records,
            "selected_records": prepared.selected_records,
            "dropped_records": prepared.dropped_records,
            "usable_sessions": 0,
        }
        empty_path = args.output_dir / f"{args.split}_empty_replay.json"
        empty_path.write_text(json.dumps(empty_replay, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "No usable sessions in this replay window; recorded an empty replay "
            f"at {empty_path} and did not produce undefined policy metrics."
        )
        return
    commitment_guard = None
    commitment_audit: dict[str, object] | None = None
    if args.apply_commitment_guard:
        declared_uncertainty = resolve_commitment_uncertainty(config)
        if declared_uncertainty is None:
            parser.error("--apply-commitment-guard requires commitment_uncertainty in the study config")
        calibration_split = str(declared_uncertainty["calibration_split"])
        if args.split == calibration_split:
            parser.error(
                "the calibration split fits the commitment guard and must not also evaluate it; "
                "use the held-out test split"
            )
        calibration = prepare_acn_split(config, config_path, calibration_split)
        candidates = set(declared_uncertainty.get("candidate_modes", ("global",)))
        if args.commitment_guard_mode not in candidates:
            parser.error(
                f"guard mode {args.commitment_guard_mode!r} is not declared in "
                "commitment_uncertainty.candidate_modes for this config"
            )
        if args.commitment_guard_mode == "global":
            commitment_guard = EarlyDepartureGuard.fit(
                calibration.sessions,
                miscoverage=float(declared_uncertainty["miscoverage"]),
            )
        elif args.commitment_guard_mode == "duration_stratified":
            duration_settings = declared_uncertainty.get("duration_stratification", {})
            minimum = int(duration_settings.get("min_group_calibration_sessions", 50))
            commitment_guard = DurationStratifiedEarlyDepartureGuard.fit(
                calibration.sessions,
                miscoverage=float(declared_uncertainty["miscoverage"]),
                min_group_calibration_sessions=minimum,
            )
        elif args.commitment_guard_mode == "adaptive":
            adaptive_settings = declared_uncertainty["adaptive_conformal"]
            learning_rate = (
                args.adaptive_learning_rate
                if args.adaptive_learning_rate is not None
                else float(adaptive_settings["learning_rate"])
            )
            commitment_guard = AdaptiveEarlyDepartureGuard.fit(
                calibration.sessions,
                miscoverage=float(declared_uncertainty["miscoverage"]),
                learning_rate=float(learning_rate),
                min_miscoverage=float(adaptive_settings.get("min_miscoverage", 0.01)),
                max_miscoverage=float(adaptive_settings.get("max_miscoverage", 0.50)),
            )
        elif args.commitment_guard_mode == "multi_rate_envelope":
            envelope_settings = declared_uncertainty["multi_rate_envelope"]
            commitment_guard = MultiRateAdaptiveEarlyDepartureGuard.fit(
                calibration.sessions,
                miscoverage=float(declared_uncertainty["miscoverage"]),
                learning_rates=tuple(float(rate) for rate in envelope_settings["learning_rates"]),
                min_miscoverage=float(envelope_settings.get("min_miscoverage", 0.01)),
                max_miscoverage=float(envelope_settings.get("max_miscoverage", 0.50)),
            )
        else:
            agaci_settings = declared_uncertainty["agaci_weighted"]
            commitment_guard = AgACIWeightedEarlyDepartureGuard.fit(
                calibration.sessions,
                miscoverage=float(declared_uncertainty["miscoverage"]),
                learning_rates=tuple(float(rate) for rate in agaci_settings["learning_rates"]),
                min_miscoverage=float(agaci_settings.get("min_miscoverage", 0.01)),
                max_miscoverage=float(agaci_settings.get("max_miscoverage", 0.50)),
                epsilon=float(agaci_settings.get("epsilon", 0.001)),
            )
        unguarded_sessions = prepared.sessions
        adaptive_audit: dict[str, object] | None = None
        if isinstance(
            commitment_guard,
            (
                AdaptiveEarlyDepartureGuard,
                MultiRateAdaptiveEarlyDepartureGuard,
                AgACIWeightedEarlyDepartureGuard,
            ),
        ):
            guarded_sessions, adaptive_audit = commitment_guard.apply_causally(unguarded_sessions)
        else:
            guarded_sessions = commitment_guard.apply(unguarded_sessions)
        prepared = type(prepared)(
            name=prepared.name,
            window=prepared.window,
            source_records=prepared.source_records,
            selected_records=prepared.selected_records,
            dropped_records=prepared.dropped_records,
            sessions=guarded_sessions,
            station_session_counts=prepared.station_session_counts,
            target_mode=prepared.target_mode,
        )
        commitment_audit = {
            "applied": True,
            "mode": args.commitment_guard_mode,
            **commitment_guard.describe(),
            "calibration_split": calibration_split,
            "evaluated_split": args.split,
            "empirical_coverage": (
                adaptive_audit["empirical_coverage"]
                if adaptive_audit is not None
                else commitment_guard.empirical_coverage(unguarded_sessions)
            ),
            "interpretation": (
                "Coverage is an offline held-out audit of whether a physical unplug occurred "
                "no earlier than the guarded deadline. It is never supplied to the controller."
            ),
        }
        if isinstance(commitment_guard, DurationStratifiedEarlyDepartureGuard):
            commitment_audit["empirical_coverage_by_group"] = (
                commitment_guard.empirical_coverage_by_group(unguarded_sessions)
            )
            commitment_audit["buffer_minutes_by_group"] = {
                group: steps * int(config["time_step_minutes"])
                for group, steps in commitment_guard.buffer_steps_by_group.items()
            }
        elif isinstance(commitment_guard, EarlyDepartureGuard):
            commitment_audit["buffer_minutes"] = (
                commitment_guard.buffer_steps * int(config["time_step_minutes"])
            )
        if adaptive_audit is not None:
            commitment_audit.update(adaptive_audit)
            buffer_audit = adaptive_audit["buffer_steps_at_decision"]
            commitment_audit["buffer_minutes_at_decision"] = {
                key: value * int(config["time_step_minutes"])
                for key, value in buffer_audit.items()
            }
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
    feeder = config["feeder_sensitivity"]
    base_import_limit_kw = float(feeder["base_import_limit_kw"])
    horizon_steps = int(feeder["forecast_horizon_steps"])
    step_minutes = int(config["time_step_minutes"])
    sources: dict[str, RobustForecastCapacitySource] = {}
    policies: dict[str, object] = {}

    def make_source(name: str) -> RobustForecastCapacitySource:
        source = RobustForecastCapacitySource(
            forecast.forecaster,
            pv_proxy,
            replay_origin=prepared.window.start,
            grid_import_limit_kw=base_import_limit_kw,
            step_minutes=step_minutes,
            metered_first_step=True,
        )
        sources[name] = source
        return source

    def capped_baseline(name, baseline):
        source = make_source(name)
        policy = feeder_capped(baseline, lambda step: float(source(step, 1)[0]))
        policies[name] = policy
        return policy

    def central_no_pv(simulation):
        policy = ACRepairedMPCPolicy(
            CentralizedFairMPC(),
            simulation.grid,
            horizon_steps=horizon_steps,
            feeder_capacity_kw=base_import_limit_kw,
        )
        policies["fair_mpc_ac_no_pv"] = policy
        return policy

    def central_robust_pv(simulation):
        source = make_source("fair_mpc_ac_robust_pv")
        policy = ACRepairedMPCPolicy(
            CentralizedFairMPC(),
            simulation.grid,
            horizon_steps=horizon_steps,
            feeder_capacity_kw=source,
        )
        policies["fair_mpc_ac_robust_pv"] = policy
        return policy

    def safe_station_marl(simulation, *, name: str, actor):
        if actor is None:
            raise RuntimeError(f"{name} builder requires a loaded checkpoint")
        source = make_source(name)
        safe_policy = SafeStationCapPolicy(
            CentralizedFairMPC(),
            simulation.grid,
            horizon_steps=horizon_steps,
            feeder_capacity_kw=source,
            feasible_cap_transform=(
                getattr(actor, "fairflex_action_parameterization", "projected_raw_request")
                == "feasible_cap_transform"
            ),
        )
        policy = SafeMAPPOStationPolicy(actor, safe_policy)
        policies[name] = policy
        return policy

    builders = {
        "uncontrolled_shared_cap": lambda simulation: capped_baseline(
            "uncontrolled_shared_cap", uncontrolled
        ),
        "fcfs_shared_cap": lambda simulation: capped_baseline("fcfs_shared_cap", first_come_first_served),
        "edf_shared_cap": lambda simulation: capped_baseline(
            "edf_shared_cap", earliest_deadline_first
        ),
        "equal_share_shared_cap": lambda simulation: capped_baseline(
            "equal_share_shared_cap", equal_share
        ),
        "round_robin_shared_cap": lambda simulation: capped_baseline(
            "round_robin_shared_cap", round_robin
        ),
        "least_laxity_first_shared_cap": lambda simulation: capped_baseline(
            "least_laxity_first_shared_cap", least_laxity_first
        ),
        "fair_mpc_ac_no_pv": central_no_pv,
        "fair_mpc_ac_robust_pv": central_robust_pv,
    }
    if mappo_actor is not None:
        builders["safe_mappo_station"] = lambda simulation: safe_station_marl(
            simulation, name="safe_mappo_station", actor=mappo_actor
        )
    if ippo_actor is not None:
        builders["safe_ippo_station"] = lambda simulation: safe_station_marl(
            simulation, name="safe_ippo_station", actor=ippo_actor
        )

    if not args.skip_distributed:
        def distributed_robust_pv(simulation):
            source = make_source("recommended_distributed_robust_pv")
            distributed = DistributedFairMPCPolicy(
                CentralizedFairMPC(),
                FeederADMMNegotiator(
                    rho=distributed_settings["rho"],
                    max_iterations=distributed_settings["max_iterations"],
                    tolerance=distributed_settings["tolerance"],
                ),
                horizon_steps=horizon_steps,
                feeder_capacity_kw=source,
                equity_debt_decay=distributed_settings["equity_debt_decay"],
                equity_debt_gain=distributed_settings["equity_debt_gain"],
            )
            policy = ACRepairedDistributedPolicy(distributed, simulation.grid)
            policies["recommended_distributed_robust_pv"] = policy
            return policy

        builders["recommended_distributed_robust_pv"] = distributed_robust_pv

    if args.only:
        unknown = sorted(set(args.only) - set(builders))
        if unknown:
            parser.error(f"unknown or disabled policy names: {unknown}")
        builders = {name: builders[name] for name in args.only}

    runner = ExperimentRunner(lambda: make_trace_simulation(config, prepared))
    runs = runner.run_with_simulation_policy(builders)
    metrics = ExperimentRunner.metrics_table(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.split}_pilot_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    policy_audits: dict[str, object] = {}
    for name, policy in policies.items():
        audit: dict[str, object] = {}
        if name in sources:
            audit["forecast"] = _forecast_audit(
                source=sources[name],
                result=runs[name].result,
                replay_origin=prepared.window.start,
                actual_pv_kw=pv_proxy,
                base_import_limit_kw=base_import_limit_kw,
                step_minutes=step_minutes,
            )
        audit["ac_repair"] = _repair_summary(policy)
        if isinstance(policy, ACRepairedMPCPolicy):
            audit["economic_tiebreak_failures"] = policy.economic_tiebreak_failures
        if isinstance(policy, ACRepairedDistributedPolicy):
            negotiations = policy.distributed_policy.negotiation_history
            audit["admm"] = {
                "settings": distributed_settings,
                "settings_profile": profile_path,
                "calls": len(negotiations),
                "max_iterations": max((item.iterations for item in negotiations), default=0),
                "max_primal_residual": max((item.primal_residual for item in negotiations), default=0.0),
                "max_dual_residual": max((item.dual_residual for item in negotiations), default=0.0),
                "equity_debt_final": policy.distributed_policy.equity_debt,
                "equity_debt_updates": len(policy.distributed_policy.debt_history),
            }
        if name in {"safe_mappo_station", "safe_ippo_station"}:
            action_history = policy.safe_policy.action_history
            checkpoint = args.mappo_checkpoint if name == "safe_mappo_station" else args.ippo_checkpoint
            audit["station_marl"] = {
                "algorithm": "mappo" if name == "safe_mappo_station" else "ippo_parameter_shared",
                "checkpoint": str(checkpoint),
                "action_calls": len(action_history),
                "raw_feeder_violation_steps": sum(
                    item.raw_feeder_violation for item in action_history
                ),
                "raw_feeder_excess_kw_sum": float(
                    sum(item.raw_feeder_excess_kw for item in action_history)
                ),
                "raw_ac_unsafe_steps": sum(item.raw_ac_unsafe for item in action_history),
                "ac_repair_activations": sum(
                    item.ac_repair.attempts > 0 for item in action_history
                ),
            }
        policy_audits[name] = audit

    summary = {
        "study_id": config["study_id"],
        "split": args.split,
        "replay_window": {
            "start": prepared.window.start.isoformat(),
            "end": prepared.window.end.isoformat(),
        },
        "assumptions": {
            "trace_selection": "sessions connect within the declared half-open split; true departures are retained",
            "pv": pv_settings["note"],
            "shared_capacity": feeder["note"],
            "ac_grid_sensitivity": config.get("grid_sensitivity", {}),
            "acn_data_protocol": config.get("acn_data_protocol", {}),
            "commitment_guard": commitment_audit
            if commitment_audit is not None
            else {"applied": False},
        },
        "forecast_evidence": {
            "train_points": forecast.train_points,
            "calibration_points": forecast.calibration_points,
            "test_points": forecast.test_points,
            "test_one_step_coverage": forecast.test_coverage,
            "test_median_mae_kw": forecast.test_median_mae_kw,
            "conformal_correction_kw": forecast.forecaster.correction_kw,
        },
        "policy_metrics": [asdict(run.metrics) for run in runs.values()],
        "policy_audits": policy_audits,
    }
    summary_path = args.output_dir / f"{args.split}_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(
        f"Forecast test: coverage={forecast.test_coverage:.3f}, "
        f"median MAE={forecast.test_median_mae_kw:.3f} kW, "
        f"correction={forecast.forecaster.correction_kw:.3f} kW"
    )
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
