"""Run the predeclared FairFlex-UC V3 policy matrix on one chronological split.

This runner keeps V1/V2 intact. It fits CQR on the declared earlier train and
calibration windows, then applies it only to the evaluated split. Actual future
unplug times are never passed to a deployable controller; the oracle policy is
explicitly labelled non-deployable and is reported only as an information upper
bound.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

from fairflex.baselines import equal_share, feeder_capped
from fairflex.commitments import MultiRateAdaptiveEarlyDepartureGuard
from fairflex.domain import EVSession
from fairflex.evaluation import summarize_outcome
from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.forecasting import RobustForecastCapacitySource
from fairflex.safety import ACRepairedMPCPolicy
from fairflex.scenarios import (
    ReplayWindow,
    build_data_manifest,
    load_pv_proxy,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
    with_replay_window,
)
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.v3.commitments import CausalCQRBufferGuard, conservative_deadline_envelope
from fairflex.v3.evaluation import continuous_replay_session_rows, summarize_early_unplug_subset
from fairflex.v3.lower_tail_mpc import LowerTailFairMPC


def _oracle_sessions(sessions: tuple[EVSession, ...]) -> tuple[EVSession, ...]:
    """Non-deployable upper bound: planning deadline is the actual unplug time."""
    return tuple(
        EVSession(
            ev.ev_id,
            ev.station_id,
            ev.arrival_step,
            ev.departure_step,
            ev.requested_energy_kwh,
            ev.max_power_kw,
            ev.delivered_energy_kwh,
            ev.departure_step,
            ev.declared_departure_step or ev.planning_departure_step,
        )
        for ev in sessions
    )


def _repair_audit(policy: ACRepairedMPCPolicy) -> dict[str, int | float]:
    history = policy.repair_history
    return {
        "calls": len(history),
        "repair_activations": sum(item.attempts > 0 for item in history),
        "maximum_attempts": max((item.attempts for item in history), default=0),
        "unsafe_initial_actions": sum(not item.initial_grid.safe for item in history),
        "unsafe_final_actions": sum(not item.final_grid.safe for item in history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="Optional policy subset for diagnosis")
    parser.add_argument(
        "--replay-start",
        help="optional ISO-8601 UTC start inside the declared split; use with --replay-end",
    )
    parser.add_argument(
        "--replay-end",
        help="optional ISO-8601 UTC end inside the declared split; use with --replay-start",
    )
    parser.add_argument(
        "--base-import-limit-kw",
        type=float,
        help="optional frozen sensitivity override for the synthetic shared import limit",
    )
    parser.add_argument(
        "--pv-dc-capacity-kw",
        type=float,
        help="optional frozen sensitivity override for the PV proxy DC capacity",
    )
    parser.add_argument(
        "--pv-ac-capacity-kw",
        type=float,
        help="optional frozen sensitivity override for the PV proxy AC capacity",
    )
    args = parser.parse_args()

    config_path = args.config
    config = load_study_config(config_path)
    if bool(args.replay_start) != bool(args.replay_end):
        parser.error("--replay-start and --replay-end must be supplied together")
    if args.replay_start:
        config = with_replay_window(config, args.split, args.replay_start, args.replay_end)
    if any(
        value is not None and value <= 0.0
        for value in (args.base_import_limit_kw, args.pv_dc_capacity_kw, args.pv_ac_capacity_kw)
    ):
        parser.error("sensitivity capacity overrides must be positive")
    v3 = config.get("v3")
    if not isinstance(v3, dict):
        parser.error("V3 config requires a v3 object")
    cqr_settings = v3.get("cqr")
    if not isinstance(cqr_settings, dict):
        parser.error("v3.cqr must be an object")
    guard_settings = v3.get("v1_multirate_guard")
    if not isinstance(guard_settings, dict):
        parser.error("v3.v1_multirate_guard must be an object")

    train = prepare_acn_split(config, config_path, "train")
    calibration = prepare_acn_split(config, config_path, "calibration")
    evaluated = prepare_acn_split(config, config_path, args.split)
    if not evaluated.sessions:
        parser.error("the declared evaluation split contains no usable commitment sessions")

    step_minutes = int(config["time_step_minutes"])
    cqr_guard = CausalCQRBufferGuard.fit(
        train.sessions,
        calibration.sessions,
        train_origin=train.window.start,
        calibration_origin=calibration.window.start,
        step_minutes=step_minutes,
        miscoverage=float(cqr_settings["miscoverage"]),
        min_train_sessions=int(cqr_settings["min_train_sessions"]),
        min_calibration_sessions=int(cqr_settings["min_calibration_sessions"]),
        max_iter=int(cqr_settings.get("max_iter", 150)),
        min_samples_leaf=int(cqr_settings.get("min_samples_leaf", 12)),
        random_state=int(cqr_settings.get("random_state", 20260903)),
    )
    cqr_sessions, cqr_audit = cqr_guard.apply_causally(
        evaluated.sessions, origin=evaluated.window.start
    )
    v1_guard = MultiRateAdaptiveEarlyDepartureGuard.fit(
        calibration.sessions,
        miscoverage=float(guard_settings["miscoverage"]),
        learning_rates=tuple(float(rate) for rate in guard_settings["learning_rates"]),
        min_miscoverage=float(guard_settings.get("min_miscoverage", 0.01)),
        max_miscoverage=float(guard_settings.get("max_miscoverage", 0.50)),
    )
    v1_sessions, v1_audit = v1_guard.apply_causally(evaluated.sessions)
    hybrid_sessions, hybrid_audit = conservative_deadline_envelope(cqr_sessions, v1_sessions)

    pv_settings = config["pv_proxy_sensitivity"]
    dc_capacity_kw = float(
        args.pv_dc_capacity_kw
        if args.pv_dc_capacity_kw is not None
        else pv_settings["dc_capacity_kw"]
    )
    ac_capacity_kw = float(
        args.pv_ac_capacity_kw
        if args.pv_ac_capacity_kw is not None
        else pv_settings["ac_capacity_kw"]
    )
    pv_proxy = load_pv_proxy(
        config,
        config_path,
        dc_capacity_kw=dc_capacity_kw,
        ac_capacity_kw=ac_capacity_kw,
    )
    windows = {name: ReplayWindow.from_config(value) for name, value in config["splits"].items()}
    forecast = fit_chronological_pv_forecaster(
        pv_proxy,
        train_window=windows["train"],
        calibration_window=windows["calibration"],
        test_window=windows[args.split],
    )
    feeder = config["feeder_sensitivity"]
    base_import_limit_kw = float(
        args.base_import_limit_kw
        if args.base_import_limit_kw is not None
        else feeder["base_import_limit_kw"]
    )
    horizon_steps = int(feeder["forecast_horizon_steps"])
    threshold_steps = int(v3["early_unplug_threshold_minutes"]) // step_minutes
    if threshold_steps <= 0:
        parser.error("v3.early_unplug_threshold_minutes must be at least one control interval")

    policy_specs = {
        "equal_share_shared_cap": (evaluated.sessions, "equal_share"),
        "v1_fair_mpc_no_guard": (evaluated.sessions, "max_min"),
        "v1_fair_mpc_multirate_guard": (v1_sessions, "max_min"),
        "v1_multirate_lower_tail_ablation": (v1_sessions, "lower_tail"),
        "v3_cqr_max_min_ablation": (cqr_sessions, "max_min"),
        "fairflex_uc_cqr_lower_tail": (cqr_sessions, "lower_tail"),
        "fairflex_uc_hybrid_max_min": (hybrid_sessions, "max_min"),
        "fairflex_uc_hybrid_lower_tail": (hybrid_sessions, "lower_tail"),
        "oracle_lower_tail_non_deployable": (_oracle_sessions(evaluated.sessions), "lower_tail"),
    }
    if args.only:
        unknown = sorted(set(args.only) - set(policy_specs))
        if unknown:
            parser.error(f"unknown policy names: {unknown}")
        policy_specs = {name: policy_specs[name] for name in args.only}

    rows: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    for name, (sessions, kind) in policy_specs.items():
        prepared = replace(evaluated, sessions=tuple(sessions))
        simulation = make_trace_simulation(config, prepared)
        source = RobustForecastCapacitySource(
            forecast.forecaster,
            pv_proxy,
            replay_origin=evaluated.window.start,
            grid_import_limit_kw=base_import_limit_kw,
            step_minutes=step_minutes,
            metered_first_step=True,
        )
        start = perf_counter()
        if kind == "equal_share":
            policy = feeder_capped(equal_share, lambda step: float(source(step, 1)[0]))
            result = simulation.run(policy)
            repair = None
            objective_mode = "baseline_equal_share"
        else:
            controller = (
                LowerTailFairMPC(tail_fraction=float(v3["tail_fraction"]))
                if kind == "lower_tail"
                else CentralizedFairMPC()
            )
            policy = ACRepairedMPCPolicy(
                controller,
                simulation.grid,
                horizon_steps=horizon_steps,
                feeder_capacity_kw=source,
            )
            result = simulation.run(policy)
            repair = _repair_audit(policy)
            objective_mode = "expected_shortfall_worst_10pct" if kind == "lower_tail" else "v1_max_min_shortfall"
        metrics = summarize_outcome(name, result, runtime_seconds=perf_counter() - start)
        subset = summarize_early_unplug_subset(
            result,
            evaluated.sessions,
            early_unplug_threshold_steps=threshold_steps,
        )
        row = {**asdict(metrics), **subset.to_dict(), "objective_mode": objective_mode}
        rows.append(row)
        # Do not restart the adaptive guard merely to obtain a daily inference
        # block.  These rows are grouped by arrival day only *after* this one
        # continuous replay has finished.
        session_rows.extend(
            continuous_replay_session_rows(
                result,
                evaluated.sessions,
                policy=name,
                replay_origin=evaluated.window.start,
                step_minutes=step_minutes,
                early_unplug_threshold_steps=threshold_steps,
            )
        )
        audits[name] = {
            "deployable": name != "oracle_lower_tail_non_deployable",
            "ac_repair": repair,
        }

    import pandas as pd

    metrics_frame = pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.split}_fairflex_uc_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    session_metrics_path = args.output_dir / f"{args.split}_fairflex_uc_session_metrics.csv"
    pd.DataFrame(session_rows).sort_values(["policy", "calendar_day", "ev_id"]).to_csv(
        session_metrics_path, index=False
    )
    summary_path = args.output_dir / f"{args.split}_fairflex_uc_summary.json"
    provenance_path = args.output_dir / f"{args.split}_fairflex_uc_provenance.json"
    provenance = build_data_manifest(
        config,
        config_path,
        {"train": train, "calibration": calibration, args.split: evaluated},
    )
    provenance.update({
        "scenario_overrides": {
            "base_import_limit_kw": args.base_import_limit_kw,
            "pv_dc_capacity_kw": args.pv_dc_capacity_kw,
            "pv_ac_capacity_kw": args.pv_ac_capacity_kw,
            "effective_base_import_limit_kw": base_import_limit_kw,
            "effective_pv_dc_capacity_kw": dc_capacity_kw,
            "effective_pv_ac_capacity_kw": ac_capacity_kw,
        },
        "v3_method": {
            "selected_deployable_policy": v3.get("selected_deployable_policy"),
            "cqr": cqr_guard.describe(),
            "lower_tail": {
                "tail_fraction": float(v3["tail_fraction"]),
                "reported_primary_metric": "raw_p10_service_ratio",
                "surrogate": "expected_shortfall_of_largest_service_deficit_tail",
            },
            "continuous_replay_session_rows": (
                "Post-hoc per-session outcomes are grouped by arrival day for day-block inference; "
                "the adaptive guard itself was not reset at a calendar boundary."
            ),
            "causal_information_boundary": (
                "Actual future unplug times are unavailable to every deployable controller."
            ),
        }
    })
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "study_id": config["study_id"],
        "split": args.split,
        "split_window": {"start": evaluated.window.start.isoformat(), "end": evaluated.window.end.isoformat()},
        "selection_status": v3.get("selection_status", "development_only"),
        "scenario_overrides": {
            "base_import_limit_kw": args.base_import_limit_kw,
            "pv_dc_capacity_kw": args.pv_dc_capacity_kw,
            "pv_ac_capacity_kw": args.pv_ac_capacity_kw,
            "effective_base_import_limit_kw": base_import_limit_kw,
            "effective_pv_dc_capacity_kw": dc_capacity_kw,
            "effective_pv_ac_capacity_kw": ac_capacity_kw,
        },
        "causal_information_boundary": (
            "CQR sees declared inputs and observable plug-in state; actual future unplug is used only "
            "for historical labels, physical replay, post-hoc evaluation, and the labelled oracle."
        ),
        "cqr_audit": cqr_audit,
        "v1_multirate_audit": v1_audit,
        "hybrid_guard_audit": hybrid_audit,
        "early_unplug_threshold_minutes": int(v3["early_unplug_threshold_minutes"]),
        "forecast": {
            "test_one_step_coverage": forecast.test_coverage,
            "test_median_mae_kw": forecast.test_median_mae_kw,
            "conformal_correction_kw": forecast.forecaster.correction_kw,
        },
        "policy_audits": audits,
        "policy_metrics": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics_frame.to_string(index=False))
    print(f"Metrics: {metrics_path}")
    print(f"Continuous-replay session rows: {session_metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
