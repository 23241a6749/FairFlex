"""Run matched FairFlex day-level studies and report paired uncertainty.

This command deliberately delegates each replay to ``run_trace_pilot.py`` so
the exact controller path, audit output and assumptions stay identical to the
single-trace pilot. It resamples matched *days*, never EV rows, for policy
comparison uncertainty summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fairflex.scenarios import ReplayWindow, load_study_config
from fairflex.statistics import paired_metric_bootstrap


# The output retains the natural treatment-minus-baseline difference.  It also
# records an oriented treatment advantage so every row has the same simple
# interpretation: positive means the treatment is preferable for that metric.
# Runtime remains a descriptive hardware-dependent measurement rather than a
# bootstrapped quality claim.
METRIC_SPECS = (
    ("mean_service_ratio", "higher"),
    ("p10_service_ratio", "higher"),
    ("worst_service_ratio", "higher"),
    ("jain_service_index", "higher"),
    ("delivered_energy_kwh", "higher"),
    ("energy_service_ratio", "higher"),
    ("completion_rate", "higher"),
    ("individually_feasible_request_rate", "higher"),
    ("unavoidable_individual_shortfall_kwh", "lower"),
    ("min_voltage_pu", "higher"),
    ("max_line_loading_percent", "lower"),
    ("unsafe_steps", "lower"),
)
CENTRAL_FAIRFLEX = "fair_mpc_ac_robust_pv"
COMPARISON_SPECS = (
    # A positive difference means centralized FairFlex is better for metrics
    # where larger is desirable. Unsafe steps are reported separately, with
    # zero as the target for every policy.
    ("central_vs_uncontrolled", CENTRAL_FAIRFLEX, "uncontrolled_shared_cap"),
    ("central_vs_fcfs", CENTRAL_FAIRFLEX, "fcfs_shared_cap"),
    ("central_vs_edf", CENTRAL_FAIRFLEX, "edf_shared_cap"),
    ("central_vs_equal_share", CENTRAL_FAIRFLEX, "equal_share_shared_cap"),
    ("central_vs_round_robin", CENTRAL_FAIRFLEX, "round_robin_shared_cap"),
    ("central_vs_least_laxity_first", CENTRAL_FAIRFLEX, "least_laxity_first_shared_cap"),
    ("central_vs_no_pv", CENTRAL_FAIRFLEX, "fair_mpc_ac_no_pv"),
    # This is an architecture trade-off, not a claim that distributed control
    # should necessarily outperform the centralized optimizer.
    ("central_vs_distributed", CENTRAL_FAIRFLEX, "recommended_distributed_robust_pv"),
)


def _oriented_comparison_fields(result, direction: str) -> dict[str, float | str]:
    """Make a paired result comparable when smaller values are preferable."""
    if direction not in {"higher", "lower"}:
        raise ValueError("metric direction must be 'higher' or 'lower'")
    sign = 1.0 if direction == "higher" else -1.0
    advantage_low = sign * result.ci_low
    advantage_high = sign * result.ci_high
    return {
        "preferred_direction": direction,
        "observed_treatment_advantage": sign * result.observed_mean_difference,
        "advantage_ci_low": min(advantage_low, advantage_high),
        "advantage_ci_high": max(advantage_low, advantage_high),
        "bootstrap_probability_treatment_better": (
            result.bootstrap_probability_difference_positive
            if direction == "higher"
            else result.bootstrap_probability_difference_negative
        ),
    }


def _utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def day_windows(window: ReplayWindow, requested_days: list[str] | None) -> list[ReplayWindow]:
    """Make non-overlapping, whole-day replay windows inside a declared split."""
    if requested_days:
        starts = [_utc_timestamp(value).normalize() for value in requested_days]
    else:
        starts = list(pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left"))
    windows: list[ReplayWindow] = []
    for start in starts:
        candidate = ReplayWindow(start, start + pd.Timedelta(days=1))
        if candidate.start < window.start or candidate.end > window.end:
            raise ValueError(
                f"requested day {start.date()} is outside declared split "
                f"{window.start.date()} to {(window.end - pd.Timedelta(seconds=1)).date()}"
            )
        windows.append(candidate)
    if len({item.start for item in windows}) != len(windows):
        raise ValueError("each requested day must be unique")
    return windows


def run_one_day(
    *,
    config_path: Path,
    split: str,
    window: ReplayWindow,
    output_dir: Path,
    skip_distributed: bool,
    policies: list[str] | None,
    distributed_settings: Path | None,
    mappo_checkpoint: Path | None,
    ippo_checkpoint: Path | None,
    apply_commitment_guard: bool,
    commitment_guard_mode: str,
) -> pd.DataFrame | None:
    run_dir = output_dir / config_path.stem / window.start.strftime("%Y%m%d")
    command = [
        sys.executable,
        "scripts/run_trace_pilot.py",
        "--config",
        str(config_path),
        "--split",
        split,
        "--replay-start",
        window.start.isoformat(),
        "--replay-end",
        window.end.isoformat(),
        "--output-dir",
        str(run_dir),
    ]
    if skip_distributed:
        command.append("--skip-distributed")
    if policies:
        command.extend(["--only", *policies])
    if distributed_settings:
        command.extend(["--distributed-settings", str(distributed_settings)])
    if mappo_checkpoint:
        command.extend(["--mappo-checkpoint", str(mappo_checkpoint)])
    if ippo_checkpoint:
        command.extend(["--ippo-checkpoint", str(ippo_checkpoint)])
    if apply_commitment_guard:
        command.append("--apply-commitment-guard")
        command.extend(["--commitment-guard-mode", commitment_guard_mode])
    # A child replay can fail after producing useful context.  Capture that
    # context so a matrix run reports the true cause rather than the opaque
    # ``CalledProcessError`` alone.  The matrix itself prints each day before
    # launching it; detailed child output is retained only when it fails.
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        failure_path = run_dir / "subprocess_failure.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr,
            encoding="utf-8",
        )
        raise RuntimeError(
            f"day replay failed with exit code {completed.returncode}; "
            f"diagnostics written to {failure_path}"
        )
    metrics_path = run_dir / f"{split}_pilot_metrics.csv"
    if not metrics_path.is_file():
        empty_path = run_dir / f"{split}_empty_replay.json"
        if empty_path.is_file():
            return None
        raise FileNotFoundError(
            f"completed replay did not write metrics or an empty-replay audit: {run_dir}"
        )
    metrics = pd.read_csv(metrics_path)
    metrics["replay_start"] = window.start.isoformat()
    metrics["replay_end"] = window.end.isoformat()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched day-level FairFlex evaluation with paired block bootstrap summaries."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="one or more study JSON files, for example configs/caltech_2019_scarce.json",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--days",
        nargs="+",
        help="optional UTC days such as 2019-05-01 2019-05-02; default is every full day in the split",
    )
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument(
        "--apply-commitment-guard",
        action="store_true",
        help=(
            "apply each configuration's locked calibration-derived early-departure guard "
            "to every test replay"
        ),
    )
    parser.add_argument(
        "--commitment-guard-mode",
        choices=("global", "duration_stratified", "adaptive", "multi_rate_envelope", "agaci_weighted"),
        default="global",
        help="guard ablation applied to every replay when --apply-commitment-guard is enabled",
    )
    parser.add_argument("--only", nargs="+", help="optional exact policy names passed to each replay")
    parser.add_argument(
        "--distributed-settings",
        type=Path,
        help="optional validation-locked distributed policy profile forwarded to every replay",
    )
    parser.add_argument(
        "--mappo-checkpoint",
        type=Path,
        help=(
            "optional frozen Safe-MAPPO checkpoint selected on validation data only; "
            "adds a matched central-vs-MAPPO comparison"
        ),
    )
    parser.add_argument(
        "--ippo-checkpoint",
        type=Path,
        help=(
            "optional frozen parameter-shared IPPO checkpoint selected on validation data only; "
            "adds a matched central-vs-IPPO comparison"
        ),
    )
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation_matrix"))
    args = parser.parse_args()

    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    if args.commitment_guard_mode != "global" and not args.apply_commitment_guard:
        parser.error("--commitment-guard-mode requires --apply-commitment-guard")
    if args.mappo_checkpoint and not args.mappo_checkpoint.is_file():
        parser.error(f"MAPPO checkpoint does not exist: {args.mappo_checkpoint}")
    if args.ippo_checkpoint and not args.ippo_checkpoint.is_file():
        parser.error(f"IPPO checkpoint does not exist: {args.ippo_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []
    manifest: dict[str, object] = {
        "purpose": "Matched policy comparisons with day-block bootstrap uncertainty summaries.",
        "unit_of_resampling": "matched day within a single study configuration, never individual EV sessions",
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "mappo_checkpoint": str(args.mappo_checkpoint) if args.mappo_checkpoint else None,
        "mappo_protocol": (
            "Frozen Safe-MAPPO actor; training and selection must use pre-test and validation cohorts only."
            if args.mappo_checkpoint
            else None
        ),
        "ippo_checkpoint": str(args.ippo_checkpoint) if args.ippo_checkpoint else None,
        "ippo_protocol": (
            "Frozen parameter-shared IPPO actor; training and selection must use pre-test and validation cohorts only."
            if args.ippo_checkpoint
            else None
        ),
        "runs": [],
        "commitment_guard": (
            "Each run applies only its configuration's calibration-derived guard; "
            "the held-out replay is never used to fit it."
            if args.apply_commitment_guard
            else None
        ),
    }

    for supplied_path in args.configs:
        config_path = Path(supplied_path)
        config = load_study_config(config_path)
        if args.split not in config["splits"]:
            parser.error(f"split {args.split!r} is not in {config_path}")
        try:
            windows = day_windows(ReplayWindow.from_config(config["splits"][args.split]), args.days)
        except ValueError as error:
            parser.error(str(error))
        for window in windows:
            print(f"Running {config['study_id']} for {window.start.date()}...")
            metrics = run_one_day(
                config_path=config_path,
                split=args.split,
                window=window,
                output_dir=args.output_dir,
                skip_distributed=args.skip_distributed,
                policies=args.only,
                distributed_settings=args.distributed_settings,
                mappo_checkpoint=args.mappo_checkpoint,
                ippo_checkpoint=args.ippo_checkpoint,
                apply_commitment_guard=args.apply_commitment_guard,
                commitment_guard_mode=args.commitment_guard_mode,
            )
            run_record = {
                "study_id": config["study_id"],
                "evaluation_group": config.get("evaluation_group", config["study_id"]),
                "config": str(config_path),
                "replay_start": window.start.isoformat(),
                "replay_end": window.end.isoformat(),
                "distributed_settings": str(args.distributed_settings) if args.distributed_settings else None,
                "mappo_checkpoint": str(args.mappo_checkpoint) if args.mappo_checkpoint else None,
                "ippo_checkpoint": str(args.ippo_checkpoint) if args.ippo_checkpoint else None,
                "apply_commitment_guard": args.apply_commitment_guard,
                "commitment_guard_mode": args.commitment_guard_mode,
            }
            if metrics is None:
                run_record["status"] = "skipped_empty_replay"
                manifest["runs"].append(run_record)
                print(f"Skipping {window.start.date()}: no usable sessions, so metrics are undefined.")
                continue
            metrics.insert(0, "study_id", config["study_id"])
            metrics.insert(1, "evaluation_group", config.get("evaluation_group", config["study_id"]))
            all_metrics.append(metrics)
            run_record["status"] = "completed"
            manifest["runs"].append(run_record)

    if not all_metrics:
        raise RuntimeError("no replay day contained usable sessions; no matched metrics can be computed")
    combined = pd.concat(all_metrics, ignore_index=True).sort_values(
        ["study_id", "replay_start", "policy"]
    )
    metrics_path = args.output_dir / "daywise_metrics.csv"
    combined.to_csv(metrics_path, index=False)

    comparisons: list[dict[str, object]] = []
    comparison_specs = COMPARISON_SPECS
    if args.mappo_checkpoint:
        comparison_specs = (*comparison_specs, ("central_vs_safe_mappo", CENTRAL_FAIRFLEX, "safe_mappo_station"))
    if args.ippo_checkpoint:
        comparison_specs = (*comparison_specs, ("central_vs_safe_ippo", CENTRAL_FAIRFLEX, "safe_ippo_station"))
    if args.mappo_checkpoint and args.ippo_checkpoint:
        comparison_specs = (
            *comparison_specs,
            ("safe_ippo_vs_safe_mappo", "safe_ippo_station", "safe_mappo_station"),
        )
    for evaluation_group, subset in combined.groupby("evaluation_group", sort=True):
        present = set(subset["policy"])
        for comparison_family, treatment, baseline in comparison_specs:
            if treatment not in present or baseline not in present:
                continue
            for metric, direction in METRIC_SPECS:
                try:
                    result = paired_metric_bootstrap(
                        subset,
                        metric=metric,
                        treatment=treatment,
                        baseline=baseline,
                        resamples=args.resamples,
                        seed=args.seed,
                    )
                except ValueError as error:
                    print(f"Skipping {evaluation_group} {treatment} vs {baseline} for {metric}: {error}")
                    continue
                comparisons.append(
                    {
                        "evaluation_group": evaluation_group,
                        "comparison_family": comparison_family,
                        **result.to_dict(),
                        **_oriented_comparison_fields(result, direction),
                    }
                )
    comparison_columns = [
        "evaluation_group", "comparison_family", "metric", "treatment", "baseline", "blocks",
        "observed_mean_difference", "ci_low", "ci_high",
        "bootstrap_probability_difference_positive", "bootstrap_probability_difference_negative",
        "preferred_direction", "observed_treatment_advantage", "advantage_ci_low",
        "advantage_ci_high", "bootstrap_probability_treatment_better", "resamples", "seed",
    ]
    comparisons_table = pd.DataFrame(comparisons, columns=comparison_columns)
    comparisons_path = args.output_dir / "paired_day_bootstrap.csv"
    comparisons_table.to_csv(comparisons_path, index=False)
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Day-level metrics: {metrics_path}")
    print(f"Paired bootstrap comparisons: {comparisons_path}")
    print("Runtime is recorded in daywise_metrics.csv but is not bootstrapped as a quality gain; lower is better.")
    print("Interpret confidence intervals cautiously for small numbers of days; expand held-out periods before paper claims.")


if __name__ == "__main__":
    main()
