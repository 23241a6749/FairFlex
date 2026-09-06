"""Run a causal FairFlex guarded-versus-unguarded MPC ablation by day.

Every condition replays exactly the same chronological day. The only
controller-information difference is whether a predeclared calibration-locked
early-departure guard is applied. Whole days, not individual EV rows, are the
resampling blocks for uncertainty summaries.
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
from run_evaluation_matrix import _oriented_comparison_fields


METRIC_SPECS = (
    ("mean_service_ratio", "higher"),
    ("p10_service_ratio", "higher"),
    ("worst_service_ratio", "higher"),
    ("jain_service_index", "higher"),
    ("energy_service_ratio", "higher"),
    ("completion_rate", "higher"),
    ("individually_feasible_request_rate", "higher"),
    ("unavoidable_individual_shortfall_kwh", "lower"),
    ("delivered_energy_kwh", "higher"),
    ("min_voltage_pu", "higher"),
    ("max_line_loading_percent", "lower"),
    ("unsafe_steps", "lower"),
)
UNGUARDED_POLICY = "fair_mpc_ac_robust_pv_unguarded"
SUPPORTED_GUARD_MODES = ("global", "duration_stratified", "adaptive", "multi_rate_envelope", "agaci_weighted")


def _condition_policy_name(guard_mode: str | None) -> str:
    return UNGUARDED_POLICY if guard_mode is None else f"fair_mpc_ac_robust_pv_{guard_mode}_guard"


def _utc_day(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC").normalize()
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC").normalize()
    )


def _day_windows(window: ReplayWindow, requested_days: list[str] | None) -> list[ReplayWindow]:
    starts = (
        [_utc_day(value) for value in requested_days]
        if requested_days
        else list(pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left"))
    )
    windows: list[ReplayWindow] = []
    for start in starts:
        candidate = ReplayWindow(start, start + pd.Timedelta(days=1))
        if candidate.start < window.start or candidate.end > window.end:
            raise ValueError(f"requested day {start.date()} is outside the declared test split")
        windows.append(candidate)
    if len({item.start for item in windows}) != len(windows):
        raise ValueError("each requested day must be unique")
    return windows


def _run_condition(
    *,
    config_path: Path,
    window: ReplayWindow,
    output_dir: Path,
    guard_mode: str | None,
) -> pd.DataFrame | None:
    variant = f"{guard_mode}_guard" if guard_mode is not None else "unguarded"
    run_dir = output_dir / config_path.stem / window.start.strftime("%Y%m%d") / variant
    command = [
        sys.executable,
        "scripts/run_trace_pilot.py",
        "--config",
        str(config_path),
        "--split",
        "test",
        "--replay-start",
        window.start.isoformat(),
        "--replay-end",
        window.end.isoformat(),
        "--skip-distributed",
        "--only",
        "fair_mpc_ac_robust_pv",
        "--output-dir",
        str(run_dir),
    ]
    if guard_mode is not None:
        command.extend(["--apply-commitment-guard", "--commitment-guard-mode", guard_mode])
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
            f"guard-ablation replay failed with exit code {completed.returncode}; "
            f"diagnostics written to {failure_path}"
        )
    metrics_path = run_dir / "test_pilot_metrics.csv"
    if not metrics_path.is_file():
        empty_path = run_dir / "test_empty_replay.json"
        if empty_path.is_file():
            return None
        raise FileNotFoundError(
            f"completed guard-ablation replay wrote neither metrics nor an empty audit: {run_dir}"
        )
    table = pd.read_csv(metrics_path)
    if len(table) != 1 or table.iloc[0]["policy"] != "fair_mpc_ac_robust_pv":
        raise RuntimeError("expected exactly one central FairFlex policy result")
    table.loc[:, "policy"] = _condition_policy_name(guard_mode)
    table["replay_start"] = window.start.isoformat()
    table["replay_end"] = window.end.isoformat()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired daily predeclared-guard versus unguarded FairFlex MPC evaluation."
    )
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--days", nargs="+", help="optional UTC days inside each final test split")
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/commitment_guard_ablation"))
    parser.add_argument(
        "--guard-modes",
        nargs="+",
        choices=SUPPORTED_GUARD_MODES,
        default=["global"],
        help=(
            "predeclared guarded conditions to compare against the unguarded condition; "
            "the first condition is always unguarded"
        ),
    )
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    if len(set(args.guard_modes)) != len(args.guard_modes):
        parser.error("each --guard-modes value must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    manifest: dict[str, object] = {
        "purpose": "Matched causal predeclared-guard versus unguarded central FairFlex MPC ablation.",
        "unit_of_resampling": "matched chronological day, never individual EV session",
        "guard": {
            "modes": args.guard_modes,
            "fitting": "configured calibration split only",
            "selection": "frozen before supplied final-season configurations",
        },
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "runs": [],
    }
    for config_path in args.configs:
        config = load_study_config(config_path)
        window = ReplayWindow.from_config(config["splits"]["test"])
        for day in _day_windows(window, args.days):
            for guard_mode in (None, *args.guard_modes):
                print(
                    f"Running {config['study_id']} {day.start.date()} "
                    f"({guard_mode if guard_mode is not None else 'unguarded'})..."
                )
                row = _run_condition(
                    config_path=config_path,
                    window=day,
                    output_dir=args.output_dir,
                    guard_mode=guard_mode,
                )
                run_record = {
                    "study_id": config["study_id"],
                    "replay_start": day.start.isoformat(),
                    "condition": f"{guard_mode}_guard" if guard_mode is not None else "unguarded",
                }
                if row is None:
                    run_record["status"] = "skipped_empty_replay"
                    manifest["runs"].append(run_record)
                    print(f"Skipping {day.start.date()} ({run_record['condition']}): no usable sessions.")
                    continue
                row.insert(0, "study_id", config["study_id"])
                row.insert(1, "evaluation_group", config.get("evaluation_group", config["study_id"]))
                all_rows.append(row)
                run_record["status"] = "completed"
                manifest["runs"].append(run_record)

    if not all_rows:
        raise RuntimeError("no guard-ablation replay day contained usable sessions")
    daywise = pd.concat(all_rows, ignore_index=True).sort_values(
        ["study_id", "replay_start", "policy"]
    )
    daywise.to_csv(args.output_dir / "daywise_guard_ablation_metrics.csv", index=False)
    summaries: list[dict[str, object]] = []
    for study_id, study_rows in daywise.groupby("study_id", sort=True):
        for guard_mode in args.guard_modes:
            treatment = _condition_policy_name(guard_mode)
            for metric, direction in METRIC_SPECS:
                try:
                    result = paired_metric_bootstrap(
                        study_rows,
                        metric=metric,
                        treatment=treatment,
                        baseline=UNGUARDED_POLICY,
                        resamples=args.resamples,
                        seed=args.seed,
                    )
                except ValueError as error:
                    print(f"Skipping {study_id} {treatment} {metric}: {error}")
                    continue
                summaries.append(
                    {
                        "study_id": study_id,
                        "comparison_family": f"{guard_mode}_guard_vs_unguarded",
                        **result.to_dict(),
                        **_oriented_comparison_fields(result, direction),
                    }
                )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "paired_day_bootstrap.csv", index=False
    )
    (args.output_dir / "guard_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Day metrics: {args.output_dir / 'daywise_guard_ablation_metrics.csv'}")
    print(f"Bootstrap comparisons: {args.output_dir / 'paired_day_bootstrap.csv'}")


if __name__ == "__main__":
    main()
