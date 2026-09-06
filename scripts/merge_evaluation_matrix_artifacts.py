"""Validate and merge completed matched-day evaluation replay artifacts.

This is deliberately a post-processing tool: it never reruns a controller or
changes an experiment.  It is useful when a long frozen matrix run has a
recoverable orchestration interruption after valid daily artifacts exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairflex.statistics import paired_metric_bootstrap
from run_evaluation_matrix import (
    CENTRAL_FAIRFLEX,
    COMPARISON_SPECS,
    METRIC_SPECS,
    _oriented_comparison_fields,
)


def _day_from_artifact(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Recover an ISO day block from the standard YYYYMMDD artifact folder."""
    try:
        start = pd.to_datetime(path.parent.name, format="%Y%m%d", utc=True)
    except ValueError as error:
        raise ValueError(f"artifact directory is not a YYYYMMDD day: {path.parent}") from error
    return start, start + pd.Timedelta(days=1)


def _load_completed_days(
    roots: list[Path], expected_policies: set[str]
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    seen_days: set[pd.Timestamp] = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"replay artifact root does not exist: {root}")
        metrics_paths = sorted(root.glob("*/test_pilot_metrics.csv"))
        if not metrics_paths:
            raise FileNotFoundError(f"no test_pilot_metrics.csv files found under: {root}")
        for metrics_path in metrics_paths:
            start, end = _day_from_artifact(metrics_path)
            if start in seen_days:
                raise ValueError(f"duplicate completed replay day found: {start.date()}")
            day = pd.read_csv(metrics_path)
            policies = set(day.get("policy", pd.Series(dtype=str)))
            if policies != expected_policies or len(day) != len(expected_policies):
                missing = sorted(expected_policies - policies)
                extra = sorted(policies - expected_policies)
                raise ValueError(
                    f"{metrics_path} does not contain exactly one row for each expected policy; "
                    f"missing={missing}, extra={extra}, rows={len(day)}"
                )
            if day["policy"].duplicated().any():
                raise ValueError(f"duplicate policy row in {metrics_path}")
            day.insert(0, "replay_start", start.isoformat())
            day.insert(1, "replay_end", end.isoformat())
            frames.append(day)
            seen_days.add(start)
            audits.append(
                {
                    "status": "completed",
                    "replay_start": start.isoformat(),
                    "replay_end": end.isoformat(),
                    "metrics_path": str(metrics_path),
                    "summary_path": str(metrics_path.parent / "test_pilot_summary.json"),
                }
            )
    return frames, audits


def _load_empty_audits(paths: list[Path]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    for path in paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        if content.get("status") != "skipped_empty_replay":
            raise ValueError(f"empty replay audit has unexpected status: {path}")
        window = content.get("replay_window")
        if not isinstance(window, dict) or not window.get("start") or not window.get("end"):
            raise ValueError(f"empty replay audit has no replay window: {path}")
        audits.append(
            {
                "status": "skipped_empty_replay",
                "replay_start": str(window["start"]),
                "replay_end": str(window["end"]),
                "reason": content.get("reason"),
                "source_records": content.get("source_records"),
                "selected_records": content.get("selected_records"),
                "dropped_records": content.get("dropped_records"),
                "audit_path": str(path),
            }
        )
    return audits


def _validate_calendar(
    completed: list[dict[str, object]],
    empty: list[dict[str, object]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    expected = set(pd.date_range(start, end, freq="1D", inclusive="left"))
    completed_days = {_timestamp(record["replay_start"]) for record in completed}
    empty_days = {_timestamp(record["replay_start"]) for record in empty}
    if completed_days & empty_days:
        raise ValueError("a calendar day cannot be both completed and an empty replay")
    observed = completed_days | empty_days
    if observed != expected:
        missing = sorted(item.date().isoformat() for item in expected - observed)
        extra = sorted(item.date().isoformat() for item in observed - expected)
        raise ValueError(f"calendar audit mismatch; missing={missing}, extra={extra}")


def _timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _comparison_table(combined: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison_family, treatment, baseline in COMPARISON_SPECS:
        if treatment not in set(combined["policy"]) or baseline not in set(combined["policy"]):
            continue
        for metric, direction in METRIC_SPECS:
            result = paired_metric_bootstrap(
                combined,
                metric=metric,
                treatment=treatment,
                baseline=baseline,
                resamples=resamples,
                seed=seed,
            )
            rows.append(
                {
                    "comparison_family": comparison_family,
                    **result.to_dict(),
                    **_oriented_comparison_fields(result, direction),
                }
            )
    columns = [
        "comparison_family", "metric", "treatment", "baseline", "blocks",
        "observed_mean_difference", "ci_low", "ci_high",
        "bootstrap_probability_difference_positive", "bootstrap_probability_difference_negative",
        "preferred_direction", "observed_treatment_advantage", "advantage_ci_low",
        "advantage_ci_high", "bootstrap_probability_treatment_better", "resamples", "seed",
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge completed matched-day FairFlex evaluation artifacts."
    )
    parser.add_argument("--replay-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--empty-replay-audits", nargs="*", type=Path, default=[])
    parser.add_argument("--expected-policies", nargs="+", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--evaluation-group", required=True)
    parser.add_argument("--calendar-start", required=True, help="inclusive ISO-8601 UTC day")
    parser.add_argument("--calendar-end", required=True, help="exclusive ISO-8601 UTC day")
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")

    start, end = _timestamp(args.calendar_start).normalize(), _timestamp(args.calendar_end).normalize()
    if end <= start:
        parser.error("--calendar-end must be later than --calendar-start")
    expected_policies = set(args.expected_policies)
    if CENTRAL_FAIRFLEX not in expected_policies:
        parser.error(f"--expected-policies must include primary treatment {CENTRAL_FAIRFLEX}")

    frames, completed_audits = _load_completed_days(args.replay_roots, expected_policies)
    empty_audits = _load_empty_audits(args.empty_replay_audits)
    _validate_calendar(completed_audits, empty_audits, start, end)
    combined = pd.concat(frames, ignore_index=True).sort_values(["replay_start", "policy"])
    combined.insert(0, "study_id", args.study_id)
    combined.insert(1, "evaluation_group", args.evaluation_group)
    comparisons = _comparison_table(combined, args.resamples, args.seed)
    comparisons.insert(0, "evaluation_group", args.evaluation_group)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "daywise_metrics.csv"
    comparison_path = args.output_dir / "paired_day_bootstrap.csv"
    combined.to_csv(metrics_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    manifest = {
        "purpose": "Validated merge of pre-existing frozen matched-day replay artifacts; no controller was rerun.",
        "study_id": args.study_id,
        "evaluation_group": args.evaluation_group,
        "calendar": {"start": start.isoformat(), "end": end.isoformat()},
        "unit_of_resampling": "matched non-empty calendar day; individual sessions are never resampled",
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "expected_policies": sorted(expected_policies),
        "completed_replays": completed_audits,
        "empty_replays": empty_audits,
        "completed_day_count": len(completed_audits),
        "empty_day_count": len(empty_audits),
    }
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Validated {len(completed_audits)} completed days and {len(empty_audits)} empty replay days.")
    print(f"Day-level metrics: {metrics_path}")
    print(f"Paired bootstrap comparisons: {comparison_path}")


if __name__ == "__main__":
    main()
