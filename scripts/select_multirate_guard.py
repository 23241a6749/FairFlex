"""Select a causal early-departure guard from frozen May/June artifacts.

This script deliberately *does not* run or tune a controller.  It only reads
the four already-created development artifacts (single-rate versus multi-rate,
across May and the unchanged June replication), applies the declared
coverage-first selection rule, and writes a traceable record for the next
unseen-site stress test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_POLICY = "fair_mpc_ac_robust_pv"
METRICS = ("p10_service_ratio", "jain_service_index", "energy_service_ratio")


def _read_run(path: Path) -> dict[str, float | str]:
    """Load the guard audit and primary-policy metrics from one frozen run."""
    summary_path = path / "test_pilot_summary.json"
    metrics_path = path / "test_pilot_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    primary = metrics.loc[metrics["policy"] == PRIMARY_POLICY]
    if len(primary) != 1:
        raise ValueError(f"{path}: expected exactly one {PRIMARY_POLICY!r} row")
    guard = summary["assumptions"]["commitment_guard"]
    if not guard.get("applied"):
        raise ValueError(f"{path}: no applied commitment guard")
    row = primary.iloc[0]
    return {
        "study_id": str(summary["study_id"]),
        "guard_mode": str(guard["mode"]),
        "empirical_coverage": float(guard["empirical_coverage"]),
        "target_coverage": float(guard["target_coverage"]),
        "mean_buffer_minutes": float(guard["buffer_minutes_at_decision"]["mean"]),
        "runtime_seconds": float(row["runtime_seconds"]),
        **{name: float(row[name]) for name in METRICS},
    }


def _aggregate(name: str, may: dict[str, float | str], june: dict[str, float | str]) -> dict[str, float | str | bool]:
    """Summarize an unchanged method over development and replication periods."""
    if may["guard_mode"] != june["guard_mode"]:
        raise ValueError(f"{name}: guard modes differ between May and June")
    target = float(may["target_coverage"])
    if float(june["target_coverage"]) != target:
        raise ValueError(f"{name}: coverage targets differ between May and June")
    return {
        "candidate": name,
        "guard_mode": str(may["guard_mode"]),
        "target_coverage": target,
        "may_coverage": float(may["empirical_coverage"]),
        "june_coverage": float(june["empirical_coverage"]),
        "minimum_coverage": min(float(may["empirical_coverage"]), float(june["empirical_coverage"])),
        "mean_coverage": (float(may["empirical_coverage"]) + float(june["empirical_coverage"])) / 2,
        "both_periods_meet_target": bool(
            float(may["empirical_coverage"]) >= target
            and float(june["empirical_coverage"]) >= target
        ),
        **{
            f"mean_{metric}": (float(may[metric]) + float(june[metric])) / 2
            for metric in METRICS
        },
        "mean_buffer_minutes": (
            float(may["mean_buffer_minutes"]) + float(june["mean_buffer_minutes"])
        )
        / 2,
        "mean_runtime_seconds": (
            float(may["runtime_seconds"]) + float(june["runtime_seconds"])
        )
        / 2,
    }


def _select(table: pd.DataFrame) -> tuple[pd.Series, str]:
    """Apply the declared coverage-first rule over both fixed periods."""
    eligible = table.loc[table["both_periods_meet_target"]]
    pool = eligible if not eligible.empty else table
    ranking = [
        "minimum_coverage",
        "mean_coverage",
        "mean_p10_service_ratio",
        "mean_jain_service_index",
        "mean_energy_service_ratio",
    ]
    rule = (
        "Require target coverage in both May and June; then maximize minimum coverage, "
        "mean coverage, mean P10 service, mean Jain index, and mean energy service."
        if not eligible.empty
        else "No candidate met the target in both periods; maximize minimum coverage, mean "
        "coverage, mean P10 service, mean Jain index, and mean energy service."
    )
    return pool.sort_values(ranking, ascending=False, kind="stable").iloc[0], rule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a coverage-first selection between frozen single- and multi-rate guards."
    )
    parser.add_argument("--single-may", type=Path, required=True)
    parser.add_argument("--single-june", type=Path, required=True)
    parser.add_argument("--multirate-may", type=Path, required=True)
    parser.add_argument("--multirate-june", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reserved-validation-note",
        required=True,
        help="A human-readable statement of the data that remains untouched by this selection.",
    )
    args = parser.parse_args()

    candidates = [
        _aggregate("single_rate_adaptive", _read_run(args.single_may), _read_run(args.single_june)),
        _aggregate(
            "multi_rate_conservative_envelope",
            _read_run(args.multirate_may),
            _read_run(args.multirate_june),
        ),
    ]
    table = pd.DataFrame(candidates)
    selected, rule = _select(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "candidate_comparison.csv", index=False)
    record = {
        "purpose": "Development-only selection of a causal early-departure guard.",
        "selection_rule": rule,
        "selected_candidate": str(selected["candidate"]),
        "selected_guard_mode": str(selected["guard_mode"]),
        "selection_metrics": {
            key: (bool(selected[key]) if key == "both_periods_meet_target" else float(selected[key]))
            for key in (
                "target_coverage",
                "may_coverage",
                "june_coverage",
                "minimum_coverage",
                "mean_coverage",
                "mean_p10_service_ratio",
                "mean_jain_service_index",
                "mean_energy_service_ratio",
                "mean_buffer_minutes",
                "mean_runtime_seconds",
                "both_periods_meet_target",
            )
        },
        "candidates": candidates,
        "implementation_note": (
            "The multi-rate candidate is a conservative envelope of independently causal, "
            "fixed-rate ACI-inspired experts. It is not AgACI and does not imply a finite-sample, "
            "per-driver, or conditional coverage guarantee."
        ),
        "reserved_validation": args.reserved_validation_note,
        "warning": (
            "This record is a development/replication decision only. Do not tune the selected "
            "candidate on any subsequently acquired new-site or later-period validation data."
        ),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"Selected candidate: {record['selected_candidate']}")
    print(f"Comparison: {args.output_dir / 'candidate_comparison.csv'}")
    print(f"Selection record: {args.output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
