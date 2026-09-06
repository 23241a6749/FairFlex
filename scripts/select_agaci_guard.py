"""Select between frozen envelope and AgACI-style May development runs.

The script only reads already-created May artifacts. It is deliberately unable
to read the reserved June or December configurations, preventing accidental
post-hoc use of those outcomes in the candidate choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_POLICY = "fair_mpc_ac_robust_pv"


def _read_candidate(label: str, run_directory: Path) -> dict[str, float | str]:
    summary = json.loads((run_directory / "test_pilot_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_directory / "test_pilot_metrics.csv")
    primary = metrics.loc[metrics["policy"] == PRIMARY_POLICY]
    if len(primary) != 1:
        raise ValueError(f"{run_directory}: expected one {PRIMARY_POLICY!r} row")
    guard = summary["assumptions"]["commitment_guard"]
    if not guard.get("applied"):
        raise ValueError(f"{run_directory}: applied commitment guard is required")
    row = primary.iloc[0]
    return {
        "candidate": label,
        "guard_mode": str(guard["mode"]),
        "run_directory": str(run_directory),
        "empirical_coverage": float(guard["empirical_coverage"]),
        "target_coverage": float(guard["target_coverage"]),
        "mean_buffer_minutes": float(guard["buffer_minutes_at_decision"]["mean"]),
        "p10_service_ratio": float(row["p10_service_ratio"]),
        "jain_service_index": float(row["jain_service_index"]),
        "energy_service_ratio": float(row["energy_service_ratio"]),
        "runtime_seconds": float(row["runtime_seconds"]),
    }


def _select(table: pd.DataFrame) -> tuple[pd.Series, str]:
    covered = table.loc[table["empirical_coverage"] >= table["target_coverage"]]
    pool = covered if not covered.empty else table
    ranking = [
        "empirical_coverage",
        "p10_service_ratio",
        "jain_service_index",
        "energy_service_ratio",
        "mean_buffer_minutes",
    ]
    ascending = [False, False, False, False, True]
    rule = (
        "Among candidates meeting the coverage target, maximize coverage, P10 service, Jain "
        "index, and energy service; then minimize mean guard buffer."
        if not covered.empty
        else "No candidate met the coverage target; maximize coverage, P10 service, Jain index, "
        "and energy service; then minimize mean guard buffer."
    )
    return pool.sort_values(ranking, ascending=ascending, kind="stable").iloc[0], rule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the coverage-first May selection record for AgACI-style development."
    )
    parser.add_argument("--envelope-run", type=Path, required=True)
    parser.add_argument("--agaci-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reserved-replication-note", required=True)
    args = parser.parse_args()

    rows = [
        _read_candidate("multi_rate_conservative_envelope", args.envelope_run),
        _read_candidate("bounded_agaci_style_weighted", args.agaci_run),
    ]
    expected = {"multi_rate_envelope", "agaci_weighted"}
    if {row["guard_mode"] for row in rows} != expected:
        raise ValueError(f"expected exactly guard modes {expected}")
    table = pd.DataFrame(rows)
    selected, rule = _select(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
    record = {
        "purpose": "Development-only guard selection from frozen May artifacts.",
        "selection_rule": rule,
        "selected_candidate": str(selected["candidate"]),
        "selected_guard_mode": str(selected["guard_mode"]),
        "selected_metrics": {
            key: float(selected[key])
            for key in (
                "empirical_coverage",
                "target_coverage",
                "mean_buffer_minutes",
                "p10_service_ratio",
                "jain_service_index",
                "energy_service_ratio",
                "runtime_seconds",
            )
        },
        "candidates": rows,
        "reserved_replication": args.reserved_replication_note,
        "warning": (
            "June and December outcomes must not be read by this selector. Run only the selected "
            "guard unchanged on the separately declared June replication."
        ),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"Selected guard mode: {record['selected_guard_mode']}")
    print(f"Candidate metrics: {args.output_dir / 'candidate_metrics.csv'}")
    print(f"Selection record: {args.output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
