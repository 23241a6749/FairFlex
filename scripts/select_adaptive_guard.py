"""Select a past-only adaptive-guard learning rate on a declared development split.

The script runs exactly the listed candidates on one development configuration,
stores every run, and writes an explicit deterministic selection record. It is
not permitted to read a final validation configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PRIMARY_POLICY = "fair_mpc_ac_robust_pv"


def _label(value: float) -> str:
    return f"gamma_{value:.6f}".replace(".", "p")


def _select(table: pd.DataFrame) -> tuple[pd.Series, str]:
    """Apply the predeclared coverage-first, fairness-second selection rule."""
    covered = table.loc[table["empirical_coverage"] >= table["target_coverage"]]
    pool = covered if not covered.empty else table
    rule = (
        "coverage at or above target, then maximize P10 service, Jain index, and energy service"
        if not covered.empty
        else "no candidate reached coverage target; maximize empirical coverage, then P10, Jain, and energy service"
    )
    ranking = (
        ["p10_service_ratio", "jain_service_index", "energy_service_ratio"]
        if not covered.empty
        else ["empirical_coverage", "p10_service_ratio", "jain_service_index", "energy_service_ratio"]
    )
    ordered = pool.sort_values(ranking, ascending=False, kind="stable")
    return ordered.iloc[0], rule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predeclared adaptive-guard candidates on one development split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reserving-final-note", required=True)
    args = parser.parse_args()
    if len(args.learning_rates) < 2 or len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("supply at least two distinct learning rates")
    if any(rate <= 0 for rate in args.learning_rates):
        parser.error("learning rates must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for rate in args.learning_rates:
        run_dir = args.output_dir / _label(rate)
        command = [
            sys.executable,
            "scripts/run_trace_pilot.py",
            "--config",
            str(args.config),
            "--split",
            "test",
            "--apply-commitment-guard",
            "--commitment-guard-mode",
            "adaptive",
            "--adaptive-learning-rate",
            str(rate),
            "--skip-distributed",
            "--only",
            PRIMARY_POLICY,
            "--output-dir",
            str(run_dir),
        ]
        subprocess.run(command, check=True)
        metrics = pd.read_csv(run_dir / "test_pilot_metrics.csv")
        metric = metrics.loc[metrics["policy"] == PRIMARY_POLICY]
        if len(metric) != 1:
            raise RuntimeError(f"expected exactly one {PRIMARY_POLICY} row in {run_dir}")
        summary = json.loads((run_dir / "test_pilot_summary.json").read_text(encoding="utf-8"))
        guard = summary["assumptions"]["commitment_guard"]
        rows.append(
            {
                "learning_rate": rate,
                "run_directory": str(run_dir),
                "empirical_coverage": float(guard["empirical_coverage"]),
                "target_coverage": float(guard["target_coverage"]),
                **metric.iloc[0].to_dict(),
            }
        )

    table = pd.DataFrame(rows).sort_values("learning_rate")
    selected, rule = _select(table)
    table.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
    selection = {
        "purpose": "Development-only learning-rate selection for a past-only adaptive guard.",
        "config": str(args.config),
        "candidates": list(args.learning_rates),
        "selection_rule": rule,
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_metrics": {
            key: float(selected[key])
            for key in (
                "empirical_coverage",
                "target_coverage",
                "p10_service_ratio",
                "jain_service_index",
                "energy_service_ratio",
            )
        },
        "reserved_final_validation": args.reserving_final_note,
        "warning": (
            "No final-validation session or outcome was read by this selection script. "
            "Run the selected rate unchanged on the separately frozen final configuration."
        ),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(f"Selected learning rate: {selection['selected_learning_rate']}")
    print(f"Candidate table: {args.output_dir / 'candidate_metrics.csv'}")
    print(f"Selection record: {args.output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
