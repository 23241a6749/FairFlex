"""Create a transparent report comparing two already completed V2 dev runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairflex.v2.comparison import development_metric_deltas


def _read_summary(directory: Path) -> dict[str, object]:
    path = directory / "development_summary.json"
    if not path.is_file():
        raise ValueError(f"missing V2 development summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(directory: Path) -> list[dict[str, object]]:
    path = directory / "development_policy_metrics.csv"
    if not path.is_file():
        raise ValueError(f"missing V2 policy metrics: {path}")
    return pd.read_csv(path).to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two isolated FairFlex V2 development outputs; never select a final method."
    )
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/development_run_comparison"),
    )
    args = parser.parse_args()
    reference_summary = _read_summary(args.reference_dir)
    candidate_summary = _read_summary(args.candidate_dir)
    if reference_summary.get("status") != "development_only_not_a_frozen_test_result":
        parser.error("reference input must be a V2 development-only report")
    if candidate_summary.get("status") != "development_only_not_a_frozen_test_result":
        parser.error("candidate input must be a V2 development-only report")
    deltas = development_metric_deltas(
        _read_metrics(args.reference_dir), _read_metrics(args.candidate_dir)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(deltas).to_csv(args.output_dir / "policy_metric_deltas.csv", index=False)
    report = {
        "status": "development_only_tradeoff_report_not_a_frozen_test_result",
        "reference_directory": str(args.reference_dir),
        "candidate_directory": str(args.candidate_dir),
        "reference_guard": reference_summary.get("guard"),
        "candidate_guard": candidate_summary.get("guard"),
        "interpretation": (
            "This report shows candidate-minus-reference outcomes on one development window. "
            "It does not select, validate, or establish the superiority of a final method."
        ),
        "required_next_step": (
            "Predeclare the final candidate and a new chronological and external-site test "
            "before reading or producing those test outcomes."
        ),
    }
    (args.output_dir / "development_tradeoff_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(deltas).to_string(index=False))
    print(f"V2 development-only comparison: {args.output_dir}")


if __name__ == "__main__":
    main()
