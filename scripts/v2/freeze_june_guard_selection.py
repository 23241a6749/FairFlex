"""Apply the documented June V2 guard decision rule reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairflex.v2.selection import apply_june_contextual_guard_rule


FAIR_MPC_POLICY = "v2_fair_mpc_ac_robust_pv"


def _summary(directory: Path) -> dict[str, object]:
    path = directory / "development_summary.json"
    if not path.is_file():
        raise ValueError(f"missing V2 development summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "development_only_not_a_frozen_test_result":
        raise ValueError(f"not a V2 development-only summary: {path}")
    return summary


def _fair_mpc_metrics(directory: Path) -> pd.Series:
    path = directory / "development_policy_metrics.csv"
    table = pd.read_csv(path).set_index("policy")
    if FAIR_MPC_POLICY not in table.index:
        raise ValueError(f"required policy {FAIR_MPC_POLICY} is absent from {path}")
    return table.loc[FAIR_MPC_POLICY]


def _mean_buffer(directory: Path) -> float:
    path = directory / "condition_guard_decision_audit.csv"
    return float(pd.read_csv(path)["buffer_minutes"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the V2 guard chosen by the predeclared June development rule."
    )
    parser.add_argument("--global-dir", required=True, type=Path)
    parser.add_argument("--contextual-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/june_guard_selection"),
    )
    args = parser.parse_args()
    global_summary = _summary(args.global_dir)
    contextual_summary = _summary(args.contextual_dir)
    global_metrics = _fair_mpc_metrics(args.global_dir)
    contextual_metrics = _fair_mpc_metrics(args.contextual_dir)
    target_coverage = float(contextual_summary["guard"]["target_coverage"])
    decision = apply_june_contextual_guard_rule(
        candidate_coverage=float(contextual_summary["guard"]["observed_coverage"]),
        target_coverage=target_coverage,
        candidate_unsafe_steps=int(contextual_metrics["unsafe_steps"]),
        p10_delta=float(contextual_metrics["p10_service_ratio"] - global_metrics["p10_service_ratio"]),
        jain_delta=float(contextual_metrics["jain_service_index"] - global_metrics["jain_service_index"]),
        mean_buffer_delta_minutes=_mean_buffer(args.contextual_dir) - _mean_buffer(args.global_dir),
    )
    decision.update(
        {
            "status": "development_guard_choice_frozen_not_a_final_test_result",
            "selection_rule_source": "docs/v2/02-development-selection-status.md",
            "global_directory": str(args.global_dir),
            "contextual_directory": str(args.contextual_dir),
            "required_next_step": (
                "Use the selected guard unchanged on a predeclared new chronological holdout "
                "and external-site replication; do not tune from their outcomes."
            ),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "june_guard_selection.json"
    output.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"V2 development guard choice: {output}")


if __name__ == "__main__":
    main()
