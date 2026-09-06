"""Generate a paper-safe report from locked analysis-held-out artifacts.

No model, guard, or metric is rerun here.  The report is derived only from
already-written manifests and result files, preserving the distinction between
development selection and prospective final evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY = "fair_mpc_ac_robust_pv"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(table: pd.DataFrame) -> str:
    """Format a small report table without requiring Pandas' optional tabulate dependency."""
    columns = list(table.columns)

    def display(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    rows = [[display(value) for value in row] for row in table.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an analysis-held-out validation report from frozen FairFlex artifacts."
    )
    parser.add_argument("--may-selection", type=Path, required=True)
    parser.add_argument("--june-summary", type=Path, required=True)
    parser.add_argument("--october-manifest", type=Path, required=True)
    parser.add_argument("--october-summary", type=Path, required=True)
    parser.add_argument("--october-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    may = _load_json(args.may_selection)
    june = _load_json(args.june_summary)
    manifest = _load_json(args.october_manifest)
    october = _load_json(args.october_summary)
    metrics = pd.read_csv(args.october_metrics)
    if PRIMARY not in set(metrics["policy"]):
        raise ValueError(f"missing primary policy {PRIMARY!r}")
    guard = october["assumptions"]["commitment_guard"]
    if not guard.get("applied") or guard.get("mode") != "adaptive":
        raise ValueError("October summary does not contain the locked adaptive guard")
    target = float(guard["target_coverage"])
    observed = float(guard["empirical_coverage"])
    safety = metrics[["policy", "unsafe_steps"]].copy()
    all_safe = bool((safety["unsafe_steps"] == 0).all())
    primary = metrics.loc[metrics["policy"] == PRIMARY].iloc[0]
    comparisons = []
    for baseline in (
        "uncontrolled_shared_cap",
        "fcfs_shared_cap",
        "equal_share_shared_cap",
        "least_laxity_first_shared_cap",
        "fair_mpc_ac_no_pv",
    ):
        row = metrics.loc[metrics["policy"] == baseline]
        if row.empty:
            continue
        row = row.iloc[0]
        comparisons.append(
            {
                "baseline": baseline,
                "p10_difference_fairflex_minus_baseline": float(
                    primary["p10_service_ratio"] - row["p10_service_ratio"]
                ),
                "energy_service_difference_fairflex_minus_baseline": float(
                    primary["energy_service_ratio"] - row["energy_service_ratio"]
                ),
            }
        )

    report = {
        "study_id": october["study_id"],
        "protocol": {
            "may_selected_learning_rate": may["selected_learning_rate"],
            "may_selection_rule": may["selection_rule"],
            "june_replication_coverage": june["assumptions"]["commitment_guard"][
                "empirical_coverage"
            ],
            "october_raw_acn_sha256": manifest["raw_input_sha256"]["acn_test"],
            "october_usable_sessions": manifest["splits"]["test"]["usable_sessions"],
        },
        "coverage": {
            "target": target,
            "observed": observed,
            "difference_observed_minus_target": observed - target,
            "passes_target": observed >= target,
            "causal_outcomes_observed_before_later_decisions": guard[
                "outcomes_observed_before_decisions"
            ],
        },
        "safety": {"all_policies_zero_unsafe_steps": all_safe},
        "primary_policy": PRIMARY,
        "policy_metrics": metrics.to_dict(orient="records"),
        "fairflex_pairwise_descriptive_differences": comparisons,
        "paper_safe_conclusion": (
            "The frozen adaptive guard did not meet its 90% coverage target on the analysis-held-out "
            "October window. Report the causal protocol, safety, and lower-tail fairness evidence, "
            "but do not claim solved or universally calibrated early-departure robustness."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prospective_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    table = metrics[
        [
            "policy",
            "sessions",
            "mean_service_ratio",
            "p10_service_ratio",
            "jain_service_index",
            "energy_service_ratio",
            "unsafe_steps",
        ]
    ].copy()
    markdown = "\n".join(
        [
            "# Locked analysis-held-out October validation",
            "",
            "This report is generated from frozen artifacts; it does not rerun or tune a method.",
            "",
            "## Protocol",
            "",
            f"- May-selected adaptive learning rate: `{may['selected_learning_rate']}`.",
            f"- June replication coverage: `{float(report['protocol']['june_replication_coverage']):.3f}`.",
            f"- October raw ACN SHA-256: `{report['protocol']['october_raw_acn_sha256']}`.",
            f"- October usable declared-commitment sessions: `{report['protocol']['october_usable_sessions']}`.",
            "",
            "## Guard result",
            "",
            f"- Target coverage: `{target:.3f}`.",
            f"- Observed coverage: `{observed:.3f}`.",
            f"- Target met: `{observed >= target}`.",
            "",
            "## Policy metrics",
            "",
            _markdown_table(table),
            "",
            "## Conclusion",
            "",
            report["paper_safe_conclusion"],
            "",
        ]
    )
    (args.output_dir / "prospective_validation_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown)
    print(f"JSON: {args.output_dir / 'prospective_validation_report.json'}")
    print(f"Markdown: {args.output_dir / 'prospective_validation_report.md'}")


if __name__ == "__main__":
    main()
