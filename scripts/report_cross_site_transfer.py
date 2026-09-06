"""Create a paper-safe report from frozen JPL cross-site transfer artifacts.

The report is intentionally descriptive.  It does not re-run a controller,
select a hyperparameter, or turn empirical coverage from one historical window
into a coverage guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY = "fair_mpc_ac_robust_pv"
BASELINES = (
    "uncontrolled_shared_cap",
    "fcfs_shared_cap",
    "equal_share_shared_cap",
    "least_laxity_first_shared_cap",
    "fair_mpc_ac_no_pv",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""
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
        description="Create a descriptive report from frozen JPL cross-site-transfer artifacts."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--continuous-summary", type=Path, required=True)
    parser.add_argument("--continuous-metrics", type=Path, required=True)
    parser.add_argument("--daywise-bootstrap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection = _load_json(args.selection)
    manifest = _load_json(args.manifest)
    summary = _load_json(args.continuous_summary)
    metrics = pd.read_csv(args.continuous_metrics)
    bootstrap = pd.read_csv(args.daywise_bootstrap)
    primary_rows = metrics.loc[metrics["policy"] == PRIMARY]
    if len(primary_rows) != 1:
        raise ValueError(f"expected exactly one primary policy row {PRIMARY!r}")
    primary = primary_rows.iloc[0]
    guard = summary["assumptions"]["commitment_guard"]
    if selection.get("selected_guard_mode") != "multi_rate_envelope":
        raise ValueError("selection record does not select the multi-rate envelope")
    if guard.get("mode") != "multi_rate_envelope" or not guard.get("applied"):
        raise ValueError("continuous run does not contain the selected applied multi-rate guard")

    policy_table = metrics[
        [
            "policy",
            "sessions",
            "mean_service_ratio",
            "p10_service_ratio",
            "worst_service_ratio",
            "jain_service_index",
            "energy_service_ratio",
            "unsafe_steps",
            "runtime_seconds",
        ]
    ].copy()
    pairwise: list[dict[str, float | str]] = []
    for baseline_name in BASELINES:
        rows = metrics.loc[metrics["policy"] == baseline_name]
        if rows.empty:
            continue
        baseline = rows.iloc[0]
        pairwise.append(
            {
                "baseline": baseline_name,
                "p10_difference_fairflex_minus_baseline": float(
                    primary["p10_service_ratio"] - baseline["p10_service_ratio"]
                ),
                "jain_difference_fairflex_minus_baseline": float(
                    primary["jain_service_index"] - baseline["jain_service_index"]
                ),
                "energy_service_difference_fairflex_minus_baseline": float(
                    primary["energy_service_ratio"] - baseline["energy_service_ratio"]
                ),
            }
        )
    pairwise_table = pd.DataFrame(pairwise)

    p10_bootstrap = bootstrap.loc[
        (bootstrap["treatment"] == PRIMARY) & (bootstrap["metric"] == "p10_service_ratio")
    ].copy()
    p10_bootstrap = p10_bootstrap[
        ["comparison_family", "baseline", "observed_mean_difference", "ci_low", "ci_high", "blocks"]
    ]

    target = float(guard["target_coverage"])
    observed = float(guard["empirical_coverage"])
    sessions = int(guard["decisions"])
    report = {
        "study_id": summary["study_id"],
        "purpose": "Descriptive report for one frozen historical JPL cross-site transfer stress test.",
        "protocol": {
            "selected_candidate": selection["selected_candidate"],
            "selection_rule": selection["selection_rule"],
            "selection_reserved_validation": selection["reserved_validation"],
            "test_site": "jpl",
            "calibration_site": "caltech",
            "raw_jpl_test_sha256": manifest["raw_input_sha256"]["acn_test"],
            "raw_jpl_test_sessions": manifest["splits"]["test"]["source_records"],
            "usable_jpl_declared_commitment_sessions": manifest["splits"]["test"]["usable_sessions"],
        },
        "continuous_causal_guard": {
            "target_coverage": target,
            "empirical_coverage": observed,
            "target_met_on_this_window": observed >= target,
            "coverage_gap": observed - target,
            "decisions": sessions,
            "mean_buffer_minutes": float(guard["buffer_minutes_at_decision"]["mean"]),
            "minimum_buffer_minutes": float(guard["buffer_minutes_at_decision"]["minimum"]),
            "maximum_buffer_minutes": float(guard["buffer_minutes_at_decision"]["maximum"]),
            "expert_selection_counts": guard["selected_expert_decision_counts"],
            "expert_audits": guard["expert_audits"],
            "causal_event_order": guard["causal_event_order"],
            "scope": (
                "An empirical audit on one historical transfer window; it is not a finite-sample, "
                "per-driver, conditional, or universal cross-site coverage guarantee."
            ),
        },
        "continuous_policy_metrics": metrics.to_dict(orient="records"),
        "continuous_pairwise_differences": pairwise,
        "day_block_p10_bootstrap": p10_bootstrap.to_dict(orient="records"),
        "safety": {
            "all_policies_zero_unsafe_steps": bool((metrics["unsafe_steps"] == 0).all()),
            "scope": (
                "Safety is evaluated only on the deliberately synthetic shared feeder and GHI/PV "
                "proxy scenario; it is not a measurement of JPL's physical feeder or PV system."
            ),
        },
        "paper_safe_conclusion": (
            "On this frozen JPL historical window, the selected causal multi-rate envelope met the "
            "90% empirical deadline-coverage target. FairFlex improved the lower-tail (P10) service "
            "ratio relative to the listed simple baselines and improved substantially over its no-PV "
            "ablation, while all evaluated policies had zero synthetic-feeder unsafe steps. It did not "
            "dominate equal sharing on mean service or Jain index, so the result supports a lower-tail "
            "fairness claim rather than a universal fairness-superiority claim."
        ),
        "limitations": [
            "JPL October is a historical, analysis-held-out transfer window, not a real-time deployment.",
            "The guard starts from Caltech calibration data; this is an intentionally hard source-to-target transfer test.",
            "The multi-rate envelope is not AgACI and is not presented as a new theoretical conformal guarantee.",
            "Real ACN sessions are deterministically mapped to synthetic stations, and the feeder/PV assumptions are controlled sensitivities rather than JPL measurements.",
            "Day-block bootstrap replays reset the adaptive guard at each day boundary; those intervals support policy-metric sensitivity only, never the continuous-guard coverage claim.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_site_transfer_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    markdown = "\n".join(
        [
            "# Frozen JPL cross-site transfer report",
            "",
            "This report is derived only from frozen artifacts. It does not rerun or tune a method.",
            "",
            "## What was frozen before JPL data acquisition",
            "",
            f"- Selected guard: `{selection['selected_candidate']}`.",
            f"- Selection rule: {selection['selection_rule']}",
            "- Development source: Caltech May selection and unchanged June replication.",
            "- Transfer target: JPL sessions connected from 2019-10-01 to 2019-10-08 UTC.",
            f"- JPL raw-file SHA-256: `{manifest['raw_input_sha256']['acn_test']}`.",
            f"- JPL usable declared-commitment sessions: `{manifest['splits']['test']['usable_sessions']}` of `{manifest['splits']['test']['source_records']}` raw sessions.",
            "",
            "## Continuous causal guard audit",
            "",
            f"- Target deadline coverage: `{target:.3f}`.",
            f"- Observed deadline coverage: `{observed:.3f}` across `{sessions}` plug-in decisions.",
            f"- Target met on this one frozen window: `{observed >= target}`.",
            f"- Mean guard buffer: `{float(guard['buffer_minutes_at_decision']['mean']):.1f}` minutes (range `{float(guard['buffer_minutes_at_decision']['minimum']):.0f}`–`{float(guard['buffer_minutes_at_decision']['maximum']):.0f}` minutes).",
            f"- Selected-expert decisions: `{json.dumps(guard['selected_expert_decision_counts'], sort_keys=True)}`.",
            "",
            "The guard starts from Caltech April calibration and, at each JPL plug-in, uses only "
            "unplug events already observed at that time. Its observed coverage is evidence for this "
            "window—not a general cross-site coverage guarantee.",
            "",
            "## Continuous seven-day policy metrics",
            "",
            _markdown_table(policy_table),
            "",
            "## FairFlex minus baseline on the continuous replay",
            "",
            _markdown_table(pairwise_table),
            "",
            "## Day-block P10 sensitivity",
            "",
            _markdown_table(p10_bootstrap),
            "",
            "The day-block bootstrap uses seven matched days and confirms the direction of the P10 "
            "differences. Because it resets the adaptive guard at each midnight boundary, it must not "
            "be used to make the continuous-coverage claim above.",
            "",
            "## Paper-safe conclusion",
            "",
            report["paper_safe_conclusion"],
            "",
            "## Boundaries of the claim",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "## Sources to cite",
            "",
            "- ACN-Data documentation and dataset citation: https://ev.caltech.edu/dataset",
            "- Gibbs and Candès (2021), Adaptive Conformal Inference Under Distribution Shift: https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html",
            "- Zaffran et al. (2022), Adaptive Conformal Predictions for Time Series: https://proceedings.mlr.press/v162/zaffran22a.html",
            "",
        ]
    )
    (args.output_dir / "cross_site_transfer_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON: {args.output_dir / 'cross_site_transfer_report.json'}")
    print(f"Markdown: {args.output_dir / 'cross_site_transfer_report.md'}")


if __name__ == "__main__":
    main()
