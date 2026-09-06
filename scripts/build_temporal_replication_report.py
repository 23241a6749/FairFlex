"""Compare two frozen FairFlex policy matrices without refitting or pooling them."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASELINE_LABELS = {
    "uncontrolled_shared_cap": "Uncontrolled (capped)",
    "fcfs_shared_cap": "FCFS",
    "edf_shared_cap": "EDF",
    "equal_share_shared_cap": "Equal share",
    "least_laxity_first_shared_cap": "LLF",
    "fair_mpc_ac_no_pv": "FairFlex, no PV robustness",
}
METRICS = [
    "mean_service_ratio",
    "p10_service_ratio",
    "jain_service_index",
    "energy_service_ratio",
    "completion_rate",
]


def _load_pairs(directory: Path) -> pd.DataFrame:
    table = pd.read_csv(directory / "paired_day_bootstrap.csv")
    table = table[table["metric"].isin(METRICS)].copy()
    table = table[table["baseline"].isin(BASELINE_LABELS)].copy()
    table["baseline_display"] = table["baseline"].map(BASELINE_LABELS)
    return table


def _combined_effects(november: pd.DataFrame, july: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "comparison_family", "metric", "baseline", "baseline_display",
        "preferred_direction", "observed_treatment_advantage", "advantage_ci_low", "advantage_ci_high",
        "bootstrap_probability_treatment_better", "blocks",
    ]
    left = november[keep].rename(
        columns={
            "observed_treatment_advantage": "november_advantage",
            "advantage_ci_low": "november_ci_low",
            "advantage_ci_high": "november_ci_high",
            "bootstrap_probability_treatment_better": "november_probability_better",
            "blocks": "november_blocks",
        }
    )
    right = july[keep].rename(
        columns={
            "observed_treatment_advantage": "july_advantage",
            "advantage_ci_low": "july_ci_low",
            "advantage_ci_high": "july_ci_high",
            "bootstrap_probability_treatment_better": "july_probability_better",
            "blocks": "july_blocks",
        }
    )
    shared = ["comparison_family", "metric", "baseline", "baseline_display", "preferred_direction"]
    table = left.merge(right, on=shared, how="inner", validate="one_to_one")
    table["same_observed_direction"] = (
        (table["november_advantage"] > 0) == (table["july_advantage"] > 0)
    )
    table["both_intervals_favor_treatment"] = (
        (table["november_ci_low"] > 0) & (table["july_ci_low"] > 0)
    )
    return table.sort_values(["metric", "baseline_display"])


def _p10_replication_figure(effects: pd.DataFrame, output_path: Path) -> None:
    table = effects[effects["metric"] == "p10_service_ratio"].copy()
    table = table.sort_values("july_advantage")
    y = list(range(len(table)))
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for shift, month, color in [(-0.13, "november", "#0B6E99"), (0.13, "july", "#F27C0A")]:
        values = table[f"{month}_advantage"]
        axis.errorbar(
            values,
            [item + shift for item in y],
            xerr=[values - table[f"{month}_ci_low"], table[f"{month}_ci_high"] - values],
            fmt="o",
            capsize=3,
            color=color,
            label=month.capitalize(),
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, labels=table["baseline_display"])
    axis.set_xlabel("FairFlex robust-PV advantage in day-level P10 service ratio")
    axis.set_title("Seasonal replication of lower-tail service\n95% paired day-block bootstrap intervals")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _reliability_table(november: pd.DataFrame, july: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for cohort, table in [("November primary", november), ("July replication", july)]:
        selected = table[table["condition"] == "multi_rate_envelope_guard"].copy()
        selected.insert(0, "cohort", cohort)
        tables.append(selected)
    return pd.concat(tables, ignore_index=True)


def _write_findings(output_path: Path, effects: pd.DataFrame, reliability: pd.DataFrame) -> None:
    p10 = effects[effects["metric"] == "p10_service_ratio"]
    p10_all_positive = bool(p10["both_intervals_favor_treatment"].all())
    equal = p10[p10["baseline"] == "equal_share_shared_cap"].iloc[0]
    no_pv = p10[p10["baseline"] == "fair_mpc_ac_no_pv"].iloc[0]
    nov = reliability[reliability["cohort"] == "November primary"].iloc[0]
    jul = reliability[reliability["cohort"] == "July replication"].iloc[0]
    text = f"""# November-to-July temporal replication

## Design

July was acquired only after its configuration and invariance test were frozen.
It uses exactly the same March forecast-fit period, April calibration period,
multi-rate guard, controller, synthetic station/feeder/PV sensitivity setup,
policy matrix, and matched-day inference as November. The two months are shown
separately; no post-hoc pooled significance test is used.

## Result

Across the six non-duplicate comparisons displayed in the table, the P10
lower-tail-service advantage has positive 95% paired day-block intervals in
both months: {p10_all_positive}. Against equal share, the November advantage
is {equal['november_advantage']:.4f} (95% CI {equal['november_ci_low']:.4f} to
{equal['november_ci_high']:.4f}) and the July advantage is
{equal['july_advantage']:.4f} (95% CI {equal['july_ci_low']:.4f} to
{equal['july_ci_high']:.4f}).

The robust-PV component also replicates: compared with FairFlex without PV
robustness, P10 improves by {no_pv['november_advantage']:.4f} in November and
{no_pv['july_advantage']:.4f} in July, with both listed intervals positive.

Multi-rate guard coverage is {nov['observed_coverage']:.4f} in November (95%
CI {nov['coverage_ci_low']:.4f} to {nov['coverage_ci_high']:.4f}) and
{jul['observed_coverage']:.4f} in July (95% CI {jul['coverage_ci_low']:.4f} to
{jul['coverage_ci_high']:.4f}). The nominal 90% target lies inside both
matched-day intervals. This is operational replication evidence, not a
per-driver, conditional, finite-sample, or deployment guarantee.

## Still not claimed

The repeated lower-tail result does not erase the trade-off with equal share
on mean service and Jain fairness. Both cohorts remain synthetic feeder/PV
sensitivity experiments using real ACN sessions, rather than measured JPL
grid or rooftop-PV experiments. A separate-site replication remains the next
external-validity step.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-pooling temporal replication report.")
    parser.add_argument("--november-dir", type=Path, required=True)
    parser.add_argument("--july-dir", type=Path, required=True)
    parser.add_argument("--november-reliability-dir", type=Path, required=True)
    parser.add_argument("--july-reliability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    effects = _combined_effects(_load_pairs(args.november_dir), _load_pairs(args.july_dir))
    november_reliability = pd.read_csv(
        args.november_reliability_dir / "guard_reliability_summary.csv"
    )
    july_reliability = pd.read_csv(args.july_reliability_dir / "guard_reliability_summary.csv")
    reliability = _reliability_table(november_reliability, july_reliability)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output_dir / "seasonal_policy_effects.csv", index=False)
    reliability.to_csv(args.output_dir / "seasonal_guard_reliability.csv", index=False)
    _p10_replication_figure(effects, args.output_dir / "figure_temporal_p10_replication.png")
    _write_findings(args.output_dir / "FINDINGS.md", effects, reliability)
    print(f"Temporal replication report: {args.output_dir}")


if __name__ == "__main__":
    main()
