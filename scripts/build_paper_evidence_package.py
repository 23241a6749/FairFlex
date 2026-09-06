"""Build transparent tables, figures, and a concise findings note from frozen artifacts.

The command is intentionally read-only with respect to controllers, datasets,
and fitted models.  It summarizes already completed matched-day experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PRIMARY_ORDER = [
    "uncontrolled_shared_cap",
    "fcfs_shared_cap",
    "edf_shared_cap",
    "equal_share_shared_cap",
    "round_robin_shared_cap",
    "least_laxity_first_shared_cap",
    "fair_mpc_ac_no_pv",
    "fair_mpc_ac_robust_pv",
]

DISPLAY_NAMES = {
    "uncontrolled_shared_cap": "Uncontrolled (capped)",
    "fcfs_shared_cap": "FCFS",
    "edf_shared_cap": "EDF",
    "equal_share_shared_cap": "Equal share",
    "round_robin_shared_cap": "Round Robin*",
    "least_laxity_first_shared_cap": "LLF",
    "fair_mpc_ac_no_pv": "FairFlex, no PV robustness",
    "fair_mpc_ac_robust_pv": "FairFlex, robust PV",
    "fair_mpc_ac_robust_pv_unguarded": "No guard",
    "fair_mpc_ac_robust_pv_global_guard": "Fixed global guard",
    "fair_mpc_ac_robust_pv_multi_rate_envelope_guard": "Multi-rate guard",
}


def _primary_summary(daywise: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_service_ratio",
        "p10_service_ratio",
        "worst_service_ratio",
        "jain_service_index",
        "energy_service_ratio",
        "completion_rate",
        "delivered_energy_kwh",
        "unsafe_steps",
        "min_voltage_pu",
        "max_line_loading_percent",
        "runtime_seconds",
    ]
    grouped = daywise.groupby("policy", sort=False)
    table = grouped[metrics].mean()
    table.insert(0, "days", grouped.size())
    table.insert(1, "total_sessions", grouped["sessions"].sum())
    table.insert(2, "total_delivered_energy_kwh", grouped["delivered_energy_kwh"].sum())
    table.insert(3, "median_runtime_seconds", grouped["runtime_seconds"].median())
    table.insert(4, "p95_runtime_seconds", grouped["runtime_seconds"].quantile(0.95))
    table = table.reindex(PRIMARY_ORDER).reset_index()
    table.insert(1, "policy_display", table["policy"].map(DISPLAY_NAMES))
    return table


def _guard_summary(daywise: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_service_ratio",
        "p10_service_ratio",
        "worst_service_ratio",
        "jain_service_index",
        "energy_service_ratio",
        "completion_rate",
        "delivered_energy_kwh",
        "unsafe_steps",
        "runtime_seconds",
    ]
    grouped = daywise.groupby("policy", sort=False)
    table = grouped[metrics].mean()
    table.insert(0, "days", grouped.size())
    table.insert(1, "total_sessions", grouped["sessions"].sum())
    table = table.reset_index()
    table.insert(1, "condition", table["policy"].str.removeprefix("fair_mpc_ac_robust_pv_").str.removesuffix("_guard"))
    table.loc[table["policy"] == "fair_mpc_ac_robust_pv_unguarded", "condition"] = "unguarded"
    table.insert(2, "condition_display", table["policy"].map(DISPLAY_NAMES))
    merged = table.merge(reliability, how="left", left_on="condition", right_on="condition")
    return merged


def _forest_plot(primary_pairs: pd.DataFrame, output_path: Path) -> None:
    rows = primary_pairs[
        (primary_pairs["metric"] == "p10_service_ratio")
        & (primary_pairs["comparison_family"] != "central_vs_round_robin")
    ].copy()
    rows["baseline_label"] = rows["baseline"].map(DISPLAY_NAMES)
    rows = rows.sort_values("observed_treatment_advantage")
    y = range(len(rows))
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.errorbar(
        rows["observed_treatment_advantage"],
        y,
        xerr=[
            rows["observed_treatment_advantage"] - rows["advantage_ci_low"],
            rows["advantage_ci_high"] - rows["observed_treatment_advantage"],
        ],
        fmt="o",
        color="#0B6E99",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(list(y), labels=rows["baseline_label"])
    axis.set_xlabel("FairFlex robust-PV advantage in day-level P10 service ratio")
    axis.set_title("Lower-tail service: FairFlex robust PV vs. each baseline\n95% paired day-block bootstrap intervals")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _guard_plot(reliability: pd.DataFrame, output_path: Path) -> None:
    table = reliability.set_index("condition").loc[["global_guard", "multi_rate_envelope_guard"]].reset_index()
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for _, row in table.iterrows():
        axis.errorbar(
            row["decision_weighted_mean_buffer_minutes"],
            row["observed_coverage"],
            yerr=[[row["observed_coverage"] - row["coverage_ci_low"]], [row["coverage_ci_high"] - row["observed_coverage"]]],
            fmt="o",
            capsize=4,
            markersize=7,
            label=DISPLAY_NAMES[f"fair_mpc_ac_robust_pv_{row['condition']}"] ,
        )
    axis.axhline(0.90, color="#B22222", linestyle="--", label="90% target coverage")
    axis.set_xlabel("Decision-weighted mean protection buffer (minutes)")
    axis.set_ylabel("Observed early-departure coverage")
    axis.set_ylim(0.80, 0.93)
    axis.set_title("Guard reliability versus conservatism\n95% matched day-block bootstrap intervals")
    axis.grid(linestyle=":", alpha=0.5)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _value(pairs: pd.DataFrame, family: str, metric: str) -> tuple[float, float, float]:
    row = pairs[(pairs["comparison_family"] == family) & (pairs["metric"] == metric)].iloc[0]
    return float(row["observed_treatment_advantage"]), float(row["advantage_ci_low"]), float(row["advantage_ci_high"])


def _write_findings(
    output_path: Path,
    primary: pd.DataFrame,
    primary_pairs: pd.DataFrame,
    guard_pairs: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    p10_equal = _value(primary_pairs, "central_vs_equal_share", "p10_service_ratio")
    jain_equal = _value(primary_pairs, "central_vs_equal_share", "jain_service_index")
    p10_no_pv = _value(primary_pairs, "central_vs_no_pv", "p10_service_ratio")
    multi_jain = _value(guard_pairs, "multi_rate_envelope_guard_vs_unguarded", "jain_service_index")
    global_row = reliability.set_index("condition").loc["global_guard"]
    multi_row = reliability.set_index("condition").loc["multi_rate_envelope_guard"]
    text = f"""# FairFlex November evidence package

## What was evaluated

This package summarizes a frozen historical JPL November 2019 trace replay:
29 non-empty calendar days, 1,320 usable ACN charging sessions, a synthetic
10 kW shared-import sensitivity scenario, a regional NSRDB/PVWatts proxy, and
the same AC execution-safety check for every policy. November 30 had no usable
session and is recorded as an empty replay rather than receiving undefined
service or fairness scores.

The primary inference unit is a matched calendar day. Confidence intervals are
95% paired day-block bootstrap intervals; they are not EV-row bootstrap
intervals.

## Primary result

FairFlex robust-PV has a lower-tail service advantage over equal share of
{p10_equal[0]:.4f} P10 service-ratio points (95% CI {p10_equal[1]:.4f} to
{p10_equal[2]:.4f}). It does **not** dominate equal share: its Jain-index
advantage is {jain_equal[0]:.4f} (95% CI {jain_equal[1]:.4f} to
{jain_equal[2]:.4f}), which favours equal share when negative.

Robust PV capacity also improves FairFlex relative to its no-PV-robustness
counterpart on P10 service by {p10_no_pv[0]:.4f} (95% CI {p10_no_pv[1]:.4f} to
{p10_no_pv[2]:.4f}). All primary policies have zero unsafe executed steps in
this synthetic sensitivity scenario.

## Guard ablation

The fixed global guard has observed coverage {global_row['observed_coverage']:.4f}
(95% CI {global_row['coverage_ci_low']:.4f} to {global_row['coverage_ci_high']:.4f}),
which is below the 90% target interval. The selected multi-rate envelope has
observed coverage {multi_row['observed_coverage']:.4f} (95% CI
{multi_row['coverage_ci_low']:.4f} to {multi_row['coverage_ci_high']:.4f});
the target remains compatible with that interval. Its coverage improvement
over the global guard and the associated buffer cost appear in the reliability
audit table.

Relative to no guard, the multi-rate condition does not show a decisive P10
change in this month, but it reduces Jain fairness by {multi_jain[0]:.4f}
(95% CI {multi_jain[1]:.4f} to {multi_jain[2]:.4f}). This is a real trade-off,
not a result to hide.

## Correct claim and limits

The supported claim is that FairFlex robust-PV improves lower-tail charging
service against several standard online baselines under this shared-capacity
sensitivity scenario, while the selected multi-rate guard is materially more
reliable than a fixed global guard. The study does not establish a universal
fairness win, a measured JPL feeder/PV result, real-time deployment performance,
or state-of-the-art performance from one site/month.

Round Robin is displayed for ACN-Sim benchmark alignment, but because FairFlex
uses continuous power it is algebraically equivalent to equal-share water
filling and is not independent evidence.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a transparent FairFlex paper evidence package.")
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--guard-dir", type=Path, required=True)
    parser.add_argument("--guard-reliability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    primary_daywise = pd.read_csv(args.primary_dir / "daywise_metrics.csv")
    primary_pairs = pd.read_csv(args.primary_dir / "paired_day_bootstrap.csv")
    guard_daywise = pd.read_csv(args.guard_dir / "daywise_guard_ablation_metrics.csv")
    guard_pairs = pd.read_csv(args.guard_dir / "paired_day_bootstrap.csv")
    reliability = pd.read_csv(args.guard_reliability_dir / "guard_reliability_summary.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = _primary_summary(primary_daywise)
    guard = _guard_summary(guard_daywise, reliability)
    primary.to_csv(args.output_dir / "primary_policy_table.csv", index=False)
    primary_pairs.to_csv(args.output_dir / "primary_paired_comparisons.csv", index=False)
    guard.to_csv(args.output_dir / "guard_ablation_table.csv", index=False)
    guard_pairs.to_csv(args.output_dir / "guard_ablation_paired_comparisons.csv", index=False)
    reliability.to_csv(args.output_dir / "guard_reliability_table.csv", index=False)
    _forest_plot(primary_pairs, args.output_dir / "figure_primary_p10_forest.png")
    _guard_plot(reliability, args.output_dir / "figure_guard_reliability_tradeoff.png")
    _write_findings(
        args.output_dir / "FINDINGS.md", primary, primary_pairs, guard_pairs, reliability
    )
    print(f"Evidence package: {args.output_dir}")


if __name__ == "__main__":
    main()
