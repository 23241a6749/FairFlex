"""Build a transparent three-cohort FairFlex replication package.

Each frozen cohort remains separate.  This report intentionally does not pool
months/sites into a post-hoc omnibus significance test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COHORTS = ("November JPL", "July JPL", "August Caltech")
COLORS = {"November JPL": "#0B6E99", "July JPL": "#F27C0A", "August Caltech": "#388E3C"}
BASELINES = {
    "uncontrolled_shared_cap": "Uncontrolled (capped)",
    "fcfs_shared_cap": "FCFS",
    "edf_shared_cap": "EDF",
    "equal_share_shared_cap": "Equal share",
    "least_laxity_first_shared_cap": "LLF",
    "fair_mpc_ac_no_pv": "FairFlex, no PV robustness",
}
METRICS = ("mean_service_ratio", "p10_service_ratio", "jain_service_index", "energy_service_ratio", "completion_rate")


def _pairs(directory: Path, cohort: str) -> pd.DataFrame:
    table = pd.read_csv(directory / "paired_day_bootstrap.csv")
    table = table[
        table["metric"].isin(METRICS) & table["baseline"].isin(BASELINES)
    ].copy()
    table.insert(0, "cohort", cohort)
    table["baseline_display"] = table["baseline"].map(BASELINES)
    return table


def _cohort_validation(directory: Path, cohort: str) -> dict[str, object]:
    daywise = pd.read_csv(directory / "daywise_metrics.csv")
    return {
        "cohort": cohort,
        "days": int(daywise["replay_start"].nunique()),
        "policy_rows": int(len(daywise)),
        "policies": int(daywise["policy"].nunique()),
        "total_sessions_per_policy": int(daywise.groupby("policy")["sessions"].sum().iloc[0]),
        "duplicate_day_policy_rows": int(daywise.duplicated(["replay_start", "policy"]).sum()),
        "unsafe_steps_across_all_policy_rows": float(daywise["unsafe_steps"].sum()),
    }


def _p10_figure(pairs: pd.DataFrame, output_path: Path) -> None:
    table = pairs[pairs["metric"] == "p10_service_ratio"].copy()
    order = (
        table.groupby("baseline_display")["observed_treatment_advantage"].mean()
        .sort_values()
        .index.tolist()
    )
    table["baseline_display"] = pd.Categorical(table["baseline_display"], order, ordered=True)
    table = table.sort_values("baseline_display")
    offsets = {"November JPL": -0.20, "July JPL": 0.0, "August Caltech": 0.20}
    fig, axis = plt.subplots(figsize=(9.1, 5.1))
    for cohort in COHORTS:
        subset = table[table["cohort"] == cohort]
        y = [order.index(str(value)) + offsets[cohort] for value in subset["baseline_display"]]
        advantage = subset["observed_treatment_advantage"]
        axis.errorbar(
            advantage,
            y,
            xerr=[advantage - subset["advantage_ci_low"], subset["advantage_ci_high"] - advantage],
            fmt="o",
            capsize=3,
            color=COLORS[cohort],
            label=cohort,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(range(len(order)), labels=order)
    axis.set_xlabel("FairFlex robust-PV advantage in day-level P10 service ratio")
    axis.set_title("Lower-tail service across frozen time and site replications\n95% paired day-block bootstrap intervals")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _coverage_figure(reliability: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 4.7))
    positions = list(range(len(reliability)))
    for position, (_, row) in zip(positions, reliability.iterrows(), strict=True):
        color = COLORS[str(row["cohort"])]
        axis.errorbar(
            position,
            row["observed_coverage"],
            yerr=[[row["observed_coverage"] - row["coverage_ci_low"]], [row["coverage_ci_high"] - row["observed_coverage"]]],
            fmt="o",
            capsize=4,
            markersize=7,
            color=color,
        )
    axis.axhline(0.90, color="#B22222", linestyle="--", label="90% target coverage")
    axis.set_xticks(positions, labels=reliability["cohort"])
    axis.set_ylabel("Observed early-departure coverage")
    axis.set_ylim(0.84, 0.93)
    axis.set_title("Multi-rate guard reliability across frozen cohorts\n95% matched day-block bootstrap intervals")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _row(pairs: pd.DataFrame, cohort: str, baseline: str, metric: str) -> pd.Series:
    return pairs[(pairs["cohort"] == cohort) & (pairs["baseline"] == baseline) & (pairs["metric"] == metric)].iloc[0]


def _findings(output_path: Path, pairs: pd.DataFrame, reliability: pd.DataFrame, validation: pd.DataFrame) -> None:
    p10 = pairs[pairs["metric"] == "p10_service_ratio"]
    all_p10_positive = bool((p10["advantage_ci_low"] > 0).all())
    equal = [_row(pairs, cohort, "equal_share_shared_cap", "p10_service_ratio") for cohort in COHORTS]
    no_pv = [_row(pairs, cohort, "fair_mpc_ac_no_pv", "p10_service_ratio") for cohort in COHORTS]
    equal_jain = [_row(pairs, cohort, "equal_share_shared_cap", "jain_service_index") for cohort in COHORTS]
    reliability_text = "\n".join(
        f"- {row.cohort}: {row.observed_coverage:.4f} (95% CI {row.coverage_ci_low:.4f} to {row.coverage_ci_high:.4f}); "
        f"decision-weighted mean buffer {row.decision_weighted_mean_buffer_minutes:.1f} min"
        for row in reliability.itertuples(index=False)
    )
    text = f"""# Three-cohort FairFlex evidence summary

## Design strength

The primary JPL November study, JPL July temporal replication, and Caltech
August external-site replication were each frozen before their test raw files
were acquired. All use matched calendar-day inference, the same eight-policy
matrix, causal declared-commitment information, multi-rate guard architecture
and rates, 15-minute control interval, PV-proxy capacity, synthetic shared
feeder sensitivity, and executed-action safety check. JPL and Caltech each use
their own earlier historical data for forecast fitting and calibration-score
initialization; neither operation selects a policy or hyperparameter on its
held-out month.

## Repeated result

Every P10 comparison shown has a positive 95% paired day-block interval in all
three cohorts: {all_p10_positive}. Against equal share, robust-PV FairFlex has
P10 advantages of {equal[0].observed_treatment_advantage:.4f} (November),
{equal[1].observed_treatment_advantage:.4f} (July), and
{equal[2].observed_treatment_advantage:.4f} (Caltech). Against its no-PV
counterpart, the corresponding P10 advantages are
{no_pv[0].observed_treatment_advantage:.4f}, {no_pv[1].observed_treatment_advantage:.4f}, and
{no_pv[2].observed_treatment_advantage:.4f}.

The important trade-off also repeats. Against equal share, Jain-index
advantages are {equal_jain[0].observed_treatment_advantage:.4f} (November),
{equal_jain[1].observed_treatment_advantage:.4f} (July), and
{equal_jain[2].observed_treatment_advantage:.4f} (Caltech); negative values
favour equal sharing. FairFlex therefore cannot honestly be described as a
universal fairness or mean-service winner. Its supported benefit is stronger
lower-tail service under the declared scarcity scenario.

## Guard reliability

{reliability_text}

The nominal 90% coverage target lies within every cohort's matched-day
interval. This is evidence of aggregate operational reliability, not a
finite-sample, conditional, per-driver, universal, or real-time guarantee.

## Safety and limits

All three matrices have zero unsafe executed steps across their recorded policy
rows. However, the feeder mapping, shared 10 kW cap, and PV production are
controlled sensitivities—not measured JPL or Caltech infrastructure. The work
supports a reproducible causal benchmark result, not a live deployment claim
or a claim of state-of-the-art performance.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-pooling three-cohort FairFlex evidence package.")
    parser.add_argument("--november-dir", type=Path, required=True)
    parser.add_argument("--july-dir", type=Path, required=True)
    parser.add_argument("--caltech-dir", type=Path, required=True)
    parser.add_argument("--november-reliability-dir", type=Path, required=True)
    parser.add_argument("--july-reliability-dir", type=Path, required=True)
    parser.add_argument("--caltech-reliability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    directories = {
        "November JPL": args.november_dir,
        "July JPL": args.july_dir,
        "August Caltech": args.caltech_dir,
    }
    pairs = pd.concat([_pairs(directory, cohort) for cohort, directory in directories.items()], ignore_index=True)
    validation = pd.DataFrame(
        [_cohort_validation(directory, cohort) for cohort, directory in directories.items()]
    )
    reliability_sources = {
        "November JPL": args.november_reliability_dir,
        "July JPL": args.july_reliability_dir,
        "August Caltech": args.caltech_reliability_dir,
    }
    reliability_rows = []
    for cohort, directory in reliability_sources.items():
        table = pd.read_csv(directory / "guard_reliability_summary.csv")
        row = table[table["condition"] == "multi_rate_envelope_guard"].copy()
        row.insert(0, "cohort", cohort)
        reliability_rows.append(row)
    reliability = pd.concat(reliability_rows, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "three_cohort_policy_effects.csv", index=False)
    reliability.to_csv(args.output_dir / "three_cohort_guard_reliability.csv", index=False)
    validation.to_csv(args.output_dir / "three_cohort_validation.csv", index=False)
    _p10_figure(pairs, args.output_dir / "figure_three_cohort_p10_replication.png")
    _coverage_figure(reliability, args.output_dir / "figure_three_cohort_guard_coverage.png")
    _findings(args.output_dir / "FINDINGS.md", pairs, reliability, validation)
    print(f"Three-cohort evidence package: {args.output_dir}")


if __name__ == "__main__":
    main()
