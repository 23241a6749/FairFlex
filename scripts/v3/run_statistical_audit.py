"""Produce paired calendar-day inference for frozen FairFlex-UC policy results.

The policy outputs are not predictions, so this command does not calculate an
"accuracy" score. It estimates a matched policy effect. Each calendar day is
one paired block because all policies replay that day's same EV arrivals and
solar trace. Individual EV rows are deliberately never used as independent
t-test or ANOVA observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairflex.statistics import paired_block_bootstrap, paired_sign_flip_test


METRICS: dict[str, dict[str, object]] = {
    "p10_service_ratio": {
        "direction": "higher",
        "support_column": "sessions",
        "minimum_support": 10,
        "role": "primary",
    },
    "early_unplug_p10_service_ratio": {
        "direction": "higher",
        "support_column": "early_unplug_sessions",
        "minimum_support": 10,
        "role": "secondary",
    },
    "all_session_expected_shortfall_10pct": {
        "direction": "lower",
        "support_column": None,
        "minimum_support": None,
        "role": "secondary",
    },
    "mean_service_ratio": {
        "direction": "higher",
        "support_column": None,
        "minimum_support": None,
        "role": "exploratory",
    },
    "jain_service_index": {
        "direction": "higher",
        "support_column": None,
        "minimum_support": None,
        "role": "exploratory",
    },
    "delivered_energy_kwh": {
        "direction": "higher",
        "support_column": None,
        "minimum_support": None,
        "role": "exploratory",
    },
}


def _metric_subset(daywise: pd.DataFrame, metric: str) -> pd.DataFrame:
    spec = METRICS[metric]
    support_column = spec["support_column"]
    if support_column is None:
        return daywise.copy()
    minimum = int(spec["minimum_support"])
    allowed = daywise.loc[daywise[str(support_column)] >= minimum, "replay_start"].unique()
    return daywise.loc[daywise["replay_start"].isin(allowed)].copy()


def _matched_values(
    table: pd.DataFrame,
    *,
    metric: str,
    treatment: str,
    baseline: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = table.loc[
        table["policy"].isin((treatment, baseline)),
        ["replay_start", "policy", metric],
    ].copy()
    if selected.duplicated(["replay_start", "policy"]).any():
        raise ValueError("each policy must occur once per calendar day")
    wide = selected.pivot(index="replay_start", columns="policy", values=metric)
    missing = [policy for policy in (treatment, baseline) if policy not in wide.columns]
    if missing:
        raise ValueError(f"requested policy is missing: {missing}")
    paired = wide[[treatment, baseline]].dropna()
    if len(paired) != len(wide):
        raise ValueError("each requested policy must occur in every eligible calendar day")
    return paired[treatment].to_numpy(dtype=float), paired[baseline].to_numpy(dtype=float)


def _holm_adjust(rows: list[dict[str, object]]) -> None:
    """Add Holm-adjusted values when multiple metrics are explicitly requested."""
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["p_value_one_sided"]))
    running = 0.0
    count = len(ordered)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row["p_value_one_sided"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_one_sided_p_value"] = running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--treatment", default="fairflex_uc_hybrid_lower_tail")
    parser.add_argument("--baseline", default="v1_fair_mpc_multirate_guard")
    parser.add_argument("--metrics", nargs="+", default=["p10_service_ratio"])
    parser.add_argument("--minimum-calendar-days", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    unknown = sorted(set(args.metrics) - set(METRICS))
    if unknown:
        parser.error(f"unknown metrics: {unknown}; available: {sorted(METRICS)}")
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    if args.minimum_calendar_days < 2:
        parser.error("--minimum-calendar-days must be at least two")

    source = args.input_dir / "daywise_metrics.csv"
    daywise = pd.read_csv(source)
    required = {"replay_start", "policy", "sessions", "early_unplug_sessions", *args.metrics}
    missing = sorted(required - set(daywise.columns))
    if missing:
        parser.error(f"input is missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    withheld: list[dict[str, object]] = []
    for metric in args.metrics:
        eligible = _metric_subset(daywise, metric)
        treatment_values, baseline_values = _matched_values(
            eligible,
            metric=metric,
            treatment=args.treatment,
            baseline=args.baseline,
        )
        if treatment_values.size < args.minimum_calendar_days:
            withheld.append({
                "metric": metric,
                "eligible_calendar_days": int(treatment_values.size),
                "reason": (
                    f"fewer than required {args.minimum_calendar_days} complete matched calendar days"
                ),
            })
            continue
        spec = METRICS[metric]
        direction = str(spec["direction"])
        bootstrap = paired_block_bootstrap(
            treatment_values,
            baseline_values,
            metric=metric,
            treatment=args.treatment,
            baseline=args.baseline,
            resamples=args.resamples,
            seed=args.seed,
        )
        try:
            sign_flip = paired_sign_flip_test(
                treatment_values,
                baseline_values,
                metric=metric,
                treatment=args.treatment,
                baseline=args.baseline,
                preferred_direction=direction,
                seed=args.seed,
            )
        except ValueError as error:
            withheld.append({
                "metric": metric,
                "eligible_calendar_days": int(treatment_values.size),
                "reason": str(error),
            })
            continue
        orientation = 1.0 if direction == "higher" else -1.0
        raw_low = orientation * bootstrap.ci_low
        raw_high = orientation * bootstrap.ci_high
        rows.append({
            "metric": metric,
            "role": spec["role"],
            "preferred_direction": direction,
            "eligible_calendar_days": int(treatment_values.size),
            "observed_treatment_advantage": sign_flip.observed_oriented_mean_advantage,
            "advantage_bootstrap_ci_low": min(raw_low, raw_high),
            "advantage_bootstrap_ci_high": max(raw_low, raw_high),
            "bootstrap_probability_treatment_better": (
                bootstrap.bootstrap_probability_difference_positive
                if direction == "higher"
                else bootstrap.bootstrap_probability_difference_negative
            ),
            "p_value_one_sided": sign_flip.p_value_one_sided,
            "p_value_two_sided": sign_flip.p_value_two_sided,
            "sign_flip_test_method": sign_flip.test_method,
            "sign_flip_permutations": sign_flip.permutations,
            "bootstrap_resamples": bootstrap.resamples,
        })
    primary_rows = [row for row in rows if row["role"] == "primary"]
    non_primary_rows = [row for row in rows if row["role"] != "primary"]
    for row in primary_rows:
        # The single predeclared primary endpoint is not part of an exploratory
        # family and must retain its raw p-value.
        row["holm_adjusted_one_sided_p_value"] = row["p_value_one_sided"]
    if len(non_primary_rows) > 1:
        _holm_adjust(non_primary_rows)
    elif non_primary_rows:
        non_primary_rows[0]["holm_adjusted_one_sided_p_value"] = non_primary_rows[0]["p_value_one_sided"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "paired_calendar_day_statistical_audit.csv"
    withheld_path = args.output_dir / "withheld_statistical_metrics.csv"
    manifest_path = args.output_dir / "statistical_audit_manifest.json"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    pd.DataFrame(withheld).to_csv(withheld_path, index=False)
    manifest_path.write_text(json.dumps({
        "source_daywise_metrics": str(source),
        "treatment": args.treatment,
        "baseline": args.baseline,
        "metrics": args.metrics,
        "unit_of_inference": "matched calendar day",
        "primary_metric": "p10_service_ratio",
        "minimum_calendar_days": args.minimum_calendar_days,
        "exact_test": {
            "name": "one-sided paired sign-flip randomisation test (exact through 16 blocks; seeded Monte Carlo thereafter)",
            "null": "within every matched day, policy labels are exchangeable under no treatment effect",
            "alternative": "treatment has positive advantage after metric orientation",
            "not_used": "individual EV-session t-test, unpaired t-test, or ordinary ANOVA",
        },
        "bootstrap": {
            "name": "paired calendar-day percentile bootstrap",
            "resamples": args.resamples,
            "seed": args.seed,
            "confidence": 0.95,
        },
        "multiple_testing": (
            "The single predeclared primary endpoint (P10) retains its raw p-value. When more than one "
            "secondary/exploratory metric is requested, their family receives a Holm adjustment."
        ),
        "interpretation": (
            "A p-value or confidence interval does not convert a historical simulation into conclusive deployment "
            "evidence. Report the registered temporal cohort, eligible day count, continuous-versus-reset guard mode, "
            "and all safety outcomes alongside this day-block uncertainty summary."
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Paired statistical audit: {result_path}")
    print(f"Withheld metrics: {withheld_path}")
    print(f"Statistical manifest: {manifest_path}")


if __name__ == "__main__":
    main()
