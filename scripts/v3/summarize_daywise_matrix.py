"""Screen and summarize completed FairFlex-UC day-block artifacts honestly.

Daily P10 needs a minimally useful number of sessions.  This command never
alters the raw daywise table: it writes a screened companion summary and skips
metrics that lack the predeclared minimum block support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairflex.statistics import paired_metric_bootstrap


SELECTED_POLICY = "fairflex_uc_hybrid_lower_tail"
COMPARISONS = (
    ("hybrid_vs_v1_multirate", SELECTED_POLICY, "v1_fair_mpc_multirate_guard"),
    ("hybrid_vs_v1_no_guard", SELECTED_POLICY, "v1_fair_mpc_no_guard"),
    ("hybrid_vs_equal_share", SELECTED_POLICY, "equal_share_shared_cap"),
)
METRICS = (
    ("p10_service_ratio", "higher"),
    ("early_unplug_p10_service_ratio", "higher"),
    ("all_session_expected_shortfall_10pct", "lower"),
    ("mean_service_ratio", "higher"),
    ("jain_service_index", "higher"),
    ("delivered_energy_kwh", "higher"),
    ("unsafe_steps", "lower"),
)


def _oriented(result, direction: str) -> dict[str, float | str]:
    sign = 1.0 if direction == "higher" else -1.0
    low = sign * result.ci_low
    high = sign * result.ci_high
    return {
        "preferred_direction": direction,
        "observed_treatment_advantage": sign * result.observed_mean_difference,
        "advantage_ci_low": min(low, high),
        "advantage_ci_high": max(low, high),
        "bootstrap_probability_treatment_better": (
            result.bootstrap_probability_difference_positive
            if direction == "higher"
            else result.bootstrap_probability_difference_negative
        ),
    }


def _eligible_subset(
    daywise: pd.DataFrame,
    *,
    metric: str,
    minimum_sessions_for_p10: int,
    minimum_early_sessions_for_p10: int,
) -> pd.DataFrame:
    if metric == "p10_service_ratio":
        allowed = daywise.loc[
            daywise["sessions"] >= minimum_sessions_for_p10, "replay_start"
        ].unique()
        return daywise.loc[daywise["replay_start"].isin(allowed)].copy()
    if metric == "early_unplug_p10_service_ratio":
        allowed = daywise.loc[
            daywise["early_unplug_sessions"] >= minimum_early_sessions_for_p10,
            "replay_start",
        ].unique()
        return daywise.loc[daywise["replay_start"].isin(allowed)].copy()
    return daywise.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-sessions-for-p10", type=int, default=10)
    parser.add_argument("--minimum-early-sessions-for-p10", type=int, default=10)
    parser.add_argument("--minimum-blocks", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if min(
        args.minimum_sessions_for_p10,
        args.minimum_early_sessions_for_p10,
        args.minimum_blocks,
    ) <= 0:
        parser.error("all support minima must be positive")
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    source_path = args.input_dir / "daywise_metrics.csv"
    daywise = pd.read_csv(source_path)
    required = {"replay_start", "policy", "sessions", "early_unplug_sessions"}
    missing = sorted(required - set(daywise.columns))
    if missing:
        parser.error(f"input daywise metrics are missing columns: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daywise.to_csv(args.output_dir / "raw_daywise_metrics_preserved.csv", index=False)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for family, treatment, baseline in COMPARISONS:
        for metric, direction in METRICS:
            eligible = _eligible_subset(
                daywise,
                metric=metric,
                minimum_sessions_for_p10=args.minimum_sessions_for_p10,
                minimum_early_sessions_for_p10=args.minimum_early_sessions_for_p10,
            )
            blocks = int(eligible["replay_start"].nunique())
            if blocks < args.minimum_blocks:
                skipped.append({
                    "comparison_family": family,
                    "metric": metric,
                    "eligible_blocks": blocks,
                    "reason": f"fewer than required {args.minimum_blocks} eligible calendar days",
                })
                continue
            try:
                result = paired_metric_bootstrap(
                    eligible,
                    metric=metric,
                    treatment=treatment,
                    baseline=baseline,
                    block_columns=("replay_start",),
                    resamples=args.resamples,
                    seed=args.seed,
                )
            except ValueError as error:
                skipped.append({
                    "comparison_family": family,
                    "metric": metric,
                    "eligible_blocks": blocks,
                    "reason": str(error),
                })
                continue
            rows.append({
                "comparison_family": family,
                "eligible_blocks": blocks,
                **result.to_dict(),
                **_oriented(result, direction),
            })

    comparisons_path = args.output_dir / "screened_paired_day_bootstrap.csv"
    pd.DataFrame(rows).to_csv(comparisons_path, index=False)
    skipped_path = args.output_dir / "screened_metrics_not_reported.csv"
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    manifest_path = args.output_dir / "screening_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_daywise_metrics": str(source_path),
        "raw_daywise_metrics": "preserved unchanged as raw_daywise_metrics_preserved.csv",
        "minimum_sessions_for_daily_p10": args.minimum_sessions_for_p10,
        "minimum_early_unplug_sessions_for_daily_p10": args.minimum_early_sessions_for_p10,
        "minimum_eligible_calendar_days": args.minimum_blocks,
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "interpretation": (
            "Screened intervals are small-sample day-block sensitivity diagnostics. "
            "They do not turn the seven-day replay into a definitive statistical claim."
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Screened comparisons: {comparisons_path}")
    print(f"Metrics not reported due to insufficient support: {skipped_path}")
    print(f"Screening manifest: {manifest_path}")


if __name__ == "__main__":
    main()
