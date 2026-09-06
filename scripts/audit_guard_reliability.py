"""Summarize completed guard-ablation coverage using matched-day bootstrap.

This post-processing command never fits a guard or reruns an EV controller.
It reads the immutable per-day audit summaries written by the ablation runner,
then treats whole calendar days as the resampling blocks for coverage and
buffer uncertainty summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _coverage_interval(
    covered: np.ndarray, decisions: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Return observed weighted coverage and a day-block percentile interval."""
    if len(covered) < 2 or np.any(decisions <= 0):
        raise ValueError("at least two non-empty matched days are required")
    observed = float(covered.sum() / decisions.sum())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(covered), size=(resamples, len(covered)))
    draws = covered[indices].sum(axis=1) / decisions[indices].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _weighted_mean_interval(
    values: np.ndarray, weights: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Return a decision-weighted mean and day-block percentile interval."""
    if len(values) < 2 or np.any(weights <= 0):
        raise ValueError("at least two non-empty matched days are required")
    observed = float(np.average(values, weights=weights))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    draws = (values[indices] * weights[indices]).sum(axis=1) / weights[indices].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _difference_interval(
    treatment: np.ndarray, baseline: np.ndarray, weights: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float, float, float]:
    """Return matched decision-weighted difference and its day-block interval."""
    if not (len(treatment) == len(baseline) == len(weights)) or len(treatment) < 2:
        raise ValueError("treatment, baseline, and weights require the same two or more days")
    observed = float(np.average(treatment - baseline, weights=weights))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(treatment), size=(resamples, len(treatment)))
    draws = (
        ((treatment[indices] - baseline[indices]) * weights[indices]).sum(axis=1)
        / weights[indices].sum(axis=1)
    )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(lower), float(upper), float(np.mean(draws > 0))


def _buffer_summary(guard: dict[str, object]) -> dict[str, float]:
    decision_summary = guard.get("buffer_minutes_at_decision")
    if isinstance(decision_summary, dict):
        return {key: float(value) for key, value in decision_summary.items()}
    fixed = guard.get("buffer_minutes")
    if fixed is None:
        raise ValueError("guard audit does not contain a buffer summary")
    value = float(fixed)
    return {"mean": value, "median": value, "p90": value, "minimum": value, "maximum": value}


def _load_daywise(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("test_pilot_summary.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        guard = document.get("assumptions", {}).get("commitment_guard")
        if not isinstance(guard, dict) or not guard.get("applied"):
            continue
        replay = document["replay_window"]
        buffers = _buffer_summary(guard)
        decisions = int(guard.get("decisions", document["policy_metrics"][0]["sessions"]))
        rows.append(
            {
                "replay_start": replay["start"],
                "condition": f"{guard['mode']}_guard",
                "decisions": decisions,
                "empirical_coverage": float(guard["empirical_coverage"]),
                "covered_decisions": float(guard["empirical_coverage"]) * decisions,
                "mean_buffer_minutes": buffers["mean"],
                "median_buffer_minutes": buffers["median"],
                "p90_buffer_minutes": buffers["p90"],
                "summary_path": str(path),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"no guarded test summaries found under {root}")
    if table.duplicated(["replay_start", "condition"]).any():
        raise ValueError("duplicate replay-day/condition guard audit")
    return table.sort_values(["replay_start", "condition"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixed guard conditions with matched-day bootstrap.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["global_guard", "multi_rate_envelope_guard"],
        help="one condition for a coverage audit, or exactly two for a matched comparison",
    )
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.target_coverage < 1:
        parser.error("--target-coverage must lie strictly between zero and one")
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    if not 1 <= len(args.conditions) <= 2 or len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions must name one or two distinct guard conditions")

    daywise = _load_daywise(args.artifact_root)
    available = set(daywise["condition"])
    missing = set(args.conditions) - available
    if missing:
        parser.error(f"guard conditions not found: {sorted(missing)}")
    required = daywise[daywise["condition"].isin(args.conditions)]
    per_day_counts = required.groupby("replay_start")["condition"].nunique()
    if not (per_day_counts == len(args.conditions)).all():
        parser.error("every completed guard-audit day must contain both requested conditions")

    summaries: list[dict[str, object]] = []
    for condition, values in required.groupby("condition", sort=True):
        values = values.sort_values("replay_start")
        coverage, low, high = _coverage_interval(
            values["covered_decisions"].to_numpy(),
            values["decisions"].to_numpy(),
            resamples=args.resamples,
            seed=args.seed,
        )
        mean_buffer, buffer_low, buffer_high = _weighted_mean_interval(
            values["mean_buffer_minutes"].to_numpy(),
            values["decisions"].to_numpy(),
            resamples=args.resamples,
            seed=args.seed,
        )
        summaries.append(
            {
                "condition": condition,
                "days": len(values),
                "decisions": int(values["decisions"].sum()),
                "observed_coverage": coverage,
                "coverage_ci_low": low,
                "coverage_ci_high": high,
                "target_coverage": args.target_coverage,
                "target_inside_coverage_ci": low <= args.target_coverage <= high,
                "decision_weighted_mean_buffer_minutes": mean_buffer,
                "mean_buffer_ci_low": buffer_low,
                "mean_buffer_ci_high": buffer_high,
                "median_of_daily_median_buffer_minutes": float(values["median_buffer_minutes"].median()),
                "p90_of_daily_p90_buffer_minutes": float(values["p90_buffer_minutes"].quantile(0.9)),
            }
        )

    difference_columns = [
        "comparison", "metric", "treatment_minus_baseline", "ci_low", "ci_high",
        "bootstrap_probability_treatment_higher",
    ]
    if len(args.conditions) == 2:
        baseline, treatment = args.conditions
        paired = required.pivot(index="replay_start", columns="condition")
        decisions = paired[("decisions", treatment)].to_numpy()
        coverage_difference = _difference_interval(
            paired[("empirical_coverage", treatment)].to_numpy(),
            paired[("empirical_coverage", baseline)].to_numpy(),
            decisions,
            resamples=args.resamples,
            seed=args.seed,
        )
        buffer_difference = _difference_interval(
            paired[("mean_buffer_minutes", treatment)].to_numpy(),
            paired[("mean_buffer_minutes", baseline)].to_numpy(),
            decisions,
            resamples=args.resamples,
            seed=args.seed,
        )
        differences = pd.DataFrame(
            [
                {
                    "comparison": f"{treatment}_vs_{baseline}",
                    "metric": "empirical_coverage",
                    "treatment_minus_baseline": coverage_difference[0],
                    "ci_low": coverage_difference[1],
                    "ci_high": coverage_difference[2],
                    "bootstrap_probability_treatment_higher": coverage_difference[3],
                },
                {
                    "comparison": f"{treatment}_vs_{baseline}",
                    "metric": "mean_buffer_minutes",
                    "treatment_minus_baseline": buffer_difference[0],
                    "ci_low": buffer_difference[1],
                    "ci_high": buffer_difference[2],
                    "bootstrap_probability_treatment_higher": buffer_difference[3],
                },
            ],
            columns=difference_columns,
        )
    else:
        differences = pd.DataFrame(columns=difference_columns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daywise.to_csv(args.output_dir / "daywise_guard_reliability.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "guard_reliability_summary.csv", index=False)
    differences.to_csv(args.output_dir / "guard_reliability_paired_comparisons.csv", index=False)
    manifest = {
        "purpose": "Post-hoc audit of already completed fixed-condition guard replays; no guard or controller fit occurs here.",
        "artifact_root": str(args.artifact_root),
        "conditions": args.conditions,
        "target_coverage": args.target_coverage,
        "unit_of_resampling": "matched chronological calendar day",
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "buffer_note": "The reported median and P90 are summaries over daily decision-level buffer summaries; mean buffer is decision-weighted across all days.",
    }
    (args.output_dir / "guard_reliability_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Daywise guard audit: {args.output_dir / 'daywise_guard_reliability.csv'}")
    print(f"Guard reliability summary: {args.output_dir / 'guard_reliability_summary.csv'}")
    print(f"Matched guard comparison: {args.output_dir / 'guard_reliability_paired_comparisons.csv'}")


if __name__ == "__main__":
    main()
