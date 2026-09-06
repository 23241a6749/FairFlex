"""Audit frozen Safe-MARL results with matched days and nested seed/day draws.

This command consumes completed continuous-replay session outcomes.  It never
trains, replays, tunes, or selects a seed.  Individual EV rows are not treated
as independent evidence because vehicles arriving on a day share conditions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fairflex.statistics import (
    nested_seed_day_bootstrap,
    paired_block_bootstrap,
    paired_sign_flip_test,
)
from fairflex.v3.evaluation import continuous_arrival_day_metrics


BASELINE = "fairflex_uc_hybrid_lower_tail"
POLICY_PATTERN = re.compile(r"^safe_(mappo|ippo)_station_seed(\d+)$")
METRICS: dict[str, dict[str, object]] = {
    "p10_service_ratio": {
        "direction": "higher", "support": "sessions", "minimum": 10, "role": "primary",
    },
    "early_unplug_p10_service_ratio": {
        "direction": "higher", "support": "early_unplug_sessions", "minimum": 10,
        "role": "secondary",
    },
    "all_session_expected_shortfall_10pct": {
        "direction": "lower", "support": None, "minimum": None, "role": "secondary",
    },
    "mean_service_ratio": {
        "direction": "higher", "support": None, "minimum": None, "role": "exploratory",
    },
    "jain_service_index": {
        "direction": "higher", "support": None, "minimum": None, "role": "exploratory",
    },
    "delivered_energy_kwh": {
        "direction": "higher", "support": None, "minimum": None, "role": "exploratory",
    },
}


def _read_session_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {
        "policy", "ev_id", "calendar_day", "requested_energy_kwh",
        "delivered_energy_kwh", "service_ratio", "is_early_unplug",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"session file {path} is missing columns: {missing}")
    return rows


def _baseline(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows.loc[rows["policy"] == BASELINE].copy()
    if baseline.empty:
        raise ValueError(f"session file does not contain baseline {BASELINE!r}")
    return baseline.sort_values(["calendar_day", "ev_id"]).reset_index(drop=True)


def _policy_metadata(policy: str) -> tuple[str, int]:
    match = POLICY_PATTERN.fullmatch(policy)
    if match is None:
        raise ValueError(f"unexpected Safe-MARL policy name: {policy!r}")
    return match.group(1), int(match.group(2))


def _eligible_wide(
    daywise: pd.DataFrame,
    *,
    metric: str,
    policies: list[str],
) -> pd.DataFrame:
    """Return complete matched days after the predeclared support screen."""
    selected = daywise.loc[
        daywise["policy"].isin(policies), ["replay_start", "policy", metric]
    ]
    wide = selected.pivot(index="replay_start", columns="policy", values=metric)
    wide = wide.reindex(columns=policies)
    spec = METRICS[metric]
    support_name = spec["support"]
    if support_name is not None:
        support = daywise.loc[
            daywise["policy"].isin(policies),
            ["replay_start", "policy", str(support_name)],
        ]
        support_wide = support.pivot(
            index="replay_start", columns="policy", values=str(support_name)
        ).reindex(columns=policies)
        wide = wide.loc[(support_wide >= int(spec["minimum"])).all(axis=1)]
    return wide.dropna(axis=0, how="any")


def _oriented_interval(low: float, high: float, direction: str) -> tuple[float, float]:
    sign = 1.0 if direction == "higher" else -1.0
    return min(sign * low, sign * high), max(sign * low, sign * high)


def _per_seed_row(
    values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    metric: str,
    policy: str,
    algorithm: str,
    seed_value: int,
    resamples: int,
    random_seed: int,
) -> dict[str, object]:
    spec = METRICS[metric]
    direction = str(spec["direction"])
    bootstrap = paired_block_bootstrap(
        values, baseline_values, metric=metric, treatment=policy,
        baseline=BASELINE, resamples=resamples, seed=random_seed,
    )
    sign_flip = paired_sign_flip_test(
        values, baseline_values, metric=metric, treatment=policy,
        baseline=BASELINE, preferred_direction=direction, seed=random_seed,
    )
    low, high = _oriented_interval(bootstrap.ci_low, bootstrap.ci_high, direction)
    return {
        "metric": metric, "role": spec["role"], "algorithm": algorithm,
        "analysis_level": "per_seed", "policy": policy, "seed": seed_value,
        "preferred_direction": direction, "eligible_calendar_days": int(values.size),
        "observed_treatment_advantage": sign_flip.observed_oriented_mean_advantage,
        "advantage_bootstrap_ci_low": low, "advantage_bootstrap_ci_high": high,
        "bootstrap_probability_treatment_better": (
            bootstrap.bootstrap_probability_difference_positive
            if direction == "higher" else bootstrap.bootstrap_probability_difference_negative
        ),
        "p_value_one_sided": sign_flip.p_value_one_sided,
        "p_value_two_sided": sign_flip.p_value_two_sided,
        "sign_flip_test_method": sign_flip.test_method,
        "sign_flip_permutations": sign_flip.permutations,
        "bootstrap_resamples": bootstrap.resamples,
    }


def _ensemble_row(
    seed_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    metric: str,
    algorithm: str,
    policies: list[str],
    resamples: int,
    random_seed: int,
) -> dict[str, object]:
    spec = METRICS[metric]
    direction = str(spec["direction"])
    seed_mean = seed_values.mean(axis=0)
    treatment = f"safe_{algorithm}_station_seed_mean"
    sign_flip = paired_sign_flip_test(
        seed_mean, baseline_values, metric=metric, treatment=treatment,
        baseline=BASELINE, preferred_direction=direction, seed=random_seed,
    )
    nested = nested_seed_day_bootstrap(
        seed_values - baseline_values[None, :], metric=metric, treatment=treatment,
        baseline=BASELINE, resamples=resamples, seed=random_seed,
    )
    low, high = _oriented_interval(nested.ci_low, nested.ci_high, direction)
    return {
        "metric": metric, "role": spec["role"], "algorithm": algorithm,
        "analysis_level": "equally_weighted_seed_ensemble", "policy": treatment,
        "seed": "all_predeclared_seeds", "seed_policies": json.dumps(policies),
        "preferred_direction": direction, "eligible_calendar_days": nested.calendar_days,
        "predeclared_seed_count": nested.seeds,
        "observed_treatment_advantage": sign_flip.observed_oriented_mean_advantage,
        "advantage_nested_seed_day_ci_low": low,
        "advantage_nested_seed_day_ci_high": high,
        "nested_bootstrap_probability_treatment_better": (
            nested.bootstrap_probability_difference_positive
            if direction == "higher" else nested.bootstrap_probability_difference_negative
        ),
        "fixed_seed_mean_day_sign_flip_p_value_one_sided": sign_flip.p_value_one_sided,
        "fixed_seed_mean_day_sign_flip_p_value_two_sided": sign_flip.p_value_two_sided,
        "sign_flip_test_method": sign_flip.test_method,
        "sign_flip_permutations": sign_flip.permutations,
        "nested_bootstrap_resamples": nested.resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappo-session-metrics", type=Path, required=True)
    parser.add_argument("--ippo-session-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--minimum-calendar-days", type=int, default=5)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    if args.minimum_calendar_days < 2:
        parser.error("--minimum-calendar-days must be at least two")

    mappo = _read_session_rows(args.mappo_session_metrics)
    ippo = _read_session_rows(args.ippo_session_metrics)
    mappo_baseline, ippo_baseline = _baseline(mappo), _baseline(ippo)
    if not mappo_baseline.equals(ippo_baseline):
        parser.error("MAPPO/IPPO files do not contain identical V3 baseline session rows")
    treatment_rows = pd.concat([
        mappo.loc[mappo["policy"] != BASELINE],
        ippo.loc[ippo["policy"] != BASELINE],
    ], ignore_index=True)
    daywise = continuous_arrival_day_metrics(
        pd.concat([mappo_baseline, treatment_rows], ignore_index=True)
    )

    seed_policies: dict[str, list[tuple[int, str]]] = {"mappo": [], "ippo": []}
    for policy in sorted(treatment_rows["policy"].unique()):
        algorithm, seed_value = _policy_metadata(str(policy))
        seed_policies[algorithm].append((seed_value, str(policy)))
    for algorithm in seed_policies:
        seed_policies[algorithm].sort()
        if len(seed_policies[algorithm]) < 2:
            parser.error(f"{algorithm} needs at least two predeclared seed policies")

    per_seed_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    withheld: list[dict[str, object]] = []
    for algorithm, indexed_policies in seed_policies.items():
        policies = [BASELINE, *[policy for _, policy in indexed_policies]]
        for metric in METRICS:
            wide = _eligible_wide(daywise, metric=metric, policies=policies)
            if len(wide) < args.minimum_calendar_days:
                withheld.append({
                    "algorithm": algorithm, "metric": metric,
                    "eligible_calendar_days": int(len(wide)),
                    "reason": f"fewer than {args.minimum_calendar_days} complete matched calendar days after support filtering",
                })
                continue
            baseline_values = wide[BASELINE].to_numpy(dtype=float)
            seed_values = np.vstack([
                wide[policy].to_numpy(dtype=float) for _, policy in indexed_policies
            ])
            for (seed_value, policy), values in zip(indexed_policies, seed_values, strict=True):
                per_seed_rows.append(_per_seed_row(
                    values, baseline_values, metric=metric, policy=policy,
                    algorithm=algorithm, seed_value=seed_value, resamples=args.resamples,
                    random_seed=args.seed,
                ))
            ensemble_rows.append(_ensemble_row(
                seed_values, baseline_values, metric=metric, algorithm=algorithm,
                policies=[policy for _, policy in indexed_policies],
                resamples=args.resamples, random_seed=args.seed,
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daywise_path = args.output_dir / "safe_marl_daywise_metrics.csv"
    per_seed_path = args.output_dir / "safe_marl_per_seed_statistical_audit.csv"
    ensemble_path = args.output_dir / "safe_marl_seed_ensemble_statistical_audit.csv"
    withheld_path = args.output_dir / "safe_marl_withheld_statistical_metrics.csv"
    manifest_path = args.output_dir / "safe_marl_statistical_audit_manifest.json"
    daywise.to_csv(daywise_path, index=False)
    pd.DataFrame(per_seed_rows).sort_values(["algorithm", "metric", "seed"]).to_csv(per_seed_path, index=False)
    pd.DataFrame(ensemble_rows).sort_values(["algorithm", "metric"]).to_csv(ensemble_path, index=False)
    pd.DataFrame(withheld).to_csv(withheld_path, index=False)
    manifest_path.write_text(json.dumps({
        "mappo_session_metrics": str(args.mappo_session_metrics),
        "ippo_session_metrics": str(args.ippo_session_metrics),
        "baseline": BASELINE,
        "baseline_rows_identical_across_files": True,
        "unit_of_inference": "matched calendar day defined by arrival date from one continuous replay",
        "primary_endpoint": "P10 service ratio with at least 10 sessions per matched day",
        "per_seed_inference": "paired calendar-day percentile bootstrap plus paired sign-flip test; exact through 16 eligible days and seeded Monte Carlo thereafter",
        "seed_ensemble_inference": "crossed nested bootstrap resampling predeclared seeds and matched calendar days; the sign-flip test conditions on the fixed five predeclared seeds",
        "not_used": "session-level t-test, ordinary ANOVA, seed picking, post-test training, or replay reset at midnight",
        "resamples": args.resamples, "seed": args.seed,
        "minimum_calendar_days": args.minimum_calendar_days,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Daywise metrics: {daywise_path}")
    print(f"Per-seed statistical audit: {per_seed_path}")
    print(f"Seed-ensemble statistical audit: {ensemble_path}")
    print(f"Withheld metrics: {withheld_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
