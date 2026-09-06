"""Development-only ranking helpers for V2 guard candidates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def calendar_day_bootstrap_coverage_interval(
    rows: Sequence[dict[str, object]],
    *,
    resamples: int = 2_000,
    seed: int = 20_260_903,
) -> tuple[float, float]:
    """Return a descriptive calendar-day cluster-bootstrap coverage interval.

    Whole days, rather than individual sessions, are resampled so that sessions
    from the same day keep their shared arrival and capacity conditions.  This
    is an uncertainty diagnostic for a chronological development replay, not a
    new distribution-free coverage guarantee.
    """
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        if "calendar_day" not in row or "covered" not in row:
            raise ValueError("coverage rows need calendar_day and covered fields")
        grouped.setdefault(str(row["calendar_day"]), []).append(bool(row["covered"]))
    if len(grouped) < 2:
        raise ValueError("calendar-day bootstrap needs at least two calendar days")
    days = tuple(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        values = [value for day in sampled_days for value in grouped[str(day)]]
        estimates[index] = float(np.mean(values))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def rank_guard_candidates(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Rank coverage-eligible V2 guard candidates by conservative buffer use.

    This helper is deliberately limited to the guard's reliability/buffer
    frontier.  It does not choose a paper method: a shortlist still requires a
    separate development policy matrix with P10, Jain, energy and runtime.
    """
    required = {
        "protocol_id",
        "observed_coverage",
        "target_coverage",
        "p90_buffer_minutes",
        "mean_buffer_minutes",
        "global_fallback_rate",
    }
    validated: list[dict[str, object]] = []
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"candidate summary is missing keys: {sorted(missing)}")
        copied = dict(row)
        copied["coverage_eligible"] = float(copied["observed_coverage"]) >= float(
            copied["target_coverage"]
        )
        validated.append(copied)
    return sorted(
        validated,
        key=lambda row: (
            not bool(row["coverage_eligible"]),
            float(row["p90_buffer_minutes"]),
            float(row["mean_buffer_minutes"]),
            float(row["global_fallback_rate"]),
            str(row["protocol_id"]),
        ),
    )


def apply_june_contextual_guard_rule(
    *,
    candidate_coverage: float,
    target_coverage: float,
    candidate_unsafe_steps: int,
    p10_delta: float,
    jain_delta: float,
    mean_buffer_delta_minutes: float,
) -> dict[str, object]:
    """Apply the June decision rule recorded before the June replay.

    This is a development freeze only.  Passing it nominates one guard for a
    new holdout; it does not validate a final method or justify a paper claim.
    """
    checks = {
        "coverage_at_least_target": candidate_coverage >= target_coverage,
        "no_executed_safety_violation": candidate_unsafe_steps == 0,
        "p10_noninferior_within_0_005": p10_delta >= -0.005,
        "jain_noninferior_within_0_010": jain_delta >= -0.010,
        "mean_buffer_within_20_minutes": mean_buffer_delta_minutes <= 20.0,
    }
    advances = all(checks.values())
    return {
        "contextual_candidate_advances": advances,
        "selected_guard": "condition_aware_min20" if advances else "global_fallback",
        "checks": checks,
        "inputs": {
            "candidate_coverage": candidate_coverage,
            "target_coverage": target_coverage,
            "candidate_unsafe_steps": candidate_unsafe_steps,
            "p10_delta": p10_delta,
            "jain_delta": jain_delta,
            "mean_buffer_delta_minutes": mean_buffer_delta_minutes,
        },
    }
