"""Reproducible, development-only comparison helpers for FairFlex V2."""

from __future__ import annotations

from collections.abc import Sequence


DEVELOPMENT_POLICY_METRICS = (
    "mean_service_ratio",
    "p10_service_ratio",
    "worst_service_ratio",
    "jain_service_index",
    "delivered_energy_kwh",
    "unsafe_steps",
    "runtime_seconds",
)


def development_metric_deltas(
    reference_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return candidate-minus-reference values, matched only by policy name.

    This function does not rank or select a candidate.  It makes every stated
    trade-off visible and keeps the comparison explicitly development-only.
    """
    def index(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            if "policy" not in row:
                raise ValueError("every policy row must include policy")
            policy = str(row["policy"])
            if policy in result:
                raise ValueError(f"duplicate policy row: {policy}")
            missing = set(DEVELOPMENT_POLICY_METRICS) - set(row)
            if missing:
                raise ValueError(f"policy row {policy} is missing metrics: {sorted(missing)}")
            result[policy] = dict(row)
        return result

    reference = index(reference_rows)
    candidate = index(candidate_rows)
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate reports must contain the same policies")
    rows: list[dict[str, object]] = []
    for policy in sorted(reference):
        row: dict[str, object] = {"policy": policy}
        for metric in DEVELOPMENT_POLICY_METRICS:
            baseline = float(reference[policy][metric])
            value = float(candidate[policy][metric])
            row[f"reference_{metric}"] = baseline
            row[f"candidate_{metric}"] = value
            row[f"candidate_minus_reference_{metric}"] = value - baseline
        rows.append(row)
    return rows
