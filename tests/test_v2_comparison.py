from __future__ import annotations

import pytest

from fairflex.v2.comparison import development_metric_deltas


def _row(policy: str, value: float) -> dict[str, object]:
    return {
        "policy": policy,
        "mean_service_ratio": value,
        "p10_service_ratio": value,
        "worst_service_ratio": value,
        "jain_service_index": value,
        "delivered_energy_kwh": value,
        "unsafe_steps": value,
        "runtime_seconds": value,
    }


def test_v2_comparison_reports_candidate_minus_reference() -> None:
    rows = development_metric_deltas([_row("fair", 1.0)], [_row("fair", 1.2)])

    assert rows[0]["candidate_minus_reference_p10_service_ratio"] == pytest.approx(0.2)


def test_v2_comparison_rejects_policy_mismatch() -> None:
    with pytest.raises(ValueError, match="same policies"):
        development_metric_deltas([_row("fair", 1.0)], [_row("fcfs", 1.0)])
