import pandas as pd

from scripts.v3.summarize_daywise_matrix import _eligible_subset


def test_daily_p10_screening_excludes_underpopulated_days_without_touching_other_metrics():
    table = pd.DataFrame({
        "replay_start": ["day-1", "day-1", "day-2", "day-2"],
        "policy": ["a", "b", "a", "b"],
        "sessions": [12, 12, 2, 2],
        "early_unplug_sessions": [11, 11, 1, 1],
    })

    p10 = _eligible_subset(
        table,
        metric="p10_service_ratio",
        minimum_sessions_for_p10=10,
        minimum_early_sessions_for_p10=10,
    )
    early = _eligible_subset(
        table,
        metric="early_unplug_p10_service_ratio",
        minimum_sessions_for_p10=10,
        minimum_early_sessions_for_p10=10,
    )
    mean = _eligible_subset(
        table,
        metric="mean_service_ratio",
        minimum_sessions_for_p10=10,
        minimum_early_sessions_for_p10=10,
    )

    assert set(p10["replay_start"]) == {"day-1"}
    assert set(early["replay_start"]) == {"day-1"}
    assert len(mean) == len(table)
