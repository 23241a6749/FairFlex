import pandas as pd

from fairflex.domain import EVSession
from fairflex.v3.evaluation import (
    continuous_arrival_day_metrics,
    continuous_replay_session_rows,
    expected_shortfall_of_deficits,
    summarize_early_unplug_subset,
)


class _Result:
    service_ratios = {"early": 0.25, "on_time": 1.0, "late": 0.5}


def test_expected_shortfall_of_deficits_uses_the_largest_tail():
    assert expected_shortfall_of_deficits((1.0, 0.5, 0.0), tail_fraction=0.10) == 1.0


def test_early_unplug_subset_metrics_use_physical_departure_after_evaluation():
    sessions = (
        EVSession("early", "s", 0, 4, 2.0, 7.2, planning_departure_step=8),
        EVSession("on_time", "s", 0, 8, 2.0, 7.2, planning_departure_step=8),
        EVSession("late", "s", 0, 9, 2.0, 7.2, planning_departure_step=8),
    )
    metrics = summarize_early_unplug_subset(_Result(), sessions, early_unplug_threshold_steps=2)

    assert metrics.early_unplug_sessions == 1
    assert metrics.early_unplug_p10_service_ratio == 0.25


def test_continuous_replay_rows_group_outcomes_without_restarting_a_guard():
    sessions = (
        EVSession("early", "s", 0, 4, 2.0, 7.2, planning_departure_step=8),
        EVSession("next-day", "s", 96, 104, 4.0, 7.2, planning_departure_step=104),
    )
    result = type("Result", (), {"service_ratios": {"early": 0.25, "next-day": 1.0}})()

    rows = continuous_replay_session_rows(
        result,
        sessions,
        policy="fairflex_uc_hybrid_lower_tail",
        replay_origin="2019-11-01T00:00:00Z",
        step_minutes=15,
        early_unplug_threshold_steps=2,
    )

    assert [row["calendar_day"] for row in rows] == ["2019-11-01", "2019-11-02"]
    assert rows[0]["is_early_unplug"]
    assert rows[0]["delivered_energy_kwh"] == 0.5


def test_continuous_day_metrics_keep_one_policy_row_per_arrival_day():
    rows = [
        {
            "policy": policy,
            "calendar_day": "2019-11-01",
            "requested_energy_kwh": 2.0,
            "delivered_energy_kwh": value * 2.0,
            "service_ratio": value,
            "is_early_unplug": early,
        }
        for policy, value, early in (
            ("v1", 0.5, True),
            ("v1", 1.0, False),
            ("v3", 0.75, True),
            ("v3", 1.0, False),
        )
    ]
    daywise = continuous_arrival_day_metrics(pd.DataFrame(rows))

    assert set(daywise["policy"]) == {"v1", "v3"}
    assert set(daywise["sessions"]) == {2}
    assert set(daywise["early_unplug_sessions"]) == {1}
    assert daywise.loc[daywise["policy"] == "v3", "p10_service_ratio"].item() > daywise.loc[
        daywise["policy"] == "v1", "p10_service_ratio"
    ].item()
