from __future__ import annotations

from fairflex.domain import EVSession
from fairflex.v2.commitments import (
    CONDITION_GROUPS,
    ConditionAwareEarlyDepartureGuard,
    declared_arrival_time_group,
    declared_condition_group,
)


def _session(
    name: str,
    *,
    arrival: int,
    declared_departure: int,
    actual_departure: int,
) -> EVSession:
    return EVSession(
        name,
        "station-a",
        arrival,
        actual_departure,
        4.0,
        7.2,
        planning_departure_step=declared_departure,
        declared_departure_step=declared_departure,
    )


def test_v2_group_uses_only_declared_duration_and_arrival_time() -> None:
    morning_short = _session("morning", arrival=24, declared_departure=40, actual_departure=25)
    afternoon_short = _session("afternoon", arrival=40, declared_departure=56, actual_departure=55)
    evening_long = _session("evening", arrival=72, declared_departure=108, actual_departure=74)

    assert declared_arrival_time_group(morning_short) == "arrival_06_to_10h"
    assert declared_arrival_time_group(afternoon_short) == "arrival_10_to_14h"
    assert declared_arrival_time_group(evening_long) == "arrival_other"
    assert declared_condition_group(morning_short) == "declared_duration_le_4h__arrival_06_to_10h"
    # Changing an actual physical unplug must never alter the plug-in group.
    altered_actual = _session("morning-2", arrival=24, declared_departure=40, actual_departure=39)
    assert declared_condition_group(altered_actual) == declared_condition_group(morning_short)


def test_v2_sparse_group_uses_global_fallback_and_preserves_physical_departure() -> None:
    calibration = [
        _session(f"cal-{index}", arrival=24, declared_departure=40, actual_departure=36)
        for index in range(3)
    ]
    guard = ConditionAwareEarlyDepartureGuard.fit(
        calibration,
        miscoverage=0.10,
        min_group_calibration_sessions=4,
    )

    assert set(guard.buffer_steps_by_group) == set(CONDITION_GROUPS)
    assert all(guard.uses_global_fallback_by_group.values())
    candidate = _session("test", arrival=24, declared_departure=40, actual_departure=35)
    guarded = guard.apply([candidate])[0]

    assert guard.buffer_steps_for(candidate) == guard.global_guard.buffer_steps
    assert guarded.departure_step == candidate.departure_step
    assert guarded.declared_departure_step == candidate.declared_departure_step
    assert guarded.planning_deadline_step <= candidate.planning_deadline_step


def test_v2_supported_condition_uses_its_own_buffer_and_audit_is_post_hoc() -> None:
    # The same declared condition has enough examples and its early-unplug
    # scores are lower than the global scores from the other condition.
    supported = [
        _session(f"supported-{index}", arrival=24, declared_departure=40, actual_departure=39)
        for index in range(5)
    ]
    other = [
        _session(f"other-{index}", arrival=72, declared_departure=108, actual_departure=76)
        for index in range(5)
    ]
    guard = ConditionAwareEarlyDepartureGuard.fit(
        [*supported, *other],
        miscoverage=0.20,
        min_group_calibration_sessions=5,
    )
    test_session = _session("test", arrival=24, declared_departure=40, actual_departure=39)
    group = guard.group_for(test_session)
    audit = guard.decision_audit([test_session])[0]

    assert not guard.uses_global_fallback_by_group[group]
    assert audit["condition_group"] == group
    assert audit["calibration_sessions"] == 5
    assert audit["buffer_steps"] == guard.buffer_steps_for(test_session)
    assert isinstance(audit["covered"], bool)
    description = guard.describe()
    assert description["condition_features"] == ["declared_duration_bin", "plug_in_time_of_day"]
    assert "per-driver" in str(description["coverage_scope"])
