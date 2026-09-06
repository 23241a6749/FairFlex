import pytest

from fairflex.domain import EVSession
from fairflex.v3.commitments import (
    CausalCQRBufferGuard,
    conservative_deadline_envelope,
    early_unplug_advance_steps,
)


def _session(index: int, *, actual: int | None = None, declared: int = 20) -> EVSession:
    arrival = index % 6
    realized = actual if actual is not None else declared - (index % 5)
    return EVSession(
        f"ev-{index}",
        "north" if index % 2 else "south",
        arrival,
        realized,
        2.0 + (index % 4),
        7.2,
        planning_departure_step=declared,
        declared_departure_step=declared,
    )


def test_cqr_guard_uses_only_training_and_calibration_sessions_and_returns_nonnegative_buffers():
    train = tuple(_session(index, declared=20 + (index % 3)) for index in range(30))
    calibration = tuple(_session(100 + index, declared=21 + (index % 3)) for index in range(20))
    test = tuple(_session(200 + index, declared=22 + (index % 3)) for index in range(6))

    guard = CausalCQRBufferGuard.fit(
        train,
        calibration,
        train_origin="2019-03-01T00:00:00Z",
        calibration_origin="2019-04-01T00:00:00Z",
        step_minutes=15,
        min_train_sessions=20,
        min_calibration_sessions=10,
        max_iter=20,
        min_samples_leaf=3,
    )
    guarded, audit = guard.apply_causally(test, origin="2019-05-01T00:00:00Z")

    assert len(guarded) == len(test)
    assert audit["train_sessions"] == len(train)
    assert audit["calibration_sessions"] == len(calibration)
    assert audit["buffer_steps"]["minimum"] >= 0
    assert audit["raw_upper_quantile_pinball_loss"] >= 0.0
    assert audit["mean_actual_early_unplug_steps"] >= 0.0
    assert audit["mean_uncovered_early_unplug_steps"] >= 0.0
    assert audit["mean_conservative_buffer_excess_steps"] >= 0.0
    assert all(item.planning_deadline_step <= item.declared_departure_step for item in guarded)
    assert all(item.departure_step == original.departure_step for item, original in zip(guarded, test))

    rows = guard.prediction_audit_rows(test, origin="2019-05-01T00:00:00Z")
    assert len(rows) == len(test)
    assert {"actual_early_unplug_steps", "integer_cqr_buffer_steps", "cqr_covered"} <= set(rows[0])
    assert all(row["integer_cqr_buffer_steps"] >= 0 for row in rows)


def test_cqr_buffer_for_a_session_does_not_depend_on_its_hidden_realized_departure():
    train = tuple(_session(index) for index in range(30))
    calibration = tuple(_session(100 + index) for index in range(20))
    guard = CausalCQRBufferGuard.fit(
        train,
        calibration,
        train_origin="2019-03-01T00:00:00Z",
        calibration_origin="2019-04-01T00:00:00Z",
        step_minutes=15,
        min_train_sessions=20,
        min_calibration_sessions=10,
        max_iter=20,
        min_samples_leaf=3,
    )
    leaves_very_early = _session(301, actual=3, declared=22)
    stays_later = _session(301, actual=18, declared=22)

    early_buffer = guard.buffer_steps((leaves_very_early,), origin="2019-05-01T00:00:00Z")
    later_buffer = guard.buffer_steps((stays_later,), origin="2019-05-01T00:00:00Z")

    assert early_buffer.tolist() == later_buffer.tolist()
    assert early_unplug_advance_steps(leaves_very_early) > early_unplug_advance_steps(stays_later)


def test_cqr_guard_rejects_undersized_declared_training_or_calibration_windows():
    sessions = tuple(_session(index) for index in range(4))
    with pytest.raises(ValueError, match="insufficient train"):
        CausalCQRBufferGuard.fit(
            sessions,
            sessions,
            train_origin="2019-03-01T00:00:00Z",
            calibration_origin="2019-04-01T00:00:00Z",
            step_minutes=15,
            min_train_sessions=5,
            min_calibration_sessions=1,
        )


def test_conservative_deadline_envelope_keeps_the_earliest_causal_deadline():
    first = _session(1, actual=14, declared=20)
    second = _session(1, actual=14, declared=20)
    first = EVSession(
        first.ev_id, first.station_id, first.arrival_step, first.departure_step,
        first.requested_energy_kwh, first.max_power_kw, planning_departure_step=17,
        declared_departure_step=20,
    )
    second = EVSession(
        second.ev_id, second.station_id, second.arrival_step, second.departure_step,
        second.requested_energy_kwh, second.max_power_kw, planning_departure_step=15,
        declared_departure_step=20,
    )

    combined, audit = conservative_deadline_envelope((first,), (second,))

    assert combined[0].planning_deadline_step == 15
    assert audit["selected_earliest_deadline_count_by_input"] == [0, 1]
    assert audit["empirical_coverage"] == 0.0
