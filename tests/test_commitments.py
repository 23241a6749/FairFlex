import pytest

from fairflex.commitments import (
    AdaptiveEarlyDepartureGuard,
    AgACIWeightedEarlyDepartureGuard,
    DurationStratifiedEarlyDepartureGuard,
    EarlyDepartureGuard,
    MultiRateAdaptiveEarlyDepartureGuard,
    declared_duration_group,
)
from fairflex.domain import EVSession


def _session(identifier: str, actual: int, declared: int, *, arrival: int = 0) -> EVSession:
    return EVSession(
        identifier,
        "station",
        arrival,
        actual,
        4.0,
        7.2,
        planning_departure_step=declared,
    )


def test_early_departure_guard_uses_split_conformal_order_statistic():
    # Early-departure advances are [0, 1, 2, 5].  At alpha=0.25, the conformal
    # rank is ceil((4 + 1) * 0.75) = 4, so the buffer is the largest score.
    calibration = (
        _session("a", 10, 10),
        _session("b", 9, 10),
        _session("c", 8, 10),
        _session("d", 5, 10),
    )

    guard = EarlyDepartureGuard.fit(calibration, miscoverage=0.25)

    assert guard.buffer_steps == 5
    assert guard.calibration_sessions == 4


def test_guarded_copy_hides_actual_unplug_from_controller_but_preserves_simulation_limit():
    session = _session("early", actual=3, declared=9)
    guard = EarlyDepartureGuard(miscoverage=0.10, buffer_steps=2, calibration_sessions=10)

    guarded = guard.apply((session,))[0]

    assert guarded.departure_step == 3
    assert guarded.planning_deadline_step == 7
    assert guarded.declared_departure_step == 9
    assert guarded.planning_deadline_step != guarded.departure_step


def test_guard_requires_declared_deadlines_and_measures_realized_coverage():
    legacy = EVSession("legacy", "station", 0, 3, 2.0, 7.2)
    with pytest.raises(ValueError, match="declared"):
        EarlyDepartureGuard.fit((legacy,))

    guard = EarlyDepartureGuard(miscoverage=0.10, buffer_steps=1, calibration_sessions=2)
    sessions = (_session("covered", 5, 6), _session("missed", 3, 6))
    assert guard.empirical_coverage(sessions) == pytest.approx(0.5)


def test_duration_stratified_guard_uses_supported_groups_and_global_fallback():
    short = tuple(_session(f"short-{index}", actual=14, declared=16) for index in range(50))
    medium = tuple(_session(f"medium-{index}", actual=20, declared=24) for index in range(50))
    long = (_session("long", actual=30, declared=40),)

    guard = DurationStratifiedEarlyDepartureGuard.fit(
        (*short, *medium, *long),
        miscoverage=0.10,
        min_group_calibration_sessions=50,
    )

    assert declared_duration_group(short[0]) == "declared_duration_le_4h"
    assert declared_duration_group(medium[0]) == "declared_duration_4_to_8h"
    assert declared_duration_group(long[0]) == "declared_duration_gt_8h"
    assert guard.buffer_steps_for(short[0]) == 2
    assert guard.buffer_steps_for(medium[0]) == 4
    # The long group has one calibration example, so it receives the global
    # buffer rather than an unreliable one-session local quantile.
    assert guard.uses_global_fallback_by_group["declared_duration_gt_8h"]
    assert guard.buffer_steps_for(long[0]) == guard.global_guard.buffer_steps
    assert guard.apply(short)[0].declared_departure_step == 16


def test_adaptive_guard_updates_only_after_a_previous_physical_unplug():
    calibration = tuple(_session(f"calibration-{index}", actual=10, declared=10) for index in range(4))
    guard = AdaptiveEarlyDepartureGuard.fit(
        calibration,
        miscoverage=0.10,
        learning_rate=0.10,
        min_miscoverage=0.01,
    )
    first = _session("first", actual=1, declared=10)
    second = _session("second", actual=8, declared=10, arrival=1)

    guarded, audit = guard.apply_causally((first, second))

    # The first EV has already physically left when the second plugs in. Its
    # miss lowers alpha and its score is then eligible for the second decision.
    assert guarded[0].planning_deadline_step == 10
    assert guarded[1].planning_deadline_step == 2
    assert guarded[1].departure_step == second.departure_step
    assert audit["outcomes_observed_before_decisions"] == 1
    assert audit["effective_miscoverage_at_decision"]["minimum"] == pytest.approx(0.01)


def test_adaptive_guard_does_not_use_an_unplug_that_has_not_occurred_yet():
    calibration = tuple(_session(f"calibration-{index}", actual=10, declared=10) for index in range(4))
    guard = AdaptiveEarlyDepartureGuard.fit(calibration, learning_rate=0.10)
    first = _session("first", actual=5, declared=10)
    second = _session("second", actual=8, declared=10, arrival=1)

    guarded, audit = guard.apply_causally((first, second))

    # Both decisions occur before the first realized departure at step five;
    # neither can use its hidden future outcome or score.
    assert guarded[0].planning_deadline_step == 10
    assert guarded[1].planning_deadline_step == 10
    assert audit["outcomes_observed_before_decisions"] == 0


def test_multi_rate_envelope_selects_the_earliest_causal_expert_deadline():
    calibration = tuple(_session(f"calibration-{index}", actual=10, declared=10) for index in range(20))
    guard = MultiRateAdaptiveEarlyDepartureGuard.fit(
        calibration,
        learning_rates=(0.005, 0.10),
        min_miscoverage=0.01,
    )
    first = _session("first", actual=1, declared=10)
    second = _session("second", actual=8, declared=10, arrival=1)

    guarded, audit = guard.apply_causally((first, second))

    # The fast expert reacts after the first physical unplug and proposes a
    # safer deadline of step two; the envelope must choose that earliest
    # causal deadline rather than an average or a future-informed value.
    assert guarded[1].planning_deadline_step == 2
    assert audit["selected_expert_decision_counts"]["0.1"] >= 1
    assert guarded[1].departure_step == second.departure_step


def test_agaci_weighted_guard_never_uses_a_future_unplug_at_plug_in():
    calibration = tuple(
        _session(f"calibration-{index}", actual=10, declared=10) for index in range(20)
    )
    guard = AgACIWeightedEarlyDepartureGuard.fit(
        calibration,
        learning_rates=(0.05, 0.10),
        min_miscoverage=0.01,
    )
    first = _session("first", actual=5, declared=10)
    second = _session("second", actual=8, declared=10, arrival=1)

    guarded, audit = guard.apply_causally((first, second))

    # The physical unplug is at step five, after both plug-ins. The second
    # decision must therefore still use only the calibration score bank.
    assert guarded[0].planning_deadline_step == 10
    assert guarded[1].planning_deadline_step == 10
    assert audit["outcomes_observed_before_decisions"] == 0


def test_agaci_weighted_guard_aggregates_causal_experts_after_observed_miss():
    calibration = tuple(
        _session(f"calibration-{index}", actual=10, declared=10) for index in range(20)
    )
    guard = AgACIWeightedEarlyDepartureGuard.fit(
        calibration,
        learning_rates=(0.05, 0.10),
        min_miscoverage=0.01,
    )
    first = _session("first", actual=1, declared=10)
    second = _session("second", actual=8, declared=10, arrival=1)

    guarded, audit = guard.apply_causally((first, second))

    # The first EV has physically departed before the second plugs in. Its
    # causal miss updates the two experts and the aggregate then selects the
    # nine-step early-departure buffer from the augmented score bank. The
    # controller deadline cannot precede the EV's arrival-plus-one step.
    assert guarded[1].planning_deadline_step == 2
    assert guarded[1].departure_step == second.departure_step
    assert audit["outcomes_observed_before_decisions"] == 1
    assert sum(audit["final_expert_probabilities"].values()) == pytest.approx(1.0)
    assert audit["aggregation_rule"] == "AgACI-style Bernstein online aggregation over ACI experts"
