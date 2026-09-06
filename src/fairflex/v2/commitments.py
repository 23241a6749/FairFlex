"""Causal, condition-aware early-departure protection for FairFlex V2.

This module is intentionally independent of V1's selected multi-rate guard.
It provides a transparent V2 candidate that uses only information available at
plug-in: declared dwell time and the time-of-day of plug-in.  The physical
unplug time is used only for calibration or after-the-fact auditing, never to
choose an action for the same session.

The group-specific guarantee is conditional only under within-group
exchangeability.  Sparse groups deliberately use the global one-sided
split-conformal guard.  The class must therefore be evaluated on a separate,
frozen test window; it is not an individual or universal guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fairflex.commitments import EarlyDepartureGuard, declared_duration_group
from fairflex.domain import EVSession


TIME_OF_DAY_GROUPS = (
    "arrival_06_to_10h",
    "arrival_10_to_14h",
    "arrival_other",
)
DURATION_GROUPS = (
    "declared_duration_le_4h",
    "declared_duration_4_to_8h",
    "declared_duration_gt_8h",
)
CONDITION_GROUPS = tuple(
    f"{duration}__{time_of_day}"
    for duration in DURATION_GROUPS
    for time_of_day in TIME_OF_DAY_GROUPS
)


def declared_arrival_time_group(
    session: EVSession,
    *,
    steps_per_day: int = 96,
) -> str:
    """Return a fixed plug-in-time group using no realised future fact.

    ``arrival_step`` is relative to the replay origin.  V2 protocol files
    require a local-midnight replay origin, so the modulo identifies local
    time-of-day at a 15-minute resolution.  The implementation keeps the
    denominator explicit in case a future experiment uses a different step
    size; it does not infer time zone from hidden record metadata.
    """
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")
    step_of_day = session.arrival_step % steps_per_day
    # 06:00-10:00 and 10:00-14:00 are workplace-relevant, interpretable
    # groups. All remaining arrivals share a single conservative fallback
    # candidate group rather than creating sparse post-hoc bins.
    if 24 <= step_of_day < 40:
        return "arrival_06_to_10h"
    if 40 <= step_of_day < 56:
        return "arrival_10_to_14h"
    return "arrival_other"


def declared_condition_group(
    session: EVSession,
    *,
    steps_per_day: int = 96,
) -> str:
    """Return the predeclared duration x plug-in-time condition label."""
    duration = declared_duration_group(session)
    time_of_day = declared_arrival_time_group(session, steps_per_day=steps_per_day)
    return f"{duration}__{time_of_day}"


@dataclass(frozen=True)
class ConditionAwareEarlyDepartureGuard:
    """One-sided condition-aware conformal guard with global fallback.

    The calibration fit is intentionally simple and auditable.  For a group
    with enough calibration sessions, V2 computes that group's finite-sample
    one-sided split-conformal deadline buffer.  Otherwise it uses the global
    buffer.  No user identifier, actual future departure, test outcome or
    target-window statistic enters the group assignment or buffer selection.
    """

    global_guard: EarlyDepartureGuard
    buffer_steps_by_group: Mapping[str, int]
    calibration_sessions_by_group: Mapping[str, int]
    uses_global_fallback_by_group: Mapping[str, bool]
    min_group_calibration_sessions: int
    steps_per_day: int = 96

    def __post_init__(self) -> None:
        if self.min_group_calibration_sessions <= 0:
            raise ValueError("min_group_calibration_sessions must be positive")
        if self.steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        expected = set(CONDITION_GROUPS)
        for name, values in {
            "buffer_steps_by_group": self.buffer_steps_by_group,
            "calibration_sessions_by_group": self.calibration_sessions_by_group,
            "uses_global_fallback_by_group": self.uses_global_fallback_by_group,
        }.items():
            if set(values) != expected:
                raise ValueError(f"{name} must declare exactly the predeclared V2 groups")
        if any(int(value) < 0 for value in self.buffer_steps_by_group.values()):
            raise ValueError("condition-aware buffers must be non-negative")
        if any(int(value) < 0 for value in self.calibration_sessions_by_group.values()):
            raise ValueError("condition-aware group counts must be non-negative")

    @classmethod
    def fit(
        cls,
        calibration_sessions: Sequence[EVSession],
        *,
        miscoverage: float = 0.10,
        min_group_calibration_sessions: int = 50,
        steps_per_day: int = 96,
    ) -> "ConditionAwareEarlyDepartureGuard":
        """Fit fixed group buffers plus an always-available global fallback."""
        if not calibration_sessions:
            raise ValueError("at least one calibration session is required")
        if min_group_calibration_sessions <= 0:
            raise ValueError("min_group_calibration_sessions must be positive")
        if steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        global_guard = EarlyDepartureGuard.fit(
            calibration_sessions,
            miscoverage=miscoverage,
        )
        grouped: dict[str, tuple[EVSession, ...]] = {
            group: tuple(
                session
                for session in calibration_sessions
                if declared_condition_group(session, steps_per_day=steps_per_day) == group
            )
            for group in CONDITION_GROUPS
        }
        counts = {group: len(sessions) for group, sessions in grouped.items()}
        fallbacks = {
            group: count < min_group_calibration_sessions
            for group, count in counts.items()
        }
        buffers = {
            group: (
                global_guard.buffer_steps
                if fallbacks[group]
                else EarlyDepartureGuard.fit(
                    grouped[group], miscoverage=miscoverage
                ).buffer_steps
            )
            for group in CONDITION_GROUPS
        }
        return cls(
            global_guard=global_guard,
            buffer_steps_by_group=buffers,
            calibration_sessions_by_group=counts,
            uses_global_fallback_by_group=fallbacks,
            min_group_calibration_sessions=min_group_calibration_sessions,
            steps_per_day=steps_per_day,
        )

    @property
    def miscoverage(self) -> float:
        return self.global_guard.miscoverage

    @property
    def calibration_sessions(self) -> int:
        return self.global_guard.calibration_sessions

    def group_for(self, session: EVSession) -> str:
        return declared_condition_group(session, steps_per_day=self.steps_per_day)

    def buffer_steps_for(self, session: EVSession) -> int:
        return int(self.buffer_steps_by_group[self.group_for(session)])

    def guarded_deadline_step(self, session: EVSession) -> int:
        if session.planning_departure_step is None:
            raise ValueError("a declared planning deadline is required for commitment guarding")
        return max(
            session.arrival_step + 1,
            session.planning_deadline_step - self.buffer_steps_for(session),
        )

    def apply(self, sessions: Sequence[EVSession]) -> tuple[EVSession, ...]:
        """Return planning-deadline copies while preserving physical replay facts."""
        return tuple(
            EVSession(
                session.ev_id,
                session.station_id,
                session.arrival_step,
                session.departure_step,
                session.requested_energy_kwh,
                session.max_power_kw,
                session.delivered_energy_kwh,
                self.guarded_deadline_step(session),
                session.declared_departure_step or session.planning_deadline_step,
            )
            for session in sessions
        )

    def decision_audit(self, sessions: Sequence[EVSession]) -> tuple[dict[str, object], ...]:
        """Return per-session, post-hoc reliability records for V2 reporting.

        ``covered`` is intentionally calculated only after constructing the
        guarded copy.  It is audit output and never a controller feature.
        """
        guarded = self.apply(sessions)
        return tuple(
            {
                "ev_id": original.ev_id,
                "condition_group": self.group_for(original),
                "used_global_fallback": self.uses_global_fallback_by_group[
                    self.group_for(original)
                ],
                "calibration_sessions": self.calibration_sessions_by_group[
                    self.group_for(original)
                ],
                "buffer_steps": self.buffer_steps_for(original),
                "buffer_minutes": self.buffer_steps_for(original) * 15,
                "covered": guarded_session.departure_step
                >= guarded_session.planning_deadline_step,
            }
            for original, guarded_session in zip(sessions, guarded, strict=True)
        )

    def empirical_coverage(self, sessions: Sequence[EVSession]) -> float:
        if not sessions:
            raise ValueError("at least one session is required")
        return float(np.mean([record["covered"] for record in self.decision_audit(sessions)]))

    def empirical_coverage_by_group(self, sessions: Sequence[EVSession]) -> dict[str, float]:
        if not sessions:
            raise ValueError("at least one session is required")
        values: dict[str, list[bool]] = {group: [] for group in CONDITION_GROUPS}
        for record in self.decision_audit(sessions):
            values[str(record["condition_group"])].append(bool(record["covered"]))
        return {
            group: float(np.mean(covered)) if covered else float("nan")
            for group, covered in values.items()
        }

    def describe(self) -> dict[str, object]:
        """Return JSON-safe protocol/audit metadata for a V2 manifest."""
        return {
            "method": "condition_aware_one_sided_split_conformal_early_departure_guard_v2",
            "version": "v2-protocol-1",
            "miscoverage": self.miscoverage,
            "target_coverage": 1.0 - self.miscoverage,
            "global_buffer_steps": self.global_guard.buffer_steps,
            "calibration_sessions": self.calibration_sessions,
            "condition_features": ["declared_duration_bin", "plug_in_time_of_day"],
            "steps_per_day": self.steps_per_day,
            "min_group_calibration_sessions": self.min_group_calibration_sessions,
            "buffer_steps_by_group": dict(self.buffer_steps_by_group),
            "calibration_sessions_by_group": dict(self.calibration_sessions_by_group),
            "uses_global_fallback_by_group": dict(self.uses_global_fallback_by_group),
            "coverage_scope": (
                "group-conditional only under within-group exchangeability for supported "
                "groups; global marginal interpretation for sparse-group fallback; no "
                "per-driver, universal, cross-site, or finite-sample conditional guarantee"
            ),
            "causal_information_boundary": (
                "group uses only declared dwell duration and plug-in time; realized physical "
                "unplug time is used only for calibration or post-hoc audit"
            ),
        }
