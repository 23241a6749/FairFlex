"""Causal user-commitment uncertainty tools for FairFlex.

The controller sees a driver's declared energy request and requested departure
time.  It must not see the historical eventual unplug time before that event
actually occurs.  This module learns a one-sided, split-conformal buffer for
early departures and converts a declared deadline into a conservative planning
deadline.  It deliberately does not alter the simulator's physical unplug time.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import ceil
from typing import Mapping, Sequence

import numpy as np

from .domain import EVSession


def _buffer_step_summary(values: Sequence[int]) -> dict[str, float | int]:
    """Describe deadline conservatism without reducing it to a mean alone."""
    if not values:
        raise ValueError("at least one buffer observation is required")
    array = np.asarray(values, dtype=float)
    return {
        "minimum": int(array.min()),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
    }


@dataclass(frozen=True)
class EarlyDepartureGuard:
    """Distribution-free one-sided guard for declared EV departure times.

    ``buffer_steps`` is fitted only from calibration sessions.  For a session
    with declared deadline ``d``, a controller receives ``max(arrival + 1,
    d - buffer_steps)``.  The historical realized departure remains hidden from
    this calculation and continues to be enforced only by the simulator.

    Under exchangeability of future and calibration sessions, the order
    statistic used in :meth:`fit` gives marginal coverage of at least
    ``1 - miscoverage`` for the event that the realized unplug time is no
    earlier than the guarded deadline.  It is not a per-driver guarantee.
    """

    miscoverage: float
    buffer_steps: int
    calibration_sessions: int

    def __post_init__(self) -> None:
        if not 0 < self.miscoverage < 1:
            raise ValueError("miscoverage must lie strictly between zero and one")
        if self.buffer_steps < 0 or self.calibration_sessions <= 0:
            raise ValueError("buffer_steps must be non-negative and calibration_sessions positive")

    @classmethod
    def fit(
        cls, calibration_sessions: Sequence[EVSession], *, miscoverage: float = 0.10
    ) -> "EarlyDepartureGuard":
        """Fit an upper conformal quantile of early-departure advances.

        The nonconformity score is ``max(0, declared - realized)`` in control
        steps.  A high score means the driver departed considerably earlier
        than promised.  Sessions without an explicit declared deadline are not
        valid calibration examples because their planning deadline may be a
        legacy fallback to the realized unplug time.
        """
        if not 0 < miscoverage < 1:
            raise ValueError("miscoverage must lie strictly between zero and one")
        if not calibration_sessions:
            raise ValueError("at least one calibration session is required")
        if any(session.planning_departure_step is None for session in calibration_sessions):
            raise ValueError("calibration sessions must have declared planning deadlines")

        advances = np.asarray(
            [
                max(0, session.planning_deadline_step - session.departure_step)
                for session in calibration_sessions
            ],
            dtype=int,
        )
        # Split-conformal order statistic: ceil((n + 1) * (1 - alpha)).
        rank = min(len(advances), ceil((len(advances) + 1) * (1.0 - miscoverage)))
        buffer_steps = int(np.partition(advances, rank - 1)[rank - 1])
        return cls(miscoverage, buffer_steps, len(advances))

    def guarded_deadline_step(self, session: EVSession) -> int:
        """Return a causal, conservative deadline for one controller decision."""
        if session.planning_departure_step is None:
            raise ValueError("a declared planning deadline is required for commitment guarding")
        return max(session.arrival_step + 1, session.planning_deadline_step - self.buffer_steps)

    def apply(self, sessions: Sequence[EVSession]) -> tuple[EVSession, ...]:
        """Return copies whose controller-visible deadline is guarded.

        ``departure_step`` and delivered energy are copied unchanged, preserving
        the actual physical replay outcome.  This makes it safe to pass the
        returned sessions directly to the existing simulation and MPC code.
        """
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

    def empirical_coverage(self, sessions: Sequence[EVSession]) -> float:
        """Measure realized coverage after a replay; never use it for planning."""
        if not sessions:
            raise ValueError("at least one session is required")
        guarded = self.apply(sessions)
        return float(
            np.mean(
                [session.departure_step >= session.planning_deadline_step for session in guarded]
            )
        )

    def describe(self) -> dict[str, object]:
        """Return credential-free fit metadata suitable for an experiment manifest."""
        return {
            "method": "one_sided_split_conformal_early_departure_guard",
            "miscoverage": self.miscoverage,
            "target_coverage": 1.0 - self.miscoverage,
            "buffer_steps": self.buffer_steps,
            "calibration_sessions": self.calibration_sessions,
            "coverage_scope": "global marginal under an exchangeability assumption",
        }


def _early_departure_advance(session: EVSession) -> int:
    """Return the one-sided early-unplug nonconformity score for a session."""
    if session.planning_departure_step is None:
        raise ValueError("a declared planning deadline is required for commitment guarding")
    declared = session.declared_departure_step or session.planning_departure_step
    return max(0, declared - session.departure_step)


def _conformal_buffer(scores: Sequence[int], *, miscoverage: float) -> int:
    """Return the finite-sample upper conformal score quantile."""
    if not scores:
        raise ValueError("at least one calibration score is required")
    if not 0 < miscoverage < 1:
        raise ValueError("miscoverage must lie strictly between zero and one")
    values = np.asarray(scores, dtype=int)
    rank = min(len(values), ceil((len(values) + 1) * (1.0 - miscoverage)))
    return int(np.partition(values, rank - 1)[rank - 1])


@dataclass(frozen=True)
class AdaptiveEarlyDepartureGuard:
    """Past-only adaptive conformal guard for a drifting departure process.

    A fixed split-conformal buffer is useful when the calibration and
    deployment sessions are exchangeable.  That assumption can fail across
    seasons.  This operational extension uses the Adaptive Conformal
    Inference (ACI) update idea: after a *previous* EV physically departs, its
    coverage error nudges the effective miscoverage risk for later EVs.  The
    next guarded deadline is then the one-sided conformal quantile of the
    original calibration scores plus all already-observed replay scores.

    ``apply_causally`` processes events in timestamp order.  At an EV's
    plug-in event it first incorporates only unplug events at or before that
    control step, predicts the guarded deadline, and only then queues the
    current EV's hidden realized departure.  Therefore, although trace replay
    holds all historical labels, it cannot leak a current or future actual
    unplug into a controller decision.

    The bounded risk interval is an engineering safety choice: it prevents an
    abrupt run of observations from requesting an undefined finite-sample
    quantile.  This means the implementation is an ACI-inspired operational
    guard, not a claim of the unbounded theoretical ACI guarantee.  Its
    coverage must be reported empirically on a fresh period.
    """

    target_miscoverage: float
    learning_rate: float
    initial_scores: tuple[int, ...]
    calibration_sessions: int
    min_miscoverage: float = 0.01
    max_miscoverage: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.target_miscoverage < 1:
            raise ValueError("target_miscoverage must lie strictly between zero and one")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.initial_scores or self.calibration_sessions <= 0:
            raise ValueError("at least one calibration score is required")
        if any(score < 0 for score in self.initial_scores):
            raise ValueError("early-departure scores must be non-negative")
        if not 0 < self.min_miscoverage <= self.target_miscoverage:
            raise ValueError("min_miscoverage must be positive and at most the target")
        if not self.target_miscoverage <= self.max_miscoverage < 1:
            raise ValueError("max_miscoverage must be at least the target and below one")

    @classmethod
    def fit(
        cls,
        calibration_sessions: Sequence[EVSession],
        *,
        miscoverage: float = 0.10,
        learning_rate: float = 0.01,
        min_miscoverage: float = 0.01,
        max_miscoverage: float = 0.50,
    ) -> "AdaptiveEarlyDepartureGuard":
        """Fit the initial score bank from the declared calibration split only."""
        if not calibration_sessions:
            raise ValueError("at least one calibration session is required")
        scores = tuple(_early_departure_advance(session) for session in calibration_sessions)
        return cls(
            target_miscoverage=miscoverage,
            learning_rate=learning_rate,
            initial_scores=scores,
            calibration_sessions=len(scores),
            min_miscoverage=min_miscoverage,
            max_miscoverage=max_miscoverage,
        )

    def _guarded_deadline_step(self, session: EVSession, *, buffer_steps: int) -> int:
        if session.planning_departure_step is None:
            raise ValueError("a declared planning deadline is required for commitment guarding")
        declared = session.declared_departure_step or session.planning_departure_step
        return max(session.arrival_step + 1, declared - buffer_steps)

    def apply_causally(
        self, sessions: Sequence[EVSession]
    ) -> tuple[tuple[EVSession, ...], dict[str, object]]:
        """Guard a replay without using an outcome before it becomes observable.

        Departures at a control step are observed before plug-ins at that same
        step.  That matches the simulator's exclusive departure convention:
        an EV with ``departure_step == t`` is no longer active at step ``t``.
        Returned sessions preserve the caller's ordering, physical departure,
        and all energy fields.
        """
        if not sessions:
            raise ValueError("at least one session is required")
        if any(session.planning_departure_step is None for session in sessions):
            raise ValueError("sessions must have declared planning deadlines")

        ordered = sorted(enumerate(sessions), key=lambda item: (item[1].arrival_step, item[1].ev_id))
        guarded: list[EVSession | None] = [None] * len(sessions)
        scores = list(self.initial_scores)
        pending: list[tuple[int, str, int, EVSession]] = []
        predictions: dict[int, tuple[int, float, int]] = {}
        alpha = self.target_miscoverage
        alphas_at_decision: list[float] = []
        buffers_at_decision: list[int] = []
        updates_before_decisions = 0

        def observe_completed(until_step: int) -> None:
            nonlocal alpha, updates_before_decisions
            while pending and pending[0][0] <= until_step:
                _, _, index, completed = heapq.heappop(pending)
                deadline, _, _ = predictions[index]
                missed = int(completed.departure_step < deadline)
                # A miss lowers alpha, selecting a more conservative upper
                # quantile later; a covered outcome makes the guard gradually
                # less conservative.  The outcome is available only now.
                alpha = float(
                    np.clip(
                        alpha + self.learning_rate * (self.target_miscoverage - missed),
                        self.min_miscoverage,
                        self.max_miscoverage,
                    )
                )
                scores.append(_early_departure_advance(completed))
                updates_before_decisions += 1

        for index, session in ordered:
            observe_completed(session.arrival_step)
            buffer_steps = _conformal_buffer(scores, miscoverage=alpha)
            deadline = self._guarded_deadline_step(session, buffer_steps=buffer_steps)
            predictions[index] = (deadline, alpha, buffer_steps)
            alphas_at_decision.append(alpha)
            buffers_at_decision.append(buffer_steps)
            guarded[index] = EVSession(
                session.ev_id,
                session.station_id,
                session.arrival_step,
                session.departure_step,
                session.requested_energy_kwh,
                session.max_power_kw,
                session.delivered_energy_kwh,
                deadline,
                session.declared_departure_step or session.planning_deadline_step,
            )
            heapq.heappush(pending, (session.departure_step, session.ev_id, index, session))

        completed_guarded = tuple(item for item in guarded if item is not None)
        coverage = float(
            np.mean(
                [
                    session.departure_step >= session.planning_deadline_step
                    for session in completed_guarded
                ]
            )
        )
        audit = {
            **self.describe(),
            "empirical_coverage": coverage,
            "decisions": len(completed_guarded),
            "outcomes_observed_before_decisions": updates_before_decisions,
            "effective_miscoverage_at_decision": {
                "initial": alphas_at_decision[0],
                "minimum": float(min(alphas_at_decision)),
                "maximum": float(max(alphas_at_decision)),
                "mean": float(np.mean(alphas_at_decision)),
            },
            "buffer_steps_at_decision": _buffer_step_summary(buffers_at_decision),
            "causal_event_order": (
                "At each plug-in, use calibration scores and realized departures at or before "
                "that control step only; update after the current EV later physically departs."
            ),
        }
        return completed_guarded, audit

    def apply(self, sessions: Sequence[EVSession]) -> tuple[EVSession, ...]:
        """Return guarded replay sessions; see :meth:`apply_causally` for its audit."""
        guarded, _ = self.apply_causally(sessions)
        return guarded

    def empirical_coverage(self, sessions: Sequence[EVSession]) -> float:
        """Audit causal replay coverage; never expose it to the controller."""
        _, audit = self.apply_causally(sessions)
        return float(audit["empirical_coverage"])

    def describe(self) -> dict[str, object]:
        return {
            "method": "aci_inspired_past_only_one_sided_conformal_early_departure_guard",
            "target_miscoverage": self.target_miscoverage,
            "target_coverage": 1.0 - self.target_miscoverage,
            "learning_rate": self.learning_rate,
            "min_miscoverage": self.min_miscoverage,
            "max_miscoverage": self.max_miscoverage,
            "calibration_sessions": self.calibration_sessions,
            "coverage_scope": (
                "operational adaptive guard; empirical coverage must be audited on a fresh "
                "period, and is not a finite-sample, per-driver, or conditional guarantee"
            ),
        }


@dataclass(frozen=True)
class MultiRateAdaptiveEarlyDepartureGuard:
    """Conservative causal envelope over several ACI-inspired adaptation rates.

    A single ACI learning rate can be too slow for an abrupt shift or too fast
    for ordinary noise.  This guard maintains several independent past-only
    adaptive guards.  At each plug-in it applies the *earliest* deadline
    proposed by any expert, equivalently the largest currently proposed
    early-departure buffer.  This is a safety-first expert envelope, not the
    learned AgACI aggregation algorithm: it is intentionally transparent and
    has no hidden weight-training or test-period hyperparameter selection.

    Every expert receives the same calibration sessions and only outcomes that
    are physically observable before the decision.  Taking the earliest of
    these causal expert deadlines is itself causal.  The approach may reduce
    energy flexibility, so it must earn its additional conservatism in a
    predeclared service-versus-coverage ablation.
    """

    experts: tuple[AdaptiveEarlyDepartureGuard, ...]

    def __post_init__(self) -> None:
        if len(self.experts) < 2:
            raise ValueError("multi-rate guard requires at least two experts")
        rates = [expert.learning_rate for expert in self.experts]
        if len(set(rates)) != len(rates):
            raise ValueError("multi-rate guard learning rates must be distinct")
        first = self.experts[0]
        if any(
            expert.target_miscoverage != first.target_miscoverage
            or expert.initial_scores != first.initial_scores
            or expert.min_miscoverage != first.min_miscoverage
            or expert.max_miscoverage != first.max_miscoverage
            for expert in self.experts[1:]
        ):
            raise ValueError("multi-rate experts must share a calibration score bank and bounds")

    @classmethod
    def fit(
        cls,
        calibration_sessions: Sequence[EVSession],
        *,
        miscoverage: float = 0.10,
        learning_rates: Sequence[float] = (0.001, 0.005, 0.01, 0.02),
        min_miscoverage: float = 0.01,
        max_miscoverage: float = 0.50,
    ) -> "MultiRateAdaptiveEarlyDepartureGuard":
        """Create fixed-rate causal experts from the calibration split only."""
        rates = tuple(float(value) for value in learning_rates)
        if len(rates) < 2 or len(set(rates)) != len(rates):
            raise ValueError("learning_rates must contain at least two distinct values")
        return cls(
            tuple(
                AdaptiveEarlyDepartureGuard.fit(
                    calibration_sessions,
                    miscoverage=miscoverage,
                    learning_rate=rate,
                    min_miscoverage=min_miscoverage,
                    max_miscoverage=max_miscoverage,
                )
                for rate in rates
            )
        )

    @property
    def target_miscoverage(self) -> float:
        return self.experts[0].target_miscoverage

    @property
    def calibration_sessions(self) -> int:
        return self.experts[0].calibration_sessions

    def apply_causally(
        self, sessions: Sequence[EVSession]
    ) -> tuple[tuple[EVSession, ...], dict[str, object]]:
        """Apply the earliest causally proposed expert deadline at each plug-in."""
        if not sessions:
            raise ValueError("at least one session is required")
        expert_runs = [expert.apply_causally(sessions) for expert in self.experts]
        expert_sessions = [run[0] for run in expert_runs]
        combined: list[EVSession] = []
        selected_counts = {str(expert.learning_rate): 0 for expert in self.experts}
        buffers: list[int] = []
        for index, original in enumerate(sessions):
            candidate_index, selected = min(
                enumerate(guarded[index] for guarded in expert_sessions),
                key=lambda item: item[1].planning_deadline_step,
            )
            selected_counts[str(self.experts[candidate_index].learning_rate)] += 1
            declared = original.declared_departure_step or original.planning_departure_step
            if declared is None:
                raise ValueError("sessions must have declared planning deadlines")
            buffers.append(max(0, declared - selected.planning_deadline_step))
            combined.append(
                EVSession(
                    original.ev_id,
                    original.station_id,
                    original.arrival_step,
                    original.departure_step,
                    original.requested_energy_kwh,
                    original.max_power_kw,
                    original.delivered_energy_kwh,
                    selected.planning_deadline_step,
                    declared,
                )
            )
        coverage = float(
            np.mean(
                [session.departure_step >= session.planning_deadline_step for session in combined]
            )
        )
        audit = {
            **self.describe(),
            "empirical_coverage": coverage,
            "decisions": len(combined),
            "selected_expert_decision_counts": selected_counts,
            "buffer_steps_at_decision": _buffer_step_summary(buffers),
            "expert_audits": {
                str(expert.learning_rate): audit for expert, (_, audit) in zip(
                    self.experts, expert_runs, strict=True
                )
            },
            "causal_event_order": (
                "Each expert uses calibration scores and realized departures at or before "
                "the plug-in step only; the controller chooses the earliest of their causal deadlines."
            ),
        }
        return tuple(combined), audit

    def apply(self, sessions: Sequence[EVSession]) -> tuple[EVSession, ...]:
        guarded, _ = self.apply_causally(sessions)
        return guarded

    def empirical_coverage(self, sessions: Sequence[EVSession]) -> float:
        _, audit = self.apply_causally(sessions)
        return float(audit["empirical_coverage"])

    def describe(self) -> dict[str, object]:
        return {
            "method": "multi_rate_conservative_aci_inspired_early_departure_envelope",
            "target_miscoverage": self.target_miscoverage,
            "target_coverage": 1.0 - self.target_miscoverage,
            "learning_rates": [expert.learning_rate for expert in self.experts],
            "calibration_sessions": self.calibration_sessions,
            "coverage_scope": (
                "operational conservative expert envelope; not AgACI itself and not a "
                "finite-sample, per-driver, or conditional coverage guarantee"
            ),
        }


@dataclass(frozen=True)
class AgACIWeightedEarlyDepartureGuard:
    """Past-only AgACI-style aggregation for one-sided EV departure guards.

    The previous multi-rate mode deliberately chooses the most conservative
    expert deadline.  This separate candidate instead follows the online
    aggregation mechanics of AgACI: each fixed-rate ACI expert maintains its
    own effective miscoverage level and a Bernstein-style online weight; the
    controller receives the weighted aggregate miscoverage level and then one
    one-sided conformal deadline.

    The published AgACI update works with sequential conformal errors.  Here
    it is mapped to a discrete one-sided early-unplug score: an expert error is
    whether the EV physically unplugged before that expert's causal deadline.
    We retain bounded miscoverage levels because an out-of-range quantile is
    undefined for the finite score bank used by this operational controller.
    Thus this is a faithful *AgACI-style update*, not a claim that the original
    paper's theory transfers unchanged to per-EV delayed outcomes.

    Every update is made only when a physical unplug is observable.  The
    algorithm must therefore be evaluated on a fresh declared period; neither
    its coverage nor its weights may be tuned after reading that test period.
    """

    target_miscoverage: float
    learning_rates: tuple[float, ...]
    initial_scores: tuple[int, ...]
    calibration_sessions: int
    min_miscoverage: float = 0.01
    max_miscoverage: float = 0.50
    epsilon: float = 0.001

    def __post_init__(self) -> None:
        if not 0 < self.target_miscoverage < 1:
            raise ValueError("target_miscoverage must lie strictly between zero and one")
        if len(self.learning_rates) < 2 or len(set(self.learning_rates)) != len(
            self.learning_rates
        ):
            raise ValueError("AgACI requires at least two distinct learning rates")
        if any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("AgACI learning rates must be positive")
        if not self.initial_scores or self.calibration_sessions <= 0:
            raise ValueError("at least one calibration score is required")
        if any(score < 0 for score in self.initial_scores):
            raise ValueError("early-departure scores must be non-negative")
        if not 0 < self.min_miscoverage <= self.target_miscoverage:
            raise ValueError("min_miscoverage must be positive and at most the target")
        if not self.target_miscoverage <= self.max_miscoverage < 1:
            raise ValueError("max_miscoverage must be at least the target and below one")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @classmethod
    def fit(
        cls,
        calibration_sessions: Sequence[EVSession],
        *,
        miscoverage: float = 0.10,
        learning_rates: Sequence[float] = (0.001, 0.005, 0.01, 0.02),
        min_miscoverage: float = 0.01,
        max_miscoverage: float = 0.50,
        epsilon: float = 0.001,
    ) -> "AgACIWeightedEarlyDepartureGuard":
        """Fit the common score bank using the declared calibration split only."""
        if not calibration_sessions:
            raise ValueError("at least one calibration session is required")
        rates = tuple(float(value) for value in learning_rates)
        return cls(
            target_miscoverage=miscoverage,
            learning_rates=rates,
            initial_scores=tuple(_early_departure_advance(session) for session in calibration_sessions),
            calibration_sessions=len(calibration_sessions),
            min_miscoverage=min_miscoverage,
            max_miscoverage=max_miscoverage,
            epsilon=epsilon,
        )

    def _guarded_deadline_step(self, session: EVSession, *, buffer_steps: int) -> int:
        if session.planning_departure_step is None:
            raise ValueError("a declared planning deadline is required for commitment guarding")
        declared = session.declared_departure_step or session.planning_departure_step
        return max(session.arrival_step + 1, declared - buffer_steps)

    def apply_causally(
        self, sessions: Sequence[EVSession]
    ) -> tuple[tuple[EVSession, ...], dict[str, object]]:
        """Run weighted expert aggregation with strictly past physical outcomes."""
        if not sessions:
            raise ValueError("at least one session is required")
        if any(session.planning_departure_step is None for session in sessions):
            raise ValueError("sessions must have declared planning deadlines")

        expert_count = len(self.learning_rates)
        rates = np.asarray(self.learning_rates, dtype=float)
        scores = list(self.initial_scores)
        expert_alphas = np.full(expert_count, self.target_miscoverage, dtype=float)
        expert_probabilities = np.full(expert_count, 1.0 / expert_count, dtype=float)
        expert_square_losses = np.zeros(expert_count, dtype=float)
        expert_l_values = np.zeros(expert_count, dtype=float)
        expert_max_losses = np.zeros(expert_count, dtype=float)
        expert_etas = np.zeros(expert_count, dtype=float)

        ordered = sorted(enumerate(sessions), key=lambda item: (item[1].arrival_step, item[1].ev_id))
        guarded: list[EVSession | None] = [None] * len(sessions)
        # arrival-time prediction state, retained only until its physical unplug
        pending: list[tuple[int, str, int, EVSession]] = []
        predictions: dict[int, tuple[int, tuple[int, ...], float, tuple[float, ...]]] = {}
        aggregate_alphas: list[float] = []
        aggregate_buffers: list[int] = []
        probability_history: list[np.ndarray] = []
        updates_before_decisions = 0

        def observe_completed(until_step: int) -> None:
            nonlocal expert_alphas, expert_probabilities, expert_etas
            nonlocal expert_square_losses, expert_l_values, expert_max_losses
            nonlocal updates_before_decisions
            while pending and pending[0][0] <= until_step:
                _, _, index, completed = heapq.heappop(pending)
                aggregate_deadline, expert_deadlines, aggregate_alpha, alpha_snapshot = predictions[index]
                aggregate_missed = int(completed.departure_step < aggregate_deadline)
                expert_missed = np.asarray(
                    [completed.departure_step < deadline for deadline in expert_deadlines], dtype=float
                )
                # This is the AgACI loss from the reference implementation:
                # (aggregate error - target alpha) * (expert alpha - aggregate alpha).
                losses = (aggregate_missed - self.target_miscoverage) * (
                    np.asarray(alpha_snapshot, dtype=float) - aggregate_alpha
                )
                expert_square_losses += losses**2
                expert_max_losses = np.maximum(expert_max_losses, np.abs(losses))
                e_values = 2.0 ** (np.ceil(np.log2(expert_max_losses + self.epsilon)) + 1.0)
                expert_l_values += 0.5 * (
                    losses * (1.0 + expert_etas * losses)
                    + e_values * (expert_etas * losses > 0.5)
                )
                sqrt_terms = np.full(expert_count, np.inf, dtype=float)
                positive = expert_square_losses > 0.0
                sqrt_terms[positive] = np.sqrt(
                    np.log(expert_count) / expert_square_losses[positive]
                )
                expert_etas = np.minimum(1.0 / e_values, sqrt_terms)
                # The original ACI expert update is applied after the outcome
                # becomes observable. Clipping is the explicitly documented
                # finite-score-bank engineering safeguard.
                expert_alphas = np.clip(
                    expert_alphas + rates * (self.target_miscoverage - expert_missed),
                    self.min_miscoverage,
                    self.max_miscoverage,
                )
                log_weights = np.log(expert_etas) - expert_etas * expert_l_values
                stabilized = np.exp(log_weights - np.max(log_weights))
                if not np.isfinite(stabilized).all() or float(stabilized.sum()) <= 0.0:
                    expert_probabilities = np.full(expert_count, 1.0 / expert_count, dtype=float)
                else:
                    expert_probabilities = stabilized / stabilized.sum()
                scores.append(_early_departure_advance(completed))
                updates_before_decisions += 1

        for index, session in ordered:
            observe_completed(session.arrival_step)
            aggregate_alpha = float(np.dot(expert_probabilities, expert_alphas))
            expert_buffers = tuple(
                _conformal_buffer(scores, miscoverage=float(alpha)) for alpha in expert_alphas
            )
            aggregate_buffer = _conformal_buffer(scores, miscoverage=aggregate_alpha)
            expert_deadlines = tuple(
                self._guarded_deadline_step(session, buffer_steps=buffer)
                for buffer in expert_buffers
            )
            aggregate_deadline = self._guarded_deadline_step(
                session, buffer_steps=aggregate_buffer
            )
            predictions[index] = (
                aggregate_deadline,
                expert_deadlines,
                aggregate_alpha,
                tuple(float(alpha) for alpha in expert_alphas),
            )
            aggregate_alphas.append(aggregate_alpha)
            aggregate_buffers.append(aggregate_buffer)
            probability_history.append(expert_probabilities.copy())
            declared = session.declared_departure_step or session.planning_departure_step
            guarded[index] = EVSession(
                session.ev_id,
                session.station_id,
                session.arrival_step,
                session.departure_step,
                session.requested_energy_kwh,
                session.max_power_kw,
                session.delivered_energy_kwh,
                aggregate_deadline,
                declared,
            )
            heapq.heappush(pending, (session.departure_step, session.ev_id, index, session))

        completed_guarded = tuple(item for item in guarded if item is not None)
        coverage = float(
            np.mean(
                [
                    session.departure_step >= session.planning_deadline_step
                    for session in completed_guarded
                ]
            )
        )
        mean_probabilities = np.mean(np.vstack(probability_history), axis=0)
        audit = {
            **self.describe(),
            "empirical_coverage": coverage,
            "decisions": len(completed_guarded),
            "outcomes_observed_before_decisions": updates_before_decisions,
            "effective_miscoverage_at_decision": {
                "initial": aggregate_alphas[0],
                "minimum": float(min(aggregate_alphas)),
                "maximum": float(max(aggregate_alphas)),
                "mean": float(np.mean(aggregate_alphas)),
            },
            "buffer_steps_at_decision": _buffer_step_summary(aggregate_buffers),
            "mean_expert_probabilities_at_decision": {
                str(rate): float(weight)
                for rate, weight in zip(self.learning_rates, mean_probabilities, strict=True)
            },
            "final_expert_probabilities": {
                str(rate): float(weight)
                for rate, weight in zip(self.learning_rates, expert_probabilities, strict=True)
            },
            "final_expert_miscoverages": {
                str(rate): float(alpha)
                for rate, alpha in zip(self.learning_rates, expert_alphas, strict=True)
            },
            "causal_event_order": (
                "At each plug-in, aggregate only experts fitted on calibration scores and "
                "realized departures observed at or before that step; update weights only after "
                "the current EV later physically departs."
            ),
        }
        return completed_guarded, audit

    def apply(self, sessions: Sequence[EVSession]) -> tuple[EVSession, ...]:
        guarded, _ = self.apply_causally(sessions)
        return guarded

    def empirical_coverage(self, sessions: Sequence[EVSession]) -> float:
        _, audit = self.apply_causally(sessions)
        return float(audit["empirical_coverage"])

    def describe(self) -> dict[str, object]:
        return {
            "method": "bounded_agaci_style_weighted_aci_one_sided_early_departure_guard",
            "target_miscoverage": self.target_miscoverage,
            "target_coverage": 1.0 - self.target_miscoverage,
            "learning_rates": list(self.learning_rates),
            "min_miscoverage": self.min_miscoverage,
            "max_miscoverage": self.max_miscoverage,
            "epsilon": self.epsilon,
            "calibration_sessions": self.calibration_sessions,
            "aggregation_rule": "AgACI-style Bernstein online aggregation over ACI experts",
            "coverage_scope": (
                "operational bounded AgACI-style guard; empirical coverage must be audited on "
                "a fresh period and is not a finite-sample, per-driver, conditional, or "
                "cross-site guarantee"
            ),
        }


def declared_duration_group(session: EVSession) -> str:
    """Classify a commitment using only plug-in-time information.

    These coarse bins are intentionally interpretable and fixed before the
    replication-period evaluation: at most four hours, four to eight hours,
    and more than eight hours. They use the original declared deadline when a
    prior guard has made the controller's working deadline earlier.
    """
    declared = session.declared_departure_step or session.planning_departure_step
    if declared is None:
        raise ValueError("a declared planning deadline is required for duration stratification")
    duration_steps = declared - session.arrival_step
    if duration_steps <= 16:
        return "declared_duration_le_4h"
    if duration_steps <= 32:
        return "declared_duration_4_to_8h"
    return "declared_duration_gt_8h"


@dataclass(frozen=True)
class DurationStratifiedEarlyDepartureGuard:
    """Predeclared duration-stratified conformal guard with global fallback.

    For groups with at least ``min_group_calibration_sessions`` calibration
    examples, the guard uses a group-specific split-conformal order statistic.
    Smaller groups use the global guard. This is deliberately a small,
    interpretable ablation: every buffer has a transparent sample count, and
    no user identity, future actual departure, or test outcome is used.

    Coverage is group-conditional only under exchangeability within a group.
    The fallback group retains the global marginal interpretation; it must not
    be advertised as an individual or universal conditional guarantee.
    """

    global_guard: EarlyDepartureGuard
    buffer_steps_by_group: Mapping[str, int]
    calibration_sessions_by_group: Mapping[str, int]
    uses_global_fallback_by_group: Mapping[str, bool]
    min_group_calibration_sessions: int

    def __post_init__(self) -> None:
        if self.min_group_calibration_sessions <= 0:
            raise ValueError("min_group_calibration_sessions must be positive")
        expected = {
            "declared_duration_le_4h",
            "declared_duration_4_to_8h",
            "declared_duration_gt_8h",
        }
        if set(self.buffer_steps_by_group) != expected:
            raise ValueError("duration guard must declare every pre-specified duration group")
        if set(self.calibration_sessions_by_group) != expected:
            raise ValueError("duration guard must record every group sample count")
        if set(self.uses_global_fallback_by_group) != expected:
            raise ValueError("duration guard must record every group fallback status")
        if any(value < 0 for value in self.buffer_steps_by_group.values()):
            raise ValueError("duration guard buffers must be non-negative")

    @classmethod
    def fit(
        cls,
        calibration_sessions: Sequence[EVSession],
        *,
        miscoverage: float = 0.10,
        min_group_calibration_sessions: int = 50,
    ) -> "DurationStratifiedEarlyDepartureGuard":
        """Fit global and sufficiently supported duration-group quantiles."""
        if min_group_calibration_sessions <= 0:
            raise ValueError("min_group_calibration_sessions must be positive")
        global_guard = EarlyDepartureGuard.fit(
            calibration_sessions, miscoverage=miscoverage
        )
        groups = (
            "declared_duration_le_4h",
            "declared_duration_4_to_8h",
            "declared_duration_gt_8h",
        )
        grouped = {
            group: tuple(
                session
                for session in calibration_sessions
                if declared_duration_group(session) == group
            )
            for group in groups
        }
        counts = {group: len(sessions) for group, sessions in grouped.items()}
        fallbacks = {
            group: counts[group] < min_group_calibration_sessions for group in groups
        }
        buffers = {
            group: (
                global_guard.buffer_steps
                if fallbacks[group]
                else EarlyDepartureGuard.fit(
                    grouped[group], miscoverage=miscoverage
                ).buffer_steps
            )
            for group in groups
        }
        return cls(global_guard, buffers, counts, fallbacks, min_group_calibration_sessions)

    @property
    def miscoverage(self) -> float:
        return self.global_guard.miscoverage

    @property
    def calibration_sessions(self) -> int:
        return self.global_guard.calibration_sessions

    def buffer_steps_for(self, session: EVSession) -> int:
        return int(self.buffer_steps_by_group[declared_duration_group(session)])

    def guarded_deadline_step(self, session: EVSession) -> int:
        if session.planning_departure_step is None:
            raise ValueError("a declared planning deadline is required for commitment guarding")
        return max(
            session.arrival_step + 1,
            session.planning_deadline_step - self.buffer_steps_for(session),
        )

    def apply(self, sessions: Sequence[EVSession]) -> tuple[EVSession, ...]:
        """Return causal guarded copies while retaining physical replay facts."""
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

    def empirical_coverage(self, sessions: Sequence[EVSession]) -> float:
        if not sessions:
            raise ValueError("at least one session is required")
        guarded = self.apply(sessions)
        return float(
            np.mean(
                [session.departure_step >= session.planning_deadline_step for session in guarded]
            )
        )

    def empirical_coverage_by_group(self, sessions: Sequence[EVSession]) -> dict[str, float]:
        """Provide descriptive held-out coverage by the fixed input groups."""
        if not sessions:
            raise ValueError("at least one session is required")
        guarded = self.apply(sessions)
        values: dict[str, list[bool]] = {
            "declared_duration_le_4h": [],
            "declared_duration_4_to_8h": [],
            "declared_duration_gt_8h": [],
        }
        for original, guarded_session in zip(sessions, guarded, strict=True):
            values[declared_duration_group(original)].append(
                guarded_session.departure_step >= guarded_session.planning_deadline_step
            )
        return {
            group: float(np.mean(covered)) if covered else float("nan")
            for group, covered in values.items()
        }

    def describe(self) -> dict[str, object]:
        return {
            "method": "duration_stratified_one_sided_split_conformal_early_departure_guard",
            "miscoverage": self.miscoverage,
            "target_coverage": 1.0 - self.miscoverage,
            "global_buffer_steps": self.global_guard.buffer_steps,
            "calibration_sessions": self.calibration_sessions,
            "min_group_calibration_sessions": self.min_group_calibration_sessions,
            "buffer_steps_by_group": dict(self.buffer_steps_by_group),
            "calibration_sessions_by_group": dict(self.calibration_sessions_by_group),
            "uses_global_fallback_by_group": dict(self.uses_global_fallback_by_group),
            "coverage_scope": (
                "group-conditional only under within-group exchangeability for non-fallback "
                "groups; global marginal for fallback groups"
            ),
        }
