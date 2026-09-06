"""Causal, calibrated early-unplug buffers for FairFlex V3.

The guard predicts only a conservative *planning buffer* from information that
is observable when a session plugs in.  The current session's realized
departure is never used as a feature; it remains a post-departure label and a
physical simulation event.  The method is one-sided split conformalized
quantile regression: a high quantile regressor is fitted on an earlier training
window and receives one residual correction from a later calibration window.

This implementation reports empirical future-period coverage.  It does not
claim per-driver or conditional coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..domain import EVSession


def _declared_deadline(session: EVSession) -> int:
    deadline = session.declared_departure_step or session.planning_departure_step
    if deadline is None:
        raise ValueError("CQR guarding requires an explicit declared planning deadline")
    return deadline


def early_unplug_advance_steps(session: EVSession) -> int:
    """One-sided label: how many steps earlier physical unplug occurred."""
    return max(0, _declared_deadline(session) - session.departure_step)


def conservative_deadline_envelope(
    *guarded_sets: Sequence[EVSession],
) -> tuple[tuple[EVSession, ...], dict[str, object]]:
    """Combine causal guards by retaining their earliest planning deadline.

    This is a deployable *AND* safety envelope: each input guard already makes
    its deadline from allowed information, and the combined controller uses the
    most conservative of those outputs.  Realized departures are read only for
    the post-hoc coverage diagnostic, never to choose a deadline.
    """
    if len(guarded_sets) < 2:
        raise ValueError("at least two guarded session sets are required")
    reference = tuple(guarded_sets[0])
    if not reference:
        raise ValueError("guarded session sets must be non-empty")
    if any(len(candidate) != len(reference) for candidate in guarded_sets[1:]):
        raise ValueError("all guarded session sets must have the same length")

    combined: list[EVSession] = []
    source_wins = [0] * len(guarded_sets)
    covered: list[bool] = []
    for index, source_sessions in enumerate(zip(*guarded_sets, strict=True)):
        base = source_sessions[0]
        if any(
            candidate.ev_id != base.ev_id
            or candidate.arrival_step != base.arrival_step
            or candidate.departure_step != base.departure_step
            or _declared_deadline(candidate) != _declared_deadline(base)
            for candidate in source_sessions[1:]
        ):
            raise ValueError("guarded session sets do not describe the same physical sessions")
        deadlines = np.asarray(
            [candidate.effective_planning_deadline_step(base.arrival_step) for candidate in source_sessions],
            dtype=int,
        )
        selected_index = int(np.argmin(deadlines))
        source_wins[selected_index] += 1
        deadline = int(deadlines[selected_index])
        combined.append(
            EVSession(
                base.ev_id,
                base.station_id,
                base.arrival_step,
                base.departure_step,
                base.requested_energy_kwh,
                base.max_power_kw,
                base.delivered_energy_kwh,
                deadline,
                _declared_deadline(base),
            )
        )
        covered.append(base.departure_step >= deadline)
    return tuple(combined), {
        "method": "causal_conservative_deadline_envelope",
        "empirical_coverage": float(np.mean(covered)),
        "selected_earliest_deadline_count_by_input": source_wins,
        "coverage_scope": "empirical future-period marginal audit; not a per-driver guarantee",
    }


def _upper_conformal_quantile(values: np.ndarray, *, miscoverage: float) -> float:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("at least one one-dimensional calibration score is required")
    if not 0 < miscoverage < 1:
        raise ValueError("miscoverage must lie strictly between zero and one")
    rank = min(values.size, ceil((values.size + 1) * (1.0 - miscoverage)))
    return float(np.partition(values, rank - 1)[rank - 1])


def _quantile_pinball_loss(
    outcomes: np.ndarray,
    predictions: np.ndarray,
    *,
    quantile: float,
) -> float:
    """Return the standard asymmetric loss for a conditional quantile forecast."""
    if outcomes.shape != predictions.shape:
        raise ValueError("outcomes and predictions must have matching shapes")
    residual = outcomes - predictions
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


@dataclass(frozen=True)
class CQRFeatureSchema:
    """Fixed, non-identity feature schema learned solely from train sessions."""

    station_ids: tuple[str, ...]
    step_minutes: int

    def __post_init__(self) -> None:
        if not self.station_ids or len(set(self.station_ids)) != len(self.station_ids):
            raise ValueError("station_ids must be a non-empty unique sequence")
        if self.step_minutes <= 0 or 1440 % self.step_minutes:
            raise ValueError("step_minutes must be a positive divisor of one day")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "requested_energy_kwh",
            "max_power_kw",
            "declared_duration_steps",
            "arrival_hour_sin",
            "arrival_hour_cos",
            "weekday_sin",
            "weekday_cos",
            "active_session_count",
            "active_same_station_count",
            *(f"station_is_{station}" for station in self.station_ids),
        )

    def matrix(
        self,
        sessions: Sequence[EVSession],
        *,
        origin: pd.Timestamp | str,
    ) -> np.ndarray:
        """Build plug-in-time features without the target's realized departure.

        A connected EV count is a valid state observation: at a new plug-in the
        controller can see which previously or simultaneously arrived EVs are
        still physically connected.  The target session itself is excluded,
        and no future delivery/departure value is included as a feature.
        """
        if not sessions:
            raise ValueError("at least one session is required")
        origin_time = pd.to_datetime(origin, utc=True)
        if pd.isna(origin_time):
            raise ValueError("origin must be a valid timestamp")
        station_index = {station: index for index, station in enumerate(self.station_ids)}
        rows: list[list[float]] = []
        for session in sessions:
            declared = _declared_deadline(session)
            arrival_time = origin_time + pd.Timedelta(minutes=self.step_minutes * session.arrival_step)
            hour_fraction = (arrival_time.hour + arrival_time.minute / 60.0) / 24.0
            weekday_fraction = arrival_time.dayofweek / 7.0
            active = [
                other
                for other in sessions
                if other.ev_id != session.ev_id
                and other.arrival_step <= session.arrival_step < other.departure_step
            ]
            row = [
                float(session.requested_energy_kwh),
                float(session.max_power_kw),
                float(declared - session.arrival_step),
                float(np.sin(2.0 * np.pi * hour_fraction)),
                float(np.cos(2.0 * np.pi * hour_fraction)),
                float(np.sin(2.0 * np.pi * weekday_fraction)),
                float(np.cos(2.0 * np.pi * weekday_fraction)),
                float(len(active)),
                float(sum(other.station_id == session.station_id for other in active)),
            ]
            row.extend(float(index == station_index.get(session.station_id, -1)) for index in range(len(self.station_ids)))
            rows.append(row)
        matrix = np.asarray(rows, dtype=float)
        if matrix.shape != (len(sessions), len(self.feature_names)):
            raise RuntimeError("CQR feature matrix shape does not match its schema")
        return matrix


@dataclass(frozen=True)
class CausalCQRBufferGuard:
    """One-sided conformalized quantile buffer for declared EV departures."""

    miscoverage: float
    schema: CQRFeatureSchema
    model: HistGradientBoostingRegressor
    conformal_correction_steps: float
    train_sessions: int
    calibration_sessions: int

    def __post_init__(self) -> None:
        if not 0 < self.miscoverage < 1:
            raise ValueError("miscoverage must lie strictly between zero and one")
        if self.train_sessions <= 0 or self.calibration_sessions <= 0:
            raise ValueError("train_sessions and calibration_sessions must be positive")
        if not np.isfinite(self.conformal_correction_steps):
            raise ValueError("conformal correction must be finite")

    @classmethod
    def fit(
        cls,
        train_sessions: Sequence[EVSession],
        calibration_sessions: Sequence[EVSession],
        *,
        train_origin: pd.Timestamp | str,
        calibration_origin: pd.Timestamp | str,
        step_minutes: int,
        miscoverage: float = 0.10,
        min_train_sessions: int = 50,
        min_calibration_sessions: int = 30,
        max_iter: int = 150,
        min_samples_leaf: int = 12,
        random_state: int = 20260903,
    ) -> "CausalCQRBufferGuard":
        """Fit on an earlier window and calibrate on a later distinct window."""
        if not 0 < miscoverage < 1:
            raise ValueError("miscoverage must lie strictly between zero and one")
        if len(train_sessions) < min_train_sessions:
            raise ValueError("insufficient train sessions for the declared CQR minimum")
        if len(calibration_sessions) < min_calibration_sessions:
            raise ValueError("insufficient calibration sessions for the declared CQR minimum")
        all_sessions = tuple(train_sessions) + tuple(calibration_sessions)
        if any((session.declared_departure_step or session.planning_departure_step) is None for session in all_sessions):
            raise ValueError("all CQR sessions require explicit declared deadlines")
        schema = CQRFeatureSchema(
            tuple(sorted({session.station_id for session in train_sessions})),
            step_minutes,
        )
        x_train = schema.matrix(train_sessions, origin=train_origin)
        y_train = np.asarray([early_unplug_advance_steps(session) for session in train_sessions], dtype=float)
        leaf = max(1, min(int(min_samples_leaf), len(train_sessions)))
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=1.0 - miscoverage,
            max_iter=int(max_iter),
            min_samples_leaf=leaf,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=int(random_state),
        )
        model.fit(x_train, y_train)
        x_calibration = schema.matrix(calibration_sessions, origin=calibration_origin)
        predicted_upper = np.asarray(model.predict(x_calibration), dtype=float)
        residuals = np.asarray(
            [early_unplug_advance_steps(session) for session in calibration_sessions], dtype=float
        ) - predicted_upper
        correction = _upper_conformal_quantile(residuals, miscoverage=miscoverage)
        return cls(
            miscoverage=miscoverage,
            schema=schema,
            model=model,
            conformal_correction_steps=correction,
            train_sessions=len(train_sessions),
            calibration_sessions=len(calibration_sessions),
        )

    def buffer_steps(
        self,
        sessions: Sequence[EVSession],
        *,
        origin: pd.Timestamp | str,
    ) -> np.ndarray:
        """Return non-negative integer buffers using no current-session outcome."""
        _, _, buffers = self._prediction_arrays(sessions, origin=origin)
        return buffers

    def _prediction_arrays(
        self,
        sessions: Sequence[EVSession],
        *,
        origin: pd.Timestamp | str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw quantiles, corrected quantiles, and integer buffers.

        The three arrays are all produced from deployable features only.  The
        helper deliberately has no access to the current sessions' realized
        departures; callers add those outcomes only after replay for auditing.
        """
        features = self.schema.matrix(sessions, origin=origin)
        raw_upper_quantiles = np.asarray(self.model.predict(features), dtype=float)
        corrected_upper_quantiles = raw_upper_quantiles + self.conformal_correction_steps
        buffers = np.maximum(
            0,
            np.ceil(corrected_upper_quantiles - 1e-12).astype(int),
        )
        return raw_upper_quantiles, corrected_upper_quantiles, buffers

    def prediction_audit_rows(
        self,
        sessions: Sequence[EVSession],
        *,
        origin: pd.Timestamp | str,
    ) -> list[dict[str, object]]:
        """Return post-hoc, per-session reliability rows for a fresh replay.

        These rows are an evaluation artifact.  Actual early-unplug advance and
        coverage are written only after a physical replay has completed; they
        must never be sent to a controller or used to choose a policy.
        """
        if not sessions:
            raise ValueError("at least one session is required")
        raw_upper, corrected_upper, buffers = self._prediction_arrays(sessions, origin=origin)
        origin_time = pd.to_datetime(origin, utc=True)
        rows: list[dict[str, object]] = []
        for session, raw, corrected, buffer in zip(
            sessions, raw_upper, corrected_upper, buffers, strict=True
        ):
            declared = _declared_deadline(session)
            deadline = max(session.arrival_step + 1, declared - int(buffer))
            arrival_time = origin_time + pd.Timedelta(minutes=self.schema.step_minutes * session.arrival_step)
            advance = early_unplug_advance_steps(session)
            rows.append({
                "ev_id": session.ev_id,
                "calendar_day": arrival_time.date().isoformat(),
                "arrival_step": session.arrival_step,
                "declared_departure_step": declared,
                "realized_departure_step": session.departure_step,
                "actual_early_unplug_steps": advance,
                "raw_upper_quantile_steps": float(raw),
                "conformal_upper_quantile_steps": float(corrected),
                "integer_cqr_buffer_steps": int(buffer),
                "cqr_planning_deadline_step": deadline,
                "cqr_covered": bool(session.departure_step >= deadline),
            })
        return rows

    def apply_causally(
        self,
        sessions: Sequence[EVSession],
        *,
        origin: pd.Timestamp | str,
    ) -> tuple[tuple[EVSession, ...], dict[str, object]]:
        if not sessions:
            raise ValueError("at least one session is required")
        raw_upper_quantiles, _, buffers = self._prediction_arrays(sessions, origin=origin)
        guarded: list[EVSession] = []
        covered: list[bool] = []
        for session, buffer_steps in zip(sessions, buffers, strict=True):
            declared = _declared_deadline(session)
            deadline = max(session.arrival_step + 1, declared - int(buffer_steps))
            guarded.append(
                EVSession(
                    session.ev_id,
                    session.station_id,
                    session.arrival_step,
                    session.departure_step,
                    session.requested_energy_kwh,
                    session.max_power_kw,
                    session.delivered_energy_kwh,
                    deadline,
                    declared,
                )
            )
            covered.append(session.departure_step >= deadline)
        buffer_array = np.asarray(buffers, dtype=float)
        actual_advances = np.asarray(
            [early_unplug_advance_steps(session) for session in sessions], dtype=float
        )
        underbuffer_steps = np.maximum(actual_advances - buffer_array, 0.0)
        overbuffer_steps = np.maximum(buffer_array - actual_advances, 0.0)
        audit = {
            **self.describe(),
            "decisions": len(sessions),
            "empirical_coverage": float(np.mean(covered)),
            "raw_upper_quantile_pinball_loss": _quantile_pinball_loss(
                actual_advances,
                raw_upper_quantiles,
                quantile=1.0 - self.miscoverage,
            ),
            "mean_actual_early_unplug_steps": float(actual_advances.mean()),
            "mean_raw_upper_quantile_steps": float(raw_upper_quantiles.mean()),
            "mean_uncovered_early_unplug_steps": float(underbuffer_steps.mean()),
            "mean_conservative_buffer_excess_steps": float(overbuffer_steps.mean()),
            "diagnostic_scope": (
                "These prediction diagnostics are computed after physical replay for evaluation only; "
                "they are never controller inputs."
            ),
            "buffer_steps": {
                "minimum": int(buffer_array.min()),
                "median": float(np.quantile(buffer_array, 0.5)),
                "p90": float(np.quantile(buffer_array, 0.9)),
                "maximum": int(buffer_array.max()),
                "mean": float(buffer_array.mean()),
            },
            "causal_feature_contract": (
                "Features contain declarations, plug-in calendar context, station and observable "
                "connected-session state only. The current session's realized departure is not a feature."
            ),
        }
        return tuple(guarded), audit

    def describe(self) -> dict[str, object]:
        return {
            "method": "one_sided_split_conformalized_quantile_regression_early_unplug_buffer",
            "target_coverage": 1.0 - self.miscoverage,
            "miscoverage": self.miscoverage,
            "conformal_correction_steps": self.conformal_correction_steps,
            "train_sessions": self.train_sessions,
            "calibration_sessions": self.calibration_sessions,
            "feature_names": list(self.schema.feature_names),
            "coverage_scope": "empirical future-period marginal and predeclared-group audit; not per-driver conditional coverage",
        }
