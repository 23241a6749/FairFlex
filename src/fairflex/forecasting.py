"""Calibrated probabilistic PV forecasting for robust charging control."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class PredictionInterval:
    lower_kw: np.ndarray
    median_kw: np.ndarray
    upper_kw: np.ndarray

    def coverage(self, actual_kw: np.ndarray) -> float:
        actual = np.asarray(actual_kw, dtype=float)
        if actual.shape != self.lower_kw.shape:
            raise ValueError("actual values must have the same shape as predictions")
        return float(np.mean((self.lower_kw <= actual) & (actual <= self.upper_kw)))


class IntervalForecaster(Protocol):
    """Small interface that permits testing a recursive forecast without XGBoost."""

    def predict(self, features: np.ndarray) -> PredictionInterval: ...


def robust_ev_import_capacity_kw(
    grid_import_limit_kw: float | np.ndarray,
    pv_interval: PredictionInterval,
) -> np.ndarray:
    """Return EV-load caps that remain safe using only guaranteed PV output.

    If feeder import is ``EV load - PV output``, allowing EV load to be at most
    ``grid limit + lower PV bound`` remains safe whenever the calibrated lower
    bound is met. Using the median or upper bound would quietly rely on sunshine
    that may not arrive. The returned profile can be passed directly to MPC or
    ADMM as its time-varying feeder-capacity constraint.
    """
    lower = np.asarray(pv_interval.lower_kw, dtype=float)
    median = np.asarray(pv_interval.median_kw, dtype=float)
    upper = np.asarray(pv_interval.upper_kw, dtype=float)
    if lower.ndim != 1 or median.shape != lower.shape or upper.shape != lower.shape:
        raise ValueError("PV interval arrays must be equally sized one-dimensional profiles")
    if np.any(lower < 0) or np.any(lower > median) or np.any(median > upper):
        raise ValueError("PV interval must be non-negative and ordered")
    grid_limit = np.asarray(grid_import_limit_kw, dtype=float)
    if grid_limit.ndim == 0:
        grid_limit = np.full(lower.shape, float(grid_limit))
    if grid_limit.shape != lower.shape or np.any(grid_limit < 0):
        raise ValueError("grid import limit must be non-negative and match the PV profile")
    return grid_limit + lower


def recursive_prediction_interval(
    forecaster: IntervalForecaster,
    observed_pv_kw: pd.Series,
    *,
    first_timestamp: pd.Timestamp,
    horizon_steps: int,
    step_minutes: int = 15,
    lag_steps: tuple[int, ...] = (1, 4, 96),
) -> PredictionInterval:
    """Forecast a PV horizon without reading realized values at or after its start.

    Each first-step feature uses observed history strictly before
    ``first_timestamp``. Later horizon features use a preceding median forecast
    whenever a short lag points inside the newly predicted horizon. This is a
    simple recursive strategy: it is honest for online control, but its
    multi-step calibration must be evaluated separately from one-step coverage.
    """
    if horizon_steps <= 0 or step_minutes <= 0 or not lag_steps or min(lag_steps) <= 0:
        raise ValueError("horizon_steps, step_minutes, and lag_steps must be positive")
    if not isinstance(observed_pv_kw.index, pd.DatetimeIndex):
        raise ValueError("observed_pv_kw must have a DatetimeIndex")
    if observed_pv_kw.index.tz is None:
        raise ValueError("observed_pv_kw timestamps must be timezone-aware")
    if observed_pv_kw.index.has_duplicates:
        raise ValueError("observed_pv_kw timestamps must be unique")
    if (observed_pv_kw < 0).any():
        raise ValueError("observed_pv_kw must be non-negative")

    start = pd.Timestamp(first_timestamp)
    if start.tzinfo is None:
        raise ValueError("first_timestamp must be timezone-aware")
    start = start.tz_convert("UTC")
    frequency = pd.Timedelta(minutes=step_minutes)
    # Crucially, this slice rejects the current target and all future outcomes.
    # Only the largest declared lag can be read by the recursive feature
    # builder. Keeping an entire multi-month history here is both unnecessary
    # and expensive because this function runs at every controller decision.
    # A bounded lag window preserves identical causal features while avoiding
    # repeated large temporary allocations on long chronological replays.
    earliest_needed = start - max(lag_steps) * frequency
    history = observed_pv_kw[
        (observed_pv_kw.index >= earliest_needed) & (observed_pv_kw.index < start)
    ]
    known = {timestamp: float(value) for timestamp, value in history.items()}
    if not known:
        raise ValueError("no observed PV history is available before first_timestamp")

    lower: list[float] = []
    median: list[float] = []
    upper: list[float] = []
    for offset in range(horizon_steps):
        timestamp = start + offset * frequency
        try:
            lag_values = [known[timestamp - lag * frequency] for lag in lag_steps]
        except KeyError as error:
            raise ValueError("insufficient contiguous PV history for requested lags") from error
        hour = timestamp.hour + timestamp.minute / 60.0
        features = np.asarray(
            [
                *lag_values,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
            ],
            dtype=float,
        ).reshape(1, -1)
        interval = forecaster.predict(features)
        if any(array.shape != (1,) for array in (interval.lower_kw, interval.median_kw, interval.upper_kw)):
            raise ValueError("forecaster must return one prediction per feature row")
        lower.append(float(interval.lower_kw[0]))
        median_value = float(interval.median_kw[0])
        median.append(median_value)
        upper.append(float(interval.upper_kw[0]))
        known[timestamp] = median_value
    return PredictionInterval(np.asarray(lower), np.asarray(median), np.asarray(upper))


class RobustForecastCapacitySource:
    """Callable MPC feeder-cap source backed by causal recursive PV forecasts."""

    def __init__(
        self,
        forecaster: IntervalForecaster,
        observed_pv_kw: pd.Series,
        *,
        replay_origin: pd.Timestamp,
        grid_import_limit_kw: float,
        step_minutes: int = 15,
        metered_first_step: bool = False,
    ) -> None:
        if grid_import_limit_kw < 0 or step_minutes <= 0:
            raise ValueError("grid_import_limit_kw must be non-negative and step_minutes positive")
        origin = pd.Timestamp(replay_origin)
        if origin.tzinfo is None:
            raise ValueError("replay_origin must be timezone-aware")
        self.forecaster = forecaster
        self.observed_pv_kw = observed_pv_kw
        self.replay_origin = origin.tz_convert("UTC")
        self.grid_import_limit_kw = float(grid_import_limit_kw)
        self.step_minutes = step_minutes
        self.metered_first_step = metered_first_step
        self.last_interval: PredictionInterval | None = None
        self.intervals_by_step: dict[int, PredictionInterval] = {}
        self.forecast_capacities_by_step: dict[int, np.ndarray] = {}
        self.capacities_by_step: dict[int, np.ndarray] = {}
        self.metered_clips_by_step: dict[int, float] = {}

    def __call__(self, current_step: int, horizon_steps: int) -> np.ndarray:
        if current_step < 0:
            raise ValueError("current_step must be non-negative")
        start = self.replay_origin + pd.Timedelta(minutes=self.step_minutes * current_step)
        interval = recursive_prediction_interval(
            self.forecaster,
            self.observed_pv_kw,
            first_timestamp=start,
            horizon_steps=horizon_steps,
            step_minutes=self.step_minutes,
        )
        self.last_interval = interval
        forecast_capacity = robust_ev_import_capacity_kw(self.grid_import_limit_kw, interval)
        capacity = forecast_capacity.copy()
        if self.metered_first_step:
            # This is a real-time execution guard, not a future information
            # feature: only the present interval's measured PV is used. The
            # remaining horizon keeps its causal forecast-derived reserve.
            measured_pv_kw = float(self.observed_pv_kw.loc[start])
            metered_capacity = self.grid_import_limit_kw + measured_pv_kw
            capacity[0] = min(capacity[0], metered_capacity)
            self.metered_clips_by_step[current_step] = max(
                0.0, float(forecast_capacity[0] - capacity[0])
            )
        self.intervals_by_step[current_step] = interval
        self.forecast_capacities_by_step[current_step] = forecast_capacity
        self.capacities_by_step[current_step] = capacity.copy()
        return capacity


class ConformalPVForecaster:
    """Quantile XGBoost forecasts with split-conformal prediction intervals.

    The model estimates lower, median, and upper conditional quantiles. A held-
    out calibration set then determines one nonconformity correction, giving
    finite-sample *marginal* coverage under exchangeability. It does not claim
    a guarantee for every weather regime; coverage is reported separately by
    season and stress scenario in the final experiments.
    """

    def __init__(
        self,
        *,
        miscoverage: float = 0.1,
        n_estimators: int = 160,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 7,
    ) -> None:
        if not 0 < miscoverage < 1:
            raise ValueError("miscoverage must be between zero and one")
        self.miscoverage = miscoverage
        self._model_config: dict[str, Any] = {
            "objective": "reg:quantileerror",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": random_state,
        }
        self.lower_model: XGBRegressor | None = None
        self.median_model: XGBRegressor | None = None
        self.upper_model: XGBRegressor | None = None
        self.correction_kw: float | None = None

    def _new_model(self, quantile: float) -> XGBRegressor:
        return XGBRegressor(**self._model_config, quantile_alpha=quantile)

    def fit(
        self,
        train_features: np.ndarray,
        train_target_kw: np.ndarray,
        calibration_features: np.ndarray,
        calibration_target_kw: np.ndarray,
    ) -> "ConformalPVForecaster":
        train_x = np.asarray(train_features, dtype=float)
        train_y = np.asarray(train_target_kw, dtype=float).reshape(-1)
        calibration_x = np.asarray(calibration_features, dtype=float)
        calibration_y = np.asarray(calibration_target_kw, dtype=float).reshape(-1)
        if train_x.ndim != 2 or calibration_x.ndim != 2:
            raise ValueError("feature arrays must be two-dimensional")
        if len(train_x) != len(train_y) or len(calibration_x) != len(calibration_y):
            raise ValueError("features and targets must have matching row counts")
        if train_x.shape[1] != calibration_x.shape[1] or not len(calibration_y):
            raise ValueError("calibration data must be non-empty and match training features")
        if np.any(train_y < 0) or np.any(calibration_y < 0):
            raise ValueError("PV power targets must be non-negative")

        self.lower_model = self._new_model(self.miscoverage / 2)
        self.median_model = self._new_model(0.5)
        self.upper_model = self._new_model(1 - self.miscoverage / 2)
        self.lower_model.fit(train_x, train_y)
        self.median_model.fit(train_x, train_y)
        self.upper_model.fit(train_x, train_y)

        lower, _, upper = self._raw_predict(calibration_x)
        scores = np.maximum.reduce([lower - calibration_y, calibration_y - upper, np.zeros_like(calibration_y)])
        sorted_scores = np.sort(scores)
        rank = min(len(sorted_scores), max(1, ceil((len(sorted_scores) + 1) * (1 - self.miscoverage))))
        self.correction_kw = float(sorted_scores[rank - 1])
        return self

    def _raw_predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not all((self.lower_model, self.median_model, self.upper_model)):
            raise RuntimeError("fit the forecaster before predicting")
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("features must be two-dimensional")
        lower = self.lower_model.predict(matrix)
        median = self.median_model.predict(matrix)
        upper = self.upper_model.predict(matrix)
        return np.minimum(lower, upper), median, np.maximum(lower, upper)

    def predict(self, features: np.ndarray) -> PredictionInterval:
        if self.correction_kw is None:
            raise RuntimeError("fit the forecaster before predicting")
        lower, median, upper = self._raw_predict(features)
        correction = self.correction_kw
        calibrated_lower = np.maximum(0.0, lower - correction)
        calibrated_upper = np.maximum(calibrated_lower, upper + correction)
        calibrated_median = np.clip(median, calibrated_lower, calibrated_upper)
        return PredictionInterval(calibrated_lower, calibrated_median, calibrated_upper)
