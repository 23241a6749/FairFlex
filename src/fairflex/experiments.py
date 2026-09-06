"""Chronological forecast fitting and evidence objects for FairFlex studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import lagged_pv_features
from .forecasting import ConformalPVForecaster, PredictionInterval
from .scenarios import ReplayWindow


@dataclass(frozen=True)
class ForecastEvaluation:
    """Held-out, chronological forecast evidence and the fitted online model."""

    forecaster: ConformalPVForecaster
    train_points: int
    calibration_points: int
    test_points: int
    test_index: pd.DatetimeIndex
    test_actual_kw: np.ndarray
    test_interval: PredictionInterval

    @property
    def test_coverage(self) -> float:
        return self.test_interval.coverage(self.test_actual_kw)

    @property
    def test_median_mae_kw(self) -> float:
        return float(np.mean(np.abs(self.test_interval.median_kw - self.test_actual_kw)))


def fit_chronological_pv_forecaster(
    pv_proxy_kw: pd.Series,
    *,
    train_window: ReplayWindow,
    calibration_window: ReplayWindow,
    test_window: ReplayWindow,
    forecaster: ConformalPVForecaster | None = None,
    lag_steps: tuple[int, ...] = (1, 4, 96),
) -> ForecastEvaluation:
    """Fit only on the train window, calibrate next, then report a held-out test.

    Lag values may come from earlier timestamps outside a labelled window: they
    are observations that would already exist at forecast time, not fitted
    targets. No target from calibration or test contributes to model fitting.
    """
    features, target, index = lagged_pv_features(pv_proxy_kw, lag_steps=lag_steps)

    def in_window(window: ReplayWindow) -> np.ndarray:
        return np.asarray((index >= window.start) & (index < window.end), dtype=bool)

    train_mask = in_window(train_window)
    calibration_mask = in_window(calibration_window)
    test_mask = in_window(test_window)
    if not all((train_mask.any(), calibration_mask.any(), test_mask.any())):
        raise ValueError("each chronological window must contain at least one feature row")
    model = forecaster or ConformalPVForecaster()
    model.fit(
        features[train_mask],
        target[train_mask],
        features[calibration_mask],
        target[calibration_mask],
    )
    test_interval = model.predict(features[test_mask])
    return ForecastEvaluation(
        model,
        int(train_mask.sum()),
        int(calibration_mask.sum()),
        int(test_mask.sum()),
        index[test_mask],
        target[test_mask],
        test_interval,
    )
