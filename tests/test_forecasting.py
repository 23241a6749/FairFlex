import numpy as np
import pandas as pd

from fairflex.forecasting import (
    ConformalPVForecaster,
    PredictionInterval,
    RobustForecastCapacitySource,
    recursive_prediction_interval,
    robust_ev_import_capacity_kw,
)


def test_conformal_forecaster_returns_ordered_nonnegative_intervals():
    rng = np.random.default_rng(7)
    features = rng.uniform(0.0, 1.0, size=(100, 3))
    target = np.maximum(0.0, 20 * features[:, 0] + 3 * features[:, 1] + rng.normal(0, 1, 100))
    forecaster = ConformalPVForecaster(miscoverage=0.2, n_estimators=24).fit(
        features[:60], target[:60], features[60:80], target[60:80]
    )
    calibration_interval = forecaster.predict(features[60:80])
    held_out_interval = forecaster.predict(features[80:])

    assert held_out_interval.lower_kw.shape == (20,)
    assert np.all(held_out_interval.lower_kw >= 0)
    assert np.all(held_out_interval.lower_kw <= held_out_interval.median_kw)
    assert np.all(held_out_interval.median_kw <= held_out_interval.upper_kw)
    # Calibration coverage is the quantity controlled by conformal prediction.
    # A future holdout rate is stochastic, so it is not asserted exactly here.
    assert calibration_interval.coverage(target[60:80]) >= 0.8


def test_robust_capacity_uses_the_lower_not_median_or_upper_pv_forecast():
    interval = PredictionInterval(
        lower_kw=np.array([1.0, 2.0]),
        median_kw=np.array([4.0, 5.0]),
        upper_kw=np.array([7.0, 8.0]),
    )

    capacity = robust_ev_import_capacity_kw([10.0, 10.0], interval)

    assert np.array_equal(capacity, np.array([11.0, 12.0]))


class _FeatureEchoForecaster:
    """Test double whose median exposes whether a recursive feature leaked."""

    def predict(self, features):
        median = np.asarray(features[:, 0], dtype=float)
        return PredictionInterval(median - 1.0, median, median + 1.0)


class _OverconfidentForecaster:
    def predict(self, features):
        values = np.full(features.shape[0], 200.0)
        return PredictionInterval(values, values + 1.0, values + 2.0)


def test_recursive_forecast_uses_past_observations_then_its_own_predictions():
    index = pd.date_range("2024-01-01", periods=110, freq="15min", tz="UTC")
    observed = pd.Series(np.arange(110, dtype=float), index=index)
    start = index[100]

    interval = recursive_prediction_interval(
        _FeatureEchoForecaster(), observed, first_timestamp=start, horizon_steps=2
    )

    # Step 0 sees observed t-1 (99); step 1 sees the preceding *prediction*
    # (99), not the hidden true target at t (100).
    assert np.allclose(interval.median_kw, [99.0, 99.0])


def test_recursive_forecast_needs_only_the_declared_lag_history():
    index = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    observed = pd.Series(np.arange(300, dtype=float), index=index)
    start = index[200]

    full_history = recursive_prediction_interval(
        _FeatureEchoForecaster(), observed, first_timestamp=start, horizon_steps=4
    )
    lag_bounded_history = recursive_prediction_interval(
        _FeatureEchoForecaster(), observed.loc[index >= index[104]], first_timestamp=start, horizon_steps=4
    )

    assert np.allclose(full_history.lower_kw, lag_bounded_history.lower_kw)
    assert np.allclose(full_history.median_kw, lag_bounded_history.median_kw)
    assert np.allclose(full_history.upper_kw, lag_bounded_history.upper_kw)


def test_robust_capacity_source_uses_the_calibrated_lower_forecast():
    index = pd.date_range("2024-01-01", periods=110, freq="15min", tz="UTC")
    source = RobustForecastCapacitySource(
        _FeatureEchoForecaster(),
        pd.Series(np.arange(110, dtype=float), index=index),
        replay_origin=index[100],
        grid_import_limit_kw=10.0,
    )

    capacity = source(0, 2)

    assert np.allclose(capacity, [108.0, 108.0])


def test_metered_first_step_clips_a_rare_overconfident_pv_reserve():
    index = pd.date_range("2024-01-01", periods=110, freq="15min", tz="UTC")
    source = RobustForecastCapacitySource(
        _OverconfidentForecaster(),
        pd.Series(np.arange(110, dtype=float), index=index),
        replay_origin=index[100],
        grid_import_limit_kw=10.0,
        metered_first_step=True,
    )

    capacity = source(0, 2)

    assert capacity[0] == 110.0  # 10 kW grid headroom + measured 100 kW PV
    assert capacity[1] == 210.0  # future slot remains forecast-derived
    assert source.metered_clips_by_step[0] == 100.0
