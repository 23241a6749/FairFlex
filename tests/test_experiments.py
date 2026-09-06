import numpy as np
import pandas as pd

from fairflex.experiments import fit_chronological_pv_forecaster
from fairflex.forecasting import ConformalPVForecaster
from fairflex.scenarios import ReplayWindow


def test_chronological_forecast_fit_keeps_test_targets_out_of_training():
    index = pd.date_range("2024-01-01", periods=288, freq="15min", tz="UTC")
    hours = index.hour.to_numpy() + index.minute.to_numpy() / 60
    pv_proxy = pd.Series(np.maximum(0.0, 20 * np.sin(np.pi * (hours - 6) / 12)), index=index)
    windows = [
        ReplayWindow(index[96], index[160]),
        ReplayWindow(index[160], index[224]),
        ReplayWindow(index[224], index[-1] + pd.Timedelta(minutes=15)),
    ]

    evaluation = fit_chronological_pv_forecaster(
        pv_proxy,
        train_window=windows[0],
        calibration_window=windows[1],
        test_window=windows[2],
        forecaster=ConformalPVForecaster(n_estimators=12, max_depth=2),
    )

    assert (evaluation.train_points, evaluation.calibration_points, evaluation.test_points) == (64, 64, 64)
    assert evaluation.test_index.min() == index[224]
    assert evaluation.test_index.max() == index[287]
    assert evaluation.test_interval.lower_kw.shape == (64,)
