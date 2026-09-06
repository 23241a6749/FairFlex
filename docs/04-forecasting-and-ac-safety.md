# Forecast uncertainty and AC-grid safety

## Why point forecasts are not enough

If a scheduler assumes that PV generation will be exactly 50 kW and clouds
reduce it to 20 kW, it can unintentionally import too much grid power. FairFlex
therefore uses lower, median, and upper PV estimates instead of one number.

XGBoost quantile regression is chosen because solar forecasting has tabular
features such as time, recent irradiance, temperature, and cloud conditions. It
trains efficiently on CPU and is easier to audit than a large neural network.
LSTM remains a paper baseline, not an unexplained second production model.

## Why conformal calibration?

Raw quantile models can be too narrow or too wide. FairFlex holds out a
calibration period, measures the amount by which real PV falls outside each raw
interval, and widens future intervals by the required ranked error. The result
has a marginal coverage target, but not a promise that every season or storm is
covered. We will report coverage by scenario instead of overstating it.

## How uncertainty reaches the controller

For G2V-only charging, feeder import equals EV load minus PV generation.
FairFlex permits EV load up to `grid import limit + lower PV prediction bound`,
not the median prediction. This is the amount of local generation that is
available even in the pessimistic calibrated case; it becomes the MPC/ADMM
time-varying feeder cap. The median remains useful for reporting and economics,
but it is not used to justify a safety-critical import allowance.

## Why AC repair after optimization?

The future distributed MPC must be fast, so it uses a linearized feeder model.
Real distribution power flow is nonlinear. After a proposed station schedule,
FairFlex runs pandapower AC power flow. If it is unsafe, the repair layer first
curtails the station with the greatest flexibility, then uses bisection to keep
the largest power that passes the AC check. The fairness MPC will subsequently
reallocate that accepted station budget among EVs; the repair layer never makes
individual EV fairness decisions by itself.
