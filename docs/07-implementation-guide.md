# FairFlex implementation guide

This guide distinguishes the working research prototype from claims that still
need a trace-driven experimental result. That distinction is essential for a
credible paper.

## What is implemented now

`FairFlex` is a CPU-first, G2V-only research simulator. It has:

- EV sessions with arrival, exclusive departure, requested energy, maximum
  power, and hard unit checks.
- Transparent uncontrolled, FCFS, and equal-share baselines.
- A three-stage fairness-first rolling MPC controller.
- Quantile XGBoost PV intervals with split-conformal calibration.
- A robust rule that converts the *lower* PV interval into feeder capacity.
- A modified IEEE 33-bus pandapower model and exact AC power-flow validation.
- A repair layer that curtails flexible station demand and re-solves EV fairness.
- Station-private local MPC coordinated by consensus ADMM.
- A reproducible trace-preparation and scenario runner, distributional fairness
  metrics, causal forecast/meter audit logs, and 44 automated tests.

It is not a field deployment, and it does not yet prove a new state-of-the-art
result. Public trace replay, controlled ablations, and statistical comparisons
are the next evidence-producing step.

## Environment: already created on D drive

The project root is `D:\saimohaneesh\FairFlex` and its isolated environment is
`D:\saimohaneesh\FairFlex\.venv`. It was validated with Python 3.12.10.

```powershell
Set-Location D:\saimohaneesh\FairFlex
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install -e ".[dev]"
pytest -q
python scripts\run_demo.py
python scripts\check_data_access.py
```

`requirements.lock` captures the exact validated package versions; `pyproject.toml`
defines the supported dependency ranges for normal development.

## How a 15-minute control interval works

1. **Observe.** Read active sessions, remaining energy, deadlines, station
   capacity, grid state, and the PV forecast interval. No future realized PV or
   future EV arrival is read.
2. **Reserve only reliable solar.** Convert the lower conformal PV bound into a
   time-varying feeder cap. Median/upper PV is intentionally not used to make a
   safety promise. At execution, a present-interval feeder/PV meter clips the
   first action to actual headroom if the probabilistic lower bound misses.
3. **Plan fairly.** MPC first minimizes the worst normalized energy shortfall,
   then maximizes delivered energy without worsening that fairness, then chooses
   a cheaper/smoother tied schedule. This avoids arbitrary reward weights.
4. **Coordinate stations.** In the distributed version, stations reveal only
   desired aggregate imports plus a scalar priority. ADMM returns caps that
   respect the shared feeder profile; local EV data remains at the station.
5. **Verify physics.** Run AC power flow for the action that will actually be
   executed. If voltage or thermal limits fail, reduce the most flexible station
   import with bisection and re-run that station's EV-level fair MPC.
6. **Execute one action and log it.** Update energy delivered, grid metrics,
   ADMM residuals, repair amount, forecast-bound coverage, meter clips, and
   solver/tie-break status. Replan at the next interval.

Only the first planned action is executed. This is why MPC responds to new
information instead of pretending forecasts are perfect.

## Why these methods, and not the tempting alternatives

| Decision | Chosen method | Reason | Deferred alternative |
| --- | --- | --- | --- |
| Hard charging and grid rules | Convex MPC | Feasibility is inspectable and testable every interval. | RL can be a later benchmark, but a reward does not guarantee safe exploration. |
| Fairness | Lexicographic max-min service, then energy, then cost | It makes the ethical priority explicit without arbitrary weights. | A weighted reward hides the fairness/cost trade-off in a tuning constant. |
| PV uncertainty | XGBoost quantiles + conformal calibration | Fast CPU tabular model with a calibrated interval. | LSTM/Transformer need more data and GPU time; use them only as evaluated baselines. |
| Feeder coupling | Consensus ADMM | Stations share aggregates, convergence/residuals are observable, capacity is hard. | A market/auction is possible later, but needs price design and strategic-behavior experiments. |
| Electrical safety | AC power flow + repair | The final action is checked against nonlinear physics. | Linear constraints alone may be optimistic; full AC OPF at every iteration is slower and is a useful accuracy benchmark. |

The scientifically defensible novelty is not “we used ADMM/XGBoost/MPC”—each is
known. The research hypothesis is that **calibrated forecast reserve +
lexicographic EV equity + privacy-preserving ADMM with auditable equity debt +
AC-verified repair** improves the fairness–energy–safety trade-off in realistic
congestion. The paper must establish that through ablation, not assertion.

## Data collection and honest use

1. Put ACN and NSRDB keys in the project-local `.env` file; it is ignored by
   Git and automatically loaded. Verify both without printing a secret:

   ```powershell
   python scripts\check_data_access.py
   ```

   Download one deliberately bounded ACN window before scaling up:

   ```powershell
   python scripts\download_acn_window.py --site caltech --start 2019-05-01T00:00:00Z --end 2019-05-08T00:00:00Z
   ```

   `ACNDataClient` downloads the official session endpoint without embedding a
   token in code. ACN's delivered kWh is an observed replay target, not always a
   declared user request; say this explicitly in the paper.
2. Discover NSRDB datasets for the actual study site, then select and download
   a fixed site/year and save its query and checksum:

   ```powershell
   python scripts\discover_nsrdb.py --lat 17.3850 --lon 78.4867
   ```

   `read_nsrdb_csv` parses raw solar-resource files. A PVWatts GHI-derived
   output is a sensitivity-scenario proxy only; prefer metered PV generation for
   forecast-accuracy claims.
3. Use IEEE 33-bus for controlled diagnostic tests and SimBench profiles/grids
   for broader reproducible grid scenarios. The mapping from a real charging
   site to a synthetic feeder bus is an experiment assumption that must be
   reported.

## Test ladder before any paper result

1. Run `pytest -q`; all tests must pass.
2. Run the deadline scenario. It is designed to show FCFS/equal-sharing failure
   and fair-MPC success—not average field performance.
3. Replay multiple held-out day windows with identical traces and scenario
   seeds for every method.
4. Use chronological training/calibration/test splits for PV; report interval
   coverage overall and by season/weather stress.
5. Ablate one element at a time: no forecast reserve, no AC repair, centralized
   MPC, ADMM without equity debt, and ADMM with equity debt.
6. Report service-ratio distribution, worst/10th-percentile service, energy,
   cost, voltage/thermal violations, ADMM residuals/iterations, repair energy,
   execution time, and failure rates. Use uncertainty intervals across days and
   seeds, not only across EVs in one day.

## Hardware

No physical EV charger, PV panel, smart meter, GPU, or microcontroller is
required for this implementation and its paper simulation study.

- **Minimum development machine:** modern 4-core CPU and 8 GB RAM. Use smaller
  scenarios and sequential AC checks.
- **Recommended experiment machine:** 4–8 CPU cores and 16 GB RAM. This is
  sufficient for trace replay, XGBoost quantile fitting, CVXPY, and pandapower.
- **Optional GPU:** only if adding neural forecasting or MARL baselines. The
  recommended controller does not require it.
- **Later hardware-in-the-loop phase:** an industrial PC or edge computer,
  OCPP-compatible charger/simulator, meter data, and a protected test feeder.
  That is a separate safety/ethics approval phase, not part of the present
  software validation.

## Important limitations to test, not conceal

- Purely online MPC cannot reserve energy for an EV that has not arrived. Test
  future-arrival uncertainty separately or add a forecast/reservation model.
- Conformal prediction provides marginal calibration under its assumptions, not
  a per-storm guarantee. The real-time meter guard is mandatory because a
  lower-bound miss can still occur.
- The current ADMM model is synchronous and cooperative; packet loss, delay,
  privacy attacks, and strategic reporting are future experiments.
- AC repair protects the executed interval; it does not replace validation of a
  future-horizon grid model.
- An optimization solver may fail on a numerical tie-break. FairFlex preserves
  a verified stage-2 fairness/energy solution and records that the optional
  economic tie-break was unavailable.
