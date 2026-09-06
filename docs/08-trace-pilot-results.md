# Trace-driven pilot: what the current evidence does and does not show

## Inputs and reproducibility

The first bounded study uses 15-minute public traces near Caltech/Pasadena:

- **Train:** 208 ACN sessions, 1--7 March 2019.
- **Calibration:** 226 ACN sessions, 1--7 April 2019.
- **Held-out test:** 214 ACN sessions, 1--7 May 2019. Sessions that begin in
  the window retain their real departure, so the replay has 688 control steps.
- **Solar sensitivity input:** 35,040 NSRDB irradiance points for 2019,
  converted to a 40 kW DC / 30 kW AC PVWatts GHI proxy. It is not measured
  Caltech PV production.

The exact raw-file hashes, session counts, split boundaries, synthetic station
assignment, and PV-proxy settings are in
`artifacts/caltech_2019_pilot/data_manifest.json`. Raw ACN sessions are mapped
by a deterministic SHA-256 session-ID hash to three synthetic IEEE-33 stations.
That is a controlled feeder scenario, not a measured Caltech topology.

Run the checks and reproduce the prepared-data manifest with:

```powershell
Set-Location D:\saimohaneesh\FairFlex
.\.venv\Scripts\Activate.ps1
pytest -q
python scripts\prepare_study.py
```

Run the severe-scarcity sensitivity and the separately timed distributed
variant with:

```powershell
python scripts\run_trace_pilot.py --config configs\caltech_2019_scarce.json --skip-distributed --output-dir artifacts\caltech_2019_scarce
python -W error::RuntimeWarning scripts\run_trace_pilot.py --config configs\caltech_2019_scarce.json --only recommended_distributed_robust_pv --output-dir artifacts\caltech_2019_scarce_distributed
```

## Why there are three study configurations

`caltech_2019_pilot.json` uses a 50 kW shared import budget and is an
end-to-end sanity check. It is too loose to distinguish policies on this week.
`caltech_2019_congested.json` uses 20 kW and detects uncontrolled charging
failures, but still has enough total energy for FCFS and equal sharing to serve
everyone. `caltech_2019_scarce.json` uses a deliberately severe 10 kW budget:
some shortfall is unavoidable, so *who* receives that shortfall becomes a real
fairness test.

The 10 kW value is **not** a claim about Caltech's measured feeder capacity.
It is a reproducible controlled sensitivity. A later study must sweep multiple
measured or benchmark-derived capacities.

## Corrected held-out scarcity result

Each policy receives a fresh clone of the same 214 sessions. This matters: EV
sessions track delivered energy and must never be reused across policy runs.
The simulator now clones session state at its boundary, and a regression test
guards against cross-policy contamination.

| Policy | Mean service | 10th percentile | Worst service | Jain index | Delivered kWh | Unsafe steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncontrolled + shared cap | 0.746 | 0.277 | 0.057 | 0.859 | 1344.9 | 0 |
| FCFS + shared cap | 0.711 | 0.080 | 0.000 | 0.802 | 1347.3 | 0 |
| Equal share + shared cap | 0.752 | 0.277 | 0.064 | 0.861 | 1344.9 | 0 |
| Central max--min MPC, no PV reserve | 0.605 | 0.268 | 0.150 | 0.802 | 1174.9 | 0 |
| Central max--min MPC + robust PV reserve | 0.714 | 0.313 | 0.210 | 0.862 | 1390.8 | 0 |
| Distributed ADMM + robust PV reserve | 0.701 | 0.325 | 0.000 | 0.864 | 1378.7 | 0 |

The central no-PV controller deliberately gives up mean energy to improve the
worst-served EV: that is the expected lexicographic max--min trade-off, not a
bug. Adding the calibrated PV reserve improves both its lower tail and energy
in this sensitivity. It is still only one week and one synthetic scenario.

The distributed variant is **not yet a superiority claim**. It has a slightly
better 10th percentile in this run, but one EV is completely unserved. This is
useful negative evidence: station-level privacy and aggregate coordination can
lose an individual-level fairness guarantee. The next experiment must tune and
ablate debt gain/decay and compare station-level versus EV-level fairness.

## Forecast and execution safety evidence

The model is trained on March, conformally calibrated on April, and evaluated
on May. Its one-step May interval coverage is 0.857, versus a nominal 0.90
target, and its median MAE is 3.693 kW. This is not sufficiently broad evidence
to claim calibrated 90% coverage; evaluate more seasons and weather regimes.

For the 688 executed test actions, the lower PV bound was met 0.996 of the
time. It was too optimistic three times by 4.55 kW in total. The meter guard
clipped those first actions to actual present headroom, leaving **zero realized
shared-import-limit violations**. The guard is an execution safety layer, not
future-information leakage: only the current meter value is used; later MPC
slots retain causal recursive forecasts.

## Grid and distributed checks

All listed policies have zero AC-invalid actions in this study. AC repair never
activated because the 10 kW shared budget is already well inside this particular
IEEE-33 sensitivity's AC headroom. That means this trace study does not validate
the *benefit* of AC repair; a separate tight-voltage/thermal grid stress sweep
is required.

The full distributed run made 688 ADMM negotiations. Its largest primal and
dual residuals were below `1e-4`, it took at most 67 iterations, and it made 140
private station-debt updates. It was run with runtime warnings promoted to
errors. The central and distributed controllers are CPU-feasible for this pilot,
but distributed coordination is slower because each interval includes local
MPC solves around the negotiation.

## Next evidence-producing work

1. Sweep several feeder capacities, PV sizes, forecast-error regimes, and
   voltage/thermal stress settings over many held-out days.
2. Run the registered ablations: no PV reserve, no meter guard (diagnostic only,
   never deployment), no AC repair, centralized MPC, ADMM without debt, and
   ADMM with a grid search over debt gain/decay.
3. Report bootstrap confidence intervals across days and fixed scenario seeds,
   not a single week or a single aggregated Jain score.
4. Add a benchmark feeder with native equipment limits or measured topology.
   The present Caltech-to-IEEE-33 mapping is only a sensitivity model.
5. Treat the PV proxy as a sensitivity input. Use metered PV production before
   making a forecast-accuracy or deployment claim.
