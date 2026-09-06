# FairFlex-UC (V3): causal early-unplug robustness pilot

## Status and scope

This is a reproducible, trace-driven **pilot**, not final paper evidence.
V1 and V2 remain unchanged. V3 code, configuration, and artifacts live under
`src/fairflex/v3`, `scripts/v3`, `configs/v3`, and `artifacts/v3`.

The operational question is: under a scarce shared EV-charging capacity, can
we improve service to the lower tail of drivers—especially drivers who unplug
at least 30 minutes before their stated deadline—without allowing an unsafe
feeder action?

Real ACN-Data sessions provide connection, declared request, declared
departure, and actual unplug observations. The shared feeder limit, station
assignment, and PV proxy are controlled sensitivity assumptions; they are not
measured Caltech feeder or rooftop-PV values. A result here must never be
described as a live-site deployment result.

## Causal controller design

At plug-in, the controller may use the EV's declared requested kWh and
departure, its charging limit, calendar context, station, and the currently
observable set of connected EVs. It must not use this EV's future actual
unplug time. Actual unplug is used only as a historical training label,
physical replay event, and post-hoc evaluation label.

V3 adds three individually auditable elements.

1. **CQR early-unplug buffer.** A gradient-boosted 90th-quantile model
   predicts how many 15-minute intervals an EV may leave early. A calibration
   month contributes one one-sided split-conformal residual correction. This
   allows the buffer to depend on observable context while retaining a clear
   empirical coverage audit.
2. **Conservative hybrid envelope.** V1's causal multi-rate online guard and
   the CQR guard each produce a planning deadline. The hybrid uses the earlier
   of the two deadlines. It is deliberately conservative: the CQR model is
   allowed to protect a context that V1's global/history-based guard misses,
   but neither may see the current driver's future unplug.
3. **Lower-tail MPC.** The MPC minimises expected shortfall of the largest
   10% service-deficit tail (a convex CVaR-style surrogate), then maximises
   delivered energy, then uses the existing economic/ramp tie-break. Raw P10
   service ratio remains the primary reported metric; the surrogate is not
   claimed to be exact finite-sample P10 optimisation.

All MPC policies retain the existing per-EV power, station capacity, shared
import, deadline, and AC feasibility checks. `ACRepairedMPCPolicy` is still
the final safety layer.

## Chronological data protocol

| Role | Data window | Usable sessions | Purpose |
| --- | --- | ---: | --- |
| Train | 2019-03-01 to 2019-07-01 (five raw extracts) | 946 | fit CQR quantile relation |
| Calibration | 2019-08-01 to 2019-09-01 | 737 | conformal correction and V1 guard history |
| Development | 2019-09-01 to 2019-09-08 | 169 | compare design alternatives once |
| Frozen holdout | 2019-10-01 to 2019-10-08 | 154 | one final temporal check |

The holdout was never used to fit a model or choose the hybrid policy.

## Development result and selection

The development comparison held the trace, synthetic feeder (10 kW import),
PV proxy, and safety checks fixed. The 47-session early-unplug subset means
actual unplug at least 30 minutes before the declaration.

| Deployable policy | All-session P10 | Early-unplug P10 | 10% expected shortfall of deficits | Unsafe steps |
| --- | ---: | ---: | ---: | ---: |
| Equal share | 0.160 | 0.097 | 0.890 | 0 |
| V1 fair MPC, no guard | 0.210 | 0.123 | 0.853 | 0 |
| V1 fair MPC + multi-rate guard | 0.259 | 0.147 | 0.824 | 0 |
| CQR + V1 max-min MPC | 0.238 | 0.145 | 0.837 | 0 |
| CQR + lower-tail MPC | 0.242 | 0.146 | 0.834 | 0 |
| Hybrid + max-min MPC | 0.259 | 0.150 | 0.823 | 0 |
| **Hybrid + lower-tail MPC** | **0.266** | **0.166** | **0.817** | **0** |

The frozen selection rule was: choose the causal deployable candidate with
the largest all-session raw P10, require zero unsafe steps, then use lower
expected shortfall as a tie-break. It selected **`fairflex_uc_hybrid_lower_tail`**.
The hybrid chose CQR's earlier deadline for 43 of 169 sessions and V1's for
126; its future-period empirical deadline coverage was 0.941. CQR alone was
0.876 against a 0.900 target on this particular week, which is reported rather
than hidden.

## Frozen October holdout

The final evaluation used the frozen CQR features, 90% nominal target,
multi-rate guard, hybrid envelope, 10% tail fraction, and selection rule. It
had 154 sessions, including 46 early-unplug sessions.

| Policy | All-session P10 | Early-unplug P10 | 10% expected shortfall of deficits | Delivered kWh | Unsafe steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal share | 0.141 | 0.089 | 0.907 | 1465.6 | 0 |
| V1 fair MPC, no guard | 0.207 | 0.110 | 0.858 | 1469.6 | 0 |
| V1 fair MPC + multi-rate guard | 0.202 | 0.107 | 0.859 | 1474.6 | 0 |
| **FairFlex-UC hybrid + lower tail** | **0.216** | **0.123** | **0.846** | **1469.5** | **0** |
| Oracle lower-tail (non-deployable) | 0.245 | 0.141 | 0.818 | 1482.3 | 0 |

Against the strongest deployable V1 comparator, V3's October differences are
`+0.0137` all-session P10, `+0.0162` early-unplug P10, and `-0.0127` expected
shortfall, with `-5.1 kWh` delivered energy. This is a fairness-versus-energy
trade-off, not a universal dominance claim. The explicitly labelled oracle
is an information upper bound only; it illegally knows actual future unplug
times and must never be compared as a deployable baseline.

The CQR-only guard had 0.916 empirical future-period coverage, close to its
0.900 nominal target. The combined conservative envelope was 0.942, so the
hybrid bought more protection than nominally required. This needs a
multi-week conservatism/energy analysis before treating it as the final guard.

## October day-block robustness diagnostic

The frozen October week was also replayed as seven matched calendar days and
resampled by complete day (2,000 paired bootstrap draws). The raw daily table
is preserved. A reporting screen requires at least ten sessions for a daily
P10 and at least five eligible days; this removes two days with only seven and
one sessions from the daily-P10 comparison. It does **not** remove them from
the raw data or from non-quantile descriptive metrics.

Against V1 multi-rate guard, the screened five-day mean P10 difference is
`+0.0164`, with a 95% paired-day percentile interval of `[-0.0061, +0.0434]`.
The interval crosses zero, so this small diagnostic does not establish a
conclusive all-session P10 advantage. Daily early-unplug P10 is not reported:
only two days have at least ten early-unplug sessions. The continuous-week
holdout remains the primary temporal result for this pilot.

The full seven-day diagnostic shows a cost of the lower-tail policy: mean
service changes by `-0.0063` (95% interval `[-0.0119, -0.0010]`) and delivered
energy by `-1.35 kWh` (`[-3.25, -0.17]`) versus guarded V1. Safety remains
zero for both policies. These daily replays reset V1's online guard from the
fixed calibration bank at each midnight, so they are policy-metric sensitivity
evidence, not continuous-week guard-coverage evidence.

## External JPL replication

The frozen design was transferred to the JPL ACN site without a JPL
development selection. It fitted CQR and the online guard only on earlier JPL
history (1,339 train and 1,437 calibration sessions), then evaluated JPL
October 1–8 (379 sessions, including 116 early-unplug sessions). This is
site-calibrated transfer of a frozen design, not zero-shot transfer.

| Policy | All-session P10 | Early-unplug P10 | 10% expected shortfall of deficits | Delivered kWh | Unsafe steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal share | 0.0437 | 0.0298 | 0.9709 | 1666.1 | 0 |
| V1 fair MPC, no guard | 0.0722 | 0.0530 | 0.9451 | 1655.3 | 0 |
| V1 fair MPC + multi-rate guard | 0.0784 | 0.0562 | 0.9439 | 1664.7 | 0 |
| **FairFlex-UC hybrid + lower tail** | **0.0859** | **0.0579** | **0.9403** | **1657.1** | **0** |

The transferred hybrid improves P10 by `+0.0076`, early-unplug P10 by
`+0.0017`, and tail shortfall by `0.0036` over guarded V1, while delivering
`7.6 kWh` less energy. CQR future-period coverage is 0.923 and hybrid
deadline coverage is 0.960. This supports the transfer hypothesis, but it
does not validate real JPL feeder physics: station placement, 10 kW shared
import limit, and the nearby Caltech NSRDB solar proxy remain sensitivity
assumptions.

## Frozen one-factor sensitivity analysis

After the policy and its hyperparameters were frozen, we changed one physical
assumption at a time on the same Caltech October replay. No scenario was used
to retune, select, or replace the policy. The differences below are FairFlex-UC
hybrid + lower-tail minus V1 fair MPC + multi-rate guard, so a positive P10 is
better, a negative shortfall is better, and a negative energy difference is an
energy cost.

| Fixed scenario | All-session P10 difference | Early-unplug P10 difference | 10% expected-shortfall difference | Delivered-energy difference (kWh) |
| --- | ---: | ---: | ---: | ---: |
| 8 kW import limit | +0.0074 | +0.0292 | -0.0248 | -11.30 |
| 10 kW import limit (base case) | +0.0137 | +0.0162 | -0.0127 | -5.08 |
| 12 kW import limit | +0.0043 | -0.0073 | -0.0099 | -3.45 |
| lower PV (20 kW DC / 15 kW AC) | +0.0142 | +0.0184 | -0.0019 | -4.89 |
| higher PV (60 kW DC / 45 kW AC) | +0.0144 | +0.0230 | -0.0299 | -8.25 |

Both MPC policies had zero unsafe steps in every listed scenario. FairFlex-UC
improves the prespecified all-session P10 and tail-shortfall metrics in all
five conditions, and early-unplug P10 in four of five conditions. It delivers
less energy in all five. The result is therefore robust evidence for the
intended lower-tail fairness trade-off, not evidence that the method dominates
V1 on every objective.

## Relationship to prior work and comparison boundary

The project is not claiming that max-min fairness, MPC, conformal prediction,
or multi-agent charging are individually new. The contribution being tested is
the *causal combination and evaluation protocol*: a per-session,
context-aware conformal early-unplug buffer; a conservative envelope with an
online past-only guard; lower-tail MPC; and declared-deadline ACN trace replay
in which each driver's future actual departure is unavailable to the policy.

| Existing line of work | What it demonstrates | Why our numerical result is not yet directly comparable |
| --- | --- | --- |
| Wang, Pi, and Tang (2017), multi-server fair queueing | Network/voltage-aware physical fairness scheduling | It uses a distinct queueing formulation and fairness construction, rather than this declared-deadline ACN replay and P10 service metric. |
| Tsaousoglou et al. (2023), fair scalable charging | Distributed max-min fairness under electrical-grid constraints | Its distributed grid model and evaluation setup differ from our uncertain-departure commitment problem. |
| Ahmad et al. (2023), two-layer workplace prediction | Caltech workplace-load/departure prediction combined with economic scheduling | It targets aggregate prediction and cost-oriented scheduling, not a per-driver causal early-unplug lower-tail guarantee. |
| MAPPO / IPPO baselines | Learned multi-agent policies for charging/control | The locally available MAPPO run is from a different pretest cohort and is **not** an apples-to-apples V3 comparison. A fair rerun must use the same traces, causal observation boundary, constraints, safety wrapper, training budget, and frozen test weeks. |

Accordingly, we may say that V3 outperforms the reproduced V1 comparator on
the two continuous held-out trace replays and the five fixed sensitivity
scenarios for the primary P10 objective, always with zero unsafe steps. We
may **not** say that it outperforms the cited papers or MAPPO until those
methods are reproduced or evaluated under the same protocol. Finding no
immediately identical evaluation protocol is not proof that no related paper
exists.

Direct sources for this comparison boundary are: Wang, Pi, and Tang (2017),
https://doi.org/10.1109/TPDS.2017.2710197; Tsaousoglou et al. (2023),
https://pure.tue.nl/ws/portalfiles/portal/316735503/Fair_and_Scalable_Electric_Vehicle_Charging_Under_Electrical_Grid_Constraints.pdf;
and Ahmad et al. (2023), https://arxiv.org/abs/2307.08311.

## Statistical evaluation clarification

The CQR guard is a one-sided quantile predictor, so it is evaluated by
coverage, pinball loss, and buffer sharpness—not classifier accuracy. The MPC
controller is evaluated by matched policy effects, not forecast accuracy. The
frozen October CQR audit found 141/154 = 0.916 coverage against the 0.900
target; its seven-day clustered 95% interval is [0.888, 0.940]. Against a
calibration-only static global conformal buffer, CQR had the same observed
coverage while using 0.247 fewer 15-minute buffer steps on average and 0.044
lower aligned pinball loss. These are modest, one-week prediction results.

For V3 versus guarded V1, the appropriate unit of inference is the matched
calendar day. An individual-EV t-test or ordinary ANOVA would be
pseudo-replication because sessions from the same day share conditions. The
five eligible daily P10 blocks give an advantage of +0.0164 with 95%
paired-day bootstrap interval [-0.0055, +0.0434] and exact one-sided paired
sign-flip p = 0.1875. It is directionally encouraging but statistically
inconclusive. The detailed definitions, results, and multi-week/MAPPO plan are
recorded in `docs/v3/02-statistical-evaluation-protocol.md`.

## How to reproduce the completed pilot

From `D:\saimohaneesh\FairFlex`:

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q

# Development ablations (already completed; do not use their output as final test evidence)
python scripts\v3\run_fairflex_uc.py `
  --config configs\v3\caltech_2019_fairflex_uc_development_september.json `
  --output-dir artifacts\v3\development_september_matrix

# Frozen temporal holdout (already completed once)
python scripts\v3\run_fairflex_uc.py `
  --config configs\v3\caltech_2019_fairflex_uc_holdout_october.json `
  --output-dir artifacts\v3\holdout_october_frozen `
  --only equal_share_shared_cap v1_fair_mpc_no_guard `
         v1_fair_mpc_multirate_guard fairflex_uc_hybrid_lower_tail `
         oracle_lower_tail_non_deployable

# Frozen one-factor sensitivity analysis (safe to resume after interruption)
python scripts\v3\run_frozen_sensitivity.py `
  --protocol configs\v3\caltech_2019_fairflex_uc_sensitivity_protocol.json `
  --output-dir artifacts\v3\caltech_october_frozen_sensitivity `
  --resume

# CQR reliability and paired-day primary statistical audit
python scripts\v3\audit_cqr_predictions.py `
  --config configs\v3\caltech_2019_fairflex_uc_holdout_october.json `
  --output-dir artifacts\v3\holdout_october_cqr_prediction_audit_v3 `
  --resamples 5000 --seed 20260903

python scripts\v3\run_statistical_audit.py `
  --input-dir artifacts\v3\holdout_october_daywise_bounded_history `
  --output-dir artifacts\v3\holdout_october_statistical_audit_primary `
  --metrics p10_service_ratio `
  --resamples 5000 --seed 20260903
```

The main outputs are:

- `artifacts/v3/development_september_matrix/test_fairflex_uc_metrics.csv`
- `artifacts/v3/development_september_hybrid_screen/test_fairflex_uc_metrics.csv`
- `artifacts/v3/holdout_october_frozen/test_fairflex_uc_metrics.csv`
- `artifacts/v3/holdout_october_daywise_bounded_history/daywise_metrics.csv`
- `artifacts/v3/holdout_october_daywise_screened/screened_paired_day_bootstrap.csv`
- `artifacts/v3/jpl_external_october_frozen/test_fairflex_uc_metrics.csv`
- `artifacts/v3/caltech_october_frozen_sensitivity/scenario_hybrid_vs_v1.csv`
- `artifacts/v3/holdout_october_cqr_prediction_audit_v3/test_cqr_prediction_summary.json`
- `artifacts/v3/holdout_october_statistical_audit_v2/paired_calendar_day_statistical_audit.csv`
- the same directories' JSON summaries, which include coverage, buffer,
  forecast, safety-repair, and causal-boundary audits.

## Necessary next evidence—not yet a completed claim

1. Evaluate multiple held-out weeks and use paired **calendar-day** block
   bootstrap intervals; do not bootstrap individual EV rows.
2. Repeat at another ACN site (for example JPL) with a separately fitted CQR
   calibration bank, and report transfer rather than assume it.
3. Vary import capacity, PV forecast error, nominal CQR coverage, and the
   early-unplug threshold. Treat each as a predeclared sensitivity analysis.
4. Replace the synthetic station assignment, feeder limit, and PV proxy with
   site-approved infrastructure data before claiming physical deployment.
5. Report all negative ablations, especially CQR-only and lower-tail-only,
   so the observed benefit is attributed to the hybrid combination rather than
   to an unsupported individual component.
