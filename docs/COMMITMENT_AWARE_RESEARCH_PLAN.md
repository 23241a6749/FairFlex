# FairFlex-C: commitment-aware, grid-safe fair EV charging

## The honest research position

This is **not** a claim that fair EV charging, MPC, AC power-flow safety,
distributed optimisation, PV forecasting, or early-departure robustness have
never been published. They have.

The paper's narrower question is:

> Can a fairness-first EV controller protect the lower tail of **driver-declared
> requested-energy service** when drivers may unplug earlier than they declared,
> while applying only AC-grid-safe actions?

The contribution is the tested combination and causal evaluation protocol:

1. Use a declared request and declared departure only when an EV plugs in.
   Do not use the future historical `disconnectTime` to schedule it.
2. Fit a one-sided, split-conformal early-departure buffer on a chronological
   calibration split. The test split cannot change the buffer.
3. Treat the actual unplug time as a hidden simulator event. After a guarded
   deadline passes, an EV that is still visibly connected gets one immediately
   observed safe slot, then MPC replans. This avoids wasting known availability
   without assuming the EV will remain connected.
4. Optimise lexicographic lower-tail fairness, then energy delivery, then an
   economic/ramp tie-break. Validate each applied action with AC power flow and
   repair unsafe proposals.
5. Report both user service and physical safety, including the fraction of
   requests that were impossible even at the EV's own maximum rate before its
   actual unplug.

This is a potentially publishable **method-and-evaluation contribution** only
if the final chronological experiments support it. We must not claim it is
the first work to use any individual ingredient.

## Closest work and how FairFlex-C differs

| Existing work | What it already establishes | FairFlex-C must not claim | Specific remaining distinction |
| --- | --- | --- | --- |
| Tsaousoglou et al. (2023), *Fair and Scalable EV Charging Under Electrical Grid Constraints* | Max-min fairness, multi-station/grid constraints, scalable distributed optimisation, and an ADMM comparison | “First fair, grid-safe, distributed EV charging method” | FairFlex-C is not centred on distributed optimisation. It evaluates declared commitments and realized early unplugging from public ACN traces under a causal protocol. |
| Wang, Pi & Tang (2017), *Scheduling of EV Charging via Multi-Server Fair Queueing* | Fair scheduling under voltage/transformer constraints; explicitly notes early departure | “First fair early-departure-aware controller” | FairFlex-C uses a calibrated coverage-controlled guard, real declared ACN commitments, and an online reactivation rule. |
| Ahmad et al. (2023), *A Two-Layers Predictive Algorithm for Workplace EV Charging* | Caltech ACN data, predicted departures, cost-aware scheduling under uncertain user inputs | “First uncertainty-aware Caltech scheduler” | FairFlex-C measures requested-energy lower-tail fairness and uses declared commitment plus calibrated early-departure protection rather than cost-oriented departure prediction alone. |
| Winschermann et al., LYNCS experiments | Earlier charging can improve robustness to early departure and QoS fairness | “First fair schedule robust to early unplugging” | FairFlex-C uses online receding-horizon action, a calibration-locked conformal buffer, AC safety auditing, and a public trace protocol with declared requests. |
| Conformal PV/MPC studies | Conformal uncertainty calibration for energy forecasts/control | “First conformal MPC for EV/PV” | The conformal element here is a one-sided *driver-commitment* guard. PV uncertainty remains a separate sensitivity component. |

Before submission, repeat this literature review in IEEE Xplore, Scopus/Web of
Science, and Google Scholar with the exact final title and method terms. A web
search can never prove that no related paper exists.

## Dataset: derived benchmark, not newly collected data

We are **not collecting or inventing a new raw EV dataset**.

The experiment creates a reproducible processed benchmark from:

- public ACN-Data Caltech session records: real plug-in time, actual unplug
  time, delivered energy, and—only for claimed sessions—driver `kWhRequested`
  and `requestedDeparture`;
- NSRDB solar-resource observations, converted to a documented PV sensitivity
  proxy; and
- a synthetic IEEE-33 feeder/station mapping used only as a controlled grid
  stress scenario.

Call this a **derived causal ACN benchmark protocol**, not a new real-world
dataset. Share code, configuration, hashes, diagnostics, and acquisition
instructions. Do not redistribute raw records until the source terms have
been checked. The manifest deliberately contains no token, email address, raw
session, or user identifier.

## Fixed chronological protocol

| Split | Dates | Permitted use |
| --- | --- | --- |
| Train | 1–7 March 2019 | Fit any feature-based commitment or PV model. Do not report final outcomes. |
| Calibration | 1–7 April 2019 | Fit the fixed 90% one-sided early-departure conformal buffer and calibrate PV uncertainty. |
| Development checks | 1–7 May and 1–7 June 2019 | Diagnose the protocol and compare the predeclared global and duration-stratified guards. |
| Final seasonal replications | 1–7 September and 1–7 December 2019 | Evaluate the frozen global guard. Do not tune a method, hyperparameter, or guard after inspecting these results. |

The primary protocol keeps claimed sessions with an initial complete user
commitment. The controller gets the initial declared request/deadline only;
the actual `disconnectTime` is hidden until it happens. Sessions without a
commitment are excluded from the primary user-satisfaction study and will be
handled later as a clearly separated missing-information robustness study.

## What is implemented now

1. Causal session model: actual unplug time and controller planning deadline
   are separate fields. This removes future-information leakage.
2. Claimed-session conversion: request energy comes from `userInputs` rather
   than historical delivered energy. The earliest complete declaration is the
   transparent plug-in approximation.
3. One-sided split-conformal guard: calibration early-departure advances give
   a fixed deadline buffer. It has a marginal coverage interpretation under
   exchangeability, not a guarantee for every driver.
4. Reactive re-planning: survival beyond the buffer is observed information,
   so the next action may charge in the current slot but assumes no later slot.
5. Fairness-first MPC: minimum worst requested-energy shortfall; maximum
   delivery without sacrificing that fairness; then a cost/ramp tie-break.
6. Exact AC action validation and repair, robust PV-capacity sensitivity, and
   reproducible summary files.
7. Expanded metrics: mean/P10/worst service, Jain index, total requested and
   delivered energy, energy-weighted service, completion rate, individual
   physical feasibility, unavoidable individual shortfall, voltage, line
   loading, safety violations, and runtime.
8. Past-only adaptive guard diagnostic: an ACI-inspired update can use an
   earlier EV's realized unplug only after it has physically happened. The
   implementation records its effective risk and buffer at every plug-in. It
   is deliberately bounded for operational safety and is labelled exploratory,
   not a replacement for a split-conformal guarantee.

## First held-out sanity check (not a final claim)

The first development-week central-MPC comparison uses the same 194 claimed May
sessions, 10 kW synthetic import cap, robust-PV sensitivity, and AC safety
check in both conditions.

| Condition | Mean service | P10 service | Jain | Delivered energy | Unsafe steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Declared deadline only | 0.430 | 0.134 | 0.655 | 1421.2 kWh | 0 |
| 90-minute conformal guard plus reactive re-planning | 0.442 | 0.146 | 0.709 | 1621.3 kWh | 0 |

The calibration-derived buffer was six 15-minute slots (90 minutes). Its
May coverage was 0.897 against the predeclared target of 0.900. The guarded
condition did **not** improve every metric: its worst service and completion
rate were lower. In the June development replication, the global guard met the
coverage target (0.927) and had better P10/Jain/energy than the coarse
duration-stratified candidate, so the global guard is now frozen for September
and December. This remains development evidence, not a headline claim.

## Frozen final seasonal evidence: report both the benefit and the failure

The following comparisons are from seven matched daily replays in each frozen
season under the common synthetic 10 kW import cap. They are sensitivity-study
results, not measurements of Caltech's physical feeder.

| Season | FairFlex robust-PV P10 | Equal-share P10 | FCFS P10 | FairFlex Jain | Unsafe applied steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| September 2019 | 0.294 | 0.234 | 0.163 | 0.788 | 0 |
| December 2019 | 0.195 | 0.184 | 0.098 | 0.735 | 0 |

The matched day-block bootstrap supports a September P10 increase of 0.0447
for the guarded controller versus its unguarded version (95% exploratory
percentile interval [0.0173, 0.0782]). In December, the corresponding P10
increase is 0.0109 with an interval crossing zero [-0.0082, 0.0347]. Thus the
guard's lower-tail benefit is promising but not consistent enough to be called
established across seasons.

More importantly, the frozen April 90% split-conformal buffer did **not**
retain nominal empirical coverage in the later seasons. This is a required
negative result, not an error to suppress:

| Protocol | September sessions / coverage | December sessions / coverage | Target |
| --- | ---: | ---: | ---: |
| Frozen fixed split-conformal baseline | 169 / 82.8% | 169 / 87.0% | 90.0% |
| Past-only adaptive diagnostic (learning rate 0.01) | 169 / 90.5% | 169 / 87.6% | 90.0% |

The adaptive September result is encouraging, but December again misses the
target. It was designed after seeing the fixed-guard seasonal failure, so it
is explicitly exploratory and cannot be presented as a final or confirmatory
solution. The transparent artifact table is
`artifacts/commitment_fairness/coverage_evidence/seasonal_commitment_coverage.csv`.

This changes the paper claim appropriately: FairFlex-C can claim a causal
benchmark and a rigorous identification of commitment-shift failure; it cannot
claim a universally calibrated early-departure protection method yet.

## Locked analysis-held-out October validation

After the above evidence, we ran a new development protocol without touching
the October raw file: May selected an adaptive learning rate from the fixed set
{0.005, 0.01, 0.02}; only 0.005 met May's 90% coverage target (90.2%). That
same value reached 92.7% in the separate June replication. We then froze the
configuration, downloaded the previously unused historical October 1--7 ACN
window through the official API, recorded its raw SHA-256, and ran it once.

This is an **analysis-held-out historical validation**, not a real-time
prospective deployment: the data occurred in 2019, but were not acquired or
examined by this project until after the May/June selection record was frozen.

| October locked result | Value |
| --- | ---: |
| Raw / usable declared-commitment sessions | 191 / 154 |
| Adaptive coverage target / observed coverage | 90.0% / 87.7% |
| FairFlex robust-PV P10 service | 0.142 |
| Equal-share / FCFS / LLF P10 service | 0.105 / 0.035 / 0.077 |
| Unsafe applied steps, every compared policy | 0 |

The controller's lower-tail service and electrical safety remain positive
findings for this controlled scarcity scenario. However, the adaptive guard
again misses its primary coverage target. Therefore this new locked result
**rejects** the claim that the current adaptive guard solves seasonal
early-departure uncertainty. It must be presented as a failed robustness
validation, not tuned further on October.

The report is generated directly from frozen artifacts at
`artifacts/commitment_fairness/prospective_october/final_report/prospective_validation_report.md`.

## Multi-rate development and frozen JPL cross-site transfer evidence

The October failure was not used to tune the earlier adaptive guard. Instead,
we designed a new **conservative multi-rate envelope** before acquiring any JPL
sessions. It runs four independently causal ACI-inspired experts with fixed
learning rates {0.001, 0.005, 0.01, 0.02} and gives the controller the
earliest (most conservative) deadline proposed by the experts. This is
inspired by the multi-rate motivation in AgACI, but is **not AgACI**: it has no
learned expert-aggregation rule and carries no new theoretical guarantee.

The selection itself was deterministic and coverage-first. It used Caltech May
only for development, ran the unchanged envelope in Caltech June, and compared
it with the previously selected single-rate (0.005) adaptive guard. Both
methods met the 90% target in May and June, but the envelope had a higher
minimum observed coverage and was selected before JPL data were acquired.

| Development/replication summary | Single-rate adaptive | Multi-rate envelope |
| --- | ---: | ---: |
| May coverage | 90.2% | 92.3% |
| June coverage | 92.7% | 92.7% |
| Mean P10 service | 0.186 | 0.189 |
| Mean Jain index | 0.719 | 0.724 |
| Mean energy service | 0.446 | 0.446 |
| Mean guard buffer | 80.2 min | 90.8 min |
| Mean runtime | 97.0 s | 137.2 s |

We then froze
`configs/jpl_2019_cross_site_transfer_multirate_october.json`, downloaded the
previously unseen JPL October 1--8, 2019 window through the official API, and
recorded a credential-free input manifest. JPL is a useful stress site because
ACN documents it as an employee workplace, whereas Caltech is a public campus
site. This is still an **analysis-held-out historical transfer test**, not a
real-time deployment.

The continuous seven-day replay deliberately did not reset the guard at
midnight. It retained Caltech April as the calibration source, then allowed
only JPL departures that had already happened to update each expert. Of 387
raw JPL records, 379 had usable initial declared commitments.

| Frozen JPL continuous result | Value |
| --- | ---: |
| Deadline-coverage target / observed coverage | 90.0% / 91.8% |
| Causal plug-in decisions | 379 |
| Mean guard buffer (min--max) | 191.5 min (30--390) |
| FairFlex robust-PV P10 service | 0.057 |
| Equal-share / FCFS / LLF P10 service | 0.032 / 0.000 / 0.000 |
| FairFlex robust-PV versus no-PV P10 | 0.057 versus 0.043 |
| Unsafe applied steps, every compared policy | 0 |

This is a positive robustness result, but it must be described precisely. The
envelope met its empirical target on this one new site/window, while its four
experts alone achieved 87.1%, 88.9%, 90.2%, and 90.2% respectively. The
conservative envelope therefore bought coverage using a much longer buffer.
It also improved P10 relative to every simple baseline and improved energy
service substantially over the no-PV ablation. It **did not** beat equal share
on mean service or Jain index (0.166/0.366 versus 0.207/0.463), so it does not
support a universal-average-fairness claim. The defensible claim is a
causally-evaluated lower-tail-service and early-departure-robustness result
under a clearly labelled synthetic scarcity scenario.

The day-block bootstrap gives positive P10 intervals versus every listed
baseline, but it resets the adaptive state every day. It is consequently a
policy-metric sensitivity analysis only; the continuous JPL replay is the sole
source of the coverage claim. Full artifacts are at:

- `artifacts/commitment_fairness/multirate_guard_selection/selection.json`
- `artifacts/commitment_fairness/jpl_cross_site_october/data_manifest.json`
- `artifacts/commitment_fairness/jpl_cross_site_october/full_week/test_pilot_summary.json`
- `artifacts/commitment_fairness/jpl_cross_site_october/final_report/cross_site_transfer_report.md`

All regression tests passed after this addition: **68 passed**.

## Predeclared AgACI-style efficiency ablation and final temporal audit

The remaining work above was completed without reopening the earlier Caltech or
JPL October evidence.  Before acquiring the new evaluation sessions, we froze
three new JPL configurations:

1. March 2019 was the training period and April 2019 was the calibration
   period for every candidate.
2. May 1--8 was the development comparison between the conservative
   multi-rate envelope and a **bounded AgACI-style weighted guard**.
3. June 1--8 was reserved to replicate *only* the May-selected guard; December
   1--8 was reserved for a single final temporal audit.

The weighted candidate follows the online-expert idea of AgACI: several
past-only adaptive-conformal experts, with rates 0.001, 0.005, 0.01, and 0.02,
are combined using online loss-based weights.  It is deliberately named
**bounded AgACI-style**, rather than claimed as a literal AgACI implementation:
the EV problem has discrete 15-minute deadlines, delayed physical unplug
outcomes, a finite calibration-score bank, and bounded miscoverage values.
Those engineering constraints make the extension useful to test but do not
transfer AgACI's theory unchanged.

The selection rule was written before May was read: first discard a candidate
below 90% empirical deadline coverage; then maximize coverage, P10 service,
Jain index, and energy service in that order; use the smaller mean buffer only
as a final tie-breaker.  The frozen May comparison was:

| JPL May development candidate | Deadline coverage | Mean buffer (min) | P10 service | Jain | Energy service | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative multi-rate envelope | 93.16% | 144.83 | 0.0465 | 0.3218 | 0.1127 | 240.20 |
| Bounded AgACI-style weighted guard | 90.53% | 119.96 | 0.0475 | 0.3357 | 0.1127 | 258.39 |

Both candidates cleared the target.  The weighted guard saved about 25 buffer
minutes and slightly improved the two fairness measures, but the envelope was
selected because the rule deliberately put reliability first (93.16% versus
90.53%).  This is a useful, honest negative result for the more complex method:
it did not win the predeclared reliability-first decision, so it was not
silently promoted after seeing later data.

The envelope alone was then replayed unchanged on June and December.  In both
periods it met the target, with zero unsafe applied-control steps under the
clearly labelled synthetic shared-import/AC scenario:

| Held-out audit | Usable sessions | Deadline coverage | Mean buffer (min) | P10 service | Jain | Energy service | Unsafe steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| June replication | 309 | 92.23% | 142.18 | 0.0595 | 0.3819 | 0.1511 | 0 |
| December final temporal audit | 322 | 91.93% | 184.05 | 0.0497 | 0.3170 | 0.1200 | 0 |

The December result is the strongest statement from this phase: an unchanged
method stayed above its 90% operational target in a later season, but required
a larger buffer.  It is **not** a finite-sample, conditional, per-driver, or
real-time deployment guarantee.  It is empirical evidence from two held-out
historical JPL windows using only outcomes available before each plug-in
decision.  The separate printed PV-forecast coverage is a different metric and
must never be confused with this early-departure coverage.

Reproducible artifacts for this phase are:

- `artifacts/commitment_fairness/agaci_development_jpl_may/candidate_metrics.csv`
- `artifacts/commitment_fairness/agaci_development_jpl_may/selection.json`
- `artifacts/commitment_fairness/agaci_replication_jpl_june/replication.json`
- `artifacts/commitment_fairness/agaci_temporal_validation_jpl_december/multi_rate_envelope/test_pilot_summary.json`

The complete regression suite after the new guard and this evaluation phase:
**71 passed in 31.38 seconds**.

## What remains before a paper-ready final claim

### Phase 1 — validate the evidence protocol

1. Keep all existing seasonal, October, and JPL results frozen. In particular,
   do not tune the selected envelope on JPL or reuse the below-target
   September/December/October coverage findings as hidden training data.
2. If buffer efficiency remains a research objective, predeclare a genuinely
   new candidate (for example, a faithful AgACI-style causal aggregation) and
   develop it on a separate source cohort. Reserve a later window or another
   site for its next test.
3. Add a data card documenting dropped sessions, user-input revisions,
   time-zone handling, and all synthetic assumptions.

Success condition: a reader can reproduce the same sessions, buffer, and
scenario without a secret or a manual spreadsheet step.

### Phase 2 — strengthen the commitment model using training data

1. Preserve the global split-conformal guard as the simple baseline.
2. Predeclare any next adaptive candidate and its learning rate using only
   development evidence, then test it on a newly acquired untouched period.
3. Compare coverage, buffer size, P10, Jain, energy service, and safety. Do
   not add a complex neural predictor unless it improves the risk/service
   trade-off on that untouched period.
4. Add a separate energy-request reliability analysis. Do not silently change
   a user's stated request into historic delivered energy.

Success condition: any extra complexity earns its place through an ablation;
otherwise retain the simple global guard.

### Phase 3 — robust controller and baselines

1. Run central FairFlex-C with and without the commitment guard.
2. Compare FCFS, equal share, least-laxity-first, unguarded MPC, robust-PV
   only MPC, and commitment-guard MPC. Keep MAPPO/IPPO and ADMM as secondary
   architecture baselines, not the paper's main novelty claim.
3. Keep a common physical simulator and AC safety wrapper for every policy.
   Label a learned policy as a station-cap allocator if it relies on the same
   MPC/safety layer for EV-level allocation.

Success condition: comparisons differ in one declared factor at a time.

### Phase 4 — rigorous held-out evaluation

1. The September/December policy matrix and day-block bootstrap are complete.
   Repeat the same evaluation on the new untouched period; never
   resample individual EVs as if they were independent days.
2. Report confidence intervals for mean, P10, worst, Jain, energy service,
   completion, and safety. Treat runtime separately because lower is better.
3. Repeat across predeclared feeder-scarcity/PV sensitivities and, if feasible,
   another chronological season/site. Never combine synthetic feeder claims
   with claims about Caltech's actual feeder.
4. Inspect the worst-decile sessions and every safety repair. Report failures,
   not just averages.

Success condition: the benefit survives uncertainty bounds and is not limited
to a single favourable week or scenario.

### Phase 5 — paper-ready artefacts

1. Generate tables/plots from CSV and JSON artifacts only; no hand-edited
   result values.
2. Write an ablation table, safety table, uncertainty-coverage table, and
   limitations section.
3. Release a reproducibility package containing environment lock file, study
   config, tests, scripts, manifests, and documentation.

## Key commands

```powershell
cd D:\saimohaneesh\FairFlex
.\.venv\Scripts\Activate.ps1

python scripts\inspect_commitment_protocol.py `
  --config configs\caltech_2019_commitment_fairness.json

python scripts\run_trace_pilot.py `
  --config configs\caltech_2019_commitment_fairness.json `
  --split test --skip-distributed `
  --apply-commitment-guard `
  --only fair_mpc_ac_robust_pv `
  --output-dir artifacts\commitment_fairness\guarded_reactive

# Exploratory drift diagnostic; do not substitute this for a fresh final test.
python scripts\run_trace_pilot.py `
  --config configs\caltech_2019_commitment_fairness_adaptive_exploratory_september.json `
  --split test --skip-distributed `
  --apply-commitment-guard --commitment-guard-mode adaptive `
  --only fair_mpc_ac_robust_pv `
  --output-dir artifacts\commitment_fairness\adaptive_exploratory_september
```

## Sources to cite and inspect

- ACN-Data: https://ev.caltech.edu/dataset.html
- Lee, Li & Low (2019), ACN-Data paper:
  https://ev.caltech.edu/assets/pub/ACN_Data_Analysis_and_Applications.pdf
- Tsaousoglou et al. (2023): https://doi.org/10.1109/TITS.2023.3311509
- Wang, Pi & Tang (2017): https://doi.org/10.1109/TPDS.2017.2710197
- Ahmad et al. (2023): https://arxiv.org/abs/2307.08311
- LYNCS experiment repository:
  https://github.com/lwinschermann/FlowbasedOfflineChargingScheduler
- Gibbs & Candès (2021), Adaptive Conformal Inference Under Distribution
  Shift: https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
- Zaffran et al. (2022), Adaptive Conformal Predictions for Time Series
  (AgACI): https://proceedings.mlr.press/v162/zaffran22a.html
- Gibbs & Candès (2024), Conformal Inference for Online Prediction with
  Arbitrary Distribution Shifts: https://www.jmlr.org/papers/v25/22-1218.html
