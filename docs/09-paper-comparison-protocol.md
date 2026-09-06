# Frozen paper-comparison protocol

## Purpose

This protocol tests a precise question, rather than treating every charging
algorithm as if it solved the same problem:

> In constrained workplace EV charging, does FairFlex's fairness-first MPC
> provide better lower-tail user service than standard online schedulers while
> every method uses the same causal departure information, PV-capacity input,
> and executed-action safety rules?

The secondary question is whether the selected multi-rate commitment guard is
worth using at all, compared with an unguarded and a fixed global-guard MPC.
These are different comparisons and must not be mixed.

## Frozen unseen cohort

`configs/jpl_2019_paper_baseline_comparison_november.json` names JPL
November 1--30, 2019 as the primary historical test. At the time this protocol
was written, `data/raw/acn_jpl_20191101_20191201.json` did not exist locally.
The raw file may be downloaded only after all items in this document are
implemented and tested.

March JPL sessions are the forecasting-fit period and April JPL sessions are
the conformal calibration period. The selected four-rate multi-rate guard is
already locked by the prior May development decision, June replication, and
December temporal audit. November therefore cannot select a new method or
adjust an existing one.

## Frozen July seasonal replication

`configs/jpl_2019_seasonal_replication_july.json` repeats the exact primary
policy matrix on JPL July 1--31, 2019. The July raw file is intentionally not
acquired or inspected until this configuration and its invariant test exist.
It uses the unchanged March forecast fit, April calibration set, multi-rate
guard, PV proxy, feeder sensitivity, station mapping, metrics, and day-block
inference. July is a second seasonal test, never a source of post-November
tuning. Results will be reported in full whether they strengthen, weaken, or
contradict the November findings.

## Frozen Caltech external-site replication

`configs/caltech_2019_cross_site_replication_august.json` repeats the same
eight-policy matrix at the public Caltech ACN site for August 1--31, 2019. Its
raw test file is not acquired or inspected until this configuration and its
invariant test pass. The controller, policy set, multi-rate architecture and
rates, PV proxy capacity, synthetic feeder sensitivity, metrics, and inference
are fixed. The only necessary site-specific inputs are Caltech March history
for the forecast fit and Caltech April history for the calibration-score bank;
they do not select a policy, guard mode, or learning rate. This evaluates
external-site transfer of the method, not measured Caltech feeder behaviour.

## Comparisons

### Primary policy matrix

All primary rows replay exactly the same sessions and use the following common
conditions:

- the first declared energy request and declared departure visible at plug-in;
- the same causal multi-rate effective deadline for every policy;
- the same present-time robust PV-capacity source, station limits, synthetic
  feeder mapping, and AC execution check;
- the same 15-minute time step; and
- a fresh simulation object per policy/day.

The frozen policies are uncontrolled charging with a physical shared-cap
wrapper, FCFS, EDF, equal share, continuous-rate Round Robin, LLF, FairFlex
without PV robustness, and FairFlex with robust PV. ACN-Sim treats
uncontrolled charging, Round Robin, FCFS, EDF, and LLF as standard online
benchmarks. FairFlex has continuous power rather than the discrete pilot
increments used by ACN-Sim, so continuous Round Robin is mathematically equal
to equal-share water filling. It remains reported for literature alignment but
is not counted as independent evidence.

No policy may gain energy by violating the shared feeder cap. For policies
that can make a raw unsafe request, record both the raw request violation and
the safe executed result. The primary result concerns the executed schedule.

### Causal-guard ablation

Using only `fair_mpc_ac_robust_pv`, compare these predeclared conditions on the
same November days:

1. no early-departure guard;
2. the fixed global conformal guard; and
3. the selected multi-rate envelope.

This isolates the value of commitment uncertainty handling. The bounded
AgACI-style guard is an already completed May development ablation, not a
candidate to be reselected after observing November.

### Deferred architecture benchmarks

ADMM, MAPPO, and IPPO are valuable secondary experiments but not the main
paper claim. A learned station-cap actor is not a pure EV-level MARL scheduler
in FairFlex: its proposal is projected onto the feeder cap, then local
fair-MPC and AC repair create the final action. It must be labelled as a
safe station-cap architecture ablation. A paper-grade learned benchmark also
requires a predeclared multi-seed training budget and a separately frozen
evaluation window; the current MAPPO evidence has one seed and IPPO has three
pre-test seeds, so neither belongs in the primary November claim yet.

## Outcomes to report

| Category | Required measures | Interpretation |
| --- | --- | --- |
| User service | mean, P10, and worst service ratio; completion; delivered/requested energy | Mean alone can hide users receiving very little charge. P10 is the primary lower-tail measure. |
| Fairness | Jain index; service-ratio distribution/CDF | Jain is a standard allocation summary, but the distribution shows who is left behind. |
| Grid safety | unsafe steps, minimum voltage, maximum line loading, raw proposal violations, repair activations | All values refer to the synthetic IEEE-33 scenario, not measured JPL feeder behaviour. |
| Commitment uncertainty | observed deadline coverage with a confidence interval; mean, median, and P90 buffer | Coverage without buffer is incomplete because an excessively early deadline is trivially safe. |
| Operational cost | median and P95 runtime per day; solver failures | Report descriptively because runtime depends on hardware. |

Each policy difference uses a paired bootstrap over calendar days, never a
bootstrap over individual EV sessions. This preserves the within-day arrival,
weather, and capacity correlation. The comparison output reports both the
natural treatment-minus-baseline difference and an oriented advantage: positive
always means the treatment is preferable, including lower-is-better metrics
such as unsafe steps or line loading.

A calendar day with no usable session after the already-declared ACN cleaning
rule is recorded in an `*_empty_replay.json` audit file and excluded from the
matched-day sample. It is not assigned zero service or zero fairness, because
those quantities have no denominator. This rule applies identically to every
policy and is reported in the matrix manifest.

## Claims that remain out of scope

- No measured JPL feeder-voltage, thermal, or rooftop-PV claim.
- No electricity-cost claim without one fixed tariff for every policy.
- No per-driver or conditional conformal guarantee.
- No claim that MAPPO is superior to MPC merely because it shares FairFlex's
  downstream safety/MPC layer.
- No "state of the art" claim from a single one-month site test. A second,
  separately frozen site/month is required for that level of evidence.

## Execution gate

Before the November file is acquired:

1. Baseline unit tests and the full regression suite must pass.
2. The configuration must parse without requiring the missing test file.
3. The matrix runner must expose the frozen policy names and correctly mark
   whether higher or lower is preferable for each reported metric.
4. The guard-ablation runner must support exactly the three listed conditions
   without retuning the guard.

Only then may the official ACN download occur, followed by a manifest with
raw-file checksums, the primary matrix, the guard ablation, and generated
tables/figures from those artifacts.
