# Safe-MAPPO Benchmark Protocol for FairFlex-UC

## Purpose

This document specifies a **secondary architecture benchmark**, not a new
primary claim.  It answers a narrow, reproducible question:

> When both methods receive the same causal charging-session information,
> CQR early-unplug buffer, physical feeder cap and action-repair rule, does a
> trained station-cap MAPPO controller improve lower-tail service over the
> deterministic FairFlex-UC controller?

The benchmark must not be described as a pure EV-level MARL controller.  A
MAPPO actor proposes station-cap fractions; deterministic projection, local
lower-tail MPC, and physical action repair then determine the EV charging
rates.  It is therefore an architecture ablation of the upper allocation layer.

## Why a new benchmark is necessary

The existing MAPPO checkpoint was trained on a different scarce-data cohort
and routes its proposed station caps through the earlier centralized fair-MPC
allocator.  It therefore differs from FairFlex-UC in the guard, downstream
objective, data split, and evaluation window.  Comparing its score directly to
V3 would mix several changes and would not show whether MAPPO itself helps.

## Common conditions (non-negotiable)

Every V1, V3, MAPPO, and IPPO run in this benchmark must use:

1. the same recorded ACN arrival, request and declared-departure information;
2. the same CQR model and calibration-only post-hoc conformal correction;
3. the same conservative hybrid deadline guard applied at arrival;
4. the same 15-minute decisions, station limits, PV-capacity proxy, feeder
   cap, AC feasibility check, and final safety repair;
5. the same train/validation/test splits and cleaning rules; and
6. a fresh simulation instance for every policy, seed, and replay day.

No test-session feature, actual unplug time, or test outcome may be used in an
actor observation, reward, normalization statistic, early stopping decision,
or hyperparameter choice.

## MAPPO architecture

### Agents and execution information

There is one homogeneous agent per charging station.  At execution time, an
actor observes only its station-local causal state: active-session demand and
remaining guarded time, connected-port count, local capacity use, station cap
from the prior step, and present PV/capacity values allowed to all policies.
It outputs a bounded cap fraction.  A shared-parameter actor is used because
stations play the same role and this makes the method scalable to a different
number of stations.

During training only, a centralized critic may read the joint station state and
the feeder state.  This is the standard **centralized training, decentralized
execution (CTDE)** structure of MAPPO.  The critic is discarded at deployment;
it cannot supply hidden global information to the actor.

### Deterministic safety and local allocation

The vector of actor cap fractions is projected deterministically onto the
physical shared feeder cap.  Each station then invokes the *same*
`LowerTailFairMPC` local allocator used in FairFlex-UC, and the shared AC
repair is applied last.  These components make every executed action feasible;
they do not make MAPPO a direct implementation of the V3 controller.

The primary deterministic comparator is FairFlex-UC V3.  An IPPO version uses
the same actor, action space, reward, projection, local allocator, and safety
layer but replaces the centralized critic with a local critic.  Thus the
MAPPO--IPPO difference isolates the critic information rather than changing
the plant or guard.

## Training and selection protocol

1. Freeze the environment, observations, action scale, reward coefficients,
   PPO update budget, network size, learning-rate schedule, and five random
   seeds in a tracked configuration before training.
2. Train MAPPO and IPPO using only the historical training period.
3. Evaluate checkpoints only on a separate validation period.  Select one
   fixed checkpoint rule before looking at the untouched test period (for
   example: maximize validation P10 subject to zero executed unsafe steps).
4. Retain all seeds and report their validation distribution.  Do not select
   the best seed after seeing test results.
5. Acquire and evaluate a separately registered, never-tuned temporal test
   window only after code, selected checkpoint rule, and all seeds are frozen.

The test result is reported per seed and as the equally weighted seed mean;
the seed spread is part of the result, not noise to hide.  Neural-policy
training does **not** have a single predictive "accuracy".  Its evidence is
the out-of-sample service, safety, runtime, and stability distribution.

## Reward design

Use an interpretable common reward at each step:

`delivered energy - lambda_tail * lower-tail service shortfall
 - lambda_unserved * guarded-deadline shortfall
 - lambda_repair * repair magnitude - lambda_unsafe * raw unsafe proposal`.

The executed-feasibility constraint is never replaced by a penalty: the
projection and repair remain hard safeguards.  Reward weights are fixed from
training/validation experiments and then locked.  The complete reward,
including units and coefficient values, must be disclosed so that the
benchmark is reproducible.

## Evaluation and statistical analysis

Primary endpoint: calendar-day P10 service ratio on a continuous multiweek
replay.  Secondary endpoints are early-unplug P10, expected shortfall below
the service threshold, Jain's index, mean/worst service, delivered energy,
unsafe steps, repair magnitude, runtime, and CQR coverage/buffer efficiency.

Within each seed, compare MAPPO and V3 on precisely the same calendar days.
Use paired calendar-day bootstrap intervals and an exact paired sign-flip test
for the predeclared primary P10 endpoint.  Do not apply a session-level t-test
or ordinary ANOVA: sessions inside a day share weather, arrivals, capacity and
control trajectory, so they are not independent observations.  For a final
multi-seed summary, resample calendar days and seeds as nested clusters; do
not treat all seed-day rows as independent.

Report the raw metric difference, its direction-oriented advantage, confidence
interval, p-value, the number of eligible matched days, and all failures.  A
confidence interval crossing zero means the result is inconclusive, even if a
single seed appears better.

## Interpretation boundaries

- A MAPPO win would show that learned station-cap coordination helped in this
  specified simulator and cohort; it would not prove a universal MARL win.
- A V3 win would support the value of deterministic lower-tail MPC under the
  same safety/uncertainty layer; it would not show MAPPO was poorly implemented
  unless all predeclared training diagnostics pass.
- Neither result establishes superiority to papers that use different sites,
  feeder models, time resolutions, data-cleaning rules, or fairness targets.
- The CQR coverage guarantee is marginal under exchangeability.  It is audited
  empirically by calendar day, and is not a guarantee for each driver.

## Evidence sources

- Yu et al., *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent
  Games* (MAPPO): https://arxiv.org/abs/2103.01955
- Official MAPPO implementation: https://github.com/marlbenchmark/on-policy
- Romano, Patterson, and Candès, *Conformalized Quantile Regression*:
  https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html
- Caltech ACN-Data documentation: https://ev.caltech.edu/dataset.html

