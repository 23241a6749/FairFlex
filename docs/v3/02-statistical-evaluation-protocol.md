# FairFlex-UC V3 statistical evaluation protocol

## The key distinction: prediction quality is not policy accuracy

FairFlex contains two different technical tasks.  They must not be compressed
into one vague "accuracy" percentage.

1. **Early-unplug prediction:** CQR predicts a conservative upper quantile of
   how much earlier a driver may unplug than their declared departure.  Here we
   can compare a prediction against the actual observed early-unplug advance.
2. **Charging control:** MPC chooses a charging action under constraints.  Its
   output is not a prediction with a single correct label, so it is evaluated
   by fairness, delivered energy, safety, cost, and runtime—not classification
   accuracy.

The frozen October test period was never used to fit CQR or select V3.  It is
the correct place to calculate the post-hoc diagnostics below.

## A. How CQR prediction quality is measured

For EV \(i\), let \(a_i = \max(0, d_i^{declared}-d_i^{actual})\) be its actual
early-unplug advance in 15-minute intervals.  Let \(b_i\) be the integer CQR
buffer selected before its outcome is known.  CQR is a one-sided 90th-quantile
predictor, not a yes/no early-unplug classifier.

| Measure | Formula / meaning | Desired result |
| --- | --- | --- |
| Empirical coverage | \(n^{-1}\sum_i 1[a_i \leq b_i]\) | Near or above the declared 0.90 target |
| Pinball loss | \(n^{-1}\sum_i\max(q(a_i-b_i),(q-1)(a_i-b_i))\), \(q=0.90\) | Smaller, when compared on the same target and split |
| Sharpness | Mean, median, P90 of \(b_i\) | Smaller buffers, but only if coverage remains adequate |
| Under-buffer amount | mean \(\max(a_i-b_i,0)\) | Smaller |
| Over-buffer amount | mean \(\max(b_i-a_i,0)\) | Smaller, subject to coverage |

This follows the purpose of conformalized quantile regression: calibrated
coverage together with useful, context-dependent intervals, rather than
ordinary point-prediction accuracy.  See [Romano, Patterson, and Candès
(2019)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).

### Current frozen October CQR audit

| Quantity | Result | Meaning |
| --- | ---: | --- |
| Target coverage | 0.900 | Declared before testing |
| Observed coverage | 141 / 154 = 0.916 | CQR protected 141 sessions against their actual early unplug |
| Calendar-day cluster-bootstrap 95% interval | [0.888, 0.940] | Includes 0.900; seven days are too few for a strong calibration claim |
| CQR mean buffer | 8.753 steps | About 131 minutes; conservatism has an energy/fairness cost |
| Static global-buffer mean | 9.000 steps | A no-context calibration-only baseline |
| CQR minus static buffer | -0.247 steps | CQR used about 3.7 fewer minutes per session on average |
| CQR conformal-buffer pinball loss | 1.151 | Lower is better, in 15-minute-step units |
| Static-buffer pinball loss | 1.195 | Same evaluation period and 90% target |

Both CQR and the static global baseline obtained 0.916 coverage in this week.
Therefore the honest statement is: *CQR matched the baseline's coverage while
being slightly sharper on this holdout.*  It is not evidence that CQR is always
more accurate, and it does not prove per-driver or subgroup coverage.

The report contains a familiar IID-session Wilson interval as a scale aid, but
the calendar-day clustered bootstrap is the appropriate descriptive interval:
EVs occurring on the same day share arrivals, PV, and capacity conditions.

## B. How charging-policy performance is tested

The primary endpoint was frozen before October testing:

\[
P10 = \text{10th percentile of per-EV service ratio}, \qquad
\text{service ratio}=\frac{\text{delivered kWh}}{\text{declared requested kWh}}.
\]

High P10 means the least-served 10% of drivers receive better service.  It is
the main fairness outcome.  The secondary outcomes are early-unplug P10,
10%-tail expected shortfall of service deficits, mean service, Jain's index,
delivered energy, cost (when available), runtime, and unsafe steps.

Jain's index, total delivered energy, cost/peak/grid-violation measures, and
individual fulfilment ratios are commonly reported in EV-charging studies.
For example, the ACN-Data paper demonstrates real workplace EV analysis and
prediction applications [Lee et al. (2019)](https://doi.org/10.1145/3307772.3328313),
and the Caltech workplace prediction paper evaluates requested-energy supply
under a different cost-oriented objective [Ahmad et al. (2023)](https://arxiv.org/abs/2307.08311).
P10 and early-unplug P10 are additional, project-specific lower-tail fairness
endpoints: they make the drivers most likely to be harmed visible.

### Why not an EV-row t-test or ordinary ANOVA?

Those are not the main tests here.  A t-test compares means; P10 is a
non-smooth tail statistic.  More importantly, the same calendar day is replayed
by both policies, and the EV sessions from that day are dependent.  Treating
154 EV rows as 154 independent policy experiments would create false
confidence (pseudo-replication).  Standard t-tests also distinguish paired
from unpaired observations; our two policy outcomes are inherently paired by
day.  See the [NIST description of paired versus unpaired t-tests](https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm).

The correct current method is:

1. Preserve the matching: V3 and V1 replay exactly the same ACN sessions, PV
   trace, and synthetic-feeder condition.  For a continuous adaptive-guard
   run, first complete the full replay and only then group its session outcomes
   by **arrival day**; do not reset the guard at midnight merely to form blocks.
2. Compute each day block’s policy-metric difference, oriented so positive means
   FairFlex is better.
3. Resample whole matched days for a 95% paired calendar-day bootstrap
   interval.  This respects within-day dependence.
4. For the predeclared P10 endpoint, enumerate all label swaps of the matched
   days when there are at most 16 blocks (an exact paired sign-flip/randomisation
   test); for a longer registered cohort, use the same label-swap null with a
   fixed-seed 100,000-draw Monte-Carlo approximation.  This avoids assuming
   that daily P10 differences are normally distributed.  A matched-pair
   permutation test works by swapping the two labels, equivalently changing
the signs of paired differences under its null hypothesis [Permutation tests for
experimental data](https://pmc.ncbi.nlm.nih.gov/articles/PMC10066020/).
5. Keep P10 as the one primary test.  If several secondary/exploratory tests
   are reported together, use Holm adjustment within that secondary family.

### Current October V3 versus guarded V1 evidence

| Endpoint | Eligible matched days | V3 advantage | 95% paired-day bootstrap interval | Exact one-sided p-value | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| **P10 service ratio (primary)** | 5 | +0.0164 | [-0.0055, +0.0434] | 0.1875 | Direction is positive but inconclusive |
| Tail expected shortfall (secondary; lower is better) | 7 | +0.0122 | [-0.0058, +0.0392] | 0.2813 | Inconclusive |
| Mean service (exploratory) | 7 | -0.0063 | [-0.0116, -0.0010] | 0.9688 | Fairness gain costs mean service |
| Delivered energy (exploratory) | 7 | -1.351 kWh | [-3.271, -0.155] | 1.0000 | Fairness gain costs energy |

The primary p-value is not below 0.05, and its interval crosses zero.  This is
not a failure of the method; it means the seven-day diagnostic is too small to
claim a confirmed V3-over-V1 advantage statistically.  That October table is
now explicitly a *midnight-reset guard sensitivity analysis*.  It does not
replace the newer continuous-replay day-block analysis, which preserves the
guard state and groups post-hoc session outcomes by arrival day.

If a future paper needs one *omnibus* comparison of three or more controllers,
use a calendar-day repeated-measures analysis: Friedman’s test is a reasonable
non-parametric exploratory screen, followed by paired, Holm-adjusted
comparisons.  Repeated-measures ANOVA or a paired t-test may be reported only
as a sensitivity analysis after there are many predeclared day blocks and the
distribution of day-level differences is plausibly suitable.  Neither should
be run on individual EV rows or treated as the primary result for daily P10.

The early-unplug daily P10 is deliberately withheld: only two days have at
least ten early-unplug sessions.  Reporting a test there would look precise but
would not be trustworthy.

## C. What training has happened, and what has not

- **CQR prediction model:** trained once on 946 earlier Caltech sessions;
  conformally calibrated on 737 later sessions; evaluated on 154 frozen October
  sessions.  This is real supervised learning.
- **MPC controller:** not "trained" in the machine-learning sense.  At every
  15-minute decision it solves a constrained optimisation using the chosen
  causal deadline and PV forecast.  Its objective is lower-tail service, then
  energy and operating tie-breaks.
- **MAPPO/IPPO:** a policy-network baseline needs separate, multi-seed training.
  It has not yet been evaluated under this exact V3 frozen protocol, so no
  V3-versus-MAPPO accuracy or significance claim is allowed.

For a fair MAPPO comparison, use at least five independently trained seeds,
the same causal observations and safety wrapper, and many frozen calendar days.
Resample days and training seeds as clusters (or use a predeclared hierarchical
model); never count policy actions or EV rows as independent seeds.

## D. Next statistical milestone

Extend the frozen evaluation to multiple contiguous weeks at Caltech and JPL,
predeclare the weeks and the primary P10 test, and keep development data
separate.  More days increase the number of genuine matched policy
replications; the needed number should be chosen by a predeclared simulation or
pilot-based power analysis, not by stopping when a p-value crosses 0.05.

## Reproducible commands

From `D:\saimohaneesh\FairFlex`:

```powershell
.\.venv\Scripts\Activate.ps1

# Prediction reliability; no charging policy is run or altered.
python scripts\v3\audit_cqr_predictions.py `
  --config configs\v3\caltech_2019_fairflex_uc_holdout_october.json `
  --output-dir artifacts\v3\holdout_october_cqr_prediction_audit_v3 `
  --resamples 5000 --seed 20260903

# Primary policy test: V3 versus guarded V1 on matched calendar days.
python scripts\v3\run_statistical_audit.py `
  --input-dir artifacts\v3\holdout_october_daywise_bounded_history `
  --output-dir artifacts\v3\holdout_october_statistical_audit_primary `
  --metrics p10_service_ratio `
  --resamples 5000 --seed 20260903

# For a continuous multiweek V3 run, keep the adaptive guard continuous and
# aggregate its already-recorded session outcomes before the same audit.
python scripts\v3\summarize_continuous_day_blocks.py `
  --session-metrics artifacts\v3\caltech_november_continuous\test_fairflex_uc_session_metrics.csv `
  --output-dir artifacts\v3\caltech_november_continuous_day_blocks

python scripts\v3\run_statistical_audit.py `
  --input-dir artifacts\v3\caltech_november_continuous_day_blocks `
  --output-dir artifacts\v3\caltech_november_continuous_statistical_audit `
  --metrics p10_service_ratio `
  --resamples 5000 --seed 20260903
```

The detailed outputs are the CQR session audit CSV, CQR summary JSON, paired
calendar-day statistical CSV, and its manifest in the two output directories.
