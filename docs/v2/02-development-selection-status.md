# FairFlex V2 development selection status

## Safeguard

V1 is frozen. Every file and result created for V2 is under `src/fairflex/v2`,
`configs/v2`, `scripts/v2`, `docs/v2`, or `artifacts/v2`. V2 cannot replace a
V1 result by name or output path.

## May development evidence

The April 1-8 calibration bank contains 183 eligible sessions. The May 1-8
development replay contains 187 eligible sessions. Three predeclared minimum
support choices were audited for the condition-aware guard:

| Candidate | Supported groups out of 9 | Point coverage | Mean / P90 buffer | Fallback rate |
| --- | ---: | ---: | ---: | ---: |
| global fallback (minimum 50) | 0 | 0.904 | 90 / 90 min | 1.000 |
| contextual candidate (minimum 20) | 6 | 0.930 | 93.5 / 225 min | 0.075 |
| contextual candidate (minimum 30) | 3 | 0.936 | 115.2 / 225 min | 0.433 |

The 95% calendar-day bootstrap lower bounds are 0.877, 0.881, and 0.893,
respectively. They are descriptive uncertainty intervals over seven calendar
days, not coverage guarantees. Since none clears the 0.90 target on that
conservative diagnostic, this experiment has not established a final V2
guard.

The minimum-30 candidate is removed from the *development shortlist*: it uses
more average buffer than the global fallback without an offsetting reason to
continue. The global fallback and minimum-20 contextual candidate remain for
the predeclared June replication.

## Development policy trade-off

For fair MPC on the same May replay, the minimum-20 candidate versus global
fallback changed the outcomes by: mean service +0.0090, P10 service +0.0147,
worst service +0.0231, Jain index -0.0030, delivered energy +3.05 kWh, unsafe
steps 0, and total runtime +5.38 s over seven days. FCFS and equal-share are
unchanged in service because they do not use planned EV deadlines.

This is a useful hypothesis, not a conclusion: it is a single development
window, its contextual guard has a much larger P90 buffer, and no final test
data have been used for V2.

## Predeclared June decision and result

The two retained alternatives were run once on the separate June 1-8
development replication using the identical calibration bank, policies,
feeder model, and metrics. The contextual candidate could advance only if all
of the following held:

1. its overall point coverage is at least 0.90;
2. it introduces no executed-safety violation;
3. fair-MPC P10 service is no more than 0.005 below global fallback;
4. fair-MPC Jain index is no more than 0.010 below global fallback; and
5. its mean buffer is no more than 20 minutes above global fallback.

The P90 buffer, group coverage, calendar-day interval, energy, worst service,
and runtime remain mandatory reported trade-offs. They are not silently
discarded. If the contextual candidate does not pass, V2 retains the global
fallback and reports that the simple condition split did not earn its added
complexity.

The June contextual candidate had point coverage 0.9148, zero unsafe steps,
Jain delta -0.0064, and mean-buffer delta +2.98 minutes. It passed those four
conditions. Its fair-MPC P10 delta was -0.0119, however, which is below the
allowed -0.005. Therefore it failed the predeclared rule and did **not**
advance. The frozen V2 development choice is the simple global fallback.

This finding is important: the condition-aware split is retained as a
reported ablation, not promoted as an optimisation or claimed as a
contribution. The transparent machine-readable decision is
`artifacts/v2/june_guard_selection/june_guard_selection.json`.

Any later chronological holdout and external-site evaluation must use the
global fallback unchanged; no final test outcome may be used to revive or
tune the rejected condition-aware candidate.
