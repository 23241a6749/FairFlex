# Evaluation protocol

## A fair comparison is paired

Each policy runs on a newly created copy of exactly the same trace and grid
scenario. Simulations mutate delivered EV energy, so reusing one object would
make the next policy benefit from the previous policy's charge and invalidate
the result. `ExperimentRunner` makes this mistake harder to make.

Each experiment reports the mean, 10th-percentile, and worst EV service ratio,
Jain's fairness index, delivered energy, and number of AC-unsafe executed
steps. Reporting only aggregate energy can conceal a policy that meets flexible
sessions while failing urgent ones.

## Required comparison matrix

Use the following policies on the same daily trace windows and scenario seeds:

- Uncoordinated charging with physical station breaker sharing.
- First-come, first-served.
- Earliest-deadline-first and least-laxity-first.
- Equal sharing and continuous-rate Round Robin. In FairFlex's continuous-power
  model these are mathematically equivalent; do not misreport their duplicate
  scores as two independent pieces of evidence.
- Centralized fairness-first MPC.
- Centralized MPC plus AC repair.
- Distributed fair MPC plus ADMM, with and without equity debt.

For every policy, repeat across at least several independent day windows, grid
loading levels, feeder capacities, PV-error levels, EV-demand multipliers, and
communication-delay/dropout settings. Report mean and uncertainty intervals
across *days/seeds*, not merely across EVs from one day.

## Non-negotiable checks

1. Split time-series training, calibration, and test periods chronologically.
2. Never use realized future GHI or PV output as a forecast feature.
3. Evaluate the executed first MPC action by AC power flow at every interval.
4. Treat a non-converged ADMM or power-flow run as a failure and report it.
5. Record every dataset version, query/filter, preprocessing rule, random seed,
   hardware/software version, and scenario-to-feeder mapping.
