# Simulator and baselines

## Why start with these policies?

Before claiming an optimizer is fair or grid-safe, the same simulator must run
simple policies whose behavior is obvious:

- **Uncontrolled charging** exposes the effect of every EV charging at its
  useful maximum rate.
- **FCFS** is common and explainable, but favors earlier arrivals even when a
  later arrival has a more urgent departure.
- **Equal share** removes arrival favoritism, but can still miss tight
  deadlines because it ignores urgency.

The later lexicographic optimizer has to outperform these policies on a stated
fairness--service--cost trade-off. It is not allowed to compare only against a
weak uncontrolled case.

## Why the modified IEEE 33-bus feeder?

The standard IEEE 33-bus feeder is a familiar benchmark, which makes our first
experiments reproducible. At its full published load, however, its voltage
minimum is about 0.913 p.u.; that violates the 0.95 p.u. operating bound before
any EV connects. FairFlex therefore starts it at 50% background demand, where
the base voltage minimum is about 0.958 p.u. This leaves realistic headroom and
makes subsequent EV-caused violations interpretable.

The source feeder has placeholder line ampacities. We derive scenario thermal
limits as 130% of the base-case currents. This is a transparent stress-test
assumption, not a claim about a deployed feeder. SimBench validation will later
replace this scenario with feeders containing native equipment ratings.
