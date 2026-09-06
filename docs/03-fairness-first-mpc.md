# Fairness-first model-predictive control

## Why optimization before reinforcement learning?

At a charging station, the physical rules are known: EV charger limits,
requested energy, declared departure, station capacity, and feeder capacity.
An optimization model can enforce them exactly. A model-free RL reward would
need manually tuned weights and could learn unsafe behavior during exploration.

The controller replans every 15 minutes and executes only its first decision.
This is model-predictive control (MPC). The next replan uses new arrivals and
measurements, so the controller does not need to predict every future event
perfectly.

That does not mean a purely online controller can protect an EV before it has
arrived. Future-arrival uncertainty must be tested as a separate scenario or
addressed with an arrival forecast/reservation extension; it must not be hidden
inside a deadline-awareness claim.

## Why lexicographic objectives instead of one reward?

A weighted reward such as `10 * fairness - cost` hides a subjective choice: why
10 rather than 5 or 50? FairFlex instead solves three small convex programs:

1. **Equity:** minimise the largest normalized requested-energy shortfall.
2. **Efficiency:** among plans with the same fairness, maximise delivered energy.
3. **Economy:** among plans tied on fairness and delivery, minimise energy cost
   and charging-power changes.

This ordering can be stated plainly in a paper and tested independently.

The first two stages are mandatory. The third is only a tie-breaker: if a
numerically degenerate cost/ramp subproblem cannot solve, FairFlex executes the
already verified stage-2 fair-and-efficient schedule and records that the
economic tie-break was not solved. It never substitutes an unsafe or unfair
fallback.

## Deadline guard

For every active EV whose full request is still individually feasible, FairFlex
calculates the minimum power needed in the current slot to keep it achievable if
all remaining slots run at maximum power. It reserves that power whenever
station capacity permits it. This is why a newly arrived urgent EV is not
automatically starved by FCFS. A request that is already impossible even at full
charger power receives no impossible lower bound; its unavoidable shortfall is
handled and reported by the fairness stage.

If the sum of all deadline guards exceeds capacity, the system cannot satisfy
every request. The max-min fairness stage then shares the unavoidable shortage
as evenly as possible; it must report this shortfall rather than pretending the
problem is feasible.
