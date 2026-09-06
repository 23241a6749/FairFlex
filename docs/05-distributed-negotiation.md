# Distributed fair charging negotiation

## What runs where

Each charging station runs a local instance of the fair MPC. It sees the EVs
connected to that station, their requested energy, deadlines, and maximum
charging rates. It creates a desired *aggregate station import profile*. The
feeder coordinator never receives individual EV identities, schedules, or
requested energy.

The coordinator receives one desired profile and one scalar priority from each
station. It projects the combined request onto the feeder-capacity limit and
returns an import-cap profile per station. Each station then re-solves its own
fair MPC under that cap. Only the first action is executed before all parties
observe the next interval and repeat the process.

## Why consensus ADMM

The shared constraint is simple: at every time interval, the sum of station
imports cannot exceed the feeder limit. ADMM separates this constraint from the
station's private charging problem. It is a better first distributed method
than multi-agent reinforcement learning because its capacity constraint is
explicit, convergence is measurable, the messages are inspectable, and there
is a meaningful centralized optimization baseline to compare against.

The current ADMM objective minimizes weighted deviation from each station's
desired import profile. A greater station priority means the system preserves
more of that request under congestion. The priority is calculated from the most
critical local deadline and can include an `equity_debt`: an auditable summary
of past under-service. This prevents the same station from losing repeatedly
without sharing raw EV histories.

## What this is and is not

This is a genuine distributed coordinator: raw EV data remains local and the
negotiated profiles satisfy feeder capacity. It is not yet a claim of an
asynchronous, adversarial, communication-fault-tolerant market. Those are
separate research extensions and must be evaluated explicitly rather than
assumed from the use of the word "multi-agent".

## Measurements to report

- ADMM primal and dual residuals, iteration count, and non-convergence rate.
- Feeder-cap violation rate (must be zero within numerical tolerance).
- EV service-ratio distribution and worst-session service ratio.
- Station-level service ratio over time, including the effect of equity debt.
- Information sent per station per interval versus a centralized controller.
