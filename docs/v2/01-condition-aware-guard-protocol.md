# FairFlex V2 protocol: condition-aware commitment guard

## Purpose

This is a separate V2 development protocol. It does not revise the frozen
FairFlex V1 controller, configurations, results, figures or claims.

V2 asks a narrower question: can a transparent guard that conditions on
information known at plug-in reduce unnecessary early-departure buffer while
retaining auditable lower-tail service and observed reliability?

## Information allowed at plug-in

The V2 guard may use declared requested energy, declared departure, plug-in
time, EV/station limits, present metering and strictly past calibration or
observed outcomes. It may not use the realised physical unplug time, future
arrival, future realised PV or any result from its target test window.

## Fixed groups

The group is the cross product of the following fixed inputs:

- declared duration: at most 4 h; 4-8 h; above 8 h;
- plug-in time: 06:00-10:00; 10:00-14:00; all other times.

With a 15-minute step, a group only receives a separate one-sided
split-conformal buffer if it has at least 50 calibration sessions. Otherwise,
it uses the global buffer. This fallback is intentional: reducing a buffer
with a tiny score bank would create an un-auditable claim.

## What the initial code does

`fairflex.v2.ConditionAwareEarlyDepartureGuard` fits the global and supported
group buffers, creates controller-visible guarded deadlines, and returns one
audit record per replayed session. The realised unplug time exists in the
audit record only after the guarded deadline is selected.

The first V2 code is a fixed split-conformal guard, not a replacement for V1's
selected adaptive multi-rate guard. Development evaluation must compare it
against the V1 global and multi-rate guards. A condition-aware method is
selected at most once on development data; a new temporal and new external
site/month then test the selected version.

## Required V2 test checks

1. The controller never sees a realised unplug time before that EV leaves.
2. Every sparse group uses the global fallback exactly.
3. Overall and group coverage have calendar-day bootstrap intervals.
4. Mean, median and P90 buffer are reported alongside coverage.
5. P10 remains the primary service outcome; P5, worst, mean, Jain,
   completion and the service CDF show trade-offs.
6. The new V2 test config and invariance checks exist before its raw test file
   is acquired or inspected.

## Claim boundary

Within-group coverage has a conditional interpretation only if future sessions
remain exchangeable within that declared group. The fallback retains only the
global marginal interpretation. Neither result is a per-driver, universal or
cross-site coverage guarantee.
