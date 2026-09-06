# FairFlex application evidence and roadshow guide

## One-sentence description

FairFlex is an offline, auditable EV-charging research demonstrator: it runs
the real fairness-first MPC and AC-feasibility repair on small deterministic
teaching fixtures, while presenting separately frozen historical evidence from
larger ACN-data trace-replay studies.

This distinction is the most important thing to say in a review. The browser
does **not** fetch live charger data or control a physical charger.

## What data is used where

| Application area | Data used at that moment | Why it exists | What it does not prove |
|---|---|---|---|
| Command Center | Deterministic FairFlex fixture | Lets a guide watch the actual V1 MPC, AC repair, allocation and explanation run safely | Population performance or field deployment |
| `deadline-stress` | 2 synthetic EV sessions, one 7.2 kW station, 15-minute steps | Makes an urgent deadline visibly compete with a flexible EV | A statistically meaningful P10 result |
| `campus-flow` | 5 synthetic EV sessions, three synthetic stations | Shows arrivals/departures and multi-station mechanics | Caltech/JPL field operation |
| Compare Lab | Fresh V1 and V3 replays of the exact same saved fixture | Isolates objective mechanics fairly | A research superiority claim |
| Audit Trail | Local SQLite run record, immutable executed-input hash, controller/safety manifest | Reproduces what the browser actually ran | A signed, remote, tamper-proof ledger |
| Evidence Vault | Frozen extracts from completed historical studies | Shows large-cohort research results without silently retuning them in the app | A new live experiment every time the app opens |

## Historical research inputs behind the Evidence Vault

The historical analyses use public ACN-Data charging-session records from the
Caltech and JPL sites. They use connection/disconnection times, energy requests,
declared departures and relevant charge traces for chronological replay.

Solar and feeder conditions are deliberately disclosed separately:

- Regional NSRDB/PVWatts values are a solar proxy.
- The shared import cap and deterministic station mapping are controlled
  sensitivity assumptions.
- The IEEE-33 feeder is a sensitivity/electrical-validation model.
- None of those is claimed to be measured Caltech or JPL feeder telemetry,
  rooftop PV, or a real charger command.

The application verifies eleven frozen source artifacts against their recorded
SHA-256 hashes before labelling the evidence locally hash-verified. The Evidence
Vault and its Markdown download expose the source paths, hashes, study scope,
and limitations.

## Research findings that can be stated safely

### Primary frozen baseline comparison

On the frozen JPL November 2019 shared-capacity replay (29 non-empty days,
1,320 usable sessions), robust-PV FairFlex had P10 service ratio 0.2119 versus
0.1915 for Equal Share. The matched calendar-day difference was +0.0204 with a
95% paired day-block bootstrap interval of [+0.0061, +0.0339].

The same frozen study compares FairFlex with Uncontrolled, FCFS, EDF, LLF,
Equal Share, and a no-PV-robustness ablation. The Evidence Vault forest plot
shows all six paired intervals rather than hiding weaker comparators.

This supports a lower-tail service improvement in this stated sensitivity
study. It does not establish universal superiority or real-world deployment
performance. A real trade-off is visible too: Jain equality is lower than
Equal Share in this primary replay, because FairFlex prioritises the
least-served tail rather than equalising every service ratio.

### V3 research extension

V3 combines a frozen CQR-style uncertainty guard, a conservative hybrid
deadline envelope, and a lower-tail MPC objective in the historical research
configuration. Its all-session P10 is descriptively higher than guarded V1 in
the three displayed cohorts:

| Cohort | Sessions | V1 P10 | V3 P10 | Correct reading |
|---|---:|---:|---:|---|
| Caltech October frozen holdout | 154 | 0.2023 | 0.2161 | Encouraging descriptive direction |
| Caltech November continuous replication | 615 | 0.1702 | 0.1713 | Small descriptive difference |
| JPL October external replication | 379 | 0.0784 | 0.0859 | Descriptive site-calibrated transfer |

The more conservative calendar-day tests are what determine the product
choice:

| Cohort | Eligible days | V3 minus V1 P10 | 95% interval | One-sided p | Reading |
|---|---:|---:|---:|---:|---|
| Caltech October | 5 | +0.0164 | [-0.0055, +0.0434] | 0.1875 | Inconclusive |
| Caltech November | 22 | -0.0068 | [-0.0175, +0.0046] | 0.8612 | No reliable V3 improvement |

Both intervals cross zero. Therefore V1 remains the live recommended policy;
V3 remains an explicitly labelled research shadow rather than a silently
promoted product controller.

### Reliability and MARL benchmarks

- The selected multi-rate early-unplug envelope achieved 0.892 coverage on
  JPL November, versus 0.855 for the global guard and a nominal 0.90 target.
  It pays for this protection with a larger mean buffer (142.125 versus 120
  minutes).
- Caltech October CQR coverage was 141/154 = 0.916 against a 0.90 nominal
  target. Coverage is calibration, not a single classifier “accuracy.”
- In the final five-seed benchmark, neither Safe-MAPPO nor Safe-IPPO established
  a daily P10 advantage over frozen V3: both paired intervals crossed zero.
  These are fair benchmarks, not deployment recommendations.

## How to explain the metrics

| Metric | Plain-language meaning | Better direction |
|---|---|---|
| Service ratio | Energy delivered to one EV divided by what it requested | Higher |
| Worst EV service | The smallest service ratio in the current small fixture | Higher |
| P10 service ratio | The lower 10% service level across a sufficiently large session cohort | Higher |
| Mean service ratio | Average fraction of requested energy delivered | Higher |
| Jain service index | Equality of service ratios; 1 means perfectly equal ratios | Higher equality, but not automatically higher minimum service |
| Expected shortfall | Deficit concentrated in the worst 10% | Lower |
| 95% paired day-block interval | Uncertainty for a policy difference when complete calendar days, not individual EVs, are resampled | An interval crossing zero is not reliable directional evidence |
| Coverage | Fraction of held-out early-unplug events protected by the prediction interval | Compare with the stated nominal target |

For the two- and five-EV live fixtures, the dashboard treats P10 as
**illustrative only** and foregrounds the worst EV and individual outcomes.

## Recommended five-minute live roadshow

1. Start on **Command Center** and say: “This is an offline teaching
   simulation. It runs our real MPC and AC safety repair; it never commands a
   physical charger.”
2. Select **Urgent-deadline teaching scenario**, then click **Run V1
   recommendation**. Point out the immediate allocation to the urgent EV and
   the safety-pass result.
3. Open the EV decision brief. Explain the difference between a tight deadline
   and flexible timing slack.
4. Open **Compare Lab** only after V1 finishes. Start V3 as a research shadow
   and point out that it uses the same immutable executed-input hash and can
   never replace V1.
5. Open **Evidence Vault**. Start with the green evidence-integrity card,
   then the JPL November paired-effect forest plot. Say what the study supports
   and the Jain trade-off.
6. Show the V3 calendar-day result and say why it stays a shadow: the interval
   crosses zero. Then show CQR coverage and the MARL non-superiority result to
   demonstrate honest negative findings.
7. Finish in **Audit Trail**, download the Markdown report, and show the input
   hash, controller, AC wrapper, and local-audit limitation.

## What remains outside the claim boundary

Before any field pilot, the project still needs measured feeder topology and
limits, metered PV, real EVSE discrete-rate behaviour, operator controls,
rollback, authentication/TLS/RBAC, OCPP threat modelling, privacy/ethics
review, and a safe read-only integration phase. Those are future validation
requirements—not features that this local demonstrator pretends to provide.
