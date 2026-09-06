# Recovered FairFlex application walkthrough

> Durable project copy created on 2026-09-06 after a chat-history display issue.
> This is a readable reconstruction of the UI/application explanation, not a
> replacement for the original Codex transcript.

## One-sentence description

FairFlex is an offline, auditable EV-charging research demonstrator. Its
interactive screens run the real fairness-first scheduler and AC-safety check
on small deterministic teaching scenarios, while the Evidence Vault shows
separately frozen results from larger historical ACN-Data studies.

The browser does not command a physical charger, download live charger data, or
train a model when **Run** is pressed.

## The two data paths

```
Interactive teaching path
Scenario Lab -> Command Center -> Compare Lab -> Audit Trail

Historical research path
ACN-Data + offline studies -> frozen, hash-checked evidence -> Evidence Vault
```

Never use a result from the small teaching fixtures as a population-level
research claim. Conversely, do not present historical evidence as a live data
feed or a physical-charger result.

## 1. Scenario Lab

Use this first to select an approved, deterministic scenario.

- **Urgent-deadline teaching scenario** (`deadline-stress`): two synthetic EVs
  (`urgent` and `flexible`) share a single 7.2 kW station. One must receive
  power quickly; the other has timing flexibility.
- **Multi-station campus flow** (`campus-flow`): five synthetic EV sessions
  across three synthetic stations, with changing arrivals and departures.

Select a scenario and choose **Run V1**. Fixed inputs are intentional: they
make a result reproducible and stop a browser user from changing an input then
misrepresenting a custom experiment as validated research.

## 2. Command Center

This is the live simulation view. Clicking **Run V1 recommendation**:

1. saves an exact snapshot of the selected EVs, stations, grid assumptions, and
   time steps;
2. runs the real fairness-first MPC;
3. applies AC-feasibility repair and validation to the executed allocation; and
4. writes the result and its input hash to the local audit record.

V1 first protects the largest normalized energy shortfall, then maximizes
delivered energy, and uses a small tie-break only where needed. Its AC wrapper
checks that the executed schedule is safe.

### What the main cards mean

- **Worst EV service**: the least-served EV's delivered/requested-energy ratio.
- **Mean service**: the average delivered/requested-energy ratio across EVs.
- **At-risk EVs**: EVs with tight timing or low expected final service.
- **Delivered energy**: the total energy supplied in this simulation.
- **Illustrative P10**: shown for teaching only in a two- or five-EV fixture;
  P10 is a real research metric only for a sufficiently large cohort.

The power chart uses a cyan line for executed charging power and an amber line
for fixture capacity. The chart is a useful visual check; the AC safety test is
the actual feasibility decision.

### Where EVs appear

After a run, the **Explainable Allocation** table lists every EV's ID, first
allocation, service ratio, deadline, available time slots, and a **Why** button.
Click **Why** to see its decision brief: requested energy, first allocation,
final service, timing slack, risk status, and a plain-language explanation.

## 3. Compare Lab

Run V1 first. **Run V3 shadow** then replays the exact immutable V1 input. The
application verifies the same SHA-256 input hash before comparison.

It compares illustrative P10, worst service, mean service, Jain fairness,
delivered energy, runtime, and safety. V3 is a research shadow: it cannot
silently replace the recorded V1 recommendation. A small fixture demonstrates
mechanism, not research superiority.

## 4. Evidence Vault

This is the source of research findings. It contains frozen, hash-verified
results from historical public ACN-Data charging sessions at Caltech and JPL.
It does not rerun, retune, or selectively choose an experiment while the app is
open.

Historical inputs include connection/disconnection times, requested energy,
declared departures, and relevant charge traces. Regional NSRDB/PVWatts values
are a solar proxy; the IEEE-33 model and shared import/station assumptions are
controlled sensitivity assumptions, not measured Caltech/JPL feeder or rooftop
PV telemetry.

Use this screen to show the integrity card, data lineage, JPL November paired
effect forest plot, cohort P10 charts, calendar-day confidence intervals and
p-values, reliability/CQR calibration, MARL benchmark results, and the
downloadable evidence ledger.

### Main finding that may be stated safely

On the frozen JPL November 2019 shared-capacity replay (29 non-empty days and
1,320 usable sessions), robust-PV FairFlex had P10 service ratio **0.2119**
versus **0.1915** for Equal Share. The matched calendar-day difference was
**+0.0204**, with a 95% paired day-block bootstrap interval of
**[+0.0061, +0.0339]**.

This supports a lower-tail service improvement in that stated sensitivity
study, not universal superiority or a field-deployment claim. Equal Share can
have higher Jain equality because it tries to equalize service ratios, while
FairFlex prioritizes EVs at risk of being least served.

V3 has encouraging descriptive results in some frozen cohorts, but its
conservative paired calendar-day intervals cross zero. Therefore V1 remains the
default recommendation; V3 remains a research extension.

## 5. Audit Trail

The Audit Trail is the local reproducibility record. It shows saved runs,
scenario ID, time, status, exact input hash, controller/safety manifest, source
revision, event history, and a downloadable Markdown report.

Records are stored locally in `data/app/fairflex.db`. This is a local SQLite
audit record, not a remotely signed or tamper-proof ledger.

## Can a user add or remove EVs?

Not in the current browser application. This is intentional, not a missing
feature. The browser may select only an allowlisted scenario; it cannot add or
remove EVs, upload files, edit deadlines, alter charger capacity, or provide
solver settings.

A future Scenario Builder can support custom EVs, but it must validate energy,
arrival/departure, station capacity, and deadlines. Its outputs must be labelled
as **custom teaching simulations**, separate from frozen research evidence.

## A five-minute guide demonstration

1. In **Scenario Lab**, select the urgent-deadline case.
2. In **Command Center**, run V1 and point out the safety pass, power/capacity
   chart, and worst-EV service.
3. Click **Why** for `urgent` and `flexible` to explain timing slack and
   priority.
4. In **Compare Lab**, run V3 shadow and explain that both methods receive the
   same input hash.
5. In **Evidence Vault**, show the hash-verified JPL study, P10 finding,
   confidence interval, and honest V3 non-promotion result.
6. In **Audit Trail**, show the stored run and download its report.

## Best one-sentence explanation

“The Command Center explains one controlled scheduling decision; the Evidence
Vault supports the research claim using larger frozen historical studies; and
the Audit Trail proves exactly what the application executed.”
