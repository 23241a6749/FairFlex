# FairFlex Demonstrator v1.0 — Build Plan

## Purpose

Build a local, auditable EV-charging decision-support application around the
existing FairFlex research engine. The interactive application executes
controlled deterministic teaching fixtures; it does **not** control a physical
charging site. Frozen historical ACN evaluation is shown separately as evidence.

The finished demonstration must let a guide see the entire causal path:

```text
Controlled fixture -> validated scenario -> V1 optimizer -> safety check
-> per-EV explanation -> persisted audit run -> V3 shadow comparison
-> benchmark/evidence view -> exported report
```

## Non-negotiable product rules

1. **V1 Fair MPC plus the AC repair wrapper is the only live recommendation
   path.** The multi-rate guard is frozen historical-evaluation evidence and is
   intentionally not fitted to the teaching fixtures.
2. **V3 is a research shadow comparison.** It uses the same input snapshot but
   never replaces V1's recommendation.
3. The interactive screens state `Offline controlled teaching simulation — no
   physical charger control`. The Evidence Vault alone labels frozen historical
   ACN evaluation.
4. A safe result is shown only after the executed allocation passes the
   feasibility check. A failure yields no allocation and a controlled error.
5. The first demo works offline with no ACN/NSRDB credentials or network.
6. The application never accepts arbitrary file paths, policy names, or solver
   flags from a browser request.

## Product surface

### Command Center

The roadshow home screen. In the first ten seconds it answers:

* Is the scenario safe?
* Which EVs are vulnerable to low service?
* What does FairFlex V1 recommend and why?

It contains capacity headroom, a time-based allocation chart, a clear
`Run V1 recommendation` action, the resulting per-EV allocation, a readable
decision brief, and an at-risk driver list. Clicking an EV opens a detail
drawer rather than navigating away.

### Scenario Lab

Lists only approved deterministic scenarios. The initial release includes a
small bundled scarce-capacity fixture. It also displays source, data hash,
assumptions, and the configuration hash.

### Compare Lab

Runs V3 only against the exact V1 scenario snapshot. It labels V1 as
`Recommended` and V3 as `Research shadow`. It shows P10, mean service, Jain
index, delivered energy, safety, runtime, and a plain-language limitation.

### Evidence Vault

Displays the frozen evaluation summaries with confidence intervals. It labels
the frozen historical-evaluation scope and differentiates descriptive pooled numbers from
calendar-day inference. MAPPO/IPPO are benchmarks, not recommendations.

### Audit Trail

Shows locally persisted, append-only application audit records and enables a
concise Markdown report export. Local SQLite is durable but is not a
tamper-proof store.
Each run retains run ID, timestamp, policy/config/data hashes, input snapshot,
allocations, feasibility output, metrics, diagnostics, and export location.

## Technical architecture

```text
React + TypeScript client
    | local HTTP/JSON only
FastAPI application
    | validated named scenario requests
FairFlex adapter service
    | existing simulator / fair MPC / safety modules
SQLite for local durable audit records
    | migration-ready repository boundary
Bundled fixture + frozen evidence JSON
```

### Why this shape

* The optimizer already exists in Python. FastAPI exposes it without copying
  scientific logic into TypeScript.
* SQLite makes the first offline demonstration one-command and reproducible.
  The persistence interface is intentionally narrow so PostgreSQL can replace
  it for a multi-user deployment.
* The initial API executes a deliberately small deterministic fixture. Long
  research evaluations stay offline/precomputed; the guide never waits for a
  complete temporal replay or MARL training run.
* The browser calls only named scenario IDs. A request cannot access a path,
  secret, or arbitrary configuration.

## API contract

* `GET /api/health` — process and demo-data readiness.
* `GET /api/scenarios` — safe scenario catalogue and provenance.
* `GET /api/scenarios/{scenario_id}` — scenario, EVs, capacity, assumptions.
* `POST /api/runs` — executes V1 for one named scenario and persists its audit.
* `POST /api/runs/{run_id}/comparisons` — produces a V3 shadow comparison from
  the exact saved input hash; it refuses mismatches.
* `GET /api/runs/{run_id}` — allocations, safety, metrics, explanations.
* `GET /api/runs` — audit history.
* `GET /api/runs/{run_id}/report` — downloadable report.
* `GET /api/evidence` — frozen V1/V3/MARL facts and limitations.

## Design system

The design direction is **Calm Energy Mission Control**:

* midnight navy base, layered ink-blue surfaces, electric cyan action,
  violet fairness, blue capacity, amber warning, and labelled red failure;
* one legible sans-serif family with tabular numeric styling;
* visual hierarchy follows `overview -> detail on demand`;
* slow, optional energy-flow motion only. Never use 3D charts, flashing,
  auto-scrolling data, or decorative loading that hides the decision;
* full keyboard operation, visible focus, accessible chart summaries, a data
  table for every chart, and `prefers-reduced-motion` support.

## Implementation order

1. Create the backend package, bundled demo fixture, schemas, hashing,
   deterministic V1 adapter, safety validation, and audit persistence.
2. Add API tests for validation, persistence, hash matching, offline operation,
   unsafe/failure behaviour, and report export.
3. Create the React application with the Command Center as its first viewport.
4. Add Scenario Lab, V3 Compare Lab, Evidence Vault, and Audit Trail.
5. Add a resilient loading/error model and a local fallback report.
6. Add browser tests, keyboard review, visual review, and complete test run.
7. Add Docker Compose and a one-page runbook after the local vertical slice
   works.

## Demo sequence

1. Select the `Urgent-deadline teaching scenario`.
2. Show that requests exceed available capacity.
3. Run V1 and explain its safe allocation.
4. Inspect an at-risk EV's explanation.
5. Run V3 shadow comparison on the saved identical snapshot.
6. Open the evidence view and explain why V1 remains default.
7. Open the persisted audit record and download its report.

## Acceptance gates

The demonstrator is ready only when:

* it starts locally without credentials or network;
* a real V1 engine call, not a hard-coded result, produces the displayed run;
* every run has provenance and a reproducible input/configuration hash;
* V3 can never silently control the V1 action;
* no unsafe allocation can display a safe status;
* one full browser demo completes using keyboard and mouse;
* unit, API, integration, and browser tests pass; and
* the product truthfully distinguishes replay validation from field deployment.

## Deferred pilot work

Physical chargers, real-time feeds, OCPP integration, role-based access,
PostgreSQL, and deployment security belong to a later controlled pilot. They
are not prerequisites for the research demonstrator and must not be simulated
as if they were live hardware.
