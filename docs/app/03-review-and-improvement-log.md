# FairFlex application review and improvement log

## Purpose

This is a transparent engineering/research review log for the local FairFlex
demonstrator. Scores judge the application as an offline research demonstrator,
not a commercial deployment or field trial.

## Independent first review (before this improvement cycle)

| Reviewer focus | Score / 10 | Strong points | Main gaps found |
|---|---:|---|---|
| Product UX and accessibility | 8.0 | Premium visual hierarchy, purposeful motion, V1/V3 governance, allocation explanation | Research provenance was too hidden; no primary statistical figure; roadshow/reconnect/accessibility polish required |
| Research evidence | 7.1 | Honest scope separation and real frozen artifacts | Evidence presentation was not yet paper-level; source artifacts were not hash-verified in the app; live small-n P10 was too prominent |
| Engineering/release | 6.8 | Real MPC/AC wrapper, local-only scope, allowlisted inputs, tests/build passed | Saved audit input did not equal executed input, V3 lifecycle races, development-server launcher, incomplete release hardening |

The scores are not averaged into a marketing number. The lowest score controls
the release judgement, because an impressive screen cannot compensate for a
provenance or reproducibility flaw.

## Closed in this improvement cycle

| Finding | Change made | Verification |
|---|---|---|
| Audit hash did not prove the executed model input | Added a complete canonical ScenarioSpec/GridSpec, hashes only engine-relevant values, and reconstructs V1/V3 strictly from the stored spec rather than from a mutable factory | New test poisons the current factory and proves the saved snapshot still executes; app tests pass |
| Grid parameters were not recorded | Snapshot now includes background load scale, voltage limit, thermal limit, line limit, stations, sessions, deadlines, initial energy, and time step | Executed-input hash appears in result, manifest, report, and Audit Trail |
| V3 could race or duplicate | Added atomic SQLite claim plus `queued → running → succeeded/failed` V3 state; one protected attempt per V1 input | Duplicate-request test asserts one queued event and second API call receives 409 |
| Restart could leave a job permanently queued/running | Startup reconciliation marks incomplete local jobs interrupted/failed while preserving the safe V1 record | Repository reconciliation runs at service creation |
| Evidence hash only covered one JSON file | Added eleven-source artifact ledger SHA-256 verification; the UI shows matched/checked count and a source table | API test asserts every bundled source artifact matches; ledger export is tested |
| Source-file hashes did not prove displayed values were extracted correctly | The API now freshly re-extracts the complete ledger from the frozen artifacts and compares every displayed field; the builder has a `--check` mode | API test runs `scripts/build_app_evidence.py --check` and asserts runtime source derivation passes |
| Paper result was not visually primary | Added paired calendar-day forest plot for the frozen JPL November baselines; includes zero line, confidence intervals, study scope, tables, and assumptions | Values are extracted from frozen source CSVs by `scripts/build_app_evidence.py` |
| V3 claims could be misunderstood | Separated live V1/V3 identities from frozen historical V1/V3 identities, shows interval-crossing negative results, and defaults live V3 to off | UI labels and evidence ledger make the scope explicit |
| Small fixture P10 could be over-read | Replaced it as the live headline with worst-EV service and labels it “Illustrative P10 (n=…)” | Dashboard copy and Compare Lab updated |
| Launcher depended on a development Vite server | `run_app.ps1` now type-checks/builds the local dashboard and FastAPI serves it on one loopback origin (`http://127.0.0.1:8000`) | App test verifies the built dashboard and baseline security headers |
| Audit export was too thin | Audit Markdown now includes controller/safety manifest, executed-input hash, V3 metrics, and local-ledger limitation | API test verifies the report contents |

## Validation at this checkpoint

- Application API and integration tests: **8 passed**.
- Frontend TypeScript build: passed.
- Frontend Vite production build: passed.
- Built dashboard route and basic security response headers: covered by app test.

The full research suite is intentionally not represented by the seven
demonstrator tests. Historical experiment artifacts remain frozen and are not
rerun or overwritten by this application work.

## Open release gates before calling it a 9.5+/10 final release

1. Repeat a cold-start browser walkthrough on the final presentation machine:
   dashboard load, V1 run, V3 run, evidence download, audit download.
2. Add browser E2E checks for keyboard navigation, mobile viewport, reduced
   motion, API-offline reconnect, and failed V3 state.
3. Create a reviewed Git baseline and record an actual commit identifier;
   no commit is created automatically because the research workspace contains
   pre-existing user-owned files.
4. Add CI, a verified dependency lock/SBOM, formatting/lint policy, and a
   clean-install test before distributing source externally.
5. Add daily-difference/ECDF and live voltage/loading traces only from
   allowlisted frozen artifacts; never manufacture data for a visual effect.
6. Treat actual charger/OCPP operation as a separate safety/security project,
   not a checkbox in this dashboard.

## Review loop to use next

1. Run the app on the target laptop with `scripts/run_app.ps1`.
2. Use the roadshow in `02-evidence-and-roadshow-guide.md`.
3. Record any UX question, unclear claim, or runtime issue in this log.
4. Fix it only with a matching test or visible disclosure.
5. Have an independent reviewer re-score product, evidence, and release
   readiness. A score above 9.5 requires all open release gates relevant to
   the presentation scope to be demonstrably closed.
