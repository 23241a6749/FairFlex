# FairFlex

FairFlex is a research implementation of fair, forecast-aware, AC-grid-safe EV charging.

## Implementation order

1. Reproducible simulator and baseline charging policies.
2. Centralized, fairness-first rolling-horizon optimization.
3. Forecast uncertainty and AC-feasibility repair.
4. Distributed station--grid negotiation using ADMM.
5. Trace-driven experiments and reproducible paper results.

The controller is optimization-first: it uses machine learning only for uncertainty-aware PV forecasts. This lets us enforce fairness and grid constraints explicitly instead of hoping a black-box policy learns them.

The first four stages are implemented with automated tests. The final stage
adds data ingestion, controlled scenarios, and reproducible evaluation—not a
claim of field deployment.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Run the deterministic deadline-congestion demonstration (where FCFS and equal
sharing miss the urgent EV, while fair MPC should serve both):

```powershell
python scripts/run_demo.py
python scripts/check_data_access.py
```

The notes under `docs/` explain the purpose of each component, its trade-offs,
and the evidence required before making a research claim.

## Local FairFlex Demonstrator

The demonstrator is a credential-free **offline controlled teaching
simulation**. It uses the actual FairFlex optimizer and AC repair wrapper, but
it does not control physical chargers. Its Evidence Vault presents the separate,
frozen historical ACN research evaluation.

Prerequisites: Python 3.11–3.12 and Node.js 20.19 or newer. pnpm 11.19.0 is
needed only once to install the pinned web dependencies; `corepack enable`
then `pnpm install --frozen-lockfile` is the recommended setup. The launcher
uses the project-local TypeScript and Vite executables afterward, so it does
not require a globally available `pnpm` command.
From the project root, install the small local application layer and pinned web
dependencies once:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,app]"
Set-Location web
corepack enable
corepack pnpm install --frozen-lockfile
corepack pnpm build
Set-Location ..
```

If `corepack` is not available on your Node installation, use the pinned
one-shot fallback instead (it does not require a global pnpm installation):

```powershell
Set-Location web
npx --yes pnpm@11.19.0 install --frozen-lockfile
npx --yes pnpm@11.19.0 build
Set-Location ..
```

Start the API and production-built dashboard together, then open
`http://127.0.0.1:8000`:

```powershell
.\scripts\run_app.ps1
```

For development in two terminals, run `python -m fairflex.app.main` in the
project root and `corepack pnpm dev` from `web` (or `& .\\node_modules\\.bin\\vite.cmd --host 127.0.0.1`). Click **Run V1 recommendation**, inspect
the safety/allocation result, optionally run V3 in research-shadow mode, then
download the locally persisted audit report. The generated SQLite database and
server logs are saved under `data/app/` and are excluded from Git.

## Next evidence step: matched day-level evaluation

After the one-trace pilot, compare policies on the same held-out **days** and
bootstrap the difference by day. This avoids treating many correlated EV
sessions from one day as independent research evidence.

Start with a bounded smoke test (two days, two central policies):

```powershell
python scripts/run_evaluation_matrix.py --configs configs/caltech_2019_scarce.json --days 2019-05-01 2019-05-02 --skip-distributed --only fair_mpc_ac_robust_pv fair_mpc_ac_no_pv --resamples 500
```

Then run the full, pre-declared scenario set only after the smoke test passes:

```powershell
python scripts/run_evaluation_matrix.py --configs configs/caltech_2019_pilot.json configs/caltech_2019_congested.json configs/caltech_2019_scarce.json --resamples 5000
```

Results are saved under `artifacts/evaluation_matrix/`: `daywise_metrics.csv`
contains each matched policy/day result, and `paired_day_bootstrap.csv`
contains paired day-level uncertainty summaries. Treat the current seven-day
window as preliminary; add more held-out periods before making paper claims.
`daywise_metrics.csv` also records policy simulation runtime in seconds; this
is kept separate from fairness/energy bootstrap comparisons because lower
runtime is better while the fairness metrics are higher-is-better.

Three later held-out ACN cohorts are configured for June, September and
December. They keep the March fitting and April calibration windows fixed, so
no later test target can enter the PV forecast fit. First validate their input
provenance, then include them in a matrix run:

```powershell
python scripts/prepare_study.py --config configs/caltech_2019_scarce_june.json --output artifacts/caltech_2019_scarce_june/data_manifest.json
python scripts/prepare_study.py --config configs/caltech_2019_scarce_september.json --output artifacts/caltech_2019_scarce_september/data_manifest.json
python scripts/prepare_study.py --config configs/caltech_2019_scarce_december.json --output artifacts/caltech_2019_scarce_december/data_manifest.json
```

When all four held-out cohorts are ready, this command produces one 28-day
comparison for the explicitly declared `caltech_2019_scarce_multiseason`
protocol. It does not pool different capacity scenarios.

```powershell
python scripts/run_evaluation_matrix.py --configs configs/caltech_2019_scarce.json configs/caltech_2019_scarce_june.json configs/caltech_2019_scarce_september.json configs/caltech_2019_scarce_december.json --skip-distributed --only fair_mpc_ac_robust_pv fair_mpc_ac_no_pv --resamples 5000 --output-dir artifacts/evaluation_matrix_scarce_multiseason
```

The next comparison includes all standard baselines and the distributed ADMM
controller. The bootstrap output labels `central_vs_distributed` separately so
the centralized accuracy/privacy trade-off is not misreported as a win/loss
claim.

Before that comparison, select the distributed controller's settings on the
separate April validation cohort. This command considers five predeclared
ADMM/equity-memory configurations, selects only among zero-unsafe-step
candidates by daily P10 service ratio, then Jain fairness, mean service, and
runtime, and writes a locked credential-free profile. It must not be rerun or
changed after inspecting the May--December test results.

```powershell
python scripts/prepare_study.py --config configs/caltech_2019_distributed_validation.json --output artifacts/distributed_validation/data_manifest.json
python scripts/tune_distributed_policy.py --config configs/caltech_2019_distributed_validation.json
```

Use the generated locked profile in the final all-policy evaluation:

```powershell
python scripts/run_evaluation_matrix.py --configs configs/caltech_2019_scarce.json configs/caltech_2019_scarce_june.json configs/caltech_2019_scarce_september.json configs/caltech_2019_scarce_december.json --distributed-settings configs/distributed_policy_profile_validation_locked.json --resamples 5000 --output-dir artifacts/evaluation_matrix_multiseason_all_policies
```

For a separate AC-repair diagnostic, use the near-limit synthetic IEEE-33
scenario. This tests whether the final AC repair actually activates; it must
not be described as a measured Caltech feeder result.

```powershell
python scripts/run_trace_pilot.py --config configs/caltech_2019_ac_stress.json --split test --only fair_mpc_ac_no_pv --skip-distributed --output-dir artifacts/ac_repair_diagnostic
```

## Safe-MAPPO and IPPO benchmarks

MAPPO and IPPO are optional learning-based benchmarks, not replacements for the
constraint-enforcing FairFlex controller. One agent requests a cap for each
station; FairFlex then projects those requests onto the feeder limit, allocates
locally with fair MPC, and applies the existing exact AC repair. This makes it
possible to report a MARL policy's raw unsafe requests separately from its safe
executed action. The training reward penalizes the amount by which a raw feeder
request exceeds the available capacity (and also any raw AC violation), so
the learning policy is trained not to rely on the safety repair. MAPPO uses a
centralized critic during training; the parameter-shared IPPO ablation uses a
local-observation critic instead. Both execute exactly the same safe action
path, so their comparison isolates the value of centralized training signals.

Install CPU-only PyTorch once (a GPU is optional for much larger training
runs), then train exclusively on the chronological March split and validate on
the separate April validation cohort. Do not train or tune using May--December.

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.8"
python scripts/train_mappo.py --episodes 40 --output-dir artifacts/mappo_training
python scripts/train_mappo.py --algorithm ippo --episodes 40 --output-dir artifacts/ippo_training
```

Use a two-episode, one-validation-day smoke test before a longer multi-seed
run:

```powershell
python scripts/train_mappo.py --episodes 2 --validation-days 2019-04-15 --output-dir artifacts/mappo_smoke
python scripts/train_mappo.py --algorithm ippo --episodes 2 --validation-days 2019-04-15 --output-dir artifacts/ippo_smoke
```

Before selecting a learned benchmark for final evaluation, use at least three
fixed seeds and compare their April validation summaries. The sweep never uses
the May--December held-out cohorts:

```powershell
python scripts/train_marl_seed_sweep.py --algorithm ippo --seeds 20260901 20260902 20260903 --episodes 40 --feasible-cap-transform --output-dir artifacts/ippo_seed_sweep
```

Start with `docs/07-implementation-guide.md` for the end-to-end build,
data, hardware, testing, and research-evidence workflow.

Raw public datasets and generated experiment artifacts are intentionally excluded from Git.
