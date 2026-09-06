"""Select and lock one distributed-policy setting profile before final tests.

The selection cohort is deliberately separate from every held-out seasonal
cohort.  This is configuration selection, not a result-generating experiment:
the final paper comparisons must use the profile produced here unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fairflex.scenarios import ReplayWindow, load_study_config, resolve_distributed_policy_settings


# A small one-factor-at-a-time candidate set, fixed before looking at the
# validation outcomes. It avoids an unprincipled large hyperparameter search.
CANDIDATES: tuple[tuple[str, dict[str, float | int]], ...] = (
    (
        "default",
        {
            "rho": 2.0,
            "max_iterations": 250,
            "tolerance": 1e-4,
            "equity_debt_decay": 0.95,
            "equity_debt_gain": 1.0,
        },
    ),
    (
        "lower_rho",
        {
            "rho": 1.0,
            "max_iterations": 250,
            "tolerance": 1e-4,
            "equity_debt_decay": 0.95,
            "equity_debt_gain": 1.0,
        },
    ),
    (
        "higher_rho",
        {
            "rho": 4.0,
            "max_iterations": 250,
            "tolerance": 1e-4,
            "equity_debt_decay": 0.95,
            "equity_debt_gain": 1.0,
        },
    ),
    (
        "longer_equity_memory",
        {
            "rho": 2.0,
            "max_iterations": 250,
            "tolerance": 1e-4,
            "equity_debt_decay": 0.99,
            "equity_debt_gain": 0.5,
        },
    ),
    (
        "more_responsive_equity",
        {
            "rho": 2.0,
            "max_iterations": 250,
            "tolerance": 1e-4,
            "equity_debt_decay": 0.90,
            "equity_debt_gain": 1.25,
        },
    ),
)
POLICY_NAME = "recommended_distributed_robust_pv"


def day_windows(window: ReplayWindow) -> list[ReplayWindow]:
    """Return every full UTC day in the declared validation split."""
    starts = pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left")
    return [ReplayWindow(start, start + pd.Timedelta(days=1)) for start in starts]


def _run_candidate_day(
    *,
    config_path: Path,
    window: ReplayWindow,
    candidate_name: str,
    settings: dict[str, float | int],
    output_dir: Path,
) -> pd.DataFrame:
    run_dir = output_dir / "runs" / candidate_name / window.start.strftime("%Y%m%d")
    command = [
        sys.executable,
        "scripts/run_trace_pilot.py",
        "--config",
        str(config_path),
        "--split",
        "test",
        "--replay-start",
        window.start.isoformat(),
        "--replay-end",
        window.end.isoformat(),
        "--only",
        POLICY_NAME,
        "--distributed-rho",
        str(settings["rho"]),
        "--distributed-max-iterations",
        str(settings["max_iterations"]),
        "--distributed-tolerance",
        str(settings["tolerance"]),
        "--equity-debt-decay",
        str(settings["equity_debt_decay"]),
        "--equity-debt-gain",
        str(settings["equity_debt_gain"]),
        "--output-dir",
        str(run_dir),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"{candidate_name} failed for {window.start.date()}; see {log_path}"
        )

    metrics = pd.read_csv(run_dir / "test_pilot_metrics.csv")
    if metrics.shape[0] != 1 or metrics.loc[0, "policy"] != POLICY_NAME:
        raise RuntimeError(f"unexpected policy metrics from {run_dir}")
    summary = json.loads((run_dir / "test_pilot_summary.json").read_text(encoding="utf-8"))
    audit = summary["policy_audits"][POLICY_NAME]["admm"]
    metrics.insert(0, "candidate", candidate_name)
    metrics.insert(1, "replay_start", window.start.isoformat())
    metrics.insert(2, "replay_end", window.end.isoformat())
    metrics["admm_max_iterations"] = int(audit["max_iterations"])
    metrics["admm_max_primal_residual"] = float(audit["max_primal_residual"])
    metrics["admm_max_dual_residual"] = float(audit["max_dual_residual"])
    for name, value in settings.items():
        metrics[f"setting_{name}"] = value
    return metrics


def rank_candidates(daywise_metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-declared lexicographic validation selection rule."""
    required = {
        "candidate",
        "p10_service_ratio",
        "jain_service_index",
        "mean_service_ratio",
        "runtime_seconds",
        "unsafe_steps",
    }
    missing = sorted(required - set(daywise_metrics))
    if missing:
        raise ValueError(f"metrics cannot be ranked; missing columns: {missing}")
    summary = (
        daywise_metrics.groupby("candidate", sort=True)
        .agg(
            days=("candidate", "size"),
            unsafe_steps_total=("unsafe_steps", "sum"),
            mean_p10_service_ratio=("p10_service_ratio", "mean"),
            mean_jain_service_index=("jain_service_index", "mean"),
            mean_service_ratio=("mean_service_ratio", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            max_admm_iterations=("admm_max_iterations", "max"),
            max_admm_primal_residual=("admm_max_primal_residual", "max"),
            max_admm_dual_residual=("admm_max_dual_residual", "max"),
        )
        .reset_index()
    )
    eligible = summary[summary["unsafe_steps_total"] == 0].copy()
    if eligible.empty:
        raise RuntimeError("no candidate is eligible: every candidate had an unsafe step")
    eligible = eligible.sort_values(
        [
            "mean_p10_service_ratio",
            "mean_jain_service_index",
            "mean_service_ratio",
            "mean_runtime_seconds",
            "candidate",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select distributed ADMM settings on a validation-only cohort and lock a profile."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/distributed_validation")
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("configs/distributed_policy_profile_validation_locked.json"),
        help="credential-free locked settings profile to use unchanged in final tests",
    )
    args = parser.parse_args()

    config = load_study_config(args.config)
    protocol = config.get("protocol", {})
    if protocol.get("role") != "validation_only":
        parser.error("the supplied configuration is not explicitly marked validation_only")
    if "test" not in config["splits"]:
        parser.error("the validation configuration must declare its replay window as split 'test'")
    windows = day_windows(ReplayWindow.from_config(config["splits"]["test"]))
    if len(windows) < 2:
        parser.error("at least two full validation days are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resolved_candidates = [
        (name, resolve_distributed_policy_settings(settings)) for name, settings in CANDIDATES
    ]
    (args.output_dir / "candidate_catalog.json").write_text(
        json.dumps(
            {
                "purpose": "Predeclared candidate settings for validation-only distributed tuning.",
                "selection_rule": protocol["selection_rule"],
                "candidates": [
                    {"candidate": name, "settings": settings}
                    for name, settings in resolved_candidates
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    all_rows: list[pd.DataFrame] = []
    total_runs = len(resolved_candidates) * len(windows)
    completed_runs = 0
    for candidate_name, settings in resolved_candidates:
        for window in windows:
            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] {candidate_name} "
                f"for {window.start.date()}...",
                flush=True,
            )
            all_rows.append(
                _run_candidate_day(
                    config_path=args.config,
                    window=window,
                    candidate_name=candidate_name,
                    settings=settings,
                    output_dir=args.output_dir,
                )
            )

    daywise = pd.concat(all_rows, ignore_index=True).sort_values(["candidate", "replay_start"])
    daywise_path = args.output_dir / "validation_daywise_metrics.csv"
    daywise.to_csv(daywise_path, index=False)
    ranking = rank_candidates(daywise)
    ranking_path = args.output_dir / "candidate_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    winning_name = str(ranking.loc[0, "candidate"])
    winning_settings = dict(dict(resolved_candidates)[winning_name])
    profile = {
        "profile_id": "distributed_admm_validation_locked_v1",
        "status": "locked_after_pretest_validation",
        "settings": winning_settings,
        "selection_provenance": {
            "validation_config": str(args.config),
            "validation_study_id": config["study_id"],
            "validation_role": protocol["role"],
            "days": [window.start.date().isoformat() for window in windows],
            "candidate_catalog": str(args.output_dir / "candidate_catalog.json"),
            "ranking": str(ranking_path),
            "selection_rule": protocol["selection_rule"],
            "warning": "Do not retune these settings on May--December held-out evaluation cohorts.",
        },
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Day-level validation metrics: {daywise_path}")
    print(f"Candidate ranking: {ranking_path}")
    print(f"Locked profile: {args.profile_output} ({winning_name})")


if __name__ == "__main__":
    main()
