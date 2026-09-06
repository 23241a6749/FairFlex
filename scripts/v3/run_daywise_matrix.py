"""Run frozen FairFlex-UC V3 policies on matched calendar-day replays.

The command resamples whole matched days, never individual EV rows.  It is a
robustness/sensitivity analysis: the adaptive V1 guard is reinitialised from
the fixed calibration bank on each day, so these intervals do not replace the
continuous-week guard-coverage audit written by ``run_fairflex_uc.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fairflex.scenarios import ReplayWindow, load_study_config
from fairflex.statistics import paired_metric_bootstrap


SELECTED_POLICY = "fairflex_uc_hybrid_lower_tail"
DEFAULT_POLICIES = (
    "equal_share_shared_cap",
    "v1_fair_mpc_no_guard",
    "v1_fair_mpc_multirate_guard",
    SELECTED_POLICY,
)
COMPARISONS = (
    ("hybrid_vs_v1_multirate", SELECTED_POLICY, "v1_fair_mpc_multirate_guard"),
    ("hybrid_vs_v1_no_guard", SELECTED_POLICY, "v1_fair_mpc_no_guard"),
    ("hybrid_vs_equal_share", SELECTED_POLICY, "equal_share_shared_cap"),
)
METRICS = (
    ("p10_service_ratio", "higher"),
    ("early_unplug_p10_service_ratio", "higher"),
    ("all_session_expected_shortfall_10pct", "lower"),
    ("mean_service_ratio", "higher"),
    ("jain_service_index", "higher"),
    ("delivered_energy_kwh", "higher"),
    ("unsafe_steps", "lower"),
)


def _day_windows(window: ReplayWindow) -> list[ReplayWindow]:
    starts = pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left")
    return [ReplayWindow(start, start + pd.Timedelta(days=1)) for start in starts]


def _oriented(result, direction: str) -> dict[str, float | str]:
    sign = 1.0 if direction == "higher" else -1.0
    low = sign * result.ci_low
    high = sign * result.ci_high
    return {
        "preferred_direction": direction,
        "observed_treatment_advantage": sign * result.observed_mean_difference,
        "advantage_ci_low": min(low, high),
        "advantage_ci_high": max(low, high),
        "bootstrap_probability_treatment_better": (
            result.bootstrap_probability_difference_positive
            if direction == "higher"
            else result.bootstrap_probability_difference_negative
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    unknown = set(args.policies) - set(DEFAULT_POLICIES)
    if unknown:
        parser.error(f"daywise frozen protocol does not allow unselected policies: {sorted(unknown)}")
    if SELECTED_POLICY not in args.policies:
        parser.error(f"--policies must retain {SELECTED_POLICY}")

    config = load_study_config(args.config)
    frozen = config.get("v3", {}).get("selected_deployable_policy")
    if frozen != SELECTED_POLICY:
        parser.error("configuration does not declare the frozen FairFlex-UC selected policy")
    windows = _day_windows(ReplayWindow.from_config(config["splits"][args.split]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    runs: list[dict[str, object]] = []
    for window in windows:
        day_name = window.start.strftime("%Y%m%d")
        run_dir = args.output_dir / day_name
        command = [
            sys.executable,
            "scripts/v3/run_fairflex_uc.py",
            "--config", str(args.config),
            "--split", args.split,
            "--replay-start", window.start.isoformat(),
            "--replay-end", window.end.isoformat(),
            "--output-dir", str(run_dir),
            "--only", *args.policies,
        ]
        print(f"Running matched day {window.start.date()}...")
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            failure_path = run_dir / "subprocess_failure.txt"
            failure_path.write_text(
                "COMMAND:\n" + " ".join(command)
                + "\n\nSTDOUT:\n" + completed.stdout
                + "\n\nSTDERR:\n" + completed.stderr,
                encoding="utf-8",
            )
            raise RuntimeError(f"day {window.start.date()} failed; diagnostics: {failure_path}")
        metrics_path = run_dir / f"{args.split}_fairflex_uc_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        expected = set(args.policies)
        if set(metrics["policy"]) != expected:
            raise RuntimeError(f"day {window.start.date()} did not return every requested policy")
        metrics.insert(0, "replay_start", window.start.isoformat())
        metrics.insert(1, "replay_end", window.end.isoformat())
        all_metrics.append(metrics)
        runs.append({
            "replay_start": window.start.isoformat(),
            "replay_end": window.end.isoformat(),
            "output_dir": str(run_dir),
            "status": "completed",
        })

    daywise = pd.concat(all_metrics, ignore_index=True).sort_values(["replay_start", "policy"])
    daywise_path = args.output_dir / "daywise_metrics.csv"
    daywise.to_csv(daywise_path, index=False)

    comparisons: list[dict[str, object]] = []
    for family, treatment, baseline in COMPARISONS:
        if treatment not in args.policies or baseline not in args.policies:
            continue
        for metric, direction in METRICS:
            try:
                result = paired_metric_bootstrap(
                    daywise,
                    metric=metric,
                    treatment=treatment,
                    baseline=baseline,
                    block_columns=("replay_start",),
                    resamples=args.resamples,
                    seed=args.seed,
                )
            except ValueError as error:
                print(f"Skipping {family} {metric}: {error}")
                continue
            comparisons.append({"comparison_family": family, **result.to_dict(), **_oriented(result, direction)})
    comparisons_path = args.output_dir / "paired_day_bootstrap.csv"
    pd.DataFrame(comparisons).to_csv(comparisons_path, index=False)
    manifest_path = args.output_dir / "matrix_manifest.json"
    manifest_path.write_text(json.dumps({
        "purpose": "Frozen-policy FairFlex-UC matched calendar-day robustness analysis.",
        "unit_of_resampling": "matched calendar day, never individual EV session",
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "selected_policy": SELECTED_POLICY,
        "policies": args.policies,
        "continuous_week_caveat": (
            "The V1 online guard is reinitialised from the fixed calibration bank on each day. "
            "These intervals are policy-metric sensitivity diagnostics, not continuous-week coverage evidence."
        ),
        "runs": runs,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Day-level metrics: {daywise_path}")
    print(f"Paired bootstrap comparisons: {comparisons_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
