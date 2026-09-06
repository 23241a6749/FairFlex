"""Run a predeclared, one-factor-at-a-time FairFlex-UC sensitivity matrix.

Each scenario uses exactly the policy selected before the October holdout and
the same fixed V1 comparator.  The scenarios are descriptive robustness
checks, never a tuning loop; their output is written separately from the
primary temporal holdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SELECTED_POLICY = "fairflex_uc_hybrid_lower_tail"
V1_COMPARATOR = "v1_fair_mpc_multirate_guard"


def _project_root(path: Path) -> Path:
    return next(
        candidate
        for candidate in (path.parent, *path.parent.parents)
        if (candidate / "pyproject.toml").is_file()
    )


def _load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {"protocol_id", "base_study_config", "selection_status", "one_factor_at_a_time", "policies", "scenarios"}
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"sensitivity protocol is missing keys: {missing}")
    if protocol["one_factor_at_a_time"] is not True:
        raise ValueError("sensitivity protocol must explicitly require one_factor_at_a_time")
    policies = tuple(str(value) for value in protocol["policies"])
    if SELECTED_POLICY not in policies or V1_COMPARATOR not in policies:
        raise ValueError("protocol policies must retain selected FairFlex-UC and V1 comparator")
    scenarios = protocol["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("protocol scenarios must be a non-empty list")
    names: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str):
            raise ValueError("each sensitivity scenario requires a string name")
        if scenario["name"] in names:
            raise ValueError("sensitivity scenario names must be unique")
        names.add(scenario["name"])
        overridden = {key for key in ("base_import_limit_kw", "pv_dc_capacity_kw", "pv_ac_capacity_kw") if key in scenario}
        if not overridden:
            raise ValueError("each sensitivity scenario must override one physical assumption")
        if "base_import_limit_kw" in overridden and len(overridden) != 1:
            raise ValueError("import-capacity scenarios must not simultaneously alter PV")
        if any(float(scenario[key]) <= 0.0 for key in overridden):
            raise ValueError("sensitivity overrides must be positive")
    return protocol


def _annotate_metrics(metrics: pd.DataFrame, scenario: dict[str, object]) -> pd.DataFrame:
    """Add a stable numeric scenario schema, including explicit missing values."""
    annotated = metrics.copy()
    annotated.insert(0, "scenario", str(scenario["name"]))
    for key in ("base_import_limit_kw", "pv_dc_capacity_kw", "pv_ac_capacity_kw"):
        annotated[key] = float(scenario[key]) if key in scenario else float("nan")
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse a complete scenario metrics file already present in the output directory",
    )
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    root = _project_root(protocol_path)
    base_config = root / str(protocol["base_study_config"])
    if not base_config.is_file():
        parser.error(f"base study config does not exist: {base_config}")
    policies = [str(value) for value in protocol["policies"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []
    runs: list[dict[str, object]] = []
    for scenario in protocol["scenarios"]:
        assert isinstance(scenario, dict)
        name = str(scenario["name"])
        run_dir = args.output_dir / name
        metrics_path = run_dir / "test_fairflex_uc_metrics.csv"
        if args.resume and metrics_path.is_file():
            metrics = pd.read_csv(metrics_path)
            if set(metrics["policy"]) != set(policies):
                raise RuntimeError(f"existing scenario {name} does not contain every frozen policy")
            print(f"Reusing complete frozen scenario {name}...")
            tables.append(_annotate_metrics(metrics, scenario))
            runs.append({"scenario": name, "overrides": scenario, "output_dir": str(run_dir), "status": "reused_completed"})
            continue
        command = [
            sys.executable,
            "scripts/v3/run_fairflex_uc.py",
            "--config", str(base_config),
            "--output-dir", str(run_dir),
            "--only", *policies,
        ]
        for argument, key in (
            ("--base-import-limit-kw", "base_import_limit_kw"),
            ("--pv-dc-capacity-kw", "pv_dc_capacity_kw"),
            ("--pv-ac-capacity-kw", "pv_ac_capacity_kw"),
        ):
            if key in scenario:
                command.extend((argument, str(scenario[key])))
        print(f"Running frozen scenario {name}...")
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            failure = run_dir / "subprocess_failure.txt"
            failure.write_text(
                "COMMAND:\n" + " ".join(command)
                + "\n\nSTDOUT:\n" + completed.stdout
                + "\n\nSTDERR:\n" + completed.stderr,
                encoding="utf-8",
            )
            raise RuntimeError(f"scenario {name} failed; diagnostics: {failure}")
        metrics = pd.read_csv(metrics_path)
        if set(metrics["policy"]) != set(policies):
            raise RuntimeError(f"scenario {name} did not return every frozen policy")
        tables.append(_annotate_metrics(metrics, scenario))
        runs.append({"scenario": name, "overrides": scenario, "output_dir": str(run_dir), "status": "completed"})

    combined = pd.concat(tables, ignore_index=True).sort_values(["scenario", "policy"])
    metrics_path = args.output_dir / "scenario_policy_metrics.csv"
    combined.to_csv(metrics_path, index=False)
    comparisons: list[dict[str, object]] = []
    for scenario, subset in combined.groupby("scenario", sort=True):
        wide = subset.set_index("policy")
        treatment = wide.loc[SELECTED_POLICY]
        baseline = wide.loc[V1_COMPARATOR]
        comparisons.append({
            "scenario": scenario,
            "hybrid_minus_v1_p10_service_ratio": float(treatment["p10_service_ratio"] - baseline["p10_service_ratio"]),
            "hybrid_minus_v1_early_unplug_p10_service_ratio": float(treatment["early_unplug_p10_service_ratio"] - baseline["early_unplug_p10_service_ratio"]),
            "hybrid_minus_v1_expected_shortfall": float(treatment["all_session_expected_shortfall_10pct"] - baseline["all_session_expected_shortfall_10pct"]),
            "hybrid_minus_v1_delivered_energy_kwh": float(treatment["delivered_energy_kwh"] - baseline["delivered_energy_kwh"]),
            "hybrid_unsafe_steps": int(treatment["unsafe_steps"]),
            "v1_unsafe_steps": int(baseline["unsafe_steps"]),
        })
    comparison_path = args.output_dir / "scenario_hybrid_vs_v1.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    manifest_path = args.output_dir / "sensitivity_manifest.json"
    manifest_path.write_text(json.dumps({
        "protocol": protocol,
        "base_study_config": str(base_config),
        "interpretation": (
            "One-factor-at-a-time, frozen-policy robustness scenarios. They are descriptive; "
            "no scenario is used for model, guard, or objective tuning."
        ),
        "runs": runs,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Scenario metrics: {metrics_path}")
    print(f"Hybrid versus V1 summary: {comparison_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
