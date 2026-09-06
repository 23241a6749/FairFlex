"""Record whether one frozen June run faithfully replicated a selected guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_POLICY = "fair_mpc_ac_robust_pv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a credential-free replication record without tuning any guard."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--replication-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserved-validation-note", required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    summary = json.loads(
        (args.replication_run / "test_pilot_summary.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(args.replication_run / "test_pilot_metrics.csv")
    primary = metrics.loc[metrics["policy"] == PRIMARY_POLICY]
    if len(primary) != 1:
        raise ValueError(f"expected one {PRIMARY_POLICY!r} row")
    guard = summary["assumptions"]["commitment_guard"]
    if guard.get("mode") != selection.get("selected_guard_mode"):
        raise ValueError(
            "replication mode does not match the May-selected mode: "
            f"{guard.get('mode')!r} != {selection.get('selected_guard_mode')!r}"
        )
    row = primary.iloc[0]
    record = {
        "purpose": "Unchanged June replication of the May-selected guard.",
        "selected_guard_mode": selection["selected_guard_mode"],
        "selection_path": str(args.selection),
        "replication_run": str(args.replication_run),
        "coverage": {
            "target": float(guard["target_coverage"]),
            "observed": float(guard["empirical_coverage"]),
            "target_met": bool(guard["empirical_coverage"] >= guard["target_coverage"]),
        },
        "service": {
            "p10_service_ratio": float(row["p10_service_ratio"]),
            "jain_service_index": float(row["jain_service_index"]),
            "energy_service_ratio": float(row["energy_service_ratio"]),
            "unsafe_steps": int(row["unsafe_steps"]),
            "runtime_seconds": float(row["runtime_seconds"]),
        },
        "mean_buffer_minutes": float(guard["buffer_minutes_at_decision"]["mean"]),
        "reserved_validation": args.reserved_validation_note,
        "warning": "This record does not choose a replacement method after seeing June.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"Replication record: {args.output}")


if __name__ == "__main__":
    main()
