"""Aggregate one continuous FairFlex-UC replay into calendar-day inference blocks.

Unlike ``run_daywise_matrix.py``, this command does not restart a simulator at
midnight.  It groups the already-recorded session outcomes by arrival day, so
the CQR/V1 hybrid guard remains exactly as it was in the continuous execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairflex.v3.evaluation import continuous_arrival_day_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    session_rows = pd.read_csv(args.session_metrics)
    daywise = continuous_arrival_day_metrics(session_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "daywise_metrics.csv"
    manifest_path = args.output_dir / "continuous_day_block_manifest.json"
    daywise.to_csv(metrics_path, index=False)
    manifest_path.write_text(json.dumps({
        "source_session_metrics": str(args.session_metrics),
        "unit_of_inference": "matched calendar day, defined by session arrival date",
        "controller_state": "one continuous replay per policy; no midnight guard reset",
        "physical_outcomes": (
            "Actual disconnect time is used only after replay to label the early-unplug subset."
        ),
        "eligible_day_rule": (
            "The statistical audit screens daily P10 blocks for at least 10 sessions and needs at least five matched days."
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Continuous-replay day metrics: {metrics_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
