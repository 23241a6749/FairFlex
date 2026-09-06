"""Aggregate already completed daily FairFlex policy replays without rerunning them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairflex.scenarios import ReplayWindow, load_study_config
from fairflex.statistics import paired_metric_bootstrap


METRICS = (
    "mean_service_ratio",
    "p10_service_ratio",
    "worst_service_ratio",
    "jain_service_index",
    "delivered_energy_kwh",
    "energy_service_ratio",
    "completion_rate",
    "min_voltage_pu",
    "unsafe_steps",
)
CENTRAL = "fair_mpc_ac_robust_pv"
COMPARISONS = (
    ("central_vs_uncontrolled", CENTRAL, "uncontrolled_shared_cap"),
    ("central_vs_fcfs", CENTRAL, "fcfs_shared_cap"),
    ("central_vs_equal_share", CENTRAL, "equal_share_shared_cap"),
    ("central_vs_least_laxity_first", CENTRAL, "least_laxity_first_shared_cap"),
    ("central_vs_no_pv", CENTRAL, "fair_mpc_ac_no_pv"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate complete daily artifacts and perform paired day-block bootstraps."
    )
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")

    rows: list[pd.DataFrame] = []
    source_files: list[str] = []
    for config_path in args.configs:
        config = load_study_config(config_path)
        window = ReplayWindow.from_config(config["splits"]["test"])
        for start in pd.date_range(window.start.normalize(), window.end, freq="1D", inclusive="left"):
            source = (
                args.artifact_root
                / config_path.stem
                / start.strftime("%Y%m%d")
                / "test_pilot_metrics.csv"
            )
            if not source.is_file():
                parser.error(f"missing completed daily artifact: {source}")
            table = pd.read_csv(source)
            table.insert(0, "study_id", config["study_id"])
            table.insert(1, "evaluation_group", config.get("evaluation_group", config["study_id"]))
            table["replay_start"] = start.isoformat()
            table["replay_end"] = (start + pd.Timedelta(days=1)).isoformat()
            rows.append(table)
            source_files.append(str(source))

    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["study_id", "replay_start", "policy"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "daywise_metrics.csv", index=False)

    comparisons: list[dict[str, object]] = []
    for study_id, subset in combined.groupby("study_id", sort=True):
        present = set(subset["policy"])
        for family, treatment, baseline in COMPARISONS:
            if treatment not in present or baseline not in present:
                continue
            for metric in METRICS:
                result = paired_metric_bootstrap(
                    subset,
                    metric=metric,
                    treatment=treatment,
                    baseline=baseline,
                    resamples=args.resamples,
                    seed=args.seed,
                )
                comparisons.append({"study_id": study_id, "comparison_family": family, **result.to_dict()})
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_day_bootstrap.csv", index=False)
    (args.output_dir / "aggregation_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "Aggregation only: no policy was rerun while constructing these tables.",
                "source_artifact_root": str(args.artifact_root),
                "source_files": source_files,
                "resamples": args.resamples,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Day metrics: {args.output_dir / 'daywise_metrics.csv'}")
    print(f"Paired bootstrap comparisons: {args.output_dir / 'paired_day_bootstrap.csv'}")


if __name__ == "__main__":
    main()
