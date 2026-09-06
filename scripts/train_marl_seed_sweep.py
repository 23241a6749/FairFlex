"""Train a fixed Safe-MARL protocol across multiple random seeds.

This script deliberately delegates to ``train_mappo.py``.  It therefore uses
the same March-only training, frozen checkpoint, separate April validation,
and safety path as a single run, then consolidates validation rows without
mixing them into the learner.  It is intended for robustness evidence before
any held-out May--December claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible March-train/April-validate Safe-MARL seed sweeps."
    )
    parser.add_argument("--algorithm", choices=("mappo", "ippo"), default="ippo")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--config", type=Path, default=Path("configs/caltech_2019_scarce.json"))
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=Path("configs/caltech_2019_distributed_validation.json"),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--feasible-cap-transform", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.hidden_dim <= 0 or args.update_epochs <= 0 or args.torch_threads <= 0:
        parser.error("episodes, hidden-dim, update-epochs and torch-threads must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("each seed must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: list[pd.DataFrame] = []
    run_manifest: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"{args.algorithm}_seed{seed}"
        command = [
            sys.executable,
            "scripts/train_mappo.py",
            "--algorithm",
            args.algorithm,
            "--config",
            str(args.config),
            "--validation-config",
            str(args.validation_config),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(seed),
            "--hidden-dim",
            str(args.hidden_dim),
            "--update-epochs",
            str(args.update_epochs),
            "--torch-threads",
            str(args.torch_threads),
            "--output-dir",
            str(seed_dir),
        ]
        if args.feasible_cap_transform:
            command.append("--feasible-cap-transform")
        print(f"Starting {args.algorithm.upper()} seed {seed}...", flush=True)
        subprocess.run(command, check=True)
        validation = pd.read_csv(seed_dir / "validation_metrics.csv")
        validation.insert(0, "seed", seed)
        validation_rows.append(validation)
        run_manifest.append(
            {
                "seed": seed,
                "output_dir": str(seed_dir),
                "checkpoint": str(seed_dir / f"safe_{args.algorithm}_actor.pt"),
                "validation_metrics": str(seed_dir / "validation_metrics.csv"),
            }
        )

    combined = pd.concat(validation_rows, ignore_index=True)
    combined_path = args.output_dir / "seed_validation_metrics.csv"
    combined.to_csv(combined_path, index=False)
    quality_columns = (
        "mean_service_ratio",
        "p10_service_ratio",
        "worst_service_ratio",
        "jain_service_index",
        "delivered_energy_kwh",
        "unsafe_steps",
        "runtime_seconds",
    )
    summary = (
        combined.groupby("seed", sort=True)[list(quality_columns)]
        .mean()
        .reset_index()
    )
    summary_path = args.output_dir / "seed_validation_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "purpose": "Multi-seed March-only Safe-MARL training with separate April validation.",
        "algorithm": args.algorithm,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "feasible_cap_transform": args.feasible_cap_transform,
        "forbidden_for_training_or_selection": "May--December held-out cohorts",
        "runs": run_manifest,
    }
    manifest_path = args.output_dir / "seed_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Combined seed validation metrics: {combined_path}")
    print(f"Per-seed validation summary: {summary_path}")
    print(f"Seed sweep manifest: {manifest_path}")


if __name__ == "__main__":
    main()
