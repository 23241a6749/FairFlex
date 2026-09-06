"""Summarize fixed and adaptive commitment-guard coverage without rerunning MPC.

This script reads existing JSON summaries only.  It deliberately keeps the
fixed-guard seasonal replication and the later adaptive extension separate so
an exploratory repair cannot overwrite or be presented as confirmatory
evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_summary(path: Path) -> tuple[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["policy_metrics"]
    if not metrics:
        raise ValueError(f"summary contains no policy metrics: {path}")
    sessions = int(metrics[0]["sessions"])
    guard = payload["assumptions"]["commitment_guard"]
    if not guard.get("applied"):
        raise ValueError(f"summary does not apply a commitment guard: {path}")
    return sessions, guard


def _fixed_season(
    *, artifact_root: Path, config_stem: str, label: str
) -> dict[str, object]:
    summaries = sorted((artifact_root / config_stem).glob("*/test_pilot_summary.json"))
    if not summaries:
        raise FileNotFoundError(
            f"no daily summaries found under {artifact_root / config_stem}"
        )
    records = [_read_summary(path) for path in summaries]
    sessions = sum(count for count, _ in records)
    coverage = sum(count * float(guard["empirical_coverage"]) for count, guard in records) / sessions
    first_guard = records[0][1]
    return {
        "season": label,
        "protocol": "frozen fixed split-conformal baseline",
        "sessions": sessions,
        "empirical_coverage": coverage,
        "target_coverage": float(first_guard["target_coverage"]),
        "guard_method": first_guard["method"],
        "learning_rate": None,
        "interpretation": (
            "Predeclared final seasonal replication. Coverage below target is a reported "
            "distribution-shift failure, not a value to retune away."
        ),
        "source_count": len(summaries),
    }


def _adaptive_season(*, summary_path: Path, label: str) -> dict[str, object]:
    sessions, guard = _read_summary(summary_path)
    return {
        "season": label,
        "protocol": "post-hoc exploratory past-only adaptive extension",
        "sessions": sessions,
        "empirical_coverage": float(guard["empirical_coverage"]),
        "target_coverage": float(guard["target_coverage"]),
        "guard_method": guard["method"],
        "learning_rate": float(guard["learning_rate"]),
        "interpretation": (
            "Exploratory diagnostic only. It uses calibration plus already-observed unplug "
            "outcomes, but requires a future untouched season for confirmatory evidence."
        ),
        "source_count": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a transparent coverage table from completed FairFlex artifacts."
    )
    parser.add_argument("--fixed-artifact-root", type=Path, required=True)
    parser.add_argument("--fixed-september-stem", required=True)
    parser.add_argument("--fixed-december-stem", required=True)
    parser.add_argument("--adaptive-september-summary", type=Path, required=True)
    parser.add_argument("--adaptive-december-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        _fixed_season(
            artifact_root=args.fixed_artifact_root,
            config_stem=args.fixed_september_stem,
            label="September 2019",
        ),
        _fixed_season(
            artifact_root=args.fixed_artifact_root,
            config_stem=args.fixed_december_stem,
            label="December 2019",
        ),
        _adaptive_season(
            summary_path=args.adaptive_september_summary,
            label="September 2019",
        ),
        _adaptive_season(
            summary_path=args.adaptive_december_summary,
            label="December 2019",
        ),
    ]
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "seasonal_commitment_coverage.csv", index=False)
    (args.output_dir / "coverage_summary_manifest.json").write_text(
        json.dumps(
            {
                "purpose": (
                    "Artifact-only coverage summary. Fixed and post-hoc adaptive outcomes "
                    "remain visibly separated."
                ),
                "fixed_artifact_root": str(args.fixed_artifact_root),
                "adaptive_summaries": [
                    str(args.adaptive_september_summary),
                    str(args.adaptive_december_summary),
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(f"Coverage table: {args.output_dir / 'seasonal_commitment_coverage.csv'}")


if __name__ == "__main__":
    main()
