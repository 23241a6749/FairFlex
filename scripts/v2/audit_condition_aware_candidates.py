"""Compare predeclared V2 guard candidates on a development-only window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairflex.scenarios import load_study_config, prepare_acn_split
from fairflex.v2.commitments import ConditionAwareEarlyDepartureGuard
from fairflex.v2.protocol import load_condition_aware_guard_protocol
from fairflex.v2.selection import (
    calendar_day_bootstrap_coverage_interval,
    rank_guard_candidates,
)


def _require_local_midnight(config: dict, split_names: tuple[str, ...]) -> None:
    for name in split_names:
        for boundary in config["splits"][name]:
            timestamp = pd.Timestamp(boundary)
            if timestamp.tzinfo is None or any(
                (timestamp.hour, timestamp.minute, timestamp.second, timestamp.microsecond)
            ):
                raise ValueError(
                    "V2 plug-in-time development windows require local-midnight ISO timestamps"
                )


def _resolver_config_path(path: Path) -> Path:
    """Use a V2-only facade for V1's stable project-root-relative data reader."""
    return path.parent.parent / path.name if path.parent.name == "v2" else path


def _summary(
    protocol,
    guard,
    decisions: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    buffers = decisions["buffer_minutes"]
    coverage_lower, coverage_upper = calendar_day_bootstrap_coverage_interval(
        decisions[["calendar_day", "covered"]].to_dict("records"),
        resamples=resamples,
        seed=seed,
    )
    return {
        "protocol_id": protocol.protocol_id,
        "minimum_group_calibration_sessions": protocol.minimum_group_calibration_sessions,
        "decisions": int(len(decisions)),
        "observed_coverage": float(decisions["covered"].mean()),
        "calendar_day_bootstrap_95_lower": coverage_lower,
        "calendar_day_bootstrap_95_upper": coverage_upper,
        "calendar_days": int(decisions["calendar_day"].nunique()),
        "target_coverage": 1.0 - protocol.miscoverage,
        "mean_buffer_minutes": float(buffers.mean()),
        "median_buffer_minutes": float(buffers.median()),
        "p90_buffer_minutes": float(np.quantile(buffers, 0.9)),
        "global_fallback_rate": float(decisions["used_global_fallback"].mean()),
        "supported_groups": int(
            sum(not value for value in guard.uses_global_fallback_by_group.values())
        ),
        "total_predeclared_groups": len(guard.uses_global_fallback_by_group),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit predeclared V2 condition-aware guard candidates on development data."
    )
    parser.add_argument("--study-config", required=True, type=Path)
    parser.add_argument("--protocols", required=True, nargs="+", type=Path)
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--split", default="test")
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_903)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2/condition_aware_candidate_audit"),
    )
    args = parser.parse_args()
    if args.calibration_split == args.split:
        parser.error("candidate calibration and development splits must differ")
    if args.resamples <= 0:
        parser.error("resamples must be positive")
    config = load_study_config(args.study_config)
    if args.calibration_split not in config["splits"] or args.split not in config["splits"]:
        parser.error("candidate splits must exist in the V2 study config")
    try:
        _require_local_midnight(config, (args.calibration_split, args.split))
    except ValueError as error:
        parser.error(str(error))
    resolver_path = _resolver_config_path(args.study_config)
    calibration = prepare_acn_split(config, resolver_path, args.calibration_split)
    development = prepare_acn_split(config, resolver_path, args.split)
    if not calibration.sessions or not development.sessions:
        parser.error("candidate audit needs non-empty calibration and development sessions")

    combined_decisions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for path in args.protocols:
        protocol = load_condition_aware_guard_protocol(path)
        if int(config["time_step_minutes"]) != protocol.time_step_minutes:
            parser.error(f"protocol {path} has an incompatible time step")
        guard = ConditionAwareEarlyDepartureGuard.fit(
            calibration.sessions,
            miscoverage=protocol.miscoverage,
            min_group_calibration_sessions=protocol.minimum_group_calibration_sessions,
        )
        decisions = pd.DataFrame(guard.decision_audit(development.sessions))
        decisions["calendar_day"] = decisions["ev_id"].str.extract(
            r"(\d{4}-\d{2}-\d{2})", expand=False
        )
        if decisions["calendar_day"].isna().any():
            parser.error("could not recover calendar day from one or more ACN session IDs")
        decisions.insert(0, "protocol_id", protocol.protocol_id)
        combined_decisions.append(decisions)
        summaries.append(
            _summary(
                protocol,
                guard,
                decisions,
                resamples=args.resamples,
                seed=args.seed,
            )
        )

    ranked = rank_guard_candidates(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_csv(args.output_dir / "candidate_guard_summary.csv", index=False)
    pd.concat(combined_decisions, ignore_index=True).to_csv(
        args.output_dir / "candidate_guard_decisions.csv", index=False
    )
    selection = {
        "status": "development_only_shortlist_not_a_frozen_test_result",
        "selection_stage": "coverage_buffer_frontier_only",
        "rule": (
            "Rank candidates with point observed coverage at or above target by P90 buffer, "
            "mean buffer, fallback rate and protocol ID. A separate development policy "
            "matrix must still compare P10, Jain, energy and runtime before one variant is frozen. "
            "Calendar-day bootstrap intervals are descriptive uncertainty diagnostics, not "
            "coverage guarantees or a final-selection rule."
        ),
        "ranked_protocol_ids": [str(row["protocol_id"]) for row in ranked],
        "best_coverage_buffer_candidate": (
            str(ranked[0]["protocol_id"]) if ranked and ranked[0]["coverage_eligible"] else None
        ),
    }
    (args.output_dir / "candidate_selection_status.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(ranked).to_string(index=False))
    print(f"V2 development-only candidate audit: {args.output_dir}")


if __name__ == "__main__":
    main()
