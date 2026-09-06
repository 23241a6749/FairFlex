"""Audit FairFlex-UC's early-unplug quantile prediction on a frozen split.

This command does not run a charging policy, tune CQR, or alter any previous
result. It fits the already configured train/calibration CQR guard and writes
post-hoc prediction rows for the declared test period. The correct metrics for
this one-sided quantile prediction task are coverage, pinball loss, and
sharpness (buffer size), not classifier accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from fairflex.commitments import EarlyDepartureGuard
from fairflex.scenarios import load_study_config, prepare_acn_split
from fairflex.statistics import clustered_binary_rate_bootstrap
from fairflex.v3.commitments import CausalCQRBufferGuard


def _wilson_interval(successes: int, observations: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Return an IID-session Wilson interval, labelled descriptive in output."""
    if observations <= 0 or not 0 < confidence < 1:
        raise ValueError("observations must be positive and confidence must lie in (0, 1)")
    rate = successes / observations
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    denominator = 1.0 + z**2 / observations
    centre = (rate + z**2 / (2.0 * observations)) / denominator
    half_width = z * ((rate * (1.0 - rate) / observations + z**2 / (4.0 * observations**2)) ** 0.5) / denominator
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def _pinball_loss(outcomes: np.ndarray, predictions: np.ndarray, *, quantile: float) -> float:
    residual = outcomes - predictions
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")

    config = load_study_config(args.config)
    v3 = config.get("v3")
    if not isinstance(v3, dict) or not isinstance(v3.get("cqr"), dict):
        parser.error("config must contain v3.cqr settings")
    cqr_settings = v3["cqr"]
    train = prepare_acn_split(config, args.config, "train")
    calibration = prepare_acn_split(config, args.config, "calibration")
    evaluated = prepare_acn_split(config, args.config, args.split)
    guard = CausalCQRBufferGuard.fit(
        train.sessions,
        calibration.sessions,
        train_origin=train.window.start,
        calibration_origin=calibration.window.start,
        step_minutes=int(config["time_step_minutes"]),
        miscoverage=float(cqr_settings["miscoverage"]),
        min_train_sessions=int(cqr_settings["min_train_sessions"]),
        min_calibration_sessions=int(cqr_settings["min_calibration_sessions"]),
        max_iter=int(cqr_settings.get("max_iter", 150)),
        min_samples_leaf=int(cqr_settings.get("min_samples_leaf", 12)),
        random_state=int(cqr_settings.get("random_state", 20260903)),
    )
    _, aggregate_audit = guard.apply_causally(evaluated.sessions, origin=evaluated.window.start)
    rows = guard.prediction_audit_rows(evaluated.sessions, origin=evaluated.window.start)
    frame = pd.DataFrame(rows)
    successes = int(frame["cqr_covered"].sum())
    observations = int(len(frame))
    wilson_low, wilson_high = _wilson_interval(successes, observations)
    clustered = clustered_binary_rate_bootstrap(
        frame["calendar_day"],
        frame["cqr_covered"],
        resamples=args.resamples,
        seed=args.seed,
    )
    # A causal no-context baseline: one static 90th-percentile buffer fitted on
    # the same calibration window. It is a planning-guard comparison, not a
    # second policy result and is never tuned on the October split.
    global_guard = EarlyDepartureGuard.fit(
        calibration.sessions,
        miscoverage=float(cqr_settings["miscoverage"]),
    )
    global_deadlines = [global_guard.guarded_deadline_step(session) for session in evaluated.sessions]
    global_covered = np.asarray(
        [session.departure_step >= deadline for session, deadline in zip(evaluated.sessions, global_deadlines, strict=True)],
        dtype=bool,
    )
    actual_advances = frame["actual_early_unplug_steps"].to_numpy(dtype=float)
    cqr_integer_prediction = frame["integer_cqr_buffer_steps"].to_numpy(dtype=float)
    static_prediction = np.full(observations, float(global_guard.buffer_steps))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{args.split}_cqr_prediction_audit.csv"
    summary_path = args.output_dir / f"{args.split}_cqr_prediction_summary.json"
    frame.to_csv(rows_path, index=False)
    summary = {
        "study_id": config["study_id"],
        "split": args.split,
        "selection_status": v3.get("selection_status", "development_only"),
        "causal_boundary": (
            "Prediction features use declarations, calendar/station context, and observable plug-in "
            "state only. Actual early-unplug advance and coverage are post-hoc labels in this audit."
        ),
        "task_type": "one_sided_upper_quantile_prediction_of_early_unplug_advance",
        "not_a_classifier_accuracy": True,
        "target_coverage": float(aggregate_audit["target_coverage"]),
        "observed_coverage": {
            "covered_sessions": successes,
            "sessions": observations,
            "rate": float(frame["cqr_covered"].mean()),
            "iid_session_wilson_95_percent_descriptive": {"low": wilson_low, "high": wilson_high},
            "calendar_day_cluster_bootstrap_95_percent_descriptive": clustered.to_dict(),
        },
        "quantile_quality": {
            "raw_upper_quantile_pinball_loss": aggregate_audit["raw_upper_quantile_pinball_loss"],
            "integer_conformal_buffer_pinball_loss": _pinball_loss(
                actual_advances,
                cqr_integer_prediction,
                quantile=1.0 - float(cqr_settings["miscoverage"]),
            ),
            "mean_actual_early_unplug_steps": aggregate_audit["mean_actual_early_unplug_steps"],
            "mean_raw_upper_quantile_steps": aggregate_audit["mean_raw_upper_quantile_steps"],
            "mean_uncovered_early_unplug_steps": aggregate_audit["mean_uncovered_early_unplug_steps"],
            "mean_conservative_buffer_excess_steps": aggregate_audit["mean_conservative_buffer_excess_steps"],
            "integer_buffer_steps": aggregate_audit["buffer_steps"],
        },
        "causal_static_global_conformal_baseline": {
            "description": (
                "Single non-contextual 90th-percentile early-unplug buffer fitted only on the same calibration "
                "sessions. This compares guard sharpness/reliability, not charging-policy performance."
            ),
            "constant_buffer_steps": global_guard.buffer_steps,
            "observed_coverage": float(global_covered.mean()),
            "pinball_loss_at_nominal_quantile": _pinball_loss(
                actual_advances,
                static_prediction,
                quantile=1.0 - float(cqr_settings["miscoverage"]),
            ),
            "cqr_minus_static_mean_integer_buffer_steps": float(
                frame["integer_cqr_buffer_steps"].mean() - global_guard.buffer_steps
            ),
            "cqr_minus_static_integer_buffer_pinball_loss": float(
                _pinball_loss(
                    actual_advances,
                    cqr_integer_prediction,
                    quantile=1.0 - float(cqr_settings["miscoverage"]),
                )
                - _pinball_loss(
                    actual_advances,
                    static_prediction,
                    quantile=1.0 - float(cqr_settings["miscoverage"]),
                )
            ),
            "interpretation": (
                "CQR is sharper only if it retains target-level coverage with a lower mean buffer. "
                "If it uses a larger buffer, it may still protect fairness but does not establish a sharpness gain."
            ),
        },
        "interpretation": (
            "Coverage should be assessed against the declared nominal target; lower pinball loss and smaller "
            "buffers are desirable only while coverage remains adequate. The Wilson interval assumes independent "
            "sessions and is shown for familiar scale only. The calendar-day clustered bootstrap respects within-day "
            "dependence but remains a small-sample descriptive interval, not a new finite-sample conformal guarantee."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CQR session audit: {rows_path}")
    print(f"CQR prediction summary: {summary_path}")


if __name__ == "__main__":
    main()
