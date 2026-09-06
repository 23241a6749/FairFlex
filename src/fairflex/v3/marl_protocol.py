"""Reusable causal V3 deadline preparation for matched MARL experiments.

The station-cap actor must receive exactly the deadline information available
to FairFlex-UC.  Keeping this transformation here prevents a future MARL
runner from silently falling back to raw declared departures or from fitting a
new uncertainty model on a test cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..commitments import MultiRateAdaptiveEarlyDepartureGuard
from ..scenarios import PreparedSplit, prepare_acn_split
from .commitments import CausalCQRBufferGuard, conservative_deadline_envelope


@dataclass(frozen=True)
class V3HybridDeadlinePreprocessor:
    """A frozen CQR plus multi-rate-envelope deadline transformation."""

    cqr_guard: CausalCQRBufferGuard
    multirate_guard: MultiRateAdaptiveEarlyDepartureGuard

    def apply(self, prepared: PreparedSplit) -> tuple[PreparedSplit, dict[str, object]]:
        """Apply both guards causally to one prepared replay split.

        The returned sessions preserve their physical disconnect times.  The
        audit is post-hoc metadata; it must not be used to select a policy
        after a registered test file is acquired.
        """
        cqr_sessions, cqr_audit = self.cqr_guard.apply_causally(
            prepared.sessions, origin=prepared.window.start
        )
        multirate_sessions, multirate_audit = self.multirate_guard.apply_causally(
            prepared.sessions
        )
        hybrid_sessions, hybrid_audit = conservative_deadline_envelope(
            cqr_sessions, multirate_sessions
        )
        return replace(prepared, sessions=hybrid_sessions), {
            "cqr": cqr_audit,
            "v1_multirate": multirate_audit,
            "hybrid": hybrid_audit,
            "causal_information_boundary": (
                "Actual future unplug times are not controller inputs; they are retained only "
                "as physical replay events and for post-hoc audit labels."
            ),
        }


def fit_v3_hybrid_deadline_preprocessor(
    config: Mapping[str, Any], config_path: Path | str
) -> tuple[PreparedSplit, PreparedSplit, V3HybridDeadlinePreprocessor]:
    """Fit V3 deadline components solely on a config's train/calibration splits."""
    v3 = config.get("v3")
    if not isinstance(v3, Mapping):
        raise ValueError("matched MARL protocol requires a v3 configuration object")
    cqr_settings = v3.get("cqr")
    guard_settings = v3.get("v1_multirate_guard")
    if not isinstance(cqr_settings, Mapping) or not isinstance(guard_settings, Mapping):
        raise ValueError("matched MARL protocol requires v3 CQR and V1 multi-rate settings")
    train = prepare_acn_split(config, config_path, "train")
    calibration = prepare_acn_split(config, config_path, "calibration")
    step_minutes = int(config["time_step_minutes"])
    cqr_guard = CausalCQRBufferGuard.fit(
        train.sessions,
        calibration.sessions,
        train_origin=train.window.start,
        calibration_origin=calibration.window.start,
        step_minutes=step_minutes,
        miscoverage=float(cqr_settings["miscoverage"]),
        min_train_sessions=int(cqr_settings["min_train_sessions"]),
        min_calibration_sessions=int(cqr_settings["min_calibration_sessions"]),
        max_iter=int(cqr_settings.get("max_iter", 150)),
        min_samples_leaf=int(cqr_settings.get("min_samples_leaf", 12)),
        random_state=int(cqr_settings.get("random_state", 20260903)),
    )
    multirate_guard = MultiRateAdaptiveEarlyDepartureGuard.fit(
        calibration.sessions,
        miscoverage=float(guard_settings["miscoverage"]),
        learning_rates=tuple(float(rate) for rate in guard_settings["learning_rates"]),
        min_miscoverage=float(guard_settings.get("min_miscoverage", 0.01)),
        max_miscoverage=float(guard_settings.get("max_miscoverage", 0.50)),
    )
    return train, calibration, V3HybridDeadlinePreprocessor(cqr_guard, multirate_guard)
