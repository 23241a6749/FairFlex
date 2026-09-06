"""Audit the causal ACN commitment protocol before controller evaluation.

This script reports properties of the *derived* research benchmark. It does
not train a controller and never writes raw ACN records, user IDs, or API keys
to its output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairflex.commitments import EarlyDepartureGuard
from fairflex.domain import EVSession
from fairflex.scenarios import (
    load_study_config,
    prepare_acn_split,
    resolve_commitment_uncertainty,
)


def _split_summary(sessions: tuple[EVSession, ...], *, step_hours: float) -> dict[str, float | int]:
    """Return non-identifying diagnostics for one chronological split."""
    if not sessions:
        raise ValueError("a commitment split must contain at least one session")
    declared = np.asarray(
        [session.declared_departure_step for session in sessions], dtype=float
    )
    if np.isnan(declared).any():
        raise ValueError("commitment diagnostics require explicit declared deadlines")
    actual = np.asarray([session.departure_step for session in sessions], dtype=float)
    advances = np.maximum(0.0, declared - actual)
    requests = np.asarray([session.requested_energy_kwh for session in sessions], dtype=float)
    individual_capacity = np.asarray(
        [
            (session.departure_step - session.arrival_step)
            * session.max_power_kw
            * step_hours
            for session in sessions
        ],
        dtype=float,
    )
    unavoidable = np.maximum(0.0, requests - individual_capacity)
    return {
        "sessions": len(sessions),
        "requested_energy_kwh": float(requests.sum()),
        "early_departure_rate": float(np.mean(advances > 0)),
        "early_departure_median_minutes": float(np.quantile(advances, 0.5) * step_hours * 60),
        "early_departure_p90_minutes": float(np.quantile(advances, 0.9) * step_hours * 60),
        "early_departure_max_minutes": float(advances.max() * step_hours * 60),
        "individually_feasible_request_rate": float(np.mean(requests <= individual_capacity + 1e-9)),
        "unavoidable_individual_shortfall_kwh": float(unavoidable.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create non-identifying diagnostics for FairFlex's causal commitment benchmark."
    )
    parser.add_argument("--config", default="configs/caltech_2019_commitment_fairness.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/commitment_fairness/commitment_protocol_diagnostics.json"),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_study_config(config_path)
    uncertainty = resolve_commitment_uncertainty(config)
    if uncertainty is None:
        parser.error("the supplied configuration has no commitment_uncertainty protocol")
    prepared = {
        split: prepare_acn_split(config, config_path, split) for split in config["splits"]
    }
    step_hours = float(config["time_step_minutes"]) / 60.0
    calibration_name = str(uncertainty["calibration_split"])
    guard = EarlyDepartureGuard.fit(
        prepared[calibration_name].sessions,
        miscoverage=float(uncertainty["miscoverage"]),
    )
    test_name = "test"
    if test_name not in prepared:
        parser.error("the commitment protocol requires a separately named test split")

    report = {
        "study_id": config["study_id"],
        "purpose": (
            "Non-identifying audit of public ACN-derived declared commitments, "
            "realized unplug outcomes, and the calibration-locked risk buffer."
        ),
        "data_interpretation": config["acn_data_protocol"],
        "commitment_uncertainty": {
            **uncertainty,
            "fitted_buffer_steps": guard.buffer_steps,
            "fitted_buffer_minutes": guard.buffer_steps * float(config["time_step_minutes"]),
            "calibration_sessions": guard.calibration_sessions,
            "held_out_test_coverage": guard.empirical_coverage(prepared[test_name].sessions),
        },
        "split_diagnostics": {
            name: _split_summary(split.sessions, step_hours=step_hours)
            for name, split in prepared.items()
        },
        "interpretation_notes": [
            "An early departure is a realized physical unplug before the driver's declared deadline.",
            "Individual feasibility ignores shared-feeder and station scarcity; it only flags requests that cannot be met even at that EV's own maximum rate before its realized unplug.",
            "The held-out coverage audit measures the guard, not controller quality; it must not be used to tune the guard after seeing test data.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Commitment protocol diagnostics: {args.output}")
    print(
        "Guard: "
        f"{guard.buffer_steps} steps ({guard.buffer_steps * int(config['time_step_minutes'])} minutes), "
        f"target coverage={1.0 - guard.miscoverage:.3f}, "
        f"held-out coverage={report['commitment_uncertainty']['held_out_test_coverage']:.3f}"
    )
    for name, summary in report["split_diagnostics"].items():
        print(
            f"{name}: {summary['sessions']} sessions; "
            f"early departures={summary['early_departure_rate']:.1%}; "
            f"individual feasibility={summary['individually_feasible_request_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
