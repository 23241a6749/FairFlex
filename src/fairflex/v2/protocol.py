"""Validation for the separately versioned FairFlex V2 guard protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .commitments import DURATION_GROUPS, TIME_OF_DAY_GROUPS


@dataclass(frozen=True)
class ConditionAwareGuardProtocol:
    """Validated choices that must be frozen before a V2 test replay."""

    protocol_id: str
    miscoverage: float
    minimum_group_calibration_sessions: int
    time_step_minutes: int


def load_condition_aware_guard_protocol(path: Path | str) -> ConditionAwareGuardProtocol:
    """Read a strict V2 protocol JSON without mutating a V1 study config."""
    protocol_path = Path(path)
    try:
        document: dict[str, Any] = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V2 protocol: {error}") from error
    required = {
        "protocol_id",
        "status",
        "time_step_minutes",
        "replay_origin_requirement",
        "controller_information_boundary",
        "guard",
        "required_reporting",
        "forbidden_claims",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"V2 protocol is missing keys: {missing}")
    if document["status"] != "development_only_not_a_frozen_test_result":
        raise ValueError("V2 protocol status must forbid interpreting development output as final")
    time_step = int(document["time_step_minutes"])
    if time_step != 15:
        raise ValueError("V2 condition groups are currently fixed for 15-minute replay steps")
    if document["replay_origin_requirement"] != "local_midnight_with_explicit_utc_offset":
        raise ValueError("V2 arrival-time groups require the declared local-midnight replay origin")
    boundary = document["controller_information_boundary"]
    if not isinstance(boundary, dict):
        raise ValueError("V2 controller_information_boundary must be an object")
    forbidden_inputs = set(boundary.get("never_visible_to_controller_before_event", []))
    if "realized_physical_unplug_time" not in forbidden_inputs:
        raise ValueError("V2 protocol must forbid physical unplug time before its event")
    guard = document["guard"]
    if not isinstance(guard, dict):
        raise ValueError("V2 guard must be an object")
    if guard.get("method") != "condition_aware_one_sided_split_conformal_early_departure_guard_v2":
        raise ValueError("unsupported V2 guard method")
    miscoverage = float(guard.get("miscoverage", -1))
    if not 0 < miscoverage < 1:
        raise ValueError("V2 guard miscoverage must lie in (0, 1)")
    minimum = guard.get("minimum_group_calibration_sessions")
    if minimum is None or isinstance(minimum, bool) or int(minimum) <= 0:
        raise ValueError("V2 minimum group calibration sessions must be positive")
    if tuple(guard.get("duration_groups", ())) != DURATION_GROUPS:
        raise ValueError("V2 duration groups must match the fixed protocol")
    if tuple(guard.get("plug_in_time_groups", ())) != TIME_OF_DAY_GROUPS:
        raise ValueError("V2 plug-in-time groups must match the fixed protocol")
    if guard.get("sparse_group_action") != "use_global_split_conformal_buffer":
        raise ValueError("V2 sparse groups must use the global split-conformal fallback")
    return ConditionAwareGuardProtocol(
        protocol_id=str(document["protocol_id"]),
        miscoverage=miscoverage,
        minimum_group_calibration_sessions=int(minimum),
        time_step_minutes=time_step,
    )
