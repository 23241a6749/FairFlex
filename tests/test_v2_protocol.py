from __future__ import annotations

import json

import pytest

from fairflex.v2.protocol import load_condition_aware_guard_protocol
from fairflex.v2.selection import (
    apply_june_contextual_guard_rule,
    calendar_day_bootstrap_coverage_interval,
    rank_guard_candidates,
)
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": "test-protocol",
        "status": "development_only_not_a_frozen_test_result",
        "time_step_minutes": 15,
        "replay_origin_requirement": "local_midnight_with_explicit_utc_offset",
        "controller_information_boundary": {
            "visible_at_plugin": ["declared_departure"],
            "never_visible_to_controller_before_event": ["realized_physical_unplug_time"],
        },
        "guard": {
            "method": "condition_aware_one_sided_split_conformal_early_departure_guard_v2",
            "miscoverage": 0.1,
            "minimum_group_calibration_sessions": 50,
            "duration_groups": [
                "declared_duration_le_4h",
                "declared_duration_4_to_8h",
                "declared_duration_gt_8h",
            ],
            "plug_in_time_groups": [
                "arrival_06_to_10h",
                "arrival_10_to_14h",
                "arrival_other",
            ],
            "sparse_group_action": "use_global_split_conformal_buffer",
        },
        "required_reporting": ["overall_coverage"],
        "forbidden_claims": ["per_driver_coverage_guarantee"],
    }


def test_v2_protocol_accepts_the_frozen_condition_design(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()), encoding="utf-8")

    loaded = load_condition_aware_guard_protocol(path)

    assert loaded.protocol_id == "test-protocol"
    assert loaded.miscoverage == 0.1
    assert loaded.minimum_group_calibration_sessions == 50


def test_v2_protocol_rejects_physical_unplug_leakage(tmp_path) -> None:
    protocol = _protocol()
    boundary = protocol["controller_information_boundary"]
    assert isinstance(boundary, dict)
    boundary["never_visible_to_controller_before_event"] = []
    path = tmp_path / "bad_protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="forbid physical unplug"):
        load_condition_aware_guard_protocol(path)


def test_v2_runner_requires_local_midnight_for_time_of_day_groups() -> None:
    script = Path(__file__).parents[1] / "scripts" / "v2" / "run_condition_aware_development.py"
    spec = spec_from_file_location("fairflex_v2_runner", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    module._require_local_midnight_windows(
        {"splits": {"test": ["2019-05-01T00:00:00-07:00", "2019-05-02T00:00:00-07:00"]}},
        ("test",),
    )
    with pytest.raises(ValueError, match="local midnight"):
        module._require_local_midnight_windows(
            {"splits": {"test": ["2019-05-01T07:00:00Z", "2019-05-02T07:00:00Z"]}},
            ("test",),
        )


def test_v2_guard_candidate_ranking_requires_coverage_before_buffer() -> None:
    ranked = rank_guard_candidates(
        [
            {"protocol_id": "unsafe-small-buffer", "observed_coverage": 0.89, "target_coverage": 0.90, "p90_buffer_minutes": 30.0, "mean_buffer_minutes": 30.0, "global_fallback_rate": 0.0},
            {"protocol_id": "reliable-larger-buffer", "observed_coverage": 0.91, "target_coverage": 0.90, "p90_buffer_minutes": 45.0, "mean_buffer_minutes": 45.0, "global_fallback_rate": 1.0},
        ]
    )

    assert ranked[0]["protocol_id"] == "reliable-larger-buffer"
    assert ranked[0]["coverage_eligible"]


def test_v2_calendar_day_bootstrap_keeps_sessions_together() -> None:
    lower, upper = calendar_day_bootstrap_coverage_interval(
        [
            {"calendar_day": "2019-05-01", "covered": True},
            {"calendar_day": "2019-05-01", "covered": True},
            {"calendar_day": "2019-05-02", "covered": False},
        ],
        resamples=100,
        seed=7,
    )

    assert 0.0 <= lower <= upper <= 1.0


def test_v2_calendar_day_bootstrap_requires_multiple_days() -> None:
    with pytest.raises(ValueError, match="at least two calendar days"):
        calendar_day_bootstrap_coverage_interval(
            [{"calendar_day": "2019-05-01", "covered": True}],
            resamples=10,
        )


def test_v2_june_rule_keeps_global_guard_when_p10_fails() -> None:
    decision = apply_june_contextual_guard_rule(
        candidate_coverage=0.92,
        target_coverage=0.90,
        candidate_unsafe_steps=0,
        p10_delta=-0.006,
        jain_delta=-0.003,
        mean_buffer_delta_minutes=4.0,
    )

    assert not decision["contextual_candidate_advances"]
    assert decision["selected_guard"] == "global_fallback"
