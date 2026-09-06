import json

import pytest

from fairflex.scenarios import (
    build_data_manifest,
    deterministic_station_assignment,
    load_study_config,
    make_trace_simulation,
    prepare_acn_split,
    resolve_commitment_uncertainty,
    resolve_distributed_policy_settings,
    stations_from_config,
    with_replay_window,
)


def _write_tiny_study(tmp_path):
    config_dir = tmp_path / "configs"
    raw_dir = tmp_path / "data" / "raw"
    config_dir.mkdir()
    raw_dir.mkdir(parents=True)
    (raw_dir / "sessions.json").write_text(
        json.dumps(
            [
                {
                    "sessionID": "included",
                    "connectionTime": "2024-01-01T00:00:00Z",
                    "disconnectTime": "2024-01-01T01:00:00Z",
                    "kWhDelivered": 3.0,
                },
                {
                    "sessionID": "at-end",
                    "connectionTime": "2024-01-02T00:00:00Z",
                    "disconnectTime": "2024-01-02T01:00:00Z",
                    "kWhDelivered": 3.0,
                },
                {
                    "sessionID": "zero-energy",
                    "connectionTime": "2024-01-01T02:00:00Z",
                    "disconnectTime": "2024-01-01T03:00:00Z",
                    "kWhDelivered": 0.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    config_path = config_dir / "study.json"
    config_path.write_text(
        json.dumps(
            {
                "study_id": "tiny",
                "time_step_minutes": 15,
                "raw_files": {"acn": {"test": "data/raw/sessions.json"}},
                "splits": {"test": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]},
                "synthetic_stations": [
                    {"station_id": "north", "bus": 6, "capacity_kw": 7.2},
                    {"station_id": "south", "bus": 30, "capacity_kw": 7.2},
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_preparation_uses_half_open_connection_window_and_records_drops(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)

    prepared = prepare_acn_split(config, config_path, "test")

    assert prepared.source_records == 3
    assert prepared.selected_records == 2
    assert prepared.dropped_records == 1
    assert [session.ev_id for session in prepared.sessions] == ["included"]
    assert prepared.sessions[0].arrival_step == 0


def test_preparation_can_represent_an_empty_replay_window(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    config["splits"]["test"] = ["2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z"]

    prepared = prepare_acn_split(config, config_path, "test")

    assert prepared.selected_records == 0
    assert prepared.sessions == ()


def test_commitment_protocol_excludes_unclaimed_sessions_and_preserves_both_deadlines(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    raw_path = config_path.parent.parent / "data" / "raw" / "sessions.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "sessionID": "claimed",
                    "connectionTime": "2024-01-01T00:00:00Z",
                    "disconnectTime": "2024-01-01T01:00:00Z",
                    "kWhDelivered": 1.0,
                    "userInputs": [
                        {
                            "modifiedAt": "2024-01-01T00:01:00Z",
                            "kWhRequested": 5.0,
                            "requestedDeparture": "2024-01-01T02:00:00Z",
                        }
                    ],
                },
                {
                    "sessionID": "unclaimed",
                    "connectionTime": "2024-01-01T03:00:00Z",
                    "disconnectTime": "2024-01-01T04:00:00Z",
                    "kWhDelivered": 1.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    config = load_study_config(config_path)
    config["acn_data_protocol"] = {"target_mode": "declared_commitment"}

    prepared = prepare_acn_split(config, config_path, "test")

    assert prepared.target_mode == "declared_commitment"
    assert len(prepared.sessions) == 1
    session = prepared.sessions[0]
    assert session.requested_energy_kwh == pytest.approx(5.0)
    assert session.departure_step == 4
    assert session.planning_deadline_step == 8


def test_station_assignment_is_stable_and_uses_a_declared_station():
    stations = stations_from_config(
        {
            "synthetic_stations": [
                {"station_id": "north", "bus": 6, "capacity_kw": 7.2},
                {"station_id": "south", "bus": 30, "capacity_kw": 7.2},
            ]
        }
    )

    first = deterministic_station_assignment("session-42", stations)

    assert first == deterministic_station_assignment("session-42", stations)
    assert first in {"north", "south"}


def test_replay_window_override_is_bounded_and_does_not_mutate_source_config(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)

    updated = with_replay_window(
        config,
        "test",
        "2024-01-01T06:00:00Z",
        "2024-01-01T18:00:00Z",
    )

    assert config["splits"]["test"] == ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
    assert updated["splits"]["test"] == [
        "2024-01-01T06:00:00+00:00",
        "2024-01-01T18:00:00+00:00",
    ]
    with pytest.raises(ValueError, match="inside the configured split"):
        with_replay_window(
            config,
            "test",
            "2023-12-31T23:45:00Z",
            "2024-01-01T01:00:00Z",
        )


def test_trace_simulation_forwards_declared_grid_sensitivity(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    config["grid_sensitivity"] = {"background_load_scale": 0.55, "min_voltage_pu": 0.94}
    prepared = prepare_acn_split(config, config_path, "test")

    simulation = make_trace_simulation(config, prepared)

    assert simulation.grid.min_voltage_pu == pytest.approx(0.94)
    assert simulation.grid.validate({}).safe


def test_data_manifest_records_declared_grid_sensitivity(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    (config_path.parent.parent / "data" / "raw" / "weather.csv").write_text(
        "placeholder solar provenance input", encoding="utf-8"
    )
    config["raw_files"]["nsrdb"] = "data/raw/weather.csv"
    config["charging_data"] = {"station_assignment": "deterministic test assignment"}
    config["grid_sensitivity"] = {"background_load_scale": 0.55, "note": "diagnostic"}
    prepared = {"test": prepare_acn_split(config, config_path, "test")}

    manifest = build_data_manifest(config, config_path, prepared)

    assert manifest["grid_sensitivity"] == config["grid_sensitivity"]


def test_data_manifest_hashes_each_file_when_a_split_uses_multiple_raw_inputs(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    raw_dir = config_path.parent.parent / "data" / "raw"
    second = raw_dir / "sessions_second.json"
    second.write_text((raw_dir / "sessions.json").read_text(encoding="utf-8"), encoding="utf-8")
    weather = raw_dir / "weather.csv"
    weather.write_text("placeholder solar provenance input", encoding="utf-8")
    config["raw_files"]["acn"]["test"] = ["data/raw/sessions.json", "data/raw/sessions_second.json"]
    config["raw_files"]["nsrdb"] = "data/raw/weather.csv"
    config["charging_data"] = {"station_assignment": "deterministic test assignment"}

    manifest = build_data_manifest(
        config,
        config_path,
        {"test": prepare_acn_split(config, config_path, "test")},
    )

    assert len(manifest["raw_input_sha256"]["acn_test"]) == 2


def test_data_manifest_hashes_each_file_when_solar_input_is_extended(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    raw_dir = config_path.parent.parent / "data" / "raw"
    first = raw_dir / "weather_first.csv"
    second = raw_dir / "weather_second.csv"
    first.write_text("first solar provenance input", encoding="utf-8")
    second.write_text("second solar provenance input", encoding="utf-8")
    config["raw_files"]["nsrdb"] = [
        "data/raw/weather_first.csv",
        "data/raw/weather_second.csv",
    ]
    config["charging_data"] = {"station_assignment": "deterministic test assignment"}

    manifest = build_data_manifest(
        config,
        config_path,
        {"test": prepare_acn_split(config, config_path, "test")},
    )

    assert len(manifest["raw_input_sha256"]["nsrdb"]) == 2


def test_distributed_policy_settings_are_explicit_and_validate_ranges():
    settings = resolve_distributed_policy_settings({"rho": 1.0, "max_iterations": 80})

    assert settings["rho"] == 1.0
    assert settings["max_iterations"] == 80
    assert settings["equity_debt_decay"] == pytest.approx(0.95)
    with pytest.raises(ValueError, match="unknown distributed"):
        resolve_distributed_policy_settings({"not_a_setting": 1})
    with pytest.raises(ValueError, match="rho"):
        resolve_distributed_policy_settings({"rho": 0})
    with pytest.raises(ValueError, match="positive integer"):
        resolve_distributed_policy_settings({"max_iterations": 0})


def test_commitment_uncertainty_requires_causal_declared_commitments(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    config["acn_data_protocol"] = {"target_mode": "declared_commitment"}
    config["commitment_uncertainty"] = {
        "method": "one_sided_split_conformal_early_departure_guard",
        "miscoverage": 0.10,
        "calibration_split": "test",
    }

    resolved = resolve_commitment_uncertainty(config)

    assert resolved is not None
    assert resolved["miscoverage"] == pytest.approx(0.10)
    config["acn_data_protocol"] = {"target_mode": "observed_delivery"}
    with pytest.raises(ValueError, match="declared_commitment"):
        resolve_commitment_uncertainty(config)


def test_agaci_weighted_configuration_requires_a_valid_expert_grid(tmp_path):
    config_path = _write_tiny_study(tmp_path)
    config = load_study_config(config_path)
    config["acn_data_protocol"] = {"target_mode": "declared_commitment"}
    config["commitment_uncertainty"] = {
        "method": "one_sided_split_conformal_early_departure_guard",
        "miscoverage": 0.10,
        "calibration_split": "test",
        "candidate_modes": ["agaci_weighted"],
        "agaci_weighted": {
            "learning_rates": [0.001, 0.01],
            "min_miscoverage": 0.01,
            "max_miscoverage": 0.50,
            "epsilon": 0.001,
        },
    }

    assert resolve_commitment_uncertainty(config) is not None
    config["commitment_uncertainty"]["agaci_weighted"]["learning_rates"] = [0.01, 0.01]
    with pytest.raises(ValueError, match="distinct positive"):
        resolve_commitment_uncertainty(config)
