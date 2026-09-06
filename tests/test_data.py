import numpy as np
import pandas as pd

from fairflex.data import (
    acn_records_to_sessions,
    lagged_pv_features,
    pvwatts_ghi_proxy_kw,
    read_nsrdb_csv,
)


def test_acn_records_become_discrete_ev_sessions_without_overlapping_time_rules():
    records = [
        {
            "sessionID": "same-id",
            "connectionTime": "2024-01-01T00:05:00Z",
            "disconnectTime": "2024-01-01T00:31:00Z",
            "kWhDelivered": 4.2,
        },
        {
            "sessionID": "same-id",
            "connectionTime": "2024-01-01T01:00:00Z",
            "disconnectTime": "2024-01-01T01:15:00Z",
            "kWhDelivered": 0.0,
        },
    ]

    sessions = acn_records_to_sessions(
        records, station_id="north", origin="2024-01-01T00:00:00Z"
    )

    assert len(sessions) == 1
    assert sessions[0].arrival_step == 0
    assert sessions[0].departure_step == 3
    assert sessions[0].requested_energy_kwh == 4.2


def test_declared_commitment_mode_uses_user_request_and_hides_actual_departure_from_planning():
    records = [
        {
            "sessionID": "declared",
            "connectionTime": "2024-01-01T00:00:00Z",
            "disconnectTime": "2024-01-01T00:45:00Z",
            "kWhDelivered": 2.0,
            "userInputs": [
                {
                    "modifiedAt": "2024-01-01T00:01:00Z",
                    "kWhRequested": 8.0,
                    "requestedDeparture": "2024-01-01T01:15:00Z",
                }
            ],
        },
        {
            "sessionID": "missing-commitment",
            "connectionTime": "2024-01-01T01:00:00Z",
            "disconnectTime": "2024-01-01T02:00:00Z",
            "kWhDelivered": 4.0,
        },
    ]

    sessions = acn_records_to_sessions(
        records,
        station_id="north",
        origin="2024-01-01T00:00:00Z",
        target_mode="declared_commitment",
    )

    assert len(sessions) == 1
    assert sessions[0].requested_energy_kwh == 8.0
    assert sessions[0].departure_step == 3  # realized unplug time: simulation only
    assert sessions[0].planning_deadline_step == 5  # declared deadline: controller only
    assert sessions[0].declared_departure_step == 5


def test_nsrdb_parser_and_pv_proxy_keep_night_generation_at_zero(tmp_path):
    path = tmp_path / "nsrdb.csv"
    path.write_text(
        "Source,example metadata\n"
        "Year,Month,Day,Hour,Minute,GHI,Temperature\n"
        "2024,1,1,0,0,0,20\n"
        "2024,1,1,12,0,1000,25\n",
        encoding="utf-8",
    )
    weather = read_nsrdb_csv(path)
    pv_kw = pvwatts_ghi_proxy_kw(weather, dc_capacity_kw=40.0, ac_capacity_kw=30.0)

    assert weather.index.tz is not None
    assert pv_kw.iloc[0] == 0.0
    assert 0.0 < pv_kw.iloc[1] <= 30.0


def test_nsrdb_parser_converts_declared_source_timezone_to_utc(tmp_path):
    path = tmp_path / "nsrdb_offset.csv"
    path.write_text(
        "Source,Time Zone,Local Time Zone\n"
        "example,-8,-8\n"
        "Year,Month,Day,Hour,Minute,GHI,Temperature\n"
        "2024,1,1,12,0,1000,25\n",
        encoding="utf-8",
    )

    weather = read_nsrdb_csv(path)

    assert weather.index[0] == pd.Timestamp("2024-01-01T20:00:00Z")


def test_lagged_features_use_only_past_generation_and_known_calendar_values():
    index = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    features, target, retained_index = lagged_pv_features(
        pd.Series(np.arange(8, dtype=float), index=index), lag_steps=(1, 2)
    )

    assert features.shape == (6, 4)
    assert target[0] == 2.0
    assert retained_index[0] == index[2]
    assert features[0, 0] == 1.0  # lag one at the first retained timestamp
