import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_guard_ablation_keeps_the_primary_scenario_fixed():
    primary = _load("configs/jpl_2019_paper_baseline_comparison_november.json")
    ablation = _load("configs/jpl_2019_paper_guard_ablation_november.json")

    for key in (
        "time_step_minutes",
        "raw_files",
        "acn_data_protocol",
        "charging_data",
        "solar_data",
        "pv_proxy_sensitivity",
        "feeder_sensitivity",
        "splits",
        "synthetic_stations",
    ):
        assert ablation[key] == primary[key]
    assert ablation["commitment_uncertainty"]["calibration_split"] == "calibration"
    assert ablation["commitment_uncertainty"]["candidate_modes"] == [
        "global",
        "multi_rate_envelope",
    ]
    assert ablation["comparison_protocol"]["frozen_guard_conditions"] == [
        "unguarded",
        "global",
        "multi_rate_envelope",
    ]


def test_july_replication_keeps_primary_operational_settings_fixed():
    primary = _load("configs/jpl_2019_paper_baseline_comparison_november.json")
    replication = _load("configs/jpl_2019_seasonal_replication_july.json")

    for key in (
        "time_step_minutes",
        "acn_data_protocol",
        "charging_data",
        "solar_data",
        "pv_proxy_sensitivity",
        "feeder_sensitivity",
        "synthetic_stations",
    ):
        assert replication[key] == primary[key]
    assert replication["raw_files"]["acn"]["train"] == primary["raw_files"]["acn"]["train"]
    assert replication["raw_files"]["acn"]["calibration"] == primary["raw_files"]["acn"]["calibration"]
    assert replication["raw_files"]["acn"]["test"] == "data/raw/acn_jpl_20190701_20190801.json"
    assert replication["splits"]["test"] == ["2019-07-01T00:00:00Z", "2019-08-01T00:00:00Z"]
    assert replication["commitment_uncertainty"]["candidate_modes"] == ["multi_rate_envelope"]
    assert replication["comparison_protocol"]["frozen_policy_set"] == primary["comparison_protocol"]["frozen_policy_set"]


def test_caltech_cross_site_replication_keeps_the_method_fixed():
    primary = _load("configs/jpl_2019_paper_baseline_comparison_november.json")
    cross_site = _load("configs/caltech_2019_cross_site_replication_august.json")

    for key in (
        "time_step_minutes",
        "acn_data_protocol",
        "synthetic_stations",
    ):
        assert cross_site[key] == primary[key]
    for key in ("solar_data", "pv_proxy_sensitivity", "feeder_sensitivity"):
        assert {
            field: value for field, value in cross_site[key].items() if field != "note"
        } == {
            field: value for field, value in primary[key].items() if field != "note"
        }
    assert cross_site["charging_data"]["site_id"] == "caltech"
    assert cross_site["raw_files"]["acn"] == {
        "train": "data/raw/acn_caltech_20190301_20190308.json",
        "calibration": "data/raw/acn_caltech_20190401_20190407.json",
        "test": "data/raw/acn_caltech_20190801_20190901.json",
    }
    assert cross_site["splits"]["test"] == ["2019-08-01T00:00:00Z", "2019-09-01T00:00:00Z"]
    assert cross_site["commitment_uncertainty"]["multi_rate_envelope"] == primary["commitment_uncertainty"]["multi_rate_envelope"]
    assert cross_site["comparison_protocol"]["frozen_policy_set"] == primary["comparison_protocol"]["frozen_policy_set"]
