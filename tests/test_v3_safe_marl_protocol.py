import json
from pathlib import Path

from fairflex.scenarios import load_study_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "v3" / "caltech_2019_safe_marl_matched_protocol.json"
V3_NOVEMBER_PATH = PROJECT_ROOT / "configs" / "v3" / "caltech_2019_fairflex_uc_temporal_replication_november.json"


def test_safe_marl_protocol_reserves_a_temporal_test_window_outside_training():
    config = load_study_config(CONFIG_PATH)
    test_raw = config["raw_files"]["acn"]["test"]

    assert config["splits"]["test"] == ["2019-12-08T00:00:00Z", "2020-01-01T00:00:00Z"]
    assert test_raw not in config["raw_files"]["acn"]["train"]
    assert test_raw != config["raw_files"]["acn"]["validation"]
    assert config["safe_marl"]["training"]["seeds"] == [
        20260910,
        20260911,
        20260912,
        20260913,
        20260914,
    ]
    assert config["safe_marl"]["training"]["action_parameterization"] == "feasible_cap_transform"
    assert "LowerTailFairMPC" in config["safe_marl"]["downstream_execution"]


def test_safe_marl_protocol_locks_the_existing_v3_uncertainty_settings():
    registered = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v3_november = json.loads(V3_NOVEMBER_PATH.read_text(encoding="utf-8"))

    assert registered["v3"]["cqr"] == v3_november["v3"]["cqr"]
    assert registered["v3"]["v1_multirate_guard"] == v3_november["v3"]["v1_multirate_guard"]
    assert registered["v3"]["tail_fraction"] == v3_november["v3"]["tail_fraction"]
    assert registered["feeder_sensitivity"] == v3_november["feeder_sensitivity"]
