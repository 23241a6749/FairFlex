from pathlib import Path

from fairflex.scenarios import load_study_config


def test_caltech_november_temporal_replication_freezes_v3_before_acquisition():
    project_root = Path(__file__).resolve().parents[1]
    config = load_study_config(
        project_root / "configs" / "v3" / "caltech_2019_fairflex_uc_temporal_replication_november.json"
    )

    assert config["study_id"] == "caltech_2019_fairflex_uc_temporal_replication_november"
    assert config["raw_files"]["acn"]["test"] == "data/raw/acn_caltech_20191101_20191201.json"
    assert config["splits"]["test"] == ["2019-11-01T00:00:00Z", "2019-12-01T00:00:00Z"]
    assert config["v3"]["selected_deployable_policy"] == "fairflex_uc_hybrid_lower_tail"
    assert config["v3"]["tail_fraction"] == 0.10
    assert config["v3"]["cqr"]["miscoverage"] == 0.10
    assert config["statistical_protocol"]["primary_endpoint"] == "all-session P10 service ratio"
    assert "forbidden_after_acquisition" in config["statistical_protocol"]
