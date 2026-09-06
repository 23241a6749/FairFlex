import json

import pytest

import pandas as pd

from scripts.v3.run_frozen_sensitivity import _annotate_metrics, _load_protocol


def test_sensitivity_protocol_requires_the_frozen_hybrid_and_one_factor_scenarios(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({
        "protocol_id": "test",
        "base_study_config": "configs/v3/base.json",
        "selection_status": "frozen",
        "one_factor_at_a_time": True,
        "policies": ["v1_fair_mpc_multirate_guard", "fairflex_uc_hybrid_lower_tail"],
        "scenarios": [{"name": "import", "base_import_limit_kw": 8.0}],
    }), encoding="utf-8")

    protocol = _load_protocol(path)

    assert protocol["protocol_id"] == "test"
    path.write_text(json.dumps({
        **protocol,
        "scenarios": [{"name": "invalid", "base_import_limit_kw": 8.0, "pv_ac_capacity_kw": 15.0}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="must not simultaneously"):
        _load_protocol(path)


def test_sensitivity_metric_annotation_keeps_missing_overrides_numeric():
    result = _annotate_metrics(
        pd.DataFrame({"policy": ["fairflex_uc_hybrid_lower_tail"]}),
        {"name": "import", "base_import_limit_kw": 8.0},
    )

    assert result["base_import_limit_kw"].iloc[0] == 8.0
    assert result["pv_ac_capacity_kw"].isna().all()
    assert str(result["pv_ac_capacity_kw"].dtype).startswith("float")
