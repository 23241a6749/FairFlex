from pathlib import Path

from fairflex.scenarios import load_study_config, prepare_acn_split
from fairflex.v3.marl_protocol import fit_v3_hybrid_deadline_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "v3" / "caltech_2019_fairflex_uc_development_september.json"


def test_matched_marl_preprocessor_reuses_the_v3_causal_hybrid_deadline():
    config = load_study_config(CONFIG_PATH)
    _, _, preprocessor = fit_v3_hybrid_deadline_preprocessor(config, CONFIG_PATH)
    unguarded = prepare_acn_split(config, CONFIG_PATH, "test")
    guarded, audit = preprocessor.apply(unguarded)

    assert len(guarded.sessions) == len(unguarded.sessions)
    assert audit["cqr"]["decisions"] == len(unguarded.sessions)
    assert audit["v1_multirate"]["decisions"] == len(unguarded.sessions)
    assert all(
        guarded_session.departure_step == original.departure_step
        and guarded_session.planning_deadline_step <= original.planning_deadline_step
        for original, guarded_session in zip(unguarded.sessions, guarded.sessions, strict=True)
    )
