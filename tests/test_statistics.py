import pandas as pd
import pytest

from fairflex.statistics import (
    clustered_binary_rate_bootstrap,
    exact_paired_sign_flip_test,
    nested_seed_day_bootstrap,
    paired_sign_flip_test,
    paired_block_bootstrap,
    paired_metric_bootstrap,
)


def test_paired_block_bootstrap_is_deterministic_and_preserves_pairing():
    first = paired_block_bootstrap(
        [0.7, 0.8, 0.9],
        [0.5, 0.6, 0.7],
        metric="p10_service_ratio",
        treatment="fair",
        baseline="fcfs",
        resamples=500,
        seed=7,
    )
    second = paired_block_bootstrap(
        [0.7, 0.8, 0.9],
        [0.5, 0.6, 0.7],
        metric="p10_service_ratio",
        treatment="fair",
        baseline="fcfs",
        resamples=500,
        seed=7,
    )

    assert first == second
    assert first.blocks == 3
    assert first.observed_mean_difference == pytest.approx(0.2)
    assert first.ci_low == pytest.approx(0.2)
    assert first.ci_high == pytest.approx(0.2)
    assert first.bootstrap_probability_difference_positive == 1.0


def test_paired_metric_bootstrap_requires_complete_matched_days():
    table = pd.DataFrame(
        [
            {"study_id": "s", "replay_start": "day-1", "policy": "fair", "p10": 0.8},
            {"study_id": "s", "replay_start": "day-1", "policy": "base", "p10": 0.6},
            {"study_id": "s", "replay_start": "day-2", "policy": "fair", "p10": 0.7},
            {"study_id": "s", "replay_start": "day-2", "policy": "base", "p10": 0.5},
        ]
    )

    result = paired_metric_bootstrap(
        table,
        metric="p10",
        treatment="fair",
        baseline="base",
        resamples=500,
        seed=5,
    )
    assert result.blocks == 2
    assert result.observed_mean_difference == pytest.approx(0.2)

    incomplete = table.loc[table["replay_start"] != "day-2"].copy()
    incomplete = pd.concat(
        [incomplete, table.loc[(table["replay_start"] == "day-2") & (table["policy"] == "fair")]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="every matched block"):
        paired_metric_bootstrap(
            incomplete,
            metric="p10",
            treatment="fair",
            baseline="base",
            resamples=500,
        )


def test_exact_sign_flip_test_is_directional_and_uses_matched_blocks():
    higher_is_better = exact_paired_sign_flip_test(
        [0.8, 0.9, 1.0],
        [0.6, 0.7, 0.8],
        metric="p10",
        treatment="fair",
        baseline="base",
        preferred_direction="higher",
    )
    lower_is_better = exact_paired_sign_flip_test(
        [0.2, 0.1, 0.0],
        [0.4, 0.3, 0.2],
        metric="shortfall",
        treatment="fair",
        baseline="base",
        preferred_direction="lower",
    )

    assert higher_is_better.blocks == 3
    assert higher_is_better.exact_permutations == 8
    assert higher_is_better.observed_oriented_mean_advantage == pytest.approx(0.2)
    assert higher_is_better.p_value_one_sided == pytest.approx(0.125)
    assert higher_is_better.p_value_two_sided == pytest.approx(0.25)
    assert lower_is_better.observed_oriented_mean_advantage == pytest.approx(0.2)


def test_clustered_binary_rate_bootstrap_keeps_day_clusters_intact():
    result = clustered_binary_rate_bootstrap(
        ["day-1", "day-1", "day-2", "day-2"],
        [True, True, False, False],
        resamples=500,
        seed=8,
    )

    assert result.observed_rate == pytest.approx(0.5)
    assert result.blocks == 2
    assert result.observations == 4
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(1.0)


def test_sign_flip_uses_seeded_monte_carlo_for_a_multiweek_day_count():
    result = paired_sign_flip_test(
        [0.8] * 20,
        [0.6] * 20,
        metric="p10",
        treatment="fair",
        baseline="base",
        preferred_direction="higher",
        monte_carlo_resamples=2_000,
        seed=42,
    )

    assert result.blocks == 20
    assert result.test_method == "seeded_monte_carlo_paired_sign_flip"
    assert result.permutations == 2_000
    assert 0.0 < result.p_value_one_sided < 0.01


def test_nested_seed_day_bootstrap_resamples_both_registered_axes():
    result = nested_seed_day_bootstrap(
        [[0.10, 0.20, 0.30], [0.05, 0.15, 0.25]],
        metric="p10_service_ratio",
        treatment="seed_mean_policy",
        baseline="baseline",
        resamples=1_000,
        seed=41,
    )

    assert result.seeds == 2
    assert result.calendar_days == 3
    assert result.observed_mean_difference == pytest.approx(0.175)
    assert result.ci_low <= result.observed_mean_difference <= result.ci_high
    assert result.bootstrap_probability_difference_positive == pytest.approx(1.0)
