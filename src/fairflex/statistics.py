"""Paired, block-level uncertainty summaries for trace-driven experiments.

Policy outcomes from the same day are paired because every policy replays the
same arrivals and solar conditions. Resampling individual EV sessions would
incorrectly treat correlated observations from a day as independent evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Percentile bootstrap result for one matched policy comparison."""

    metric: str
    treatment: str
    baseline: str
    blocks: int
    observed_mean_difference: float
    ci_low: float
    ci_high: float
    bootstrap_probability_difference_positive: float
    bootstrap_probability_difference_negative: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NestedSeedDayBootstrapResult:
    """Crossed-cluster bootstrap summary for a multi-seed policy result.

    A neural-policy seed and a calendar day are distinct sources of
    variability. The bootstrap therefore resamples both axes, rather than
    treating the seed-by-day cells as independent observations.
    """

    metric: str
    treatment: str
    baseline: str
    seeds: int
    calendar_days: int
    observed_mean_difference: float
    ci_low: float
    ci_high: float
    bootstrap_probability_difference_positive: float
    bootstrap_probability_difference_negative: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactPairedSignFlipResult:
    """Exact randomisation test on matched, independently sampled blocks.

    This test is deliberately defined at the calendar-day block level.  It is
    not an EV-row t-test: sessions sharing a day have common arrival, solar,
    and capacity conditions and cannot be treated as independent replications.
    """

    metric: str
    treatment: str
    baseline: str
    preferred_direction: str
    blocks: int
    observed_oriented_mean_advantage: float
    exact_permutations: int
    p_value_one_sided: float
    p_value_two_sided: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairedSignFlipResult:
    """Exact or seeded Monte-Carlo paired sign-flip evidence."""

    metric: str
    treatment: str
    baseline: str
    preferred_direction: str
    blocks: int
    observed_oriented_mean_advantage: float
    test_method: str
    permutations: int
    p_value_one_sided: float
    p_value_two_sided: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClusteredRateBootstrapResult:
    """Cluster-preserving percentile interval for a binary event rate."""

    observed_rate: float
    blocks: int
    observations: int
    ci_low: float
    ci_high: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def paired_block_bootstrap(
    treatment_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    metric: str,
    treatment: str,
    baseline: str,
    resamples: int = 5_000,
    seed: int = 20260831,
    confidence: float = 0.95,
) -> PairedBootstrapResult:
    """Bootstrap matched block differences by resampling whole paired blocks.

    ``treatment_values[i]`` and ``baseline_values[i]`` must describe the same
    day/scenario block. A percentile interval is an uncertainty summary, not a
    claim of formal significance from a small pilot.
    """
    treatment_array = np.asarray(list(treatment_values), dtype=float)
    baseline_array = np.asarray(list(baseline_values), dtype=float)
    if treatment_array.ndim != 1 or baseline_array.ndim != 1:
        raise ValueError("paired inputs must be one-dimensional")
    if treatment_array.size != baseline_array.size or treatment_array.size < 2:
        raise ValueError("at least two matched treatment/baseline blocks are required")
    if not np.isfinite(treatment_array).all() or not np.isfinite(baseline_array).all():
        raise ValueError("paired inputs must be finite")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    differences = treatment_array - baseline_array
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, differences.size, size=(resamples, differences.size))
    bootstrap_means = differences[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return PairedBootstrapResult(
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        blocks=int(differences.size),
        observed_mean_difference=float(differences.mean()),
        ci_low=float(np.quantile(bootstrap_means, tail)),
        ci_high=float(np.quantile(bootstrap_means, 1.0 - tail)),
        bootstrap_probability_difference_positive=float(np.mean(bootstrap_means > 0.0)),
        bootstrap_probability_difference_negative=float(np.mean(bootstrap_means < 0.0)),
        resamples=resamples,
        seed=seed,
    )


def nested_seed_day_bootstrap(
    seed_day_differences: np.ndarray | Iterable[Iterable[float]],
    *,
    metric: str,
    treatment: str,
    baseline: str,
    resamples: int = 5_000,
    seed: int = 20260903,
    confidence: float = 0.95,
) -> NestedSeedDayBootstrapResult:
    """Resample predeclared training seeds and matched calendar days jointly.

    ``seed_day_differences`` has shape ``(seeds, calendar_days)`` and contains
    raw treatment-minus-baseline values. Each replicate samples seeds with
    replacement, samples days with replacement, and averages their crossed
    cells. This preserves the repeated common V3 baseline across neural seeds
    without treating the seed-by-day cells as independent observations.
    """
    differences = np.asarray(seed_day_differences, dtype=float)
    if differences.ndim != 2:
        raise ValueError("seed_day_differences must be a two-dimensional array")
    seed_count, day_count = differences.shape
    if seed_count < 2 or day_count < 2:
        raise ValueError("at least two seeds and two calendar days are required")
    if not np.isfinite(differences).all():
        raise ValueError("seed_day_differences must be finite")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    generator = np.random.default_rng(seed)
    sampled_seeds = generator.integers(0, seed_count, size=(resamples, seed_count))
    sampled_days = generator.integers(0, day_count, size=(resamples, day_count))
    sampled_means = differences[
        sampled_seeds[:, :, None], sampled_days[:, None, :]
    ].mean(axis=(1, 2))
    tail = (1.0 - confidence) / 2.0
    return NestedSeedDayBootstrapResult(
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        seeds=int(seed_count),
        calendar_days=int(day_count),
        observed_mean_difference=float(differences.mean()),
        ci_low=float(np.quantile(sampled_means, tail)),
        ci_high=float(np.quantile(sampled_means, 1.0 - tail)),
        bootstrap_probability_difference_positive=float(np.mean(sampled_means > 0.0)),
        bootstrap_probability_difference_negative=float(np.mean(sampled_means < 0.0)),
        resamples=resamples,
        seed=seed,
    )


def exact_paired_sign_flip_test(
    treatment_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    metric: str,
    treatment: str,
    baseline: str,
    preferred_direction: str,
    max_exact_blocks: int = 16,
) -> ExactPairedSignFlipResult:
    """Test a directional matched-block advantage by exact label swapping.

    Under the sharp null that either policy could have occupied each matched
    block, swapping their labels changes the sign of that block's difference.
    We enumerate all sign assignments, avoiding a normality assumption that is
    unsuitable for small-sample daily P10 values.  ``preferred_direction`` is
    ``"higher"`` for metrics such as P10 and ``"lower"`` for deficits.

    This is meaningful only when the blocks were fixed before the test and are
    reasonably independent (for this project: separate calendar days).  It is
    intentionally limited to 16 blocks so an exact calculation cannot quietly
    turn into an uncontrolled approximation.
    """
    treatment_array = np.asarray(list(treatment_values), dtype=float)
    baseline_array = np.asarray(list(baseline_values), dtype=float)
    if treatment_array.ndim != 1 or baseline_array.ndim != 1:
        raise ValueError("paired inputs must be one-dimensional")
    if treatment_array.size != baseline_array.size or treatment_array.size < 2:
        raise ValueError("at least two matched treatment/baseline blocks are required")
    if not np.isfinite(treatment_array).all() or not np.isfinite(baseline_array).all():
        raise ValueError("paired inputs must be finite")
    if preferred_direction not in {"higher", "lower"}:
        raise ValueError("preferred_direction must be 'higher' or 'lower'")
    if treatment_array.size > max_exact_blocks:
        raise ValueError(
            f"exact sign-flip test supports at most {max_exact_blocks} blocks; "
            "predeclare a Monte-Carlo or asymptotic alternative for larger studies"
        )

    sign = 1.0 if preferred_direction == "higher" else -1.0
    oriented_differences = sign * (treatment_array - baseline_array)
    observed = float(oriented_differences.mean())
    signs = np.asarray(tuple(product((-1.0, 1.0), repeat=oriented_differences.size)), dtype=float)
    permuted_means = (signs * oriented_differences).mean(axis=1)
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 8.0
    return ExactPairedSignFlipResult(
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        preferred_direction=preferred_direction,
        blocks=int(oriented_differences.size),
        observed_oriented_mean_advantage=observed,
        exact_permutations=int(permuted_means.size),
        p_value_one_sided=float(np.mean(permuted_means >= observed - tolerance)),
        p_value_two_sided=float(
            np.mean(np.abs(permuted_means) >= abs(observed) - tolerance)
        ),
    )


def paired_sign_flip_test(
    treatment_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    metric: str,
    treatment: str,
    baseline: str,
    preferred_direction: str,
    max_exact_blocks: int = 16,
    monte_carlo_resamples: int = 100_000,
    seed: int = 20260903,
) -> PairedSignFlipResult:
    """Use exact sign flips when feasible, otherwise a seeded MC approximation.

    A month can supply more than 16 eligible day blocks, making ``2**n`` exact
    swaps impractical.  The Monte-Carlo branch retains the identical matched
    day-label-swap null and a fixed random seed; it only approximates the tail
    probability.  The +1 correction avoids reporting an impossible p-value of
    zero from a finite random sample.
    """
    treatment_array = np.asarray(list(treatment_values), dtype=float)
    baseline_array = np.asarray(list(baseline_values), dtype=float)
    if treatment_array.ndim != 1 or baseline_array.ndim != 1:
        raise ValueError("paired inputs must be one-dimensional")
    if treatment_array.size != baseline_array.size or treatment_array.size < 2:
        raise ValueError("at least two matched treatment/baseline blocks are required")
    if not np.isfinite(treatment_array).all() or not np.isfinite(baseline_array).all():
        raise ValueError("paired inputs must be finite")
    if preferred_direction not in {"higher", "lower"}:
        raise ValueError("preferred_direction must be 'higher' or 'lower'")
    if monte_carlo_resamples < 1_000:
        raise ValueError("monte_carlo_resamples must be at least 1000")
    if treatment_array.size <= max_exact_blocks:
        exact = exact_paired_sign_flip_test(
            treatment_array,
            baseline_array,
            metric=metric,
            treatment=treatment,
            baseline=baseline,
            preferred_direction=preferred_direction,
            max_exact_blocks=max_exact_blocks,
        )
        return PairedSignFlipResult(
            metric=exact.metric,
            treatment=exact.treatment,
            baseline=exact.baseline,
            preferred_direction=exact.preferred_direction,
            blocks=exact.blocks,
            observed_oriented_mean_advantage=exact.observed_oriented_mean_advantage,
            test_method="exact_paired_sign_flip",
            permutations=exact.exact_permutations,
            p_value_one_sided=exact.p_value_one_sided,
            p_value_two_sided=exact.p_value_two_sided,
        )

    sign = 1.0 if preferred_direction == "higher" else -1.0
    differences = sign * (treatment_array - baseline_array)
    observed = float(differences.mean())
    generator = np.random.default_rng(seed)
    signs = generator.choice(np.array([-1.0, 1.0]), size=(monte_carlo_resamples, differences.size))
    permuted_means = (signs * differences).mean(axis=1)
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 8.0
    one_sided_exceedances = int(np.sum(permuted_means >= observed - tolerance))
    two_sided_exceedances = int(np.sum(np.abs(permuted_means) >= abs(observed) - tolerance))
    return PairedSignFlipResult(
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        preferred_direction=preferred_direction,
        blocks=int(differences.size),
        observed_oriented_mean_advantage=observed,
        test_method="seeded_monte_carlo_paired_sign_flip",
        permutations=monte_carlo_resamples,
        p_value_one_sided=float((one_sided_exceedances + 1) / (monte_carlo_resamples + 1)),
        p_value_two_sided=float((two_sided_exceedances + 1) / (monte_carlo_resamples + 1)),
    )


def clustered_binary_rate_bootstrap(
    block_labels: Iterable[object],
    events: Iterable[bool],
    *,
    resamples: int = 5_000,
    seed: int = 20260903,
    confidence: float = 0.95,
) -> ClusteredRateBootstrapResult:
    """Bootstrap a binary rate while keeping all observations from a block.

    In FairFlex a calendar day is the block: sessions within it share weather,
    arrivals, and scarcity.  This produces a descriptive uncertainty interval,
    not a new conformal guarantee or a substitute for a longer holdout.
    """
    labels = list(block_labels)
    values = np.asarray(list(events), dtype=bool)
    if len(labels) != values.size or values.size == 0:
        raise ValueError("block_labels and events must have the same non-zero length")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    grouped: dict[object, np.ndarray] = {}
    for label, value in zip(labels, values, strict=True):
        grouped.setdefault(label, np.asarray([], dtype=bool))
        grouped[label] = np.append(grouped[label], value)
    if len(grouped) < 2:
        raise ValueError("at least two blocks are required for a clustered bootstrap")
    block_values = tuple(grouped.values())
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        choices = generator.integers(0, len(block_values), size=len(block_values))
        sampled = np.concatenate([block_values[index] for index in choices])
        estimates[draw] = float(sampled.mean())
    tail = (1.0 - confidence) / 2.0
    return ClusteredRateBootstrapResult(
        observed_rate=float(values.mean()),
        blocks=len(block_values),
        observations=int(values.size),
        ci_low=float(np.quantile(estimates, tail)),
        ci_high=float(np.quantile(estimates, 1.0 - tail)),
        resamples=resamples,
        seed=seed,
    )


def paired_metric_bootstrap(
    day_metrics: pd.DataFrame,
    *,
    metric: str,
    treatment: str,
    baseline: str,
    block_columns: tuple[str, ...] = ("study_id", "replay_start"),
    resamples: int = 5_000,
    seed: int = 20260831,
) -> PairedBootstrapResult:
    """Validate a complete matched table, then bootstrap a policy difference."""
    required = {"policy", metric, *block_columns}
    missing = sorted(required - set(day_metrics.columns))
    if missing:
        raise ValueError(f"day metrics are missing columns: {missing}")
    selected = day_metrics.loc[
        day_metrics["policy"].isin((treatment, baseline)), [*block_columns, "policy", metric]
    ].copy()
    if selected.empty:
        raise ValueError("neither requested policy is present")
    if selected.duplicated([*block_columns, "policy"]).any():
        raise ValueError("each policy must appear once per matched block")
    wide = selected.pivot(index=list(block_columns), columns="policy", values=metric)
    missing_policies = [name for name in (treatment, baseline) if name not in wide.columns]
    if missing_policies:
        raise ValueError(f"requested policy is absent: {missing_policies}")
    paired = wide[[treatment, baseline]].dropna()
    if len(paired) != len(wide):
        raise ValueError("each compared policy must be present in every matched block")
    return paired_block_bootstrap(
        paired[treatment].to_numpy(),
        paired[baseline].to_numpy(),
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        resamples=resamples,
        seed=seed,
    )
