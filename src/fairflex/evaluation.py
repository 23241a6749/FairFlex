"""Reproducible policy evaluation and fairness metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .simulation import ChargingSimulation, Policy, SimulationResult


@dataclass(frozen=True)
class OutcomeMetrics:
    policy: str
    sessions: int
    mean_service_ratio: float
    p10_service_ratio: float
    worst_service_ratio: float
    jain_service_index: float
    delivered_energy_kwh: float
    requested_energy_kwh: float
    energy_service_ratio: float
    completion_rate: float
    individually_feasible_request_rate: float
    unavoidable_individual_shortfall_kwh: float
    min_voltage_pu: float
    max_line_loading_percent: float
    unsafe_steps: int
    runtime_seconds: float


@dataclass(frozen=True)
class PolicyRun:
    metrics: OutcomeMetrics
    result: SimulationResult


def jain_index(values: Sequence[float]) -> float:
    """Return Jain's fairness index, with 1 denoting equal non-negative values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or np.any(array < 0):
        raise ValueError("values must be a non-empty one-dimensional non-negative sequence")
    denominator = array.size * float(np.square(array).sum())
    return 0.0 if denominator == 0 else float(array.sum() ** 2 / denominator)


def summarize_outcome(
    policy: str, result: SimulationResult, *, runtime_seconds: float = 0.0
) -> OutcomeMetrics:
    """Report distributional service and safety, not only total energy."""
    ratios = np.fromiter(result.service_ratios.values(), dtype=float)
    if ratios.size == 0:
        raise ValueError("cannot summarize an experiment with no EV sessions")
    if runtime_seconds < 0:
        raise ValueError("runtime_seconds must be non-negative")
    min_voltages = [step.grid.min_voltage_pu for step in result.steps if step.grid.min_voltage_pu is not None]
    max_loadings = [
        step.grid.max_line_loading_percent
        for step in result.steps
        if step.grid.max_line_loading_percent is not None
    ]
    return OutcomeMetrics(
        policy=policy,
        sessions=ratios.size,
        mean_service_ratio=float(ratios.mean()),
        p10_service_ratio=float(np.quantile(ratios, 0.1)),
        worst_service_ratio=float(ratios.min()),
        jain_service_index=jain_index(ratios),
        delivered_energy_kwh=result.delivered_energy_kwh,
        requested_energy_kwh=result.requested_energy_kwh,
        energy_service_ratio=float(result.delivered_energy_kwh / result.requested_energy_kwh),
        completion_rate=float(np.mean(ratios >= 1.0 - 1e-6)),
        individually_feasible_request_rate=float(
            result.individually_feasible_sessions / ratios.size
        ),
        unavoidable_individual_shortfall_kwh=result.unavoidable_individual_shortfall_kwh,
        min_voltage_pu=float(min(min_voltages)) if min_voltages else float("nan"),
        max_line_loading_percent=float(max(max_loadings)) if max_loadings else float("nan"),
        unsafe_steps=result.unsafe_steps,
        runtime_seconds=float(runtime_seconds),
    )


class ExperimentRunner:
    """Run every policy on an independently constructed but identical scenario.

    The scenario factory is required because a simulation mutates delivered EV
    energy. Reusing one simulation across policies is a common and serious
    experiment bug: the second policy would inherit the first policy's work.
    """

    def __init__(self, scenario_factory: Callable[[], ChargingSimulation]) -> None:
        self.scenario_factory = scenario_factory

    def run(
        self, policy_factories: Mapping[str, Callable[[], Policy]]
    ) -> dict[str, PolicyRun]:
        if not policy_factories:
            raise ValueError("at least one policy is required")
        outputs: dict[str, PolicyRun] = {}
        for name, make_policy in policy_factories.items():
            simulation = self.scenario_factory()
            start = perf_counter()
            result = simulation.run(make_policy())
            outputs[name] = PolicyRun(
                summarize_outcome(name, result, runtime_seconds=perf_counter() - start), result
            )
        return outputs

    def run_with_simulation_policy(
        self, policy_builders: Mapping[str, Callable[[ChargingSimulation], Policy]]
    ) -> dict[str, PolicyRun]:
        """Run policies whose construction needs the scenario's AC-grid object."""
        if not policy_builders:
            raise ValueError("at least one policy is required")
        outputs: dict[str, PolicyRun] = {}
        for name, make_policy in policy_builders.items():
            simulation = self.scenario_factory()
            start = perf_counter()
            result = simulation.run(make_policy(simulation))
            outputs[name] = PolicyRun(
                summarize_outcome(name, result, runtime_seconds=perf_counter() - start), result
            )
        return outputs

    @staticmethod
    def metrics_table(runs: Mapping[str, PolicyRun]) -> pd.DataFrame:
        """A stable table suitable for CSV export or paper figures."""
        return (
            pd.DataFrame([asdict(run.metrics) for run in runs.values()])
            .sort_values("policy")
            .reset_index(drop=True)
        )
