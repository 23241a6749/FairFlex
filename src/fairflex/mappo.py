"""Safe station-level MAPPO benchmark for FairFlex.

The learned agents never allocate EV power directly.  One agent per station
requests an aggregate station cap in ``[0, 1]``; a deterministic adapter then
projects those requests onto the feeder limit, allocates locally with the same
fair MPC used elsewhere in FairFlex, and applies the exact AC repair.  This
keeps the MARL comparison meaningful without allowing a learned policy to
silently violate electrical constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from .domain import EVSession
from .fair_mpc import CentralizedFairMPC, MPCResult
from .grid import IEEE33Grid
from .safety import ACFeasibilityRepair, ACRepairResult, FeederCapacitySource
from .simulation import ChargingSimulation


@dataclass(frozen=True)
class StationActionAudit:
    """Auditable difference between a raw MARL request and safe execution."""

    raw_station_cap_requests_kw: Mapping[str, float]
    feeder_projected_caps_kw: Mapping[str, float]
    raw_feeder_violation: bool
    raw_feeder_excess_kw: float
    raw_ac_unsafe: bool
    ac_repair: ACRepairResult


def _first_capacity(value: float | Sequence[float]) -> float:
    if isinstance(value, (float, int)):
        result = float(value)
    else:
        profile = np.asarray(value, dtype=float)
        if profile.ndim != 1 or profile.size == 0:
            raise ValueError("feeder capacity profile must be a non-empty one-dimensional array")
        result = float(profile[0])
    if not np.isfinite(result) or result < 0:
        raise ValueError("feeder capacity must be finite and non-negative")
    return result


def _project_station_caps(
    requested_caps_kw: Mapping[str, float],
    station_capacities_kw: Mapping[str, float],
    feeder_cap_kw: float,
) -> dict[str, float]:
    """Project station cap requests onto physical and shared-feeder limits."""
    if set(requested_caps_kw) != set(station_capacities_kw):
        raise ValueError("requested and physical station caps must share station IDs")
    station_ids = tuple(sorted(station_capacities_kw))
    values = np.array([requested_caps_kw[station_id] for station_id in station_ids], dtype=float)
    upper = np.array([station_capacities_kw[station_id] for station_id in station_ids], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(upper < 0):
        raise ValueError("station caps must be finite and non-negative")
    clipped = np.minimum(values, upper)
    available = min(float(feeder_cap_kw), float(upper.sum()))
    if available < 0:
        raise ValueError("feeder_cap_kw must be non-negative")
    if float(clipped.sum()) <= available + 1e-9:
        projected = clipped
    else:
        # Exact Euclidean projection onto {x >= 0, sum(x) = available}.
        # With upper clipping already applied, the solution is x_i=max(v_i-t,0).
        # Sorting identifies the active set directly and avoids the previous
        # 60-step bisection/allocation loop at every control interval.
        descending = np.sort(clipped)[::-1]
        cumulative = np.cumsum(descending)
        threshold = 0.0
        for count, value in enumerate(descending, start=1):
            candidate = (float(cumulative[count - 1]) - available) / count
            next_value = descending[count] if count < descending.size else -np.inf
            if candidate >= next_value - 1e-12:
                threshold = candidate
                break
        projected = np.maximum(clipped - threshold, 0.0)
    return {station_id: float(projected[index]) for index, station_id in enumerate(station_ids)}


class SafeStationCapPolicy:
    """Safely execute station-level cap fractions supplied by a MARL policy."""

    def __init__(
        self,
        controller: CentralizedFairMPC,
        grid: IEEE33Grid,
        *,
        horizon_steps: int,
        feeder_capacity_kw: FeederCapacitySource,
        feasible_cap_transform: bool = False,
    ) -> None:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        self.controller = controller
        self.grid = grid
        self.horizon_steps = horizon_steps
        self.feeder_capacity_kw = feeder_capacity_kw
        self.feasible_cap_transform = feasible_cap_transform
        self.repairer = ACFeasibilityRepair(grid)
        self._action_fractions: dict[str, float] | None = None
        self.last_audit: StationActionAudit | None = None
        self.action_history: list[StationActionAudit] = []
        self.last_station_results: Mapping[str, MPCResult] = {}
        self.last_applied_grid_validation = None
        self.last_applied_station_powers_kw: Mapping[str, float] | None = None

    def first_feeder_capacity_kw(self, current_step: int) -> float:
        value = (
            self.feeder_capacity_kw(current_step, self.horizon_steps)
            if callable(self.feeder_capacity_kw)
            else self.feeder_capacity_kw
        )
        return _first_capacity(value)

    def set_action_fractions(self, fractions: Mapping[str, float]) -> None:
        station_ids = set(self.grid.station_load_indices)
        if set(fractions) != station_ids:
            raise ValueError("MAPPO actions must specify every station exactly once")
        normalized = {station_id: float(value) for station_id, value in fractions.items()}
        if any(not np.isfinite(value) or value < 0 or value > 1 for value in normalized.values()):
            raise ValueError("MAPPO station-action fractions must be finite values in [0, 1]")
        self._action_fractions = normalized

    def _solve_locally(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        first_step_caps_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, MPCResult]:
        return {
            station_id: self.controller.solve(
                list(evs),
                {
                    station_id: [first_step_caps_kw[station_id]]
                    + [station_capacities_kw[station_id]] * (self.horizon_steps - 1)
                },
                current_step=current_step,
                horizon_steps=self.horizon_steps,
                step_hours=step_hours,
            )
            for station_id, evs in station_evs.items()
        }

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        if set(station_capacities_kw) != set(self.grid.station_load_indices):
            raise ValueError("policy station capacities must exactly match the AC grid stations")
        if self._action_fractions is None:
            raise RuntimeError("set_action_fractions must be called before executing a MARL step")
        priority_caps = {
            station_id: self._action_fractions[station_id] * station_capacities_kw[station_id]
            for station_id in station_capacities_kw
        }
        feeder_cap = self.first_feeder_capacity_kw(current_step)
        projected_caps = _project_station_caps(priority_caps, station_capacities_kw, feeder_cap)
        # In the feasible variant, station agents emit priorities and this
        # deterministic capped-simplex transform is their action.  Thus the
        # learned action itself obeys physical station and shared-feeder caps.
        # The original variant preserves the untransformed request for the
        # safety-ablation audit.
        raw_caps = projected_caps if self.feasible_cap_transform else priority_caps
        # Audit the learning action before the repair path.  This preserves the
        # distinction between raw policy feasibility and final execution, while
        # allowing the final applied AC result to remain cached for the
        # simulator's immediate integrity check.
        raw_grid_validation = self.grid.validate(raw_caps)
        initial_results = self._solve_locally(
            station_evs,
            station_capacities_kw,
            projected_caps,
            step_hours,
            current_step,
        )
        requested_power = {
            station_id: float(result.schedule_kw[:, 0].sum())
            for station_id, result in initial_results.items()
        }
        repair = self.repairer.repair(
            requested_power,
            flexibility_scores={
                station_id: sum(
                    max(
                        0.0,
                        (ev.effective_planning_deadline_step(current_step) - current_step)
                        * ev.max_power_kw
                        * step_hours
                        - ev.remaining_energy_kwh,
                    )
                    for ev in evs
                )
                for station_id, evs in station_evs.items()
            },
        )
        if repair.attempts:
            repaired_caps = {
                station_id: min(projected_caps[station_id], repair.applied_powers_kw[station_id])
                for station_id in station_capacities_kw
            }
            station_results = self._solve_locally(
                station_evs,
                station_capacities_kw,
                repaired_caps,
                step_hours,
                current_step,
            )
        else:
            station_results = initial_results
        applied_power = {
            station_id: float(result.schedule_kw[:, 0].sum())
            for station_id, result in station_results.items()
        }
        applied_grid_validation = self.grid.validate(applied_power)
        if not applied_grid_validation.safe:
            raise RuntimeError("safe MARL caps did not produce an AC-safe action")
        raw_feeder_excess_kw = max(0.0, float(sum(raw_caps.values())) - feeder_cap)
        audit = StationActionAudit(
            raw_station_cap_requests_kw=raw_caps,
            feeder_projected_caps_kw=projected_caps,
            raw_feeder_violation=raw_feeder_excess_kw > 1e-9,
            raw_feeder_excess_kw=raw_feeder_excess_kw,
            raw_ac_unsafe=not raw_grid_validation.safe,
            ac_repair=repair,
        )
        self.last_audit = audit
        self.action_history.append(audit)
        self.last_station_results = station_results
        self.last_applied_grid_validation = applied_grid_validation
        self.last_applied_station_powers_kw = dict(applied_power)
        return {
            ev_id: allocation
            for result in station_results.values()
            for ev_id, allocation in result.first_step_allocations().items()
        }


def station_observations(
    station_evs: Mapping[str, Sequence[EVSession]],
    station_capacities_kw: Mapping[str, float],
    *,
    current_step: int,
    step_hours: float,
    feeder_capacity_kw: float,
) -> dict[str, np.ndarray]:
    """Build bounded, station-local observations for a shared MAPPO actor.

    No EV identifiers leave a station.  The centralized critic receives the
    concatenation of these station summaries during training only.
    """
    total_capacity = max(float(sum(station_capacities_kw.values())), 1e-9)
    feeder_fraction = np.clip(feeder_capacity_kw / total_capacity, 0.0, 1.0)
    observations: dict[str, np.ndarray] = {}
    for station_id in sorted(station_capacities_kw):
        evs = station_evs[station_id]
        capacity = station_capacities_kw[station_id]
        urgency = max(
            (
                ev.remaining_energy_kwh
                / max(
                    (ev.effective_planning_deadline_step(current_step) - current_step)
                    * ev.max_power_kw
                    * step_hours,
                    1e-9,
                )
                for ev in evs
            ),
            default=0.0,
        )
        observations[station_id] = np.array(
            [
                min(len(evs) / 10.0, 1.0),
                min(sum(ev.remaining_energy_kwh for ev in evs) / max(2.0 * capacity, 1e-9), 1.0),
                min(urgency / 2.0, 1.0),
                float(np.mean([ev.service_ratio() for ev in evs])) if evs else 0.0,
                capacity / total_capacity,
                feeder_fraction,
            ],
            dtype=np.float32,
        )
    return observations


class StationMARLEnvironment:
    """Small Gym-like environment with a common reward for all station agents."""

    def __init__(
        self,
        scenario_factory: Callable[[], ChargingSimulation],
        feeder_capacity_factory: Callable[[ChargingSimulation], FeederCapacitySource],
        *,
        controller_factory: Callable[[], CentralizedFairMPC] = CentralizedFairMPC,
        horizon_steps: int = 4,
        fairness_reward_weight: float = 0.35,
        raw_feeder_violation_penalty: float = 0.10,
        raw_ac_violation_penalty: float = 0.10,
        feasible_cap_transform: bool = False,
    ) -> None:
        if (
            horizon_steps <= 0
            or fairness_reward_weight < 0
            or raw_feeder_violation_penalty < 0
            or raw_ac_violation_penalty < 0
        ):
            raise ValueError("MARL horizon and reward weights must be non-negative with a positive horizon")
        self.scenario_factory = scenario_factory
        self.feeder_capacity_factory = feeder_capacity_factory
        # The V3 LowerTailFairMPC is a CentralizedFairMPC subclass, so this
        # factory permits a matched downstream allocator without duplicating
        # the station-cap safety wrapper.  The default retains every existing
        # MAPPO/IPPO artifact exactly as it was trained.
        self.controller_factory = controller_factory
        self.horizon_steps = horizon_steps
        self.fairness_reward_weight = fairness_reward_weight
        self.raw_feeder_violation_penalty = raw_feeder_violation_penalty
        self.raw_ac_violation_penalty = raw_ac_violation_penalty
        self.feasible_cap_transform = feasible_cap_transform
        self.simulation: ChargingSimulation | None = None
        self.policy: SafeStationCapPolicy | None = None
        self._through_step = 0

    @property
    def agent_ids(self) -> tuple[str, ...]:
        if self.simulation is None:
            raise RuntimeError("reset must be called before reading agent IDs")
        return tuple(sorted(self.simulation.stations))

    @staticmethod
    def _utility(simulation: ChargingSimulation) -> float:
        # Concave service utility makes early service to an entirely neglected
        # EV more valuable than an equal increment to an already well-served EV.
        ratios = np.fromiter((ev.service_ratio() for ev in simulation.evs.values()), dtype=float)
        return float(np.mean(np.log(0.05 + ratios))) if ratios.size else 0.0

    def _observations(self) -> dict[str, np.ndarray]:
        if self.simulation is None or self.policy is None:
            raise RuntimeError("reset must be called before obtaining observations")
        return station_observations(
            self.simulation.active_evs_by_station(),
            self.simulation.station_capacities_kw,
            current_step=self.simulation.step,
            step_hours=self.simulation.step_hours,
            feeder_capacity_kw=self.policy.first_feeder_capacity_kw(self.simulation.step),
        )

    def reset(self) -> dict[str, np.ndarray]:
        self.simulation = self.scenario_factory()
        self.policy = SafeStationCapPolicy(
            self.controller_factory(),
            self.simulation.grid,
            horizon_steps=self.horizon_steps,
            feeder_capacity_kw=self.feeder_capacity_factory(self.simulation),
            feasible_cap_transform=self.feasible_cap_transform,
        )
        self._through_step = max((ev.departure_step for ev in self.simulation.evs.values()), default=0)
        return self._observations()

    def step(
        self, action_fractions: Mapping[str, float]
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self.simulation is None or self.policy is None:
            raise RuntimeError("reset must be called before step")
        before_energy = sum(ev.delivered_energy_kwh for ev in self.simulation.evs.values())
        before_utility = self._utility(self.simulation)
        self.policy.set_action_fractions(action_fractions)
        result = self.simulation.advance(self.policy)
        after_energy = sum(ev.delivered_energy_kwh for ev in self.simulation.evs.values())
        after_utility = self._utility(self.simulation)
        capacity_energy = max(sum(self.simulation.station_capacities_kw.values()) * self.simulation.step_hours, 1e-9)
        reward = (after_energy - before_energy) / capacity_energy + self.fairness_reward_weight * (
            after_utility - before_utility
        )
        audit = self.policy.last_audit
        assert audit is not None
        # Projection and AC repair guarantee actual feasibility.  Penalize the
        # *normalized excess*, rather than only its Boolean indicator, so the
        # unrepaired policy receives a learning signal for moving closer to the
        # feeder boundary instead of treating every over-limit request alike.
        feeder_cap = max(self.policy.first_feeder_capacity_kw(self.simulation.step - 1), 1e-9)
        raw_feeder_excess_fraction = audit.raw_feeder_excess_kw / feeder_cap
        reward -= self.raw_feeder_violation_penalty * raw_feeder_excess_fraction
        reward -= self.raw_ac_violation_penalty * float(audit.raw_ac_unsafe)
        done = self.simulation.step >= self._through_step
        return self._observations(), float(reward), done, {
            "actual_grid_safe": result.grid.safe,
            "raw_feeder_violation": audit.raw_feeder_violation,
            "raw_feeder_excess_kw": audit.raw_feeder_excess_kw,
            "raw_feeder_excess_fraction": raw_feeder_excess_fraction,
            "raw_ac_unsafe": audit.raw_ac_unsafe,
            "ac_repair_activated": audit.ac_repair.attempts > 0,
        }


class BetaActor(nn.Module):
    """Shared continuous-action policy: one Beta-distributed cap fraction per station."""

    def __init__(self, observation_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if observation_dim <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Beta:
        hidden = self.network(observations)
        alpha = torch.nn.functional.softplus(self.alpha_head(hidden)).squeeze(-1) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(hidden)).squeeze(-1) + 1.0
        return torch.distributions.Beta(alpha, beta)

    def deterministic_actions(self, observations: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(observations)
        return distribution.mean


class CentralCritic(nn.Module):
    """Centralized critic used only during training (CTDE)."""

    def __init__(self, global_state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state).squeeze(-1)


class LocalCritic(nn.Module):
    """Per-agent value estimator for the parameter-shared IPPO ablation.

    The same network is evaluated independently for every station.  It sees
    only that station's six local features, so it cannot use other stations'
    loads, urgency, or service history during training.
    """

    def __init__(self, observation_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


@dataclass(frozen=True)
class MAPPOConfig:
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    update_epochs: int = 4

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.learning_rate <= 0 or self.update_epochs <= 0:
            raise ValueError("MAPPO hidden_dim, learning_rate and update_epochs must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("MAPPO gamma and gae_lambda must lie in (0, 1] and [0, 1]")
        if self.clip_ratio <= 0 or self.entropy_weight < 0:
            raise ValueError("MAPPO clip_ratio must be positive and entropy_weight non-negative")


class MAPPOTrainer:
    """Auditable parameter-shared PPO trainer for controlled MARL ablations.

    ``critic_mode='centralized'`` is MAPPO: the shared actor sees only a local
    station observation, while its training-only critic sees all stations.
    ``critic_mode='local'`` is parameter-shared IPPO: both the actor and the
    value estimator see one station observation only.  Keeping the Beta actor,
    reward, optimizer and safe action adapter identical makes the difference
    attributable to centralized versus decentralized value information.
    """

    def __init__(
        self,
        observation_dim: int,
        num_agents: int,
        *,
        config: MAPPOConfig | None = None,
        seed: int = 0,
        action_parameterization: str = "projected_raw_request",
        torch_threads: int = 1,
        critic_mode: str = "centralized",
    ) -> None:
        if observation_dim <= 0 or num_agents <= 0:
            raise ValueError("observation_dim and num_agents must be positive")
        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
        self.observation_dim = observation_dim
        self.num_agents = num_agents
        self.config = config or MAPPOConfig()
        if action_parameterization not in {"projected_raw_request", "feasible_cap_transform"}:
            raise ValueError("unknown MAPPO action parameterization")
        if critic_mode not in {"centralized", "local"}:
            raise ValueError("critic_mode must be 'centralized' or 'local'")
        self.action_parameterization = action_parameterization
        self.critic_mode = critic_mode
        self.algorithm = "mappo" if critic_mode == "centralized" else "ippo_parameter_shared"
        self.seed = seed
        self.torch_threads = torch_threads
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.actor = BetaActor(observation_dim, self.config.hidden_dim)
        self.critic = (
            CentralCritic(observation_dim * num_agents, self.config.hidden_dim)
            if critic_mode == "centralized"
            else LocalCritic(observation_dim, self.config.hidden_dim)
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
        )

    @staticmethod
    def _stack_observations(observations: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray]:
        agent_ids = tuple(sorted(observations))
        return agent_ids, np.stack([observations[agent_id] for agent_id in agent_ids]).astype(np.float32)

    def _sample_actions(
        self, observations: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        agent_ids, local = self._stack_observations(observations)
        if len(agent_ids) != self.num_agents or local.shape[1] != self.observation_dim:
            raise ValueError("environment observation dimensions do not match the MAPPO trainer")
        local_tensor = torch.as_tensor(local, dtype=torch.float32)
        global_tensor = local_tensor.flatten().unsqueeze(0)
        with torch.no_grad():
            distribution = self.actor.distribution(local_tensor)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            if self.critic_mode == "centralized":
                value = np.array([float(self.critic(global_tensor).item())], dtype=np.float32)
            else:
                value = self.critic(local_tensor).numpy().astype(np.float32)
        return (
            {agent_id: float(actions[index].item()) for index, agent_id in enumerate(agent_ids)},
            local,
            actions.numpy(),
            value,
            log_probs.numpy(),
        )

    def _update(self, trajectory: Mapping[str, np.ndarray]) -> dict[str, float]:
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        dones = trajectory["dones"]
        if self.critic_mode == "centralized":
            values = values.reshape(-1)
            reward_matrix = rewards
        else:
            if values.shape != (len(rewards), self.num_agents):
                raise ValueError("local IPPO values must have one value per station and time step")
            reward_matrix = np.repeat(rewards[:, None], self.num_agents, axis=1)
        advantages = np.zeros_like(values, dtype=np.float32)
        gae = np.zeros(values.shape[1:], dtype=np.float32)
        next_value = np.zeros(values.shape[1:], dtype=np.float32)
        for index in range(len(rewards) - 1, -1, -1):
            mask = 1.0 - dones[index]
            delta = reward_matrix[index] + self.config.gamma * next_value * mask - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        local = torch.as_tensor(trajectory["local"], dtype=torch.float32)
        global_states = torch.as_tensor(trajectory["global"], dtype=torch.float32)
        actions = torch.as_tensor(trajectory["actions"], dtype=torch.float32)
        old_log_probs = torch.as_tensor(trajectory["log_probs"], dtype=torch.float32)
        actor_advantages = torch.as_tensor(
            np.repeat(normalized_advantages, self.num_agents)
            if self.critic_mode == "centralized"
            else normalized_advantages.reshape(-1),
            dtype=torch.float32,
        )
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        actor_loss_value = critic_loss_value = entropy_value = 0.0
        for _ in range(self.config.update_epochs):
            distribution = self.actor.distribution(local.reshape(-1, self.observation_dim))
            new_log_probs = distribution.log_prob(actions.reshape(-1))
            ratio = torch.exp(new_log_probs - old_log_probs.reshape(-1))
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
            )
            actor_loss = -torch.minimum(
                ratio * actor_advantages, clipped_ratio * actor_advantages
            ).mean()
            entropy = distribution.entropy().mean()
            if self.critic_mode == "centralized":
                critic_values = self.critic(global_states)
            else:
                critic_values = self.critic(local.reshape(-1, self.observation_dim)).reshape(
                    -1, self.num_agents
                )
            critic_loss = torch.nn.functional.mse_loss(critic_values, returns_tensor)
            loss = actor_loss + 0.5 * critic_loss - self.config.entropy_weight * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 1.0
            )
            self.optimizer.step()
            actor_loss_value = float(actor_loss.item())
            critic_loss_value = float(critic_loss.item())
            entropy_value = float(entropy.item())
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "entropy": entropy_value,
            "mean_return": float(returns.mean()),
        }

    def train(
        self,
        environment_factory: Callable[[int], StationMARLEnvironment],
        *,
        episodes: int,
        progress_callback: Callable[[Mapping[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        history: list[dict[str, float]] = []
        for episode in range(episodes):
            episode_start = perf_counter()
            environment = environment_factory(episode)
            observations = environment.reset()
            local_rows: list[np.ndarray] = []
            global_rows: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            log_probs: list[np.ndarray] = []
            rewards: list[float] = []
            values: list[np.ndarray] = []
            dones: list[float] = []
            raw_feeder_violations = raw_ac_unsafe = ac_repairs = 0
            raw_feeder_excess_kw_sum = 0.0
            done = False
            while not done:
                action_mapping, local, action, value, log_prob = self._sample_actions(observations)
                next_observations, reward, done, info = environment.step(action_mapping)
                local_rows.append(local)
                global_rows.append(local.reshape(-1))
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(float(done))
                raw_feeder_violations += int(info["raw_feeder_violation"])
                raw_feeder_excess_kw_sum += float(info["raw_feeder_excess_kw"])
                raw_ac_unsafe += int(info["raw_ac_unsafe"])
                ac_repairs += int(info["ac_repair_activated"])
                observations = next_observations
            update = self._update(
                {
                    "local": np.asarray(local_rows, dtype=np.float32),
                    "global": np.asarray(global_rows, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "log_probs": np.asarray(log_probs, dtype=np.float32),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                "values": np.asarray(values, dtype=np.float32),
                    "dones": np.asarray(dones, dtype=np.float32),
                }
            )
            solve_timings = environment.policy.controller.solve_timing_history if environment.policy else []
            record = {
                "episode": float(episode),
                "steps": float(len(rewards)),
                "episode_reward": float(sum(rewards)),
                "raw_feeder_violations": float(raw_feeder_violations),
                "raw_feeder_excess_kw_sum": raw_feeder_excess_kw_sum,
                "raw_ac_unsafe": float(raw_ac_unsafe),
                "ac_repair_activations": float(ac_repairs),
                "episode_seconds": perf_counter() - episode_start,
                "local_mpc_seconds": float(sum(item["total_seconds"] for item in solve_timings)),
                **update,
            }
            history.append(record)
            if progress_callback is not None:
                progress_callback(record)
        return history

    def save_checkpoint(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "fairflex_safe_marl_v2",
                "observation_dim": self.observation_dim,
                "num_agents": self.num_agents,
                "config": asdict(self.config),
                "seed": self.seed,
                "algorithm": self.algorithm,
                "critic_mode": self.critic_mode,
                "action_parameterization": self.action_parameterization,
                "torch_threads": self.torch_threads,
                "actor_state_dict": self.actor.state_dict(),
            },
            target,
        )


def load_station_marl_actor(path: Path | str) -> BetaActor:
    """Load a frozen station-MARL actor for decentralized execution.

    Version-1 MAPPO checkpoints remain supported so prior validation evidence
    stays reproducible after adding the IPPO ablation.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") not in {"fairflex_safe_mappo_v1", "fairflex_safe_marl_v2"}:
        raise ValueError("unrecognized FairFlex station-MARL checkpoint")
    actor = BetaActor(int(payload["observation_dim"]), int(payload["config"]["hidden_dim"]))
    actor.load_state_dict(payload["actor_state_dict"])
    actor.fairflex_action_parameterization = payload.get(
        "action_parameterization", "projected_raw_request"
    )
    actor.fairflex_algorithm = payload.get("algorithm", "mappo")
    actor.eval()
    return actor


def load_mappo_actor(path: Path | str) -> BetaActor:
    """Backward-compatible name for loading a frozen station-MARL actor."""
    return load_station_marl_actor(path)


class SafeMAPPOStationPolicy:
    """Frozen MAPPO actor plus the same deterministic feeder/AC safety layer."""

    def __init__(self, actor: BetaActor, safe_policy: SafeStationCapPolicy) -> None:
        self.actor = actor
        self.safe_policy = safe_policy
        self.action_fractions_history: list[Mapping[str, float]] = []

    @property
    def repair_history(self) -> list[ACRepairResult]:
        return [audit.ac_repair for audit in self.safe_policy.action_history]

    def __call__(
        self,
        station_evs: Mapping[str, Sequence[EVSession]],
        station_capacities_kw: Mapping[str, float],
        step_hours: float,
        current_step: int,
    ) -> dict[str, float]:
        observations = station_observations(
            station_evs,
            station_capacities_kw,
            current_step=current_step,
            step_hours=step_hours,
            feeder_capacity_kw=self.safe_policy.first_feeder_capacity_kw(current_step),
        )
        agent_ids = tuple(sorted(observations))
        local = np.stack([observations[agent_id] for agent_id in agent_ids])
        if local.shape[1] != self.actor.observation_dim:
            raise ValueError("checkpoint observation dimension does not match SafeMAPPOStationPolicy")
        with torch.no_grad():
            action_tensor = self.actor.deterministic_actions(
                torch.as_tensor(local, dtype=torch.float32)
            )
        fractions = {
            station_id: float(action_tensor[index].item())
            for index, station_id in enumerate(agent_ids)
        }
        self.safe_policy.set_action_fractions(fractions)
        self.action_fractions_history.append(fractions)
        return self.safe_policy(station_evs, station_capacities_kw, step_hours, current_step)
