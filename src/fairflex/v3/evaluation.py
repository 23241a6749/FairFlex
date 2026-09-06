"""Evaluation helpers specific to the V3 early-unplug research question."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..domain import EVSession
from ..evaluation import jain_index
from ..simulation import SimulationResult
from .commitments import early_unplug_advance_steps


def expected_shortfall_of_deficits(service_ratios: Sequence[float], *, tail_fraction: float = 0.10) -> float:
    """Mean of the largest requested fraction of non-negative service deficits."""
    ratios = np.asarray(service_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.size == 0 or np.any(~np.isfinite(ratios)):
        raise ValueError("service_ratios must be a non-empty finite vector")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1]")
    deficits = np.maximum(0.0, 1.0 - ratios)
    count = max(1, ceil(tail_fraction * deficits.size))
    return float(np.sort(deficits)[-count:].mean())


@dataclass(frozen=True)
class EarlyUnplugSubsetMetrics:
    early_unplug_sessions: int
    early_unplug_p10_service_ratio: float
    early_unplug_mean_service_ratio: float
    early_unplug_worst_service_ratio: float
    all_session_expected_shortfall_10pct: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize_early_unplug_subset(
    result: SimulationResult,
    original_sessions: Sequence[EVSession],
    *,
    early_unplug_threshold_steps: int,
) -> EarlyUnplugSubsetMetrics:
    """Evaluate a predeclared physical-early-unplug subset after simulation."""
    if early_unplug_threshold_steps <= 0:
        raise ValueError("early_unplug_threshold_steps must be positive")
    early_ids = {
        session.ev_id
        for session in original_sessions
        if early_unplug_advance_steps(session) >= early_unplug_threshold_steps
    }
    ratios: Mapping[str, float] = result.service_ratios
    subset = np.asarray([ratios[ev_id] for ev_id in sorted(early_ids) if ev_id in ratios], dtype=float)
    all_ratios = np.asarray(list(ratios.values()), dtype=float)
    if subset.size == 0:
        return EarlyUnplugSubsetMetrics(
            early_unplug_sessions=0,
            early_unplug_p10_service_ratio=float("nan"),
            early_unplug_mean_service_ratio=float("nan"),
            early_unplug_worst_service_ratio=float("nan"),
            all_session_expected_shortfall_10pct=expected_shortfall_of_deficits(all_ratios),
        )
    return EarlyUnplugSubsetMetrics(
        early_unplug_sessions=int(subset.size),
        early_unplug_p10_service_ratio=float(np.quantile(subset, 0.10)),
        early_unplug_mean_service_ratio=float(subset.mean()),
        early_unplug_worst_service_ratio=float(subset.min()),
        all_session_expected_shortfall_10pct=expected_shortfall_of_deficits(all_ratios),
    )


def continuous_replay_session_rows(
    result: SimulationResult,
    original_sessions: Sequence[EVSession],
    *,
    policy: str,
    replay_origin: pd.Timestamp | str,
    step_minutes: int,
    early_unplug_threshold_steps: int,
) -> list[dict[str, object]]:
    """Return post-hoc session rows while retaining continuous-policy state.

    Calendar-day inference needs a matched daily block, but restarting an
    adaptive guard at midnight changes the executed controller.  This helper
    instead groups *final outcomes* from one continuous replay by session
    arrival day.  The physical disconnect time appears only in the post-hoc
    early-unplug label; it is never available to the policy.
    """
    if not policy:
        raise ValueError("policy must be non-empty")
    if step_minutes <= 0 or early_unplug_threshold_steps <= 0:
        raise ValueError("step_minutes and early_unplug_threshold_steps must be positive")
    origin = pd.to_datetime(replay_origin, utc=True)
    ratios: Mapping[str, float] = result.service_ratios
    rows: list[dict[str, object]] = []
    for session in original_sessions:
        if session.ev_id not in ratios:
            raise ValueError(f"simulation result is missing session {session.ev_id!r}")
        service_ratio = float(ratios[session.ev_id])
        arrival = origin + pd.Timedelta(minutes=step_minutes * session.arrival_step)
        rows.append({
            "policy": policy,
            "ev_id": session.ev_id,
            "calendar_day": arrival.date().isoformat(),
            "arrival_time": arrival.isoformat(),
            "station_id": session.station_id,
            "requested_energy_kwh": float(session.requested_energy_kwh),
            "delivered_energy_kwh": float(service_ratio * session.requested_energy_kwh),
            "service_ratio": service_ratio,
            "early_unplug_advance_steps": int(early_unplug_advance_steps(session)),
            "is_early_unplug": bool(
                early_unplug_advance_steps(session) >= early_unplug_threshold_steps
            ),
        })
    return rows


def continuous_arrival_day_metrics(session_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate continuous-replay outcomes into matched arrival-day blocks.

    This is deliberately an analysis-only aggregation.  It never starts a
    second simulation and consequently cannot reset or otherwise alter the
    causal guard that generated the session outcomes.
    """
    required = {
        "policy",
        "calendar_day",
        "requested_energy_kwh",
        "delivered_energy_kwh",
        "service_ratio",
        "is_early_unplug",
    }
    missing = sorted(required - set(session_rows.columns))
    if missing:
        raise ValueError(f"session rows are missing required columns: {missing}")
    if session_rows.empty:
        raise ValueError("session rows must be non-empty")

    rows: list[dict[str, object]] = []
    grouped = session_rows.groupby(["policy", "calendar_day"], sort=True)
    for (policy, calendar_day), group in grouped:
        ratios = group["service_ratio"].to_numpy(dtype=float)
        if ratios.size == 0 or not np.isfinite(ratios).all():
            raise ValueError("each policy/day block must contain finite service ratios")
        early = group.loc[group["is_early_unplug"].astype(bool), "service_ratio"].to_numpy(dtype=float)
        rows.append({
            "replay_start": f"{calendar_day}T00:00:00+00:00",
            "policy": str(policy),
            "sessions": int(ratios.size),
            "mean_service_ratio": float(ratios.mean()),
            "p10_service_ratio": float(np.quantile(ratios, 0.10)),
            "worst_service_ratio": float(ratios.min()),
            "jain_service_index": jain_index(ratios),
            "delivered_energy_kwh": float(group["delivered_energy_kwh"].sum()),
            "requested_energy_kwh": float(group["requested_energy_kwh"].sum()),
            "early_unplug_sessions": int(early.size),
            "early_unplug_p10_service_ratio": (
                float(np.quantile(early, 0.10)) if early.size else float("nan")
            ),
            "all_session_expected_shortfall_10pct": expected_shortfall_of_deficits(ratios),
        })
    return pd.DataFrame(rows).sort_values(["replay_start", "policy"]).reset_index(drop=True)
