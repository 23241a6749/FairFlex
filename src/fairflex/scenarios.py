"""Reproducible conversion of raw trace files into FairFlex experiments.

This module deliberately keeps real observations and modelling assumptions
separate. ACN sessions are observed charging traces, whereas the allocation to
three IEEE-33 feeder stations and the GHI-to-PV conversion are controlled
scenario assumptions that the experiment manifest records explicitly.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import acn_records_to_sessions, pvwatts_ghi_proxy_kw, read_nsrdb_csv
from .domain import ChargingStation, EVSession
from .grid import IEEE33Grid
from .simulation import ChargingSimulation


DEFAULT_DISTRIBUTED_POLICY_SETTINGS: dict[str, float | int] = {
    "rho": 2.0,
    "max_iterations": 250,
    "tolerance": 1e-4,
    "equity_debt_decay": 0.95,
    "equity_debt_gain": 1.0,
}


def resolve_distributed_policy_settings(
    declared_settings: Mapping[str, Any] | None,
) -> dict[str, float | int]:
    """Return validated, explicit ADMM and station-equity settings.

    Keeping these settings in a versioned, credential-free JSON file lets a
    validation study lock them before the held-out seasonal comparisons.  The
    defaults preserve the original implementation when no profile is supplied.
    """
    supplied = dict(declared_settings or {})
    unknown = set(supplied) - set(DEFAULT_DISTRIBUTED_POLICY_SETTINGS)
    if unknown:
        raise ValueError(f"unknown distributed policy settings: {sorted(unknown)}")
    resolved = {**DEFAULT_DISTRIBUTED_POLICY_SETTINGS, **supplied}
    if float(resolved["rho"]) <= 0:
        raise ValueError("distributed rho must be positive")
    if isinstance(resolved["max_iterations"], bool) or int(resolved["max_iterations"]) <= 0:
        raise ValueError("distributed max_iterations must be a positive integer")
    if float(resolved["tolerance"]) <= 0:
        raise ValueError("distributed tolerance must be positive")
    if not 0 <= float(resolved["equity_debt_decay"]) <= 1:
        raise ValueError("distributed equity_debt_decay must be in [0, 1]")
    if float(resolved["equity_debt_gain"]) < 0:
        raise ValueError("distributed equity_debt_gain must be non-negative")
    return {
        "rho": float(resolved["rho"]),
        "max_iterations": int(resolved["max_iterations"]),
        "tolerance": float(resolved["tolerance"]),
        "equity_debt_decay": float(resolved["equity_debt_decay"]),
        "equity_debt_gain": float(resolved["equity_debt_gain"]),
    }


@dataclass(frozen=True)
class ReplayWindow:
    """A half-open UTC interval: ``start <= connection_time < end``."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("replay-window timestamps must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("replay-window end must be after its start")

    @classmethod
    def from_config(cls, values: Sequence[str]) -> "ReplayWindow":
        if len(values) != 2:
            raise ValueError("each split must contain exactly start and end timestamps")
        start, end = (pd.Timestamp(value) for value in values)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("split timestamps must include an explicit timezone")
        return cls(start.tz_convert("UTC"), end.tz_convert("UTC"))

    def includes_connection(self, value: Any) -> bool:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        return not pd.isna(timestamp) and self.start <= timestamp < self.end


@dataclass(frozen=True)
class PreparedSplit:
    """Validated sessions for one chronological replay window."""

    name: str
    window: ReplayWindow
    source_records: int
    selected_records: int
    dropped_records: int
    sessions: tuple[EVSession, ...]
    station_session_counts: Mapping[str, int]
    target_mode: str

    @property
    def requested_energy_kwh(self) -> float:
        return sum(session.requested_energy_kwh for session in self.sessions)


def load_study_config(path: Path | str) -> dict[str, Any]:
    """Load and minimally validate a versioned JSON study description."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"study_id", "time_step_minutes", "raw_files", "splits", "synthetic_stations"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"study config is missing keys: {missing}")
    if int(config["time_step_minutes"]) <= 0:
        raise ValueError("time_step_minutes must be positive")
    if not isinstance(config["raw_files"].get("acn"), Mapping):
        raise ValueError("raw_files.acn must map each split name to one JSON file or a list of JSON files")
    if not config["synthetic_stations"]:
        raise ValueError("synthetic_stations must not be empty")
    resolve_acn_target_mode(config)
    resolve_commitment_uncertainty(config)
    return config


def resolve_acn_target_mode(config: Mapping[str, Any]) -> str:
    """Return the declared ACN replay target protocol.

    Older study files remain reproducible through the explicit legacy default.
    New paper studies must declare ``declared_commitment`` so their controller
    never treats historical delivery or realized unplug time as future input.
    """
    protocol = config.get("acn_data_protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("acn_data_protocol must be an object when provided")
    target_mode = protocol.get("target_mode", "observed_delivery")
    if target_mode not in {"observed_delivery", "declared_commitment"}:
        raise ValueError("acn_data_protocol.target_mode is invalid")
    return str(target_mode)


def resolve_commitment_uncertainty(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a validated, explicit early-departure uncertainty protocol.

    Conditional and adaptive extensions must be separately specified and
    evaluated; silently treating an arbitrary learned model as calibrated
    would make final fairness claims difficult to audit.
    """
    declared = config.get("commitment_uncertainty")
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise ValueError("commitment_uncertainty must be an object when provided")
    required = {"method", "miscoverage", "calibration_split"}
    missing = sorted(required - set(declared))
    if missing:
        raise ValueError(f"commitment_uncertainty is missing keys: {missing}")
    if declared["method"] != "one_sided_split_conformal_early_departure_guard":
        raise ValueError("unsupported commitment_uncertainty.method")
    miscoverage = float(declared["miscoverage"])
    if not 0 < miscoverage < 1:
        raise ValueError("commitment_uncertainty.miscoverage must lie in (0, 1)")
    calibration_split = str(declared["calibration_split"])
    if calibration_split not in config["splits"]:
        raise ValueError("commitment_uncertainty.calibration_split must name a declared split")
    if resolve_acn_target_mode(config) != "declared_commitment":
        raise ValueError("commitment uncertainty requires acn_data_protocol.target_mode='declared_commitment'")
    candidates = declared.get("candidate_modes", ["global"])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("commitment_uncertainty.candidate_modes must be an array")
    allowed_candidates = {
        "global",
        "duration_stratified",
        "adaptive",
        "multi_rate_envelope",
        "agaci_weighted",
    }
    if not candidates or not set(candidates).issubset(allowed_candidates):
        raise ValueError("commitment_uncertainty.candidate_modes is invalid")
    if "duration_stratified" in candidates:
        duration = declared.get("duration_stratification")
        if not isinstance(duration, Mapping):
            raise ValueError("duration_stratification is required for the duration-stratified candidate")
        minimum = duration.get("min_group_calibration_sessions")
        if minimum is None or isinstance(minimum, bool) or int(minimum) <= 0:
            raise ValueError("duration_stratification.min_group_calibration_sessions must be positive")
    if "adaptive" in candidates:
        adaptive = declared.get("adaptive_conformal")
        if not isinstance(adaptive, Mapping):
            raise ValueError("adaptive_conformal is required for the adaptive candidate")
        learning_rate = adaptive.get("learning_rate")
        if learning_rate is None or isinstance(learning_rate, bool) or float(learning_rate) <= 0:
            raise ValueError("adaptive_conformal.learning_rate must be positive")
        minimum = float(adaptive.get("min_miscoverage", 0.01))
        maximum = float(adaptive.get("max_miscoverage", 0.50))
        if not 0 < minimum <= miscoverage <= maximum < 1:
            raise ValueError(
                "adaptive_conformal bounds must satisfy 0 < min <= miscoverage <= max < 1"
            )
    if "multi_rate_envelope" in candidates:
        envelope = declared.get("multi_rate_envelope")
        if not isinstance(envelope, Mapping):
            raise ValueError("multi_rate_envelope is required for the multi-rate candidate")
        rates = envelope.get("learning_rates")
        if (
            not isinstance(rates, Sequence)
            or isinstance(rates, (str, bytes))
            or len(rates) < 2
            or len({float(value) for value in rates}) != len(rates)
            or any(float(value) <= 0 for value in rates)
        ):
            raise ValueError(
                "multi_rate_envelope.learning_rates must contain at least two distinct positive values"
            )
    if "agaci_weighted" in candidates:
        agaci = declared.get("agaci_weighted")
        if not isinstance(agaci, Mapping):
            raise ValueError("agaci_weighted is required for the AgACI-style candidate")
        rates = agaci.get("learning_rates")
        if (
            not isinstance(rates, Sequence)
            or isinstance(rates, (str, bytes))
            or len(rates) < 2
            or len({float(value) for value in rates}) != len(rates)
            or any(float(value) <= 0 for value in rates)
        ):
            raise ValueError(
                "agaci_weighted.learning_rates must contain at least two distinct positive values"
            )
        minimum = float(agaci.get("min_miscoverage", 0.01))
        maximum = float(agaci.get("max_miscoverage", 0.50))
        epsilon = float(agaci.get("epsilon", 0.001))
        if not 0 < minimum <= miscoverage <= maximum < 1:
            raise ValueError(
                "agaci_weighted bounds must satisfy 0 < min <= miscoverage <= max < 1"
            )
        if epsilon <= 0:
            raise ValueError("agaci_weighted.epsilon must be positive")
    return dict(declared)


def with_replay_window(
    config: Mapping[str, Any],
    split_name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    """Return a copy whose declared replay split is a validated sub-window.

    The raw source file stays unchanged. This permits matched day-level
    evaluation while preventing an accidental replay outside the configured,
    documented split. It is deliberately a copy so a loop over days cannot
    mutate the configuration used by the next run.
    """
    if split_name not in config["splits"]:
        raise ValueError(f"unknown split: {split_name}")
    declared = ReplayWindow.from_config(config["splits"][split_name])
    candidate = ReplayWindow.from_config([str(start), str(end)])
    if candidate.start < declared.start or candidate.end > declared.end:
        raise ValueError(
            "replay window must be inside the configured split: "
            f"{declared.start.isoformat()} to {declared.end.isoformat()}"
        )
    updated = deepcopy(dict(config))
    updated["splits"][split_name] = [candidate.start.isoformat(), candidate.end.isoformat()]
    return updated


def resolve_study_path(config_path: Path | str, configured_path: str) -> Path:
    """Resolve study paths from the project root, not the shell CWD.

    Historical configurations sit directly in ``configs/`` while versioned
    studies (for example ``configs/v3/``) sit below it.  Locate the first
    ancestor containing ``pyproject.toml`` so both layouts resolve the same
    ``data/...`` inputs without copying raw files.
    """
    config_dir = Path(config_path).resolve().parent
    root = next((candidate for candidate in (config_dir, *config_dir.parents) if (candidate / "pyproject.toml").is_file()), None)
    # Unit tests deliberately build a minimal temporary ``configs/`` tree
    # without a project manifest.  Preserve the historic one-level fallback
    # for that self-contained layout while production V3 configs use the
    # explicit manifest-root search above.
    if root is None:
        root = config_dir.parent
    resolved = (root / configured_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"study input does not exist: {resolved}")
    return resolved


def stations_from_config(config: Mapping[str, Any]) -> tuple[ChargingStation, ...]:
    stations = tuple(
        ChargingStation(
            str(item["station_id"]),
            int(item["bus"]),
            float(item["capacity_kw"]),
        )
        for item in config["synthetic_stations"]
    )
    if len({station.station_id for station in stations}) != len(stations):
        raise ValueError("synthetic station IDs must be unique")
    return stations


def deterministic_station_assignment(session_id: str, stations: Sequence[ChargingStation]) -> str:
    """Map a trace session to one synthetic station reproducibly across runs."""
    if not session_id or not stations:
        raise ValueError("session_id and stations are required")
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big") % len(stations)
    return stations[index].station_id


def prepare_acn_split(
    config: Mapping[str, Any],
    config_path: Path | str,
    split_name: str,
) -> PreparedSplit:
    """Convert a declared ACN raw file to discrete sessions for one split.

    We select sessions by *connection* time. A session that begins in the
    window may retain its true departure after the boundary, avoiding a false
    deadline created solely by the experiment cut-off. Sessions already active
    at the boundary are excluded and this left-censoring rule is reported in
    the manifest.
    """
    if split_name not in config["splits"]:
        raise ValueError(f"unknown split: {split_name}")
    raw_file = config["raw_files"]["acn"].get(split_name)
    if not raw_file:
        raise ValueError(f"no ACN raw file configured for split {split_name!r}")
    if isinstance(raw_file, str):
        configured_files = (raw_file,)
    elif isinstance(raw_file, Sequence) and not isinstance(raw_file, (str, bytes)):
        configured_files = tuple(str(value) for value in raw_file)
    else:
        raise ValueError(f"ACN raw input for split {split_name!r} must be a string or a list of strings")
    if not configured_files:
        raise ValueError(f"ACN raw input list for split {split_name!r} must not be empty")

    records: list[Mapping[str, Any]] = []
    for configured_file in configured_files:
        raw_path = resolve_study_path(config_path, configured_file)
        loaded = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"ACN raw file must contain a JSON list: {raw_path}")
        records.extend(loaded)

    window = ReplayWindow.from_config(config["splits"][split_name])
    selected = [record for record in records if window.includes_connection(record.get("connectionTime"))]
    step_minutes = int(config["time_step_minutes"])
    stations = stations_from_config(config)
    target_mode = resolve_acn_target_mode(config)
    sessions: list[EVSession] = []
    seen_ids: set[str] = set()

    for record_index, record in enumerate(selected):
        raw_id = str(record.get("sessionID", record.get("id", f"record-{record_index}")))
        station_id = deterministic_station_assignment(raw_id, stations)
        converted = acn_records_to_sessions(
            [record],
            station_id=station_id,
            origin=window.start,
            step_minutes=step_minutes,
            target_mode=target_mode,
        )
        if not converted:
            continue
        session = converted[0]
        session_id = session.ev_id
        suffix = 1
        while session_id in seen_ids:
            suffix += 1
            session_id = f"{session.ev_id}-{suffix}"
        seen_ids.add(session_id)
        if session_id != session.ev_id:
            session = EVSession(
                session_id,
                session.station_id,
                session.arrival_step,
                session.departure_step,
                session.requested_energy_kwh,
                session.max_power_kw,
                planning_departure_step=session.planning_departure_step,
                declared_departure_step=session.declared_departure_step,
            )
        sessions.append(session)

    sessions.sort(key=lambda item: (item.arrival_step, item.departure_step, item.ev_id))
    station_counts = {station.station_id: 0 for station in stations}
    for session in sessions:
        station_counts[session.station_id] += 1
    return PreparedSplit(
        split_name,
        window,
        len(records),
        len(selected),
        len(selected) - len(sessions),
        tuple(sessions),
        station_counts,
        target_mode,
    )


def make_trace_simulation(
    config: Mapping[str, Any], prepared_split: PreparedSplit
) -> ChargingSimulation:
    """Create a fresh grid and mutable replay simulation for one prepared split."""
    stations = stations_from_config(config)
    grid_settings = config.get("grid_sensitivity", {})
    if not isinstance(grid_settings, Mapping):
        raise ValueError("grid_sensitivity must be an object when supplied")
    allowed_grid_settings = {
        "background_load_scale",
        "min_voltage_pu",
        "max_line_loading_percent",
        "thermal_headroom",
    }
    metadata_grid_settings = {"note"}
    unknown_grid_settings = set(grid_settings) - allowed_grid_settings - metadata_grid_settings
    if unknown_grid_settings:
        raise ValueError(f"unknown grid_sensitivity keys: {sorted(unknown_grid_settings)}")
    grid = IEEE33Grid(
        {station.station_id: station.bus for station in stations},
        **{
            key: float(value)
            for key, value in grid_settings.items()
            if key in allowed_grid_settings
        },
    )
    return ChargingSimulation(
        stations,
        prepared_split.sessions,
        grid,
        step_hours=float(config["time_step_minutes"]) / 60.0,
    )


def load_pv_proxy(
    config: Mapping[str, Any],
    config_path: Path | str,
    *,
    dc_capacity_kw: float,
    ac_capacity_kw: float,
) -> pd.Series:
    """Return a sensitivity-scenario PV proxy from the configured NSRDB input."""
    configured = config["raw_files"]["nsrdb"]
    if isinstance(configured, str):
        configured_files = (configured,)
    elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        configured_files = tuple(str(item) for item in configured)
    else:
        raise ValueError("NSRDB input must be a path or a non-empty list of paths")
    if not configured_files:
        raise ValueError("NSRDB input list must not be empty")
    weather = pd.concat(
        [read_nsrdb_csv(resolve_study_path(config_path, item)) for item in configured_files]
    ).sort_index()
    if not weather.index.is_unique:
        raise ValueError("configured NSRDB inputs contain duplicate timestamps")
    return pvwatts_ghi_proxy_kw(
        weather,
        dc_capacity_kw=dc_capacity_kw,
        ac_capacity_kw=ac_capacity_kw,
    )


def sha256_file(path: Path | str) -> str:
    """Hash a raw input so an experiment can later prove its exact source file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_data_manifest(
    config: Mapping[str, Any],
    config_path: Path | str,
    prepared_splits: Mapping[str, PreparedSplit],
) -> dict[str, Any]:
    """Build JSON-safe provenance without exposing credentials or raw records."""
    raw_files = config["raw_files"]
    acn_files: dict[str, tuple[Path, ...]] = {}
    for name, value in raw_files["acn"].items():
        if isinstance(value, str):
            configured_files = (value,)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            configured_files = tuple(str(item) for item in value)
        else:
            raise ValueError(f"ACN raw input for split {name!r} must be a string or a list of strings")
        acn_files[name] = tuple(
            resolve_study_path(config_path, configured_file)
            for configured_file in configured_files
        )
    configured_nsrdb = raw_files["nsrdb"]
    if isinstance(configured_nsrdb, str):
        configured_nsrdb_files = (configured_nsrdb,)
    elif isinstance(configured_nsrdb, Sequence) and not isinstance(configured_nsrdb, (str, bytes)):
        configured_nsrdb_files = tuple(str(item) for item in configured_nsrdb)
    else:
        raise ValueError("NSRDB input must be a path or a non-empty list of paths")
    if not configured_nsrdb_files:
        raise ValueError("NSRDB input list must not be empty")
    nsrdb_files = tuple(
        resolve_study_path(config_path, configured_file)
        for configured_file in configured_nsrdb_files
    )
    return {
        "study_id": config["study_id"],
        "time_step_minutes": config["time_step_minutes"],
        "selection_rule": "connection_time in [split_start, split_end); keep true departure; exclude boundary-active sessions",
        "acn_data_protocol": dict(config.get("acn_data_protocol", {"target_mode": "observed_delivery"})),
        "commitment_uncertainty": resolve_commitment_uncertainty(config),
        "station_assignment": config["charging_data"]["station_assignment"],
        "raw_input_sha256": {
            **{
                f"acn_{name}": (
                    sha256_file(paths[0])
                    if len(paths) == 1
                    else [sha256_file(path) for path in paths]
                )
                for name, paths in acn_files.items()
            },
            "nsrdb": (
                sha256_file(nsrdb_files[0])
                if len(nsrdb_files) == 1
                else [sha256_file(path) for path in nsrdb_files]
            ),
        },
        "splits": {
            name: {
                "start": prepared.window.start.isoformat(),
                "end": prepared.window.end.isoformat(),
                "source_records": prepared.source_records,
                "selected_records": prepared.selected_records,
                "dropped_records": prepared.dropped_records,
                "usable_sessions": len(prepared.sessions),
                "requested_energy_kwh": prepared.requested_energy_kwh,
                "station_session_counts": dict(prepared.station_session_counts),
                "target_mode": prepared.target_mode,
            }
            for name, prepared in prepared_splits.items()
        },
        "synthetic_stations": [asdict(station) for station in stations_from_config(config)],
        "solar_data": config.get("solar_data", {}),
        "grid_sensitivity": config.get("grid_sensitivity", {}),
    }
