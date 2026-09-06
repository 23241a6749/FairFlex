"""Trace and solar-data adapters with explicit provenance assumptions."""

from __future__ import annotations

import os
from csv import reader as csv_reader
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pvlib.inverter import pvwatts as pvwatts_inverter
from pvlib.pvsystem import pvwatts_dc

from .domain import EVSession


# Load a developer's project-local keys when present. The file is ignored by
# Git; deployed environments can supply the same variables conventionally.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class DatasetProvenance:
    """Facts which must accompany every result generated from public data."""

    source_name: str
    source_url: str
    access_date: str
    transformations: tuple[str, ...]


class ACNDataClient:
    """Minimal authenticated client for the official ACN-Data sessions API.

    API tokens are intentionally read from ``ACN_API_TOKEN`` or passed by the
    caller; they are never written to source code, datasets, or experiment logs.
    The API's non-timeseries endpoint is the default because it is sufficient
    for arrival/departure/energy trace replay and avoids unnecessary downloads.
    """

    def __init__(self, api_token: str | None = None, *, timeout_seconds: float = 30.0) -> None:
        self.api_token = api_token or os.environ.get("ACN_API_TOKEN")
        if not self.api_token:
            raise ValueError("provide an ACN token or set ACN_API_TOKEN")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://ev.caltech.edu/api/v1/"

    def fetch_sessions(
        self,
        site_id: str,
        *,
        where: str | None = None,
        include_timeseries: bool = False,
        max_pages: int = 100,
    ) -> list[Mapping[str, Any]]:
        """Retrieve one finite API query, following the API's pagination links."""
        if not site_id or max_pages <= 0:
            raise ValueError("site_id must be non-empty and max_pages positive")
        endpoint = f"sessions/{quote(site_id, safe='')}"
        if include_timeseries:
            endpoint += "/ts"
        url = urljoin(self.base_url, endpoint)
        params = {"where": where} if where else None
        sessions: list[Mapping[str, Any]] = []

        for _ in range(max_pages):
            response = requests.get(
                url,
                params=params,
                auth=(self.api_token, ""),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            records = payload.get("_items", payload.get("items", []))
            if not isinstance(records, list):
                raise ValueError("unexpected ACN-Data response: items is not a list")
            sessions.extend(records)
            next_link = payload.get("_links", {}).get("next", {}).get("href")
            if not next_link:
                return sessions
            url = urljoin(self.base_url, next_link)
            params = None  # Pagination HATEOAS links contain their own query.
        raise RuntimeError("ACN-Data pagination exceeded max_pages")

    def ping(self, site_id: str = "caltech") -> int:
        """Verify authentication with one page, returning only its record count."""
        if not site_id:
            raise ValueError("site_id must be non-empty")
        response = requests.get(
            urljoin(self.base_url, f"sessions/{quote(site_id, safe='')}"),
            auth=(self.api_token, ""),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("_items", payload.get("items", []))
        if not isinstance(records, list):
            raise ValueError("unexpected ACN-Data response: items is not a list")
        return len(records)


class NSRDBDataClient:
    """Authenticated discovery client for the current NSRDB data-query API."""

    def __init__(self, api_key: str | None = None, *, timeout_seconds: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("NSRDB_API_KEY")
        if not self.api_key:
            raise ValueError("provide an NSRDB key or set NSRDB_API_KEY")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.query_url = "https://developer.nlr.gov/api/solar/nsrdb_data_query.json"

    def discover(self, latitude: float, longitude: float) -> Mapping[str, Any]:
        """Find NSRDB datasets available nearest a site without downloading data."""
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude/longitude are out of range")
        response = requests.get(
            self.query_url,
            params={"api_key": self.api_key, "lat": latitude, "lon": longitude},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"NSRDB rejected the query: {payload['errors']}")
        if not isinstance(payload.get("outputs"), list):
            raise ValueError("unexpected NSRDB response: outputs is not a list")
        return payload

    def download_url(
        self,
        *,
        dataset: str,
        year: int,
        interval_minutes: int,
        latitude: float,
        longitude: float,
        email: str | None = None,
    ) -> str:
        """Build the documented NSRDB CSV request URL for a fixed site/year."""
        if not dataset or year < 1990 or interval_minutes <= 0:
            raise ValueError("dataset, a plausible year, and a positive interval are required")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude/longitude are out of range")
        contact_email = email or os.environ.get("NSRDB_EMAIL")
        if not contact_email:
            raise ValueError("provide the registered NSRDB email or set NSRDB_EMAIL")
        endpoint = f"https://developer.nlr.gov/api/nsrdb/v2/solar/{quote(dataset, safe='-')}-download.csv"
        return f"{endpoint}?{urlencode({
            'names': year,
            'wkt': f'POINT({longitude} {latitude})',
            'interval': interval_minutes,
            'api_key': self.api_key,
            'email': contact_email,
        })}"

    def download_csv(
        self,
        destination: Path | str,
        *,
        dataset: str,
        year: int,
        interval_minutes: int,
        latitude: float,
        longitude: float,
        email: str | None = None,
    ) -> Path:
        """Download one explicit NSRDB request to local, ignored raw storage."""
        url = self.download_url(
            dataset=dataset,
            year=year,
            interval_minutes=interval_minutes,
            latitude=latitude,
            longitude=longitude,
            email=email,
        )
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return target


def _timestamp(value: Any, field_name: str) -> pd.Timestamp:
    if isinstance(value, Mapping):
        value = value.get("$date", value.get("date"))
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"ACN record has an invalid {field_name}")
    return parsed


def acn_records_to_sessions(
    records: Iterable[Mapping[str, Any]],
    *,
    station_id: str,
    origin: pd.Timestamp | str,
    step_minutes: int = 15,
    fallback_max_power_kw: float = 6.6,
    target_mode: str = "observed_delivery",
) -> list[EVSession]:
    """Convert ACN session records to discrete simulation sessions.

    ``target_mode='observed_delivery'`` preserves the legacy trace-replay
    behavior: historical delivered energy is used as an observed-energy target.
    It is useful for backward-compatible diagnostics, but it is *not* a user
    request study.

    ``target_mode='declared_commitment'`` uses the first valid user-submitted
    ``kWhRequested`` and ``requestedDeparture`` values as the requested energy
    and controller-visible deadline. The observed ``disconnectTime`` remains
    the physical end of availability in the simulator. Records without both
    declared fields are excluded. This is the causal protocol used for the
    commitment-aware FairFlex study.

    Mapping a real site to a synthetic feeder station remains a scenario
    assumption, not a measured grid connection.
    """
    if not station_id or step_minutes <= 0 or fallback_max_power_kw <= 0:
        raise ValueError("station_id, step_minutes, and fallback_max_power_kw must be positive")
    if target_mode not in {"observed_delivery", "declared_commitment"}:
        raise ValueError("target_mode must be 'observed_delivery' or 'declared_commitment'")
    origin_time = _timestamp(origin, "origin")
    step_seconds = step_minutes * 60
    sessions: list[EVSession] = []
    seen_ids: set[str] = set()

    for row_index, record in enumerate(records):
        connection = _timestamp(record.get("connectionTime"), "connectionTime")
        disconnection = _timestamp(record.get("disconnectTime"), "disconnectTime")
        planning_departure_step: int | None = None
        if target_mode == "declared_commitment":
            commitment = _initial_declared_commitment(record, connection)
            if commitment is None:
                continue
            energy, declared_departure = commitment
            if declared_departure <= connection:
                continue
        else:
            energy = float(record.get("kWhDelivered", record.get("requestedEnergy", 0.0)) or 0.0)

        if disconnection <= connection or energy <= 0:
            continue
        arrival = max(0, int(np.floor((connection - origin_time).total_seconds() / step_seconds)))
        departure = max(
            arrival + 1,
            int(np.ceil((disconnection - origin_time).total_seconds() / step_seconds)),
        )
        if target_mode == "declared_commitment":
            planning_departure_step = max(
                arrival + 1,
                int(np.ceil((declared_departure - origin_time).total_seconds() / step_seconds)),
            )
        session_id = str(record.get("sessionID", record.get("id", f"session-{row_index}")))
        while session_id in seen_ids:
            session_id = f"{session_id}-{row_index}"
        seen_ids.add(session_id)
        max_power = float(record.get("maxPowerKW", fallback_max_power_kw) or fallback_max_power_kw)
        sessions.append(
            EVSession(
                session_id,
                station_id,
                arrival,
                departure,
                energy,
                max_power,
                planning_departure_step=planning_departure_step,
                declared_departure_step=planning_departure_step,
            )
        )
    return sessions


def _initial_declared_commitment(
    record: Mapping[str, Any], connection: pd.Timestamp
) -> tuple[float, pd.Timestamp] | None:
    """Extract the earliest complete user commitment from an ACN record.

    ACN stores a list because users may later modify their input.  A controller
    cannot use a later amendment at plug-in, so this protocol deliberately uses
    the earliest complete entry.  It is a transparent approximation at the
    15-minute control resolution; later work can model amendment events
    explicitly without changing the causal data contract.
    """
    raw_inputs = record.get("userInputs")
    if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
        return None

    candidates: list[tuple[pd.Timestamp, float, pd.Timestamp]] = []
    for item in raw_inputs:
        if not isinstance(item, Mapping):
            continue
        try:
            energy = float(item.get("kWhRequested", 0.0) or 0.0)
            departure = _timestamp(item.get("requestedDeparture"), "requestedDeparture")
            modified = _timestamp(item.get("modifiedAt", connection), "modifiedAt")
        except (TypeError, ValueError):
            continue
        if energy > 0 and departure > connection:
            candidates.append((modified, energy, departure))

    if not candidates:
        return None
    _, energy, departure = min(candidates, key=lambda candidate: candidate[0])
    return energy, departure


def read_nsrdb_csv(path: Path | str) -> pd.DataFrame:
    """Read an NSRDB CSV despite its variable number of metadata rows.

    The returned frame has a UTC timestamp index and canonical ``ghi_wm2`` and
    ``air_temperature_c`` columns. This routine only parses solar-resource
    observations; it never presents GHI as a future forecast feature.
    """
    csv_path = Path(path)
    # NSRDB files contain many meteorological columns, while this project uses
    # only timestamps, GHI and (when present) temperature. Scanning the small
    # preamble and selecting those columns prevents unnecessary parsing and
    # avoids avoidable memory pressure on student laptops.
    lines: list[str] = []
    header_index: int | None = None
    header_columns: list[str] | None = None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, line in enumerate(handle):
            lines.append(line.rstrip("\r\n"))
            parsed = next(csv_reader([line]), [])
            normalized = {str(column).strip().lower() for column in parsed}
            if {"year", "month", "day"}.issubset(normalized):
                header_index = index
                header_columns = [str(column) for column in parsed]
                break
    if header_index is None or header_columns is None:
        raise ValueError("could not find an NSRDB Year/Month/Day header")
    columns = {str(column).strip().lower(): column for column in header_columns}
    required = ("year", "month", "day", "hour", "ghi")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"NSRDB CSV is missing columns: {missing}")
    used_names = set(required) | {"minute", "temperature"}
    selected_columns = [column for name, column in columns.items() if name in used_names]
    numeric_dtypes: dict[str, str] = {
        columns["year"]: "int16",
        columns["month"]: "int8",
        columns["day"]: "int8",
        columns["hour"]: "int8",
        columns["ghi"]: "float32",
    }
    if "minute" in columns:
        numeric_dtypes[columns["minute"]] = "int8"
    if "temperature" in columns:
        numeric_dtypes[columns["temperature"]] = "float32"
    raw = pd.read_csv(
        csv_path,
        skiprows=header_index,
        usecols=selected_columns,
        dtype=numeric_dtypes,
        # The C parser and explicit compact dtypes avoid constructing a full
        # object-valued matrix. That is material for a year of 15-minute NSRDB
        # observations on memory-constrained student machines.
        engine="c",
    )
    minute = raw[columns["minute"]] if "minute" in columns else 0
    metadata: dict[str, str] = {}
    if header_index >= 2:
        metadata_keys = next(csv_reader([lines[0]]), [])
        metadata_values = next(csv_reader([lines[1]]), [])
        metadata = {
            key.strip().lower(): value.strip()
            for key, value in zip(metadata_keys, metadata_values, strict=False)
        }
    try:
        time_zone_offset_hours = float(metadata.get("time zone", "0"))
    except ValueError as error:
        raise ValueError("NSRDB metadata has a non-numeric Time Zone") from error
    if not -14 <= time_zone_offset_hours <= 14:
        raise ValueError("NSRDB metadata Time Zone is out of range")

    naive_timestamps = pd.to_datetime(
        {
            "year": raw[columns["year"]],
            "month": raw[columns["month"]],
            "day": raw[columns["day"]],
            "hour": raw[columns["hour"]],
            "minute": minute,
        },
        errors="coerce",
    )
    if naive_timestamps.isna().any():
        raise ValueError("NSRDB CSV contains invalid timestamp values")
    source_timezone = timezone(timedelta(hours=time_zone_offset_hours))
    timestamps = naive_timestamps.dt.tz_localize(source_timezone).dt.tz_convert(UTC)
    temperature = raw[columns["temperature"]] if "temperature" in columns else 25.0
    return pd.DataFrame(
        {
            # Convert to arrays before assigning the DatetimeIndex. Retaining
            # the CSV RangeIndex would make pandas align labels and silently
            # fill this frame with NaNs.
            "ghi_wm2": pd.to_numeric(raw[columns["ghi"]], errors="raise").to_numpy(),
            "air_temperature_c": np.asarray(
                pd.to_numeric(temperature, errors="raise"), dtype=float
            ),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    ).sort_index()


def pvwatts_ghi_proxy_kw(
    weather: pd.DataFrame,
    *,
    dc_capacity_kw: float,
    ac_capacity_kw: float,
    gamma_pdc: float = -0.004,
) -> pd.Series:
    """Create a documented PVWatts-based *scenario proxy* from GHI.

    GHI is not plane-of-array irradiance; without tilt, azimuth, and measured
    plant information it cannot reconstruct real plant output. This function is
    intentionally named a proxy and is suitable for controlled sensitivity
    experiments only. Use metered PV output when evaluating forecast accuracy.
    """
    if dc_capacity_kw <= 0 or ac_capacity_kw <= 0:
        raise ValueError("PV capacities must be positive")
    if not {"ghi_wm2", "air_temperature_c"}.issubset(weather.columns):
        raise ValueError("weather requires ghi_wm2 and air_temperature_c columns")
    effective_irradiance = weather["ghi_wm2"].clip(lower=0.0)
    pdc_w = pvwatts_dc(
        effective_irradiance,
        weather["air_temperature_c"],
        dc_capacity_kw * 1_000.0,
        gamma_pdc,
    )
    inverter_dc_limit_w = ac_capacity_kw * 1_000.0 / 0.96
    # The inverter efficiency equation divides by DC power and is undefined at
    # night. Physically, zero irradiance means zero AC output, not a missing
    # observation that could propagate into the forecasting dataset.
    pac_kw = np.nan_to_num(
        pvwatts_inverter(pdc_w, inverter_dc_limit_w) / 1_000.0,
        nan=0.0,
        posinf=ac_capacity_kw,
        neginf=0.0,
    )
    pac_kw = np.clip(pac_kw, 0.0, ac_capacity_kw)
    return pd.Series(pac_kw, index=weather.index, name="pv_proxy_kw")


def lagged_pv_features(
    pv_kw: pd.Series,
    *,
    lag_steps: Sequence[int] = (1, 4, 96),
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Make calendar-plus-past-power features without future-information leakage."""
    if not isinstance(pv_kw.index, pd.DatetimeIndex):
        raise ValueError("pv_kw must use a DatetimeIndex")
    if not lag_steps or any(lag <= 0 for lag in lag_steps):
        raise ValueError("lag_steps must contain positive integers")
    features = pd.DataFrame(index=pv_kw.index)
    for lag in lag_steps:
        features[f"lag_{lag}"] = pv_kw.shift(lag)
    hour = pv_kw.index.hour + pv_kw.index.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    joined = features.assign(target=pv_kw).dropna()
    return (
        joined.drop(columns="target").to_numpy(dtype=float),
        joined["target"].to_numpy(dtype=float),
        joined.index,
    )
