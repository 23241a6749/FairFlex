"""Allowlisted, credential-free scenarios for the demonstrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from ..domain import ChargingStation, EVSession
from ..examples import make_deadline_stress_simulation, make_demo_simulation
from ..grid import IEEE33Grid
from ..simulation import ChargingSimulation


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    label: str
    subtitle: str
    kind: str
    estimated_runtime_seconds: float
    source: str
    disclosure: str
    limitations: tuple[str, ...]
    factory: Callable[[], ChargingSimulation]

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("factory")
        value["limitations"] = list(self.limitations)
        return value


_SCENARIOS = (
    ScenarioDefinition(
        scenario_id="deadline-stress",
        label="Urgent-deadline teaching scenario",
        subtitle="Two EVs share 7.2 kW. One must charge immediately.",
        kind="controlled_teaching_demo",
        estimated_runtime_seconds=3.0,
        source="FairFlex deterministic scenario fixture",
        disclosure="Controlled teaching scenario — not historical field evidence and not live charger operation.",
        limitations=(
            "The scenario teaches deadline-aware allocation; it does not estimate real site performance.",
            "The grid model is the FairFlex IEEE-33 sensitivity model, not a Caltech feeder model.",
            "The app does not control a physical charger.",
        ),
        factory=make_deadline_stress_simulation,
    ),
    ScenarioDefinition(
        scenario_id="campus-flow",
        label="Multi-station campus flow",
        subtitle="Five EV sessions across three stations with changing arrivals.",
        kind="controlled_teaching_demo",
        estimated_runtime_seconds=8.0,
        source="FairFlex deterministic scenario fixture",
        disclosure="Controlled teaching scenario — not historical field evidence and not live charger operation.",
        limitations=(
            "This fixture illustrates the simulator workflow and uses controlled arrival/departure data.",
            "It is not a measured Caltech, JPL, or physical feeder result.",
            "The app does not control a physical charger.",
        ),
        factory=make_demo_simulation,
    ),
)


class ScenarioCatalog:
    """Expose only pre-approved factories; never accept a browser path."""

    def __init__(self) -> None:
        self._by_id = {scenario.scenario_id: scenario for scenario in _SCENARIOS}

    def list_public(self) -> list[dict[str, Any]]:
        return [scenario.public() for scenario in _SCENARIOS]

    def get(self, scenario_id: str) -> ScenarioDefinition:
        try:
            return self._by_id[scenario_id]
        except KeyError as exc:
            raise KeyError(f"unknown approved scenario: {scenario_id}") from exc

    def snapshot(self, scenario_id: str) -> dict[str, Any]:
        scenario = self.get(scenario_id)
        simulation = scenario.factory()
        return {
            **scenario.public(),
            "step_hours": simulation.step_hours,
            "grid_model": {
                "name": "IEEE-33 sensitivity model",
                "background_load_scale": simulation.grid.background_load_scale,
                "min_voltage_pu": simulation.grid.min_voltage_pu,
                "max_line_loading_percent": simulation.grid.max_line_loading_percent,
                "thermal_headroom": simulation.grid.thermal_headroom,
            },
            "stations": [
                {
                    "station_id": station.station_id,
                    "bus": station.bus,
                    "capacity_kw": station.capacity_kw,
                }
                for station in simulation.stations.values()
            ],
            "sessions": [
                {
                    "ev_id": ev.ev_id,
                    "station_id": ev.station_id,
                    "arrival_step": ev.arrival_step,
                    "departure_step": ev.departure_step,
                    "planning_deadline_step": ev.planning_deadline_step,
                    "planning_departure_step": ev.planning_departure_step,
                    "declared_departure_step": ev.declared_departure_step,
                    "requested_energy_kwh": ev.requested_energy_kwh,
                    "max_power_kw": ev.max_power_kw,
                    "initial_delivered_energy_kwh": ev.delivered_energy_kwh,
                }
                for ev in simulation.evs.values()
            ],
            "simulation_statement": "Offline controlled teaching simulation — not historical field evidence and not live charger operation.",
        }

    @staticmethod
    def execution_input(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return the canonical, engine-relevant portion of a saved snapshot.

        The user interface text is deliberately excluded from this hash.  Only
        values that can alter the simulated allocation or AC feasibility result
        are included, in a deterministic order.
        """

        grid = snapshot.get("grid_model")
        if not isinstance(grid, dict):
            raise ValueError("saved scenario is missing its grid model")
        stations = snapshot.get("stations")
        sessions = snapshot.get("sessions")
        if not isinstance(stations, list) or not isinstance(sessions, list):
            raise ValueError("saved scenario is missing stations or EV sessions")
        return {
            "schema_version": 1,
            "step_hours": snapshot["step_hours"],
            "grid_model": {
                "name": grid["name"],
                "background_load_scale": grid["background_load_scale"],
                "min_voltage_pu": grid["min_voltage_pu"],
                "max_line_loading_percent": grid["max_line_loading_percent"],
                "thermal_headroom": grid["thermal_headroom"],
            },
            "stations": [
                {
                    "station_id": station["station_id"],
                    "bus": station["bus"],
                    "capacity_kw": station["capacity_kw"],
                }
                for station in sorted(stations, key=lambda item: str(item["station_id"]))
            ],
            "sessions": [
                {
                    "ev_id": session["ev_id"],
                    "station_id": session["station_id"],
                    "arrival_step": session["arrival_step"],
                    "departure_step": session["departure_step"],
                    "planning_departure_step": session.get("planning_departure_step"),
                    "declared_departure_step": session.get("declared_departure_step"),
                    "requested_energy_kwh": session["requested_energy_kwh"],
                    "max_power_kw": session["max_power_kw"],
                    "initial_delivered_energy_kwh": session.get("initial_delivered_energy_kwh", 0.0),
                }
                for session in sorted(sessions, key=lambda item: str(item["ev_id"]))
            ],
        }

    @classmethod
    def simulation_from_snapshot(cls, snapshot: dict[str, Any]) -> ChargingSimulation:
        """Rebuild a fresh simulation only from a recorded immutable input.

        This prevents later edits to a scenario factory from silently changing
        what an already-audited run executes.  Browser callers never supply a
        snapshot: it is created by this allowlisted catalog and stored locally.
        """

        spec = cls.execution_input(snapshot)
        grid_spec = spec["grid_model"]
        if grid_spec["name"] != "IEEE-33 sensitivity model":
            raise ValueError("unsupported saved grid model")
        stations = [
            ChargingStation(
                station_id=str(station["station_id"]),
                bus=int(station["bus"]),
                capacity_kw=float(station["capacity_kw"]),
            )
            for station in spec["stations"]
        ]
        sessions = [
            EVSession(
                ev_id=str(session["ev_id"]),
                station_id=str(session["station_id"]),
                arrival_step=int(session["arrival_step"]),
                departure_step=int(session["departure_step"]),
                requested_energy_kwh=float(session["requested_energy_kwh"]),
                max_power_kw=float(session["max_power_kw"]),
                delivered_energy_kwh=float(session["initial_delivered_energy_kwh"]),
                planning_departure_step=(
                    int(session["planning_departure_step"])
                    if session["planning_departure_step"] is not None
                    else None
                ),
                declared_departure_step=(
                    int(session["declared_departure_step"])
                    if session["declared_departure_step"] is not None
                    else None
                ),
            )
            for session in spec["sessions"]
        ]
        grid = IEEE33Grid(
            {station.station_id: station.bus for station in stations},
            background_load_scale=float(grid_spec["background_load_scale"]),
            min_voltage_pu=float(grid_spec["min_voltage_pu"]),
            max_line_loading_percent=float(grid_spec["max_line_loading_percent"]),
            thermal_headroom=float(grid_spec["thermal_headroom"]),
        )
        return ChargingSimulation(stations, sessions, grid, step_hours=float(spec["step_hours"]))
