"""Pandapower-backed feeder model and AC-grid validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandapower as pp
import pandapower.networks as pn


@dataclass(frozen=True)
class GridValidation:
    converged: bool
    min_voltage_pu: float | None
    max_line_loading_percent: float | None
    violations: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return self.converged and not self.violations


class IEEE33Grid:
    """A modified IEEE 33-bus feeder with controllable station loads.

    The stock case has a 0.913 p.u. voltage minimum at full background demand,
    already below our 0.95 p.u. safety limit. We therefore use a documented
    50% background-demand operating point. Thermal ratings are set to 30%
    headroom above the resulting base currents because the source case omits
    usable ampacity limits (it uses placeholder values of 99,999 kA).
    """

    def __init__(
        self,
        station_buses: Mapping[str, int],
        *,
        background_load_scale: float = 0.5,
        min_voltage_pu: float = 0.95,
        max_line_loading_percent: float = 100.0,
        thermal_headroom: float = 1.3,
    ) -> None:
        if not 0 < background_load_scale <= 1:
            raise ValueError("background_load_scale must be in (0, 1]")
        if not 0 < min_voltage_pu <= 1:
            raise ValueError("min_voltage_pu must be in (0, 1]")
        if thermal_headroom <= 1:
            raise ValueError("thermal_headroom must be greater than one")

        # Preserve every parameter that changes the electrical replay.  The
        # demonstrator serialises these values into its immutable run spec so
        # an audit hash identifies the *executed* feeder configuration rather
        # than only the friendly scenario description.
        self.background_load_scale = background_load_scale
        self.thermal_headroom = thermal_headroom
        self.net = pn.case33bw()
        self.net.load[["p_mw", "q_mvar"]] *= background_load_scale
        self.min_voltage_pu = min_voltage_pu
        self.max_line_loading_percent = max_line_loading_percent
        self.station_load_indices: dict[str, int] = {}

        pp.runpp(self.net, numba=False)
        self.net.line["max_i_ka"] = np.maximum(
            self.net.res_line.i_ka.to_numpy() * thermal_headroom, 0.02
        )

        for station_id, bus in station_buses.items():
            if bus not in self.net.bus.index:
                raise ValueError(f"station {station_id!r} references unknown bus {bus}")
            self.station_load_indices[station_id] = pp.create_load(
                self.net,
                bus=bus,
                p_mw=0.0,
                q_mvar=0.0,
                name=f"station:{station_id}",
            )

        base_result = self.validate({})
        if not base_result.safe:
            raise RuntimeError(
                "The chosen base feeder state is unsafe; adjust the scenario before scheduling."
            )

    def validate(self, station_powers_kw: Mapping[str, float]) -> GridValidation:
        """Apply aggregate station import powers and run an AC power flow."""
        unknown_stations = set(station_powers_kw) - set(self.station_load_indices)
        if unknown_stations:
            raise ValueError(f"unknown station powers: {sorted(unknown_stations)}")

        for station_id, load_index in self.station_load_indices.items():
            power_kw = station_powers_kw.get(station_id, 0.0)
            if power_kw < 0:
                raise ValueError("G2V-only experiments do not allow negative station power")
            self.net.load.at[load_index, "p_mw"] = power_kw / 1_000.0

        try:
            pp.runpp(self.net, numba=False)
        except pp.LoadflowNotConverged:
            return GridValidation(False, None, None, ("power_flow_not_converged",))

        min_voltage = float(self.net.res_bus.vm_pu.min())
        max_loading = float(self.net.res_line.loading_percent.max())
        violations: list[str] = []
        if min_voltage < self.min_voltage_pu:
            violations.append("undervoltage")
        if max_loading > self.max_line_loading_percent:
            violations.append("line_overload")
        return GridValidation(True, min_voltage, max_loading, tuple(violations))
