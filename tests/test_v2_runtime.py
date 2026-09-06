from __future__ import annotations

from types import SimpleNamespace

from fairflex.domain import EVSession
from fairflex.v2.runtime import ProfiledACRepairedMPCPolicy


class _FakeDelegate:
    def __init__(self) -> None:
        self.controller = SimpleNamespace(solve_timing_history=[])
        self.repair_history = []
        self.economic_tiebreak_failures = 0
        self.last_repair = None

    def __call__(self, station_evs, station_capacities_kw, step_hours, current_step):
        del station_capacities_kw, step_hours
        self.controller.solve_timing_history.append(
            {
                "stage_one_seconds": 0.01,
                "stage_two_seconds": 0.02,
                "stage_three_seconds": 0.03,
                "total_seconds": 0.06,
            }
        )
        repair = SimpleNamespace(
            attempts=0,
            final_grid=SimpleNamespace(safe=True),
            applied_powers_kw={"station-a": 1.0},
        )
        self.repair_history.append(repair)
        self.last_repair = repair
        return {ev.ev_id: 1.0 for evs in station_evs.values() for ev in evs}


def test_v2_runtime_wrapper_records_stage_times_without_changing_action() -> None:
    session = EVSession("ev", "station-a", 0, 4, 2.0, 7.2)
    wrapper = ProfiledACRepairedMPCPolicy(_FakeDelegate(), soft_latency_seconds=10.0)

    result = wrapper({"station-a": [session]}, {"station-a": 7.2}, 0.25, 0)

    assert result == {"ev": 1.0}
    assert len(wrapper.decision_runtimes) == 1
    record = wrapper.decision_runtimes[0]
    assert record.active_evs == 1
    assert record.mpc_solves == 1
    assert record.mpc_stage_one_seconds == 0.01
    assert not record.soft_latency_exceeded
    assert wrapper.last_applied_grid_validation.safe
    assert wrapper.runtime_summary()["decisions"] == 1
    assert wrapper.runtime_summary()["controller_calls"] == 1


def test_v2_runtime_wrapper_skips_an_idle_step_without_calling_delegate() -> None:
    delegate = _FakeDelegate()
    wrapper = ProfiledACRepairedMPCPolicy(delegate, soft_latency_seconds=10.0)

    result = wrapper({"station-a": []}, {"station-a": 7.2}, 0.25, 0)

    assert result == {}
    assert delegate.controller.solve_timing_history == []
    assert wrapper.decision_runtimes[0].idle_step_skipped
    assert wrapper.runtime_summary()["controller_calls"] == 0
    assert wrapper.runtime_summary()["idle_steps_skipped"] == 1
