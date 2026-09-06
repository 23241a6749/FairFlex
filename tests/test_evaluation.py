import pytest

from fairflex.baselines import equal_share, first_come_first_served, uncontrolled
from fairflex.domain import ChargingStation, EVSession
from fairflex.evaluation import ExperimentRunner, jain_index
from fairflex.examples import make_deadline_stress_simulation, make_demo_simulation
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.grid import IEEE33Grid
from fairflex.safety import ACRepairedMPCPolicy
from fairflex.simulation import ChargingSimulation


def test_jain_index_has_expected_limits():
    assert jain_index([1.0, 1.0, 1.0]) == 1.0
    assert jain_index([1.0, 0.0]) == 0.5


def test_experiment_runner_constructs_a_fresh_scenario_per_policy():
    runs = ExperimentRunner(make_demo_simulation).run(
        {"fcfs": lambda: first_come_first_served, "equal": lambda: equal_share}
    )
    table = ExperimentRunner.metrics_table(runs)

    assert set(runs) == {"fcfs", "equal"}
    assert (table["sessions"] == 5).all()
    assert (table["unsafe_steps"] == 0).all()
    assert (table["runtime_seconds"] >= 0).all()
    assert table["jain_service_index"].between(0, 1).all()
    assert table["energy_service_ratio"].between(0, 1).all()
    assert table["completion_rate"].between(0, 1).all()
    assert table["individually_feasible_request_rate"].between(0, 1).all()
    assert (table["requested_energy_kwh"] > 0).all()
    assert (table["min_voltage_pu"] > 0).all()
    assert (table["max_line_loading_percent"] >= 0).all()


def test_experiment_runner_can_build_a_policy_that_uses_its_own_grid():
    runs = ExperimentRunner(make_demo_simulation).run_with_simulation_policy(
        {
            "ac_repaired": lambda simulation: ACRepairedMPCPolicy(
                CentralizedFairMPC(), simulation.grid, horizon_steps=4
            )
        }
    )

    assert runs["ac_repaired"].metrics.unsafe_steps == 0


def test_simulation_clones_mutable_sessions_when_a_factory_reuses_trace_objects():
    shared_ev = EVSession("shared", "north", 0, 1, 1.8, 7.2)
    station = ChargingStation("north", bus=6, capacity_kw=7.2)

    def factory():
        return ChargingSimulation([station], [shared_ev], IEEE33Grid({"north": 6}))

    runs = ExperimentRunner(factory).run(
        {"first": lambda: uncontrolled, "second": lambda: uncontrolled}
    )

    assert runs["first"].metrics.delivered_energy_kwh == pytest.approx(1.8)
    assert runs["second"].metrics.delivered_energy_kwh == pytest.approx(1.8)
    assert shared_ev.delivered_energy_kwh == 0.0


def test_deadline_stress_scenario_distinguishes_deadline_aware_mpc_from_fcfs():
    runner = ExperimentRunner(make_deadline_stress_simulation)
    runs = runner.run_with_simulation_policy(
        {
            "fcfs": lambda _simulation: first_come_first_served,
            "fair_mpc_ac": lambda simulation: ACRepairedMPCPolicy(
                CentralizedFairMPC(), simulation.grid, horizon_steps=4
            ),
        }
    )

    assert runs["fair_mpc_ac"].metrics.worst_service_ratio == pytest.approx(1.0, abs=1e-4)
    assert runs["fcfs"].metrics.worst_service_ratio == 0.0
