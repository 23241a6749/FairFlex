"""Run a small end-to-end FairFlex demonstration from the project root."""

from fairflex.admm import DistributedFairMPCPolicy, FeederADMMNegotiator
from fairflex.baselines import equal_share, first_come_first_served, uncontrolled
from fairflex.evaluation import ExperimentRunner
from fairflex.examples import make_deadline_stress_simulation
from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.safety import ACRepairedDistributedPolicy, ACRepairedMPCPolicy


def fair_mpc_with_ac(simulation):
    return ACRepairedMPCPolicy(CentralizedFairMPC(), simulation.grid, horizon_steps=4)


def recommended_distributed_policy(simulation):
    distributed = DistributedFairMPCPolicy(
        CentralizedFairMPC(),
        FeederADMMNegotiator(),
        horizon_steps=4,
        feeder_capacity_kw=50.0,
    )
    return ACRepairedDistributedPolicy(distributed, simulation.grid)


def main() -> None:
    runner = ExperimentRunner(make_deadline_stress_simulation)
    baseline_runs = runner.run(
        {
            "uncontrolled": lambda: uncontrolled,
            "fcfs": lambda: first_come_first_served,
            "equal_share": lambda: equal_share,
        }
    )
    optimized_runs = runner.run_with_simulation_policy(
        {
            "fair_mpc_ac": fair_mpc_with_ac,
            "recommended_distributed": recommended_distributed_policy,
        }
    )
    print(ExperimentRunner.metrics_table({**baseline_runs, **optimized_runs}).to_string(index=False))


if __name__ == "__main__":
    main()
