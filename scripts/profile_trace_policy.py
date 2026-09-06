"""Measure per-interval controller latency on one declared trace scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from fairflex.fair_mpc import CentralizedFairMPC
from fairflex.safety import ACRepairedMPCPolicy
from fairflex.scenarios import load_study_config, make_trace_simulation, prepare_acn_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/caltech_2019_congested.json")
    parser.add_argument("--split", default="test", choices=("train", "calibration", "test"))
    parser.add_argument("--feeder-capacity-kw", type=float, default=20.0)
    parser.add_argument("--report-over-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.feeder_capacity_kw < 0 or args.report_over_seconds <= 0:
        parser.error("capacity must be non-negative and report threshold positive")

    config_path = Path(args.config)
    config = load_study_config(config_path)
    simulation = make_trace_simulation(config, prepare_acn_split(config, config_path, args.split))
    policy = ACRepairedMPCPolicy(
        CentralizedFairMPC(),
        simulation.grid,
        horizon_steps=int(config["feeder_sensitivity"]["forecast_horizon_steps"]),
        feeder_capacity_kw=args.feeder_capacity_kw,
    )
    through_step = max(ev.departure_step for ev in simulation.evs.values())
    total_start = perf_counter()
    slow_steps = []
    while simulation.step < through_step:
        active = sum(len(items) for items in simulation.active_evs_by_station().values())
        start = perf_counter()
        simulation.advance(policy)
        elapsed = perf_counter() - start
        if elapsed >= args.report_over_seconds:
            slow_steps.append((simulation.step - 1, active, elapsed))
            print(
                f"slow step={simulation.step - 1} active_evs={active} seconds={elapsed:.3f}",
                flush=True,
            )
        if simulation.step % 96 == 0:
            print(f"progress step={simulation.step}/{through_step}", flush=True)
    print(
        f"completed steps={through_step}; seconds={perf_counter() - total_start:.3f}; "
        f"slow_steps={len(slow_steps)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
