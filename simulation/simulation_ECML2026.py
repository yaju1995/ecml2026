import os
import sys
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from tqdm.contrib.concurrent import process_map

# Restore real streams explicitly
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

from simulation.simulation import Simulation, SimulationConfig
from strategies.strategy import Strategy
from strategies.baseline import Baseline
from strategies.mab_ts import MABTS
from strategies.mab_ftpl_R import MABFTPL
from strategies.prior_opt import PriorOpt
from strategies.milp_forecast_price import MilpForecastPrice


# ==========================================================
# Strategy registration
# ==========================================================

Strategy.register("GREEDY", Baseline)
Strategy.register("PRIOR OPT", PriorOpt)
Strategy.register("Inde-TS", MABTS)
Strategy.register("FTPL-IRS-F2", MABFTPL)
Strategy.register("FTPL-IRS-EXP", MABFTPL)
Strategy.register("FTPL-IRS", MABFTPL)
Strategy.register("MILP_Price_Forecast", MilpForecastPrice)


# ==========================================================
# Base config
# ==========================================================

def get_base_config():
    return {
        "T_episode_seconds": 24 * 60 * 60,
        "dt": 15 * 60,
        "num_ev_agents": 30,
        "num_nf_agents": 20,
        "pv_area": 3000,
        "num_episodes": 1460,
        "congestion_limit": 173000,
        "part_of_telecommuters": 0.0,
        "strategy_name": "GREEDY",
        "ev_agent_type": "default",
        "ev_physics_type": "default",
        "price_generator": "default",
        "seed": 0,
        "verbose": False,
        "save": True,
        "path": "",
    }


# ==========================================================
# Experiment grid
# ==========================================================

def build_experiment_grid(strategies, num_samples_classic, start_seed, num_agents, mode="default"):
    base_path = "results_ECML2026/"

    num_ev_agents = np.atleast_1d(num_agents)
    num_nf_agents = (num_ev_agents * 2) / 3
    pv_areas = (3000.0 / 30.0) * num_ev_agents
    congestion_limits = (173000.0 / 30.0) * num_ev_agents

    tasks = []

    for sim_idx in range(len(num_ev_agents)):
        if num_ev_agents[sim_idx] > 500:
            num_samples = 1
        else:
            num_samples = num_samples_classic

        if mode == "centralized_ca" and num_ev_agents[sim_idx] > 480:
            num_samples = 0

        for strategy in strategies:
            for seed in range(num_samples):
                tasks.append(
                    {
                        "num_ev_agents": int(num_ev_agents[sim_idx]),
                        "num_nf_agents": int(num_nf_agents[sim_idx]),
                        "pv_area": float(pv_areas[sim_idx]),
                        "congestion_limit": float(congestion_limits[sim_idx]),
                        "strategy": strategy,
                        "seed": seed + start_seed,
                        "base_path": base_path,
                    }
                )

    tasks = sorted(tasks, key=lambda t: t["num_ev_agents"], reverse=True)
    return tasks


# ==========================================================
# Parallel decentralized runs
# ==========================================================

def run_simulation(task):
    config_dict = get_base_config()

    config_dict["num_ev_agents"] = task["num_ev_agents"]
    config_dict["num_nf_agents"] = task["num_nf_agents"]
    config_dict["pv_area"] = task["pv_area"]
    config_dict["congestion_limit"] = task["congestion_limit"]
    config_dict["strategy_name"] = task["strategy"]
    config_dict["seed"] = task["seed"]

    result_path = (
        f"{task['base_path']}default/"
        f"EVs_{task['num_ev_agents']}/"
        f"{task['strategy']}/"
        f"{task['seed']}/"
    )

    config_dict["path"] = result_path
    config_dict["ev_agent_type"] = "default"

    os.makedirs(result_path, exist_ok=True)

    simulation_config = SimulationConfig(**config_dict)
    simulation = Simulation(simulation_config)

    # Suppress noisy worker output only locally
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            simulation.run()

    return 1


def run_simulation_CA(task):
    config_dict = get_base_config()

    config_dict["num_ev_agents"] = task["num_ev_agents"]
    config_dict["num_nf_agents"] = task["num_nf_agents"]
    config_dict["pv_area"] = task["pv_area"]
    config_dict["congestion_limit"] = task["congestion_limit"]
    config_dict["strategy_name"] = task["strategy"]
    config_dict["seed"] = task["seed"]

    result_path = (
        f"{task['base_path']}CA/"
        f"EVs_{task['num_ev_agents']}/"
        f"{task['strategy']}/"
        f"{task['seed']}/"
    )

    config_dict["path"] = result_path
    config_dict["ev_agent_type"] = "congestion_aware"

    os.makedirs(result_path, exist_ok=True)

    simulation_config = SimulationConfig(**config_dict)
    simulation = Simulation(simulation_config)

    # Suppress noisy worker output only locally
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            simulation.run()

    return 1


# ==========================================================
# Sequential centralized MILP run
# ==========================================================


    print("All centralized CA simulations completed.", file=sys.__stdout__)


# ==========================================================
# Main
# ==========================================================

def main():
    strat_decentralized = ["GREEDY", "Inde-TS", "FTPL-IRS-F2", "FTPL-IRS-EXP"]
    strat_centralized_ca = ["MILP_Price_Forecast"]

    num_agents = np.array([5])
    num_samples = 2
    max_workers_decentralized = 3

    max_workers_centralized = 1


    # -------------------------------
    # Decentralized default
    # -------------------------------
    # for strategy in strat_decentralized:
    #     tasks = build_experiment_grid(
    #         strategies=[strategy],
    #         start_seed=0,
    #         num_samples_classic=5,
    #         mode="default",
    #         num_agents=num_agents,
    #     )
    
    #     print(f"Total simulations to run {strategy}-default: {len(tasks)}")
    
    #     process_map(
    #         run_simulation,
    #         tasks,
    #         max_workers=max_workers_decentralized,
    #         chunksize=1,
    #     )
    
    #     print(f"All simulations completed - {strategy}-default.")

    # # -------------------------------
    # # Decentralized congestion-aware
    # # -------------------------------
    # for strategy in strat_decentralized:
    #     tasks = build_experiment_grid(
    #         strategies=[strategy],
    #         start_seed=0,
    #         num_samples_classic=num_samples,
    #         mode="ca",
    #         num_agents=num_agents,
    #     )
    
    #     print(f"Total simulations to run - {strategy}-CA: {len(tasks)}")
    
    #     process_map(
    #         run_simulation_CA,
    #         tasks,
    #         max_workers=max_workers_decentralized,
    #         chunksize=1,
    #     )
    
    #     print(f"All simulations completed - {strategy}-CA.")

        

    # -------------------------------
    # Centralized congestion-aware MILP
    # -------------------------------
    for strategy in strat_centralized_ca:
        tasks = build_experiment_grid(
            strategies=[strategy],
            start_seed=0,
            num_samples_classic=num_samples,
            mode="ca",
            num_agents=num_agents,
        )
    
        print(f"Total simulations to run - {strategy}-CA: {len(tasks)}")
    
        process_map(
            run_simulation_CA,
            tasks,
            max_workers=max_workers_centralized,
            chunksize=1,
        )
    
        print(f"All simulations completed - {strategy}-CA.")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()