import csv
import os

import numpy as np
import pandas as pd
import tqdm.notebook as tqdm


def import_agent_config(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        data = {
            "id": [],
            "t_a": [],
            "t_b": [],
            "soc_initial": [],
            "soc_target": [],
            "telecommuting_days": [],
            "strategy": [],
            "ev_config": [],
        }

        for row in reader:
            data["id"].append(int(row["id"]))
            data["t_a"].append(float(row["t_a"]))
            data["t_b"].append(float(row["t_b"]))
            data["soc_initial"].append(float(row["soc_initial"]))
            data["soc_target"].append(float(row["soc_target"]))
            # Convert string representation of list back to actual list
            telecommuting_days = eval(row["telecommuting_days"])
            data["telecommuting_days"].append(telecommuting_days)
            data["strategy"].append(row["strategy"])
            data["ev_config"].append(row["ev_config"])

    return data


def import_simulation_config(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        data = next(reader)  # Only one row of data
        config = {
            "n_evs": int(data["n_evs"]),
            "T": int(data["T"]),
            "T_seconds": int(data["T_seconds"]),
            "dt": float(data["dt"]),
            "n_episodes": int(data["n_episodes"]),
            "n_samples": int(data["n_samples"]),
            "congestion_limit": float(data["congestion_limit"]),
        }

        return config


def import_simulation_results(path, config, strategy):
    """
    Import simulation results using pandas with detailed progress tracking.

    This implementation combines pandas' efficiency with progress bars to give
    visibility into each stage of data processing.
    """
    # print(f"\nProcessing data for strategy: {strategy}")

    # First progress bar: Reading the CSV file
    with tqdm.tqdm(total=1, desc=f"Reading CSV file {strategy}", leave=False) as pbar:
        df = pd.read_csv(
            path,
            dtype={
                "agent": "int32",
                "sample": "int32",
                "episode": "int32",
                "timestep": "int32",
                "power": "float32",
                "soc": "float32",
                "reward": "float32",
                "price": "float32",
                "computing_time": "float32",
            },
            engine="c",
        )
        pbar.update(1)

    # Get dimensions from config
    n_evs = config["n_evs"]
    T = config["T"]
    n_episodes = config["n_episodes"]
    n_samples = config["n_samples"]

    # Second progress bar: Array initialization
    power_history = np.zeros((n_evs, n_samples, n_episodes, T), dtype=np.float32)
    soc_history = np.zeros((n_evs, n_samples, n_episodes, T), dtype=np.float32)
    reward_history = np.zeros((n_evs, n_samples, n_episodes, T), dtype=np.float32)
    price_history = np.zeros((n_samples, n_episodes, T), dtype=np.float32)
    computing_times = np.zeros((n_samples, n_episodes), dtype=np.float32)

    # Third progress bar: Data processing
    with tqdm.tqdm(total=5, desc="Processing data arrays", leave=False) as pbar:
        # Process power history
        power_history[
            df["agent"].values, df["sample"].values, df["episode"].values, df["timestep"].values
        ] = df["power"].values
        pbar.update(1)

        # Process SOC history
        soc_history[
            df["agent"].values, df["sample"].values, df["episode"].values, df["timestep"].values
        ] = df["soc"].values
        pbar.update(1)

        # Process reward history
        reward_history[
            df["agent"].values, df["sample"].values, df["episode"].values, df["timestep"].values
        ] = df["reward"].values
        pbar.update(1)

        # Process price history (unique combinations only)
        price_df = df.drop_duplicates(["sample", "episode", "timestep"])[
            ["sample", "episode", "timestep", "price"]
        ]
        price_history[
            price_df["sample"].values, price_df["episode"].values, price_df["timestep"].values
        ] = price_df["price"].values
        pbar.update(1)

        # Process computing times (timestep = 0 only)
        computing_df = df[df["timestep"] == 0][["sample", "episode", "computing_time"]]
        computing_times[computing_df["sample"].values, computing_df["episode"].values] = (
            computing_df["computing_time"].values
        )
        pbar.update(1)

    return power_history, soc_history, reward_history, price_history, computing_times


def import_all_results(results_path):
    """
    Import simulation config and results for all strategies in the results directory.

    Parameters:
        results_path (str): Path to the main results directory

    Returns:
        tuple: (config, power_data, soc_data, reward_data, computing_times)
            where *_data are dictionaries with strategy names as keys
    """
    # Import simulation config
    config = import_simulation_config(os.path.join(results_path, "simulation_config.csv"))

    # Initialize dictionaries to store results
    strategies = [
        d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))
    ]
    power_data = {}
    soc_data = {}
    reward_data = {}
    price_data = {}
    computing_times = {}

    # Import results for each strategy
    for strategy in tqdm.tqdm(strategies, desc="Importing Results"):
        strategy_path = os.path.join(results_path, strategy, "results.csv")
        (
            power_data[strategy],
            soc_data[strategy],
            reward_data[strategy],
            price_data[strategy],
            computing_times[strategy],
        ) = import_simulation_results(strategy_path, config, strategy)

    print("Simulation Configuration:")
    print("-" * 50)
    print(f"Total Agents: {config['n_evs']}")
    print(f"Total Timesteps Per Episode: {config['T']}")
    print(f"Total Episodes: {config['n_episodes']}")
    print(f"Total Samples: {config['n_samples']}")
    print(f"Congestion Limit: {config['congestion_limit']:.2f} W")
    print("-" * 50)
    print(f"Imported results for {len(strategies)} strategies:")
    print(sorted(strategies))
    return config, power_data, soc_data, reward_data, price_data, computing_times, strategies
