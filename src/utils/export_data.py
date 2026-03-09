import csv

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def export_agent_config(ev_agents, path):
    data = {
        "id": [ev_agent.id for ev_agent in ev_agents],
        "t_a": [ev_agent.t_a for ev_agent in ev_agents],
        "t_b": [ev_agent.t_b for ev_agent in ev_agents],
        "soc_initial": [ev_agent.soc_initial for ev_agent in ev_agents],
        "soc_target": [ev_agent.soc_target for ev_agent in ev_agents],
        "telecommuting_days": [ev_agent.telecommuting_days for ev_agent in ev_agents],
        "strategy": [ev_agent.strategy_config.name for ev_agent in ev_agents],
        "ev_config": [ev_agent.ev_config for ev_agent in ev_agents],
    }
    export_data(data, path)


def export_strategy_config(strategy_config, path):

    data = {}
    for key, value in strategy_config.__dict__.items():
        data[key] = [value]

    export_data(data, path)


def export_simulation_config(
    n_evs, T, T_seconds, dt, n_episodes, n_samples, congestion_limit, path
):
    data = {
        "n_evs": [n_evs],
        "T": [T],
        "T_seconds": [T_seconds],
        "dt": [dt],
        "n_episodes": [n_episodes],
        "n_samples": [n_samples],
        "congestion_limit": [congestion_limit],
    }
    export_data(data, path)


def export_simulation_results(
    power_history,
    soc_history,
    reward_history,
    price_history,
    computing_times,
    path,
    chunk_size=1000,
):
    """
    Export simulation results in chunks with proper Jupyter notebook progress bars.
    """
    # Get the dimensions of the arrays
    n_evs, n_samples, n_episodes, T = power_history.shape
    # Write headers first
    headers = [
        "agent",
        "sample",
        "episode",
        "timestep",
        "power",
        "soc",
        "reward",
        "price",
        "computing_time",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    # Process data in chunks of agents
    for chunk_start in tqdm(range(0, n_evs, chunk_size), desc="Processing chunks", leave=False):
        chunk_end = min(chunk_start + chunk_size, n_evs)

        # Create index arrays for this chunk
        n_idx, s_idx, e_idx, t_idx = np.meshgrid(
            np.arange(chunk_start, chunk_end),
            np.arange(n_samples),
            np.arange(n_episodes),
            np.arange(T),
            indexing="ij",
        )

        # Create DataFrame for this chunk
        chunk_dict = {
            "agent": n_idx.ravel().astype(np.int32),
            "sample": s_idx.ravel().astype(np.int32),
            "episode": e_idx.ravel().astype(np.int32),
            "timestep": t_idx.ravel().astype(np.int32),
            "power": power_history[chunk_start:chunk_end].ravel(),
            "soc": soc_history[chunk_start:chunk_end].ravel(),
            "reward": reward_history[chunk_start:chunk_end].ravel(),
            "price": np.broadcast_to(
                price_history[None, :, :, :], (chunk_end - chunk_start, n_samples, n_episodes, T)
            ).ravel(),
        }

        df_chunk = pd.DataFrame(chunk_dict)

        # Add computing times for this chunk
        timestep_0_mask = df_chunk["timestep"] == 0
        df_chunk["computing_time"] = np.zeros(len(df_chunk), dtype=np.float32)
        df_chunk.loc[timestep_0_mask, "computing_time"] = computing_times[
            df_chunk.loc[timestep_0_mask, "sample"], df_chunk.loc[timestep_0_mask, "episode"]
        ]

        # Append this chunk to the CSV
        df_chunk.to_csv(path, mode="a", header=False, index=False, float_format="%.6f")

        # Clean up chunk data
        del df_chunk, chunk_dict


def export_data(data, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data.keys())

        # Convert any numpy arrays to lists and handle special data types
        processed_values = []
        for value in data.values():
            if isinstance(value, np.ndarray):
                processed_values.append(value.tolist())
            elif isinstance(value, list):
                # Handle nested numpy arrays in lists
                processed_value = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        processed_value.append(item.tolist())
                    else:
                        processed_value.append(item)
                processed_values.append(processed_value)
            else:
                processed_values.append(value)

        writer.writerows(zip(*processed_values))
