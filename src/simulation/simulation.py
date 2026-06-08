import datetime
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":  # Terminal IPython
            return False
        else:
            return False
    except NameError:  # Likely standard Python interpreter
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import json

import seaborn as sns
from matplotlib import pyplot as plt

from agents.ev_agent import EVAgent, EVAgentConfig, Observation
from physics.ev_physics import  EVPhysicsConfig
from strategies.baseline import  BaselineConfig
from strategies.mab_ftpl_R import MABFTPLConfig
from strategies.prior_opt import PriorOptConfig
from strategies.mab_ts import MABTSConfig
# from strategies.mab_cf_ts import MABCFTS, MABCFTSConfig
from strategies.milp import MilpConfig
from strategies.drl import DRLConfig
from strategies.milp_forecast_price import MilpForecastPriceConfig
from strategies.milp_optimization import milp_daily_optimization
from strategies.strategy import StrategyConfig
from utils.generate_price import generate_price_from_net_energy

STRATEGY_NAMES = [
    "GREEDY",
    "Inde-TS",
    "MILP",
    "FTPL-IRS-EXP",
    "FTPL-IRS-F2",
    "MILP_Price_Forecast",
    "PRIOR OPT",
    "EV_DRL",
]

EV_AGENT_TYPES = ["default","congestion_aware"]
EV_PHYSICS_TYPES = ["default"]


@dataclass
class SimulationConfig:
    """Configuration for the simulation"""

    T_episode_seconds: int  # Duration of the episode in seconds
    dt: int  # Time step in seconds
    num_ev_agents: int  # Number of EV agents
    num_nf_agents: int # Number of Non-flexibles agents
    pv_area: int # Area of the PV farm (in m^2)
    num_episodes: int  # Number of episodes to run
    congestion_limit: int  # Congestion limit in W
    part_of_telecommuters: float  # Part of telecommuters in the population

    strategy_name: str  # Name of the strategy to use
    ev_agent_type: str  # Type of EV agents
    ev_physics_type: str  # Type of EV agents

    seed: int  # Seed for the random number generator
    verbose: bool  # Flag to enable verbose mode
    path: str  # Path to save the results
    save: bool  # Flag to save the results

    price_generator: str # Price Computation Function name


    @classmethod
    def print_config(cls, config):
        """Print the configuration with physical units"""
        print(f"T_episode: {config.T_episode_seconds} s")
        print(f"dt: {config.dt} s")
        print(f"num_ev_agents: {config.num_ev_agents}")
        print(f"num_nf_agents: {config.num_nf_agents}")
        print(f"PV_area: {config.pv_area}")
        print(f"num_episodes: {config.num_episodes}")
        print(f"congestion_limit: {config.congestion_limit:.2f} W")
        print(f"part_of_telecommuters: {config.part_of_telecommuters}")
        print(f"strategy_name: {config.strategy_name}")
        print(f"ev_agent_type: {config.ev_agent_type}")
        print(f"ev_physics_type: {config.ev_physics_type}")
        print(f"seed: {config.seed}")
        print(f"verbose: {config.verbose}")
        print(f"price_generator: {config.verbose}")
        print(f"path: {config.path}")
        print(f"save: {config.save}")

    @classmethod
    def get_config_dict(cls, config):
        """Return the configuration as a dictionary"""
        return config.__dict__


def load_simulation_config(path, verbose=False) -> SimulationConfig:
    """Load the simulation parameters"""
    if not os.path.exists(f"{path}/simulation_parameters.h5"):
        raise FileNotFoundError(f"File not found: {path}/simulation_parameters.h5")

    if verbose:
        print("Loading the simulation parameters...")

    try:
        with h5py.File(f"{path}/simulation_parameters.h5", "r") as f:
            # Display metadata if verbose
            if verbose and "saved_timestamp" in f.attrs:
                print(f"File created: {f.attrs['saved_timestamp']}")

            config_dict = {}
            for key in f.keys():
                if "json_serialized" in f[key].attrs and f[key].attrs["json_serialized"]:
                    # Deserialize JSON data
                    config_dict[key] = json.loads(f[key][()])
                else:
                    # Load simple values directly
                    value = f[key][()]
                    # Convert byte strings to Python strings
                    if isinstance(value, bytes):
                        value = value.decode()
                    config_dict[key] = value

        # Create SimulationConfig object from the dictionary
        config = SimulationConfig(**config_dict)
        if verbose:
            print(f"Simulation parameters loaded from {path}")
            print("----------------------------")

        return config
    except Exception as e:
        print(f"Error loading simulation parameters: {e}")
        raise


@dataclass
class SimulationResults:
    """Results of the simulation"""

    p_history: np.ndarray  # Power history
    soc_history: np.ndarray  # SOC history
    reward_history: np.ndarray  # Reward history
    price_history: np.ndarray  # Price history
    congestion_history: np.ndarray  # Congestion history
    telecommute_history: np.ndarray  # Telecommute history
    t_a_history: np.ndarray #arrival time history
    t_b_history: np.ndarray # departure time history
    soc_target_history: np.ndarray # soc target history
    soc_init_history: np.ndarray # soc init history
    action_history: np.ndarray # action history of agents
    computing_time: float # Computing time
    price_data: np.ndarray # Encountered_price_history.
    pv_data: np.ndarray # Encountered solar power production
    price_f6pm_data: np.ndarray # Price Forecast dayahead 6pm history
    pv_f6pm_data: np.ndarray # PV Forecast dayahead 6pm history
    price_f11am_data: np.ndarray # Price Forecast dayahead 11am history
    pv_f11am_data: np.ndarray # PV Forecast dayahead 11am history
    n_flexible_power: np.ndarray # Encountered non flexible history
    

    @classmethod
    def print_data_structure(cls):
        print("SimulationResults data structure:")
        print("p_history: (num_episodes, T, num_ev_agents)")
        print("soc_history: (num_episodes, T, num_ev_agents)")
        print("reward_history: (num_episodes, T, num_ev_agents)")
        print("price_history: (num_episodes, T)")
        print("congestion_history: (num_episodes, T, num_ev_agents)")
        print("telecommute_history: (num_episodes, T)")
        print("arrival_time_history: (num_ev_agent, num_episode)")
        print("departure_time_history: (num_ev_agent, num_episode)")
        print("soc_target_history: (num_ev_agent, num_episode)")
        print("soc_init_history: (num_ev_agents, num_episode)")
        print("action_history: (num_ev_agents, num_episode)")
        print("price_data: (num_episode, T)")
        print("pv_data: (num_episode, T)")
        print("price_f6pm_data: (num_episode, T)")
        print("pv_f6pm_data: (num_episode, T)")
        print("price_f11am_data: (num_episode, T)")
        print("pv_f11am_data: (num_episode, T)")
        print("n_flexible_power: (num_episode, T)")
        print("----------------------------")

    @classmethod
    def get_result_df(cls, results):
        """Return the results as a pandas DataFrame with proper indexing"""
        # Get dimensions from the data
        num_episodes, T, num_ev_agents = results.p_history.shape

        # Create an empty list to store DataFrames for each episode
        episode_dfs = []

        for episode in range(num_episodes):
            # Create DataFrame for each timestep and agent
            data = []
            for t in range(T):
                for agent in range(num_ev_agents):
                    row = {
                        "episode": episode,
                        "timestep": t,
                        "agent": agent,
                        "power": results.p_history[episode, t, agent],
                        "soc": results.soc_history[episode, t, agent],
                        "reward": results.reward_history[episode, t, agent],
                        "congestion": results.congestion_history[episode, t, agent],
                        "price": results.price_history[episode, t, agent],
                        "t_a": results.t_a_history[agent,episode],
                        "t_b": results.t_b_history[agent,episode],
                        "soc_init": results.soc_init_history[agent,episode],
                        "soc_target": results.soc_target_history[agent,episode],
                        "action": results.action_history[episode, t, agent],
                        "telecommute": results.telecommute_history[episode, t],              
                    }
                    data.append(row)

            # Create a DataFrame for this episode
            episode_df = pd.DataFrame(data)
            episode_dfs.append(episode_df)

        # Concatenate all episode DataFrames
        result_df = pd.concat(episode_dfs, ignore_index=True)

        return result_df


class Simulation:
    """Class to run the simulation"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        if self.config.verbose:
            print("Simulation configuration:")
            SimulationConfig.print_config(self.config)
            print("----------------------------")

        self.T = self.config.T_episode_seconds // self.config.dt
        self.ev_agents = None
        self.results = None
        self.num_ev_agents = self.config.num_ev_agents
        self.num_episodes = self.config.num_episodes
        self.num_nf_agents = self.config.num_nf_agents
        self.pv_area = self.config.pv_area


        # Set the seed for the random number generator
        np.random.seed(self.config.seed)

        # Create the strategy config
        self.strategy_config = self.set_strategy_config()

        # Create the EV physics config
        self.ev_physics_config = self.set_ev_physics_config()

        # Create the EV agent default config
        if self.config.ev_agent_type != "default":
            if self.config.ev_agent_type == "congestion_aware":
                self.ev_agent_config = EVAgentConfig.congestion_aware_default_config()
        else:
            self.ev_agent_config = EVAgentConfig.default_config()

        self.price_generator = self.config.price_generator

        # Create price temporal data for the simulation [num_episodes, T]
        self.price_data, self.pv_data, self.price_f6pm_data, self.pv_f6pm_data, self.price_f11am_data, self.pv_f11am_data, self.n_flexible_power = generate_price_from_net_energy(num_episodes= self.config.num_episodes, T = self.T, seed = self.config.seed, n_conso=self.num_nf_agents, PV_area = self.pv_area)

        # Pregenerate behavior history:
        self.t_a_history, self.t_b_history, self.soc_init_history, self.soc_target_history = generate_ev_fleet_uniform_behavior_history(t_a_range = [32,36],
                                                                                                                                       t_b_range = [72,76],
                                                                                                                                        soc_init_range=[0.3,0.8],
                                                                                                                                        soc_target_range = [0.8, 0.8],
                                                                                                                                        num_episodes= self.num_episodes,
                                                                                                                                        num_ev_agents=self.num_ev_agents,
                                                                                                                                        random_seed=self.config.seed )

        # Initialize the simulation
        self.initialisation()

    def set_strategy_config(self) -> StrategyConfig:
        """Set the strategy config"""
        if self.config.strategy_name == "GREEDY":
            return BaselineConfig.default_config()
        elif self.config.strategy_name == "MILP":
            return MilpConfig.default_config()
        elif self.config.strategy_name == "FTPL-IRS-EXP":
            return MABFTPLConfig.default_config()
        elif self.config.strategy_name == "Inde-TS":
            return MABTSConfig.default_config()
        elif self.config.strategy_name == "FTPL-IRS-F2":
            return MABFTPLConfig.F2_perturbed_config()
        elif self.config.strategy_name == "MILP_Price_Forecast":
            return MilpForecastPriceConfig.default_config()
        elif self.config.strategy_name == "PRIOR OPT":
            return PriorOptConfig.default_config()
        elif self.config.strategy_name == "EV_DRL":
            return DRLConfig.default_config()
        else:
            raise ValueError(f"Unknown strategy name: {self.config.strategy_name}")

    def set_ev_physics_config(self) -> EVPhysicsConfig:
        """Set the EV physics config"""
        if self.config.ev_physics_type == "default":
            return EVPhysicsConfig.default_config()
        else:
            raise ValueError(f"Unknown EV physics type: {self.config.ev_physics_type}")

    def create_ev_agents(self) -> List[EVAgent]:
        """Create the EV agents"""
        if self.config.verbose:
            print("Creating the EV agents...")

        ev_agents = []
        # Create a base dictionary outside the loop
        base_dict = self.ev_agent_config.__dict__.copy()
        base_dict["ev_config"] = self.ev_physics_config
        base_dict["strategy_config"] = self.strategy_config
        
        for i in range(self.config.num_ev_agents):
            # Create a NEW copy for each agent
            agent_dict = base_dict.copy()
            agent_dict["id"] = i  # Set unique ID

            if i < self.config.num_ev_agents * self.config.part_of_telecommuters:
                agent_dict["telecommuting_days"] = [1, 3]
            else:
                agent_dict["telecommuting_days"] = []  # Ensure non-telecommuters have empty list

            ev_agent_config = EVAgentConfig(**agent_dict)
            ev_agent = EVAgent(ev_agent_config)

            if ev_agent.strategy.name == "PRIOR OPT":
                ev_agent.strategy.init_reward_list(self.price_f6pm_data)
            ev_agents.append(ev_agent)

        
        return ev_agents

    def delete_ev_agents(self):
        """Delete the EV agents"""
        for ev_agent in self.ev_agents:
            del ev_agent
        del self.ev_agents

    def initialisation(self):
        """Initialise the simulation"""
        if self.config.verbose:
            print("Initialising the simulation...")
        if self.config.save:
            if not os.path.exists(self.config.path):
                os.makedirs(self.config.path)
        if self.ev_agents is not None:
            self.delete_ev_agents()

        num_episodes = self.num_episodes
        T = self.T
        num_ev_agents = self.num_ev_agents

        # Create the EV agents
        self.ev_agents = self.create_ev_agents()
        # 
        # Create the agents actions
        self.agents_actions = np.zeros(num_ev_agents)

        # Initialize the simulation variables
        self.t = 0
        self.episode = 0
        self.congestion_list_t = np.zeros(num_ev_agents)
        self.price_t = 0
        self.price_day = np.zeros(num_episodes)
        self.telecommute_day = 0
        self.congestion_list_t = np.zeros(num_ev_agents)
        self.disconnection_list_t = np.zeros(num_ev_agents)
        self.congestion_limit = np.zeros(self.T)

        # Initialize the history
        self.p_history = np.zeros((num_episodes, T, num_ev_agents))
        self.soc_history = np.zeros((num_episodes, T, num_ev_agents))
        self.reward_history = np.zeros((num_episodes, T, num_ev_agents))
        self.price_history = np.zeros((num_episodes,T,num_ev_agents ))
        self.congestion_history = np.zeros((num_episodes, T, num_ev_agents))
        self.telecommute_history = np.zeros((num_episodes, T))
        self.action_history = np.zeros((self.num_episodes, self.T , self.num_ev_agents))

        if self.config.verbose:
            print("Simulation initialised.")

    def run(self) -> SimulationResults:
        """Run the simulation"""
        if self.config.verbose:
            print("Running the simulation...")
        num_episodes = self.config.num_episodes
        num_ev_agents = self.config.num_ev_agents
        T = self.T

        # Start measuring computing time
        start_time = time.time()

        for episode in tqdm(range(num_episodes), leave=False, desc="Episodes"):
            # Reset observation context
            self.episode = episode
            self.t = 0
            self.congestion_list_t = np.zeros(num_ev_agents)
            self.price_day = self.price_data[episode]
            self.telecommute_day = 0
            self.price_t = self.price_day[self.t]

            # Reset the simulation and the agents for a new episode
            self.reset()

            for i in range (self.T):
                self.congestion_limit[i] = self.config.congestion_limit - min(self.n_flexible_power[self.episode,i],self.config.congestion_limit)
            
            actions_matrix = None

            # If the strategy is MILP, run the centralized optimization
            if self.config.strategy_name == "MILP":
                (
                    p_matrix
                
                ) = milp_daily_optimization(
                    episode, self.ev_agents, self.price_day, self.congestion_limit
                )

                actions_matrix = np.where(p_matrix > 0, 1, 0)

            elif self.config.strategy_name == "MILP_Price_Forecast":
                (
                    p_matrix
                
                ) = milp_daily_optimization(
                    episode, self.ev_agents, self.price_f6pm_data[episode], self.congestion_limit
                )

                actions_matrix = np.where(p_matrix > 0, 1, 0)

            #no else the how does other strategy work ? [Demand Resposne]


            # Run the simulation for the episode fro T time period
            for t in range(T):
                # Observe the current global context
                self.t = t
                self.price_t = self.price_day[t]

                # Run the simulation for the time step
                self.step(actions_matrix=actions_matrix)

        computing_time = time.time() - start_time
        if self.config.verbose:
            print("Simulation completed.")
            print(f"Computing time: {computing_time:.2f} s")

        self.results = SimulationResults(
            p_history=self.p_history,
            soc_history=self.soc_history,
            reward_history=self.reward_history,
            price_history=self.price_history,
            congestion_history=self.congestion_history,
            telecommute_history=self.telecommute_history,
            t_a_history = self.t_a_history,
            t_b_history = self.t_b_history,
            computing_time=computing_time,
            price_data = self.price_data,
            pv_data = self.pv_data,
            n_flexible_power= self.n_flexible_power,
            action_history=self.action_history,
            soc_init_history= self.soc_init_history,
            soc_target_history= self.soc_target_history,
            price_f6pm_data =  self.price_f6pm_data,
            pv_f6pm_data = self.pv_f6pm_data,
            price_f11am_data =  self.price_f11am_data,
            pv_f11am_data = self.pv_f11am_data,
        )

        if self.config.save:
            self.save_results(self.results)
            if self.config.verbose:
                print(f"Results saved at {self.config.path}")

        return self.results

    def reset(self):
        """Reset the simulation for a new episode"""
        for ev_agent in self.ev_agents:

            # SOC INIT SETUP
            ev_agent.state.soc = np.random.uniform(0.3,ev_agent.config.soc_target)
            ev_agent.config.soc_initial = self.soc_init_history[ev_agent.id, self.episode]
            
            # SOC TARGET SETUP
            ev_agent.config.soc_target = self.soc_target_history[ev_agent.id, self.episode]

            # DEPARTURE AND ARRIVAL TIME SETUP
            ev_agent.state.t_b = self.t_b_history[ev_agent.id, self.episode]
            ev_agent.config.t_b = ev_agent.state.t_b*3600
            ev_agent.t_b = ev_agent.state.t_b

            ev_agent.state.t_a = self.t_a_history[ev_agent.id, self.episode]
            # ev_agent.state.t_a = int(32)
            ev_agent.config.t_a = ev_agent.state.t_a*3600
            ev_agent.t_a = ev_agent.state.t_a

            ev_obs = self.ev_get_observation(ev_agent)

            ev_agent.reset(obs=ev_obs)



    def step(self, actions_matrix=None):
        """Run the simulation for a time step"""
        self.congestion_list_t = np.zeros(self.num_ev_agents)
        self.disconnection_list_t = np.zeros(self.num_ev_agents)
        for ev_agent in self.ev_agents:
            # get observation for all ev agent
            ev_obs = self.ev_get_observation(ev_agent)
            
            # need to store this obseravation in the list or dataclass
            ev_agent.ins_state = ev_obs

            id = ev_agent.id
            # If MILP, use the precomputed actions
            # get action 
            if actions_matrix is not None:
                self.agents_actions[id] = ev_agent.act(
                    obs=ev_obs, milp_action=actions_matrix[self.t, id]
                )
            # Otherwise, use the agent's strategy
            else:
                self.agents_actions[id] = ev_agent.act(obs=ev_obs)
            
            # need to store the actions in to list or data class

        # Congestion management
        total_power = np.sum(self.agents_actions)
        flex_t = min(self.n_flexible_power[self.episode,self.t],self.config.congestion_limit)
        random_idx = []
        if total_power + flex_t > self.config.congestion_limit:
            # Identify agents contributing to congestion
            responsible_agents_idx = np.where(self.agents_actions > 0)[0]
            num_resp_agents = len(responsible_agents_idx)

            # Calculate how many agents to reduce power
            p_max_mean = total_power / num_resp_agents            
            number_needed_agent = int(np.ceil((total_power - (self.config.congestion_limit - flex_t))/p_max_mean))

            # Randomly select agents to reduce power
            random_idx = np.random.choice(responsible_agents_idx, number_needed_agent, replace=False)
            self.congestion_list_t[responsible_agents_idx] = 1
            self.disconnection_list_t[random_idx] = 1 
        
        # Update Power after congestion management
        updated_total_power = total_power
        for id in random_idx:
            updated_total_power -= self.agents_actions[id]

        #Price computation:
        if self.price_generator != "default":
            if self.price_generator == "valley_filling":
                self.price_t = 1.0 - float(max(
                    min(
                        self.pv_data[self.episode][self.t]-self.n_flexible_power[self.episode][self.t], 
                        self.congestion_limit) - updated_total_power,0))
            if self.price_generator == "valley_filling_cong_regu":
                self.price_t = 1.0 - float(max(
                        self.pv_data[self.episode][self.t]
                        -self.n_flexible_power[self.episode][self.t]) - updated_total_power,0)
        
        self.price_data[self.episode, self.t] = self.price_t
        
        # Update the agents' state
        for ev_agent in self.ev_agents:
            ev_obs = self.ev_get_observation(ev_agent) # this observation is next state

            # getting next state s'
            ev_agent.ins_next_state = ev_obs

            # print(f'Agent {ev_agent.id}->[{ev_obs}]')
            _, reward = ev_agent.update(obs=ev_obs) # update to get reward not udpdate
            
            # need to store reward 
            ev_agent.ins_reward = reward
            
            ev_agent.ins_terminate = False # Not true temination become true when episode termiate [Defining final Reward]

            new_state = ev_obs   
            # Record the state and reward
            self.p_history[self.episode, self.t, ev_agent.id] = ev_agent.state.p
            self.soc_history[self.episode, self.t, ev_agent.id] = ev_agent.state.soc
            self.reward_history[self.episode, self.t, ev_agent.id] = reward
            self.congestion_history[self.episode, self.t, ev_agent.id] = self.congestion_list_t[
                ev_agent.id
            ]
            if self.disconnection_list_t[ev_agent.id] == 0 and self.agents_actions[ev_agent.id] > 0:
                self.price_history[self.episode, self.t, ev_agent.id] = self.price_t
            self.action_history[self.episode, self.t, ev_agent.id] = self.agents_actions[ev_agent.id]

            # use this loop than creating a new loop to store infromation 
            # if stragegy is 'DRL': then get next state, reward and upload in the collector 

        self.telecommute_history[self.episode, self.t] = self.telecommute_day

        # do a for loop in ev_agents and then store the information. 

    def ev_get_observation(self, ev_agent: EVAgent, return_type = None):

        return Observation(

            day=self.episode,
            price_t=self.price_t,
            congestion_signal_t=self.congestion_list_t[ev_agent.id],
            telecommute=self.telecommute_day,
            price_day=self.price_day,
            disconnect_t=self.disconnection_list_t[ev_agent.id],
        )
    
    def ev_get_observation_array(self, ev_agent: EVAgent):
        """Return EV observation as a flat NumPy array"""
        return np.concatenate([
            np.array([self.episode], dtype=np.float32),
            np.array([self.price_t], dtype=np.float32),
            np.array([self.congestion_list_t[ev_agent.id]], dtype=np.float32),
            np.array([self.telecommute_day], dtype=np.float32),
            self.price_day.astype(np.float32),
            np.array([self.disconnection_list_t[ev_agent.id]], dtype=np.float32),
        ])

    # Enhanced save methods
    def save_results(self, results: SimulationResults):
        """Save the results of the simulation"""
        if self.config.verbose:
            print("Saving the results...")

        # Create directory if it doesn't exist
        if not os.path.exists(self.config.path):
            os.makedirs(self.config.path)

        try:
            self.save_simulation_parameters()
            self.save_agent_config()
            self.save_simulation_results(results)

            if self.config.verbose:
                print(f"Results successfully saved to {self.config.path}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def save_simulation_parameters(self):
        """Save the simulation parameters"""
        try:
            config_dict = SimulationConfig.get_config_dict(self.config)

            with h5py.File(f"{self.config.path}/simulation_parameters.h5", "w") as f:
                # Add metadata
                f.attrs["saved_timestamp"] = str(datetime.datetime.now())
                f.attrs["config_type"] = "SimulationConfig"
                f.attrs["version"] = "1.0"

                # Save configuration items
                for key, value in config_dict.items():
                    # Handle different value types
                    if isinstance(value, (list, dict)) or (
                        hasattr(np.array(value), "dtype") and np.array(value).dtype == "O"
                    ):
                        # Convert complex objects to JSON
                        f.create_dataset(key, data=json.dumps(value))
                        f[key].attrs["json_serialized"] = True
                    else:
                        # Store simple values directly
                        f.create_dataset(key, data=value)
        except Exception as e:
            print(f"Error saving simulation parameters: {e}")
            raise

    def save_agent_config(self):
        """Save all individual agent configurations"""
        try:
            with h5py.File(f"{self.config.path}/agent_configs.h5", "w") as f:
                f.attrs["saved_timestamp"] = str(datetime.datetime.now())
                f.attrs["config_type"] = "EVAgentConfigs"
                f.attrs["version"] = "1.0"
                f.attrs["num_agents"] = len(self.ev_agents)

                # Create a group for each agent
                for i, ev_agent in enumerate(self.ev_agents):
                    agent_group = f.create_group(f"agent_{i}")
                    agent_config_dict = EVAgentConfig.get_config_dict(ev_agent.config)

                    # Save each attribute
                    for key, value in agent_config_dict.items():
                        if key == "ev_config" and hasattr(value, "__dict__"):
                            # Handle nested EVPhysicsConfig
                            ev_config_dict = EVPhysicsConfig.get_config_dict(value)
                            agent_group.create_dataset(key, data=json.dumps(ev_config_dict))
                            agent_group[key].attrs["json_serialized"] = True
                            agent_group[key].attrs["object_type"] = "EVPhysicsConfig"
                        elif key == "strategy_config" and hasattr(value, "__dict__"):
                            # Handle nested StrategyConfig
                            strategy_config_dict = StrategyConfig.get_config_dict(value)
                            agent_group.create_dataset(key, data=json.dumps(strategy_config_dict))
                            agent_group[key].attrs["json_serialized"] = True
                            agent_group[key].attrs["object_type"] = value.__class__.__name__
                        elif isinstance(value, (list, dict)) or (
                            hasattr(np.array(value), "dtype") and np.array(value).dtype == "O"
                        ):
                            # Handle complex objects
                            agent_group.create_dataset(key, data=json.dumps(value))
                            agent_group[key].attrs["json_serialized"] = True
                        else:
                            # Handle simple values
                            agent_group.create_dataset(key, data=value)

        except Exception as e:
            print(f"Error saving agent configurations: {e}")
            raise

    def save_simulation_results(self, results: SimulationResults):
        """Save the simulation results"""
        try:
            with h5py.File(f"{self.config.path}/simulation_results.h5", "w") as f:
                # Add metadata
                f.attrs["saved_timestamp"] = str(datetime.datetime.now())
                f.attrs["data_type"] = "SimulationResults"
                f.attrs["version"] = "1.0"
                f.attrs["shape_info"] = (
                    f"Episodes: {results.p_history.shape[0]}, Timesteps: {results.p_history.shape[1]}, Agents: {results.p_history.shape[2]}"
                )

                # Save all values from the results object
                for key, value in results.__dict__.items():
                    if value is not None:
                        # Check if value is an array/has shape attribute
                        if hasattr(value, "shape") and value.shape:
                            # This is an array - apply compression
                            f.create_dataset(key, data=value, compression="gzip", compression_opts=4)
                            # Store array shape information as attributes
                            f[key].attrs["shape"] = value.shape
                        else:
                            # This is a scalar value - no compression
                            f.create_dataset(key, data=value)
        except Exception as e:
            print(f"Error saving simulation results: {e}")
            raise

    # Enhanced loading methods
    def load_simulation_parameters(self) -> SimulationConfig:
        """Load the simulation parameters"""
        if not os.path.exists(f"{self.config.path}/simulation_parameters.h5"):
            raise FileNotFoundError(f"File not found: {self.config.path}/simulation_parameters.h5")

        if self.config.verbose:
            print("Loading the simulation parameters...")

        try:
            with h5py.File(f"{self.config.path}/simulation_parameters.h5", "r") as f:
                # Display metadata if verbose
                if self.config.verbose and "saved_timestamp" in f.attrs:
                    print(f"File created: {f.attrs['saved_timestamp']}")

                config_dict = {}
                for key in f.keys():
                    if "json_serialized" in f[key].attrs and f[key].attrs["json_serialized"]:
                        # Deserialize JSON data
                        config_dict[key] = json.loads(f[key][()])
                    else:
                        # Load simple values directly
                        value = f[key][()]
                        # Convert byte strings to Python strings
                        if isinstance(value, bytes):
                            value = value.decode()
                        config_dict[key] = value

            # Create SimulationConfig object from the dictionary
            config = SimulationConfig(**config_dict)

            if self.config.verbose:
                print(f"Simulation parameters loaded from {self.config.path}")
                print("----------------------------")

            return config
        except Exception as e:
            print(f"Error loading simulation parameters: {e}")
            raise

    def load_agent_configs(self) -> List[EVAgentConfig]:
        """Load all individual agent configurations"""
        if not os.path.exists(f"{self.config.path}/agent_configs.h5"):
            raise FileNotFoundError(f"File not found: {self.config.path}/agent_configs.h5")

        if self.config.verbose:
            print("Loading agent configurations...")

        try:
            with h5py.File(f"{self.config.path}/agent_configs.h5", "r") as f:
                # Display metadata if verbose
                if self.config.verbose and "saved_timestamp" in f.attrs:
                    print(f"File created: {f.attrs['saved_timestamp']}")

                num_agents = f.attrs.get("num_agents", self.config.num_ev_agents)
                agent_configs = []

                # Load each agent's configuration
                for i in range(min(num_agents, self.config.num_ev_agents)):
                    agent_group = f[f"agent_{i}"]
                    agent_config_dict = {}

                    # Load all attributes
                    for key in agent_group.keys():
                        if (
                            "json_serialized" in agent_group[key].attrs
                            and agent_group[key].attrs["json_serialized"]
                        ):
                            # Deserialize JSON data
                            value = json.loads(agent_group[key][()])

                            # Handle specially marked nested objects
                            if "object_type" in agent_group[key].attrs:
                                object_type = agent_group[key].attrs["object_type"]
                                if object_type == "EVPhysicsConfig":
                                    value = EVPhysicsConfig(**value)
                                elif "Config" in object_type:
                                    # Convert to appropriate strategy config
                                    strategy_name = value.get("name", self.config.strategy_name)
                                    value = self.create_strategy_config_from_dict(
                                        strategy_name, value
                                    )

                            agent_config_dict[key] = value
                        else:
                            # Load simple values
                            value = agent_group[key][()]
                            # Convert byte strings to Python strings
                            if isinstance(value, bytes):
                                value = value.decode()
                            agent_config_dict[key] = value

                    # Create the agent config object
                    agent_config = EVAgentConfig(**agent_config_dict)
                    agent_configs.append(agent_config)

                if self.config.verbose:
                    print(f"Loaded {len(agent_configs)} agent configurations")

                return agent_configs

        except Exception as e:
            print(f"Error loading agent configurations: {e}")
            raise

    def create_strategy_config_from_dict(self, strategy_name, config_dict):
        """Create a strategy config object from a dictionary based on strategy name"""
        if strategy_name == "GREEDY":
            return BaselineConfig(**config_dict)
        elif strategy_name == "MABTS":
            return MABTSConfig(**config_dict)
        elif strategy_name == "FTPL-IRS-EXP":
            return MABFTPLConfig(**config_dict)
        elif strategy_name == "FTPL-IRS-F2":
            return MABFTPLConfig(**config_dict)
        elif strategy_name == "MILP":
            return MilpConfig(**config_dict)
        elif strategy_name == "PRIOR OPT":
            return PriorOptConfig(**config_dict)
        elif strategy_name == "MILP_Price_Forecast":
            return MilpForecastPriceConfig(**config_dict)
        elif strategy_name == "EV_DRL":
            return DRLConfig(**config_dict)
        else:
            raise ValueError(f"Unknown strategy name: {strategy_name}")

    def load_agents_configs_df(self) -> pd.DataFrame:
        """Load the agent configurations as a DataFrame with ALL parameters"""
        agent_configs = self.load_agent_configs()

        # Create a list to store dictionaries
        agent_configs_dicts = []

        for agent_config in agent_configs:
            # Start with a basic dictionary for direct attributes
            config_dict = {}

            # Add all direct attributes of the EVAgentConfig object
            for key, value in agent_config.__dict__.items():
                # Skip complex objects that will be handled separately
                if key not in ["ev_config", "strategy_config"]:
                    config_dict[key] = value

            # Add EV physics config attributes if available
            if hasattr(agent_config, "ev_config") and agent_config.ev_config is not None:
                if hasattr(agent_config.ev_config, "__dict__"):
                    for key, value in agent_config.ev_config.__dict__.items():
                        config_dict[f"ev_{key}"] = value
                elif isinstance(agent_config.ev_config, dict):
                    for key, value in agent_config.ev_config.items():
                        config_dict[f"ev_{key}"] = value

            # Add strategy config attributes if available
            if (
                hasattr(agent_config, "strategy_config")
                and agent_config.strategy_config is not None
            ):
                # Add the strategy name separately for clarity
                if hasattr(agent_config.strategy_config, "name"):
                    config_dict["strategy_name"] = agent_config.strategy_config.name
                elif (
                    isinstance(agent_config.strategy_config, dict)
                    and "name" in agent_config.strategy_config
                ):
                    config_dict["strategy_name"] = agent_config.strategy_config["name"]
                else:
                    config_dict["strategy_name"] = self.config.strategy_name

                # Add all other strategy config attributes
                if hasattr(agent_config.strategy_config, "__dict__"):
                    for key, value in agent_config.strategy_config.__dict__.items():
                        # Skip adding the name field to avoid duplication
                        if key != "name":
                            config_dict[f"strategy_{key}"] = value
                elif isinstance(agent_config.strategy_config, dict):
                    for key, value in agent_config.strategy_config.items():
                        # Skip adding the name field to avoid duplication
                        if key != "name":
                            config_dict[f"strategy_{key}"] = value

            agent_configs_dicts.append(config_dict)

        agent_configs_df = pd.DataFrame(agent_configs_dicts)
        return agent_configs_df
    
    def load_results(self, interpolate_telecommuter_reward=False) -> SimulationResults:
        """
        Load the results of the simulation.

        Parameters:
        -----------
        interpolate_telecommuter_reward : bool, optional
            If True, interpolates the reward values for telecommuting agents on telecommuting days
            to allow for fair comparison between telecommuting and non-telecommuting agents.
            Default is False.

        Returns:
        --------
        SimulationResults
            The loaded simulation results.
        """
        if not os.path.exists(f"{self.config.path}/simulation_results.h5"):
            raise FileNotFoundError(f"File not found: {self.config.path}/simulation_results.h5")

        if self.config.verbose:
            print("Loading the results...")

        try:
            with h5py.File(f"{self.config.path}/simulation_results.h5", "r") as f:
                # Display metadata if verbose
                if self.config.verbose and "saved_timestamp" in f.attrs:
                    print(f"File created: {f.attrs['saved_timestamp']}")
                    if "shape_info" in f.attrs:
                        print(f"Data info: {f.attrs['shape_info']}")

                # Create a dictionary to store all the loaded values
                results_dict = {}

                # Load all datasets into the dictionary
                for key in f.keys():
                    # Check if the dataset is scalar
                    if f[key].shape == ():
                        # For scalar values, load directly without slicing
                        results_dict[key] = f[key][()]
                    else:
                        # For arrays, use slicing as before
                        results_dict[key] = f[key][:]
                
                # Add computing_time with default value if it doesn't exist
                if 'computing_time' not in results_dict:
                    results_dict['computing_time'] = 0.0
                    if self.config.verbose:
                        print("Computing time field not found, using default value of 0.0 seconds")

            # Create SimulationResults object
            results = SimulationResults(**results_dict)

            # Apply reward interpolation if requested
            if interpolate_telecommuter_reward:
                if self.config.verbose:
                    print("----------------------------")
                results = self._interpolate_telecommuter_rewards(results)

            if self.config.verbose:
                print(f"Results loaded from {self.config.path}")
                print(f"Computing time: {results.computing_time:.2f} seconds")
                print("----------------------------")

            return results
        except Exception as e:
            print(f"Error loading simulation results: {e}")
            raise

    def _interpolate_telecommuter_rewards(self, results: SimulationResults) -> SimulationResults:
        """
        Interpolate reward values for telecommuting agents on telecommuting days.

        Parameters:
        -----------
        results : SimulationResults
            The original simulation results.

        Returns:
        --------
        SimulationResults
            The simulation results with interpolated rewards.
        """
        if self.config.verbose:
            print("Interpolating rewards for telecommuting agents...")

        # Create a copy of the reward history to modify
        interpolated_reward_history = results.reward_history.copy()

        # Get dimensions from the data
        num_episodes, T, num_ev_agents = results.reward_history.shape

        # Determine telecommuting agents
        telecommuting_agents = []

        # First, we need to load agent configs to identify telecommuting agents
        try:
            agent_configs = self.load_agent_configs()
            for i, agent_config in enumerate(agent_configs):
                if hasattr(agent_config, "telecommuting_days") and agent_config.telecommuting_days:
                    telecommuting_agents.append(i)
        except:
            # If we can't load agent configs, use the first part_of_telecommuters fraction of agents
            telecommuting_agents = list(
                range(int(num_ev_agents * self.config.part_of_telecommuters))
            )

        if not telecommuting_agents:
            if self.config.verbose:
                print("No telecommuting agents found. No interpolation performed.")
            return results

        if self.config.verbose:
            print(f"Found {len(telecommuting_agents)} telecommuting agents.")

        # Process each episode
        for episode in range(num_episodes):
            # Check if this is a telecommuting day
            telecommute_day = bool(results.telecommute_history[episode, 0])

            if not telecommute_day:
                continue

            # For each telecommuting agent
            for agent_idx in telecommuting_agents:
                # Find non-telecommuting days for this agent
                non_telecommute_episodes = []
                for e in range(num_episodes):
                    if not results.telecommute_history[e, 0]:
                        non_telecommute_episodes.append(e)

                if not non_telecommute_episodes:
                    continue

                # Calculate the average reward for this agent on non-telecommuting days
                avg_rewards = []
                for e in non_telecommute_episodes:
                    # Only consider timesteps when the agent is available/charging
                    available_mask = results.p_history[e, :, agent_idx] > 0
                    if available_mask.any():
                        avg_rewards.append(
                            np.mean(results.reward_history[e, available_mask, agent_idx])
                        )

                if avg_rewards:
                    avg_reward = np.mean(avg_rewards)

                    # Replace 0 rewards with the average reward during periods when the agent would typically charge
                    # Use the availability pattern from non-telecommuting days
                    for non_tele_episode in non_telecommute_episodes:
                        charging_pattern = results.p_history[non_tele_episode, :, agent_idx] > 0
                        if charging_pattern.any():
                            # Apply this pattern to the telecommuting day
                            telecommute_mask = charging_pattern.copy()
                            interpolated_reward_history[episode, telecommute_mask, agent_idx] = (
                                avg_reward
                            )
                            break

        # Create a new SimulationResults object with the interpolated rewards
        interpolated_results = SimulationResults(
            p_history=results.p_history,
            soc_history=results.soc_history,
            reward_history=interpolated_reward_history,
            price_history=results.price_history,
            congestion_history=results.congestion_history,
            telecommute_history=results.telecommute_history,
            computing_time=results.computing_time,
        )

        if self.config.verbose:
            print("Reward interpolation complete.")

        return interpolated_results

    # Optional utility method for validating loaded configurations
    def validate_loaded_config(self, config):
        """Validate a loaded configuration to ensure all expected fields are present"""
        if isinstance(config, SimulationConfig):
            required_fields = [
                "T_episode_seconds",
                "dt",
                "num_ev_agents",
                "num_episodes",
                "congestion_limit",
                "part_of_telecommuters",
                "strategy_name",
            ]

        elif isinstance(config, EVAgentConfig):
            required_fields = [
                "type",
                "id",
                "T",
                "t_a",
                "t_b",
                "soc_initial",
                "soc_target",
                "ev_config",
                "strategy_config",
            ]

        elif isinstance(config, EVPhysicsConfig):
            required_fields = ["type", "dt", "e_max", "e_min", "p_max", "p_min", "eta_c", "eta_d"]

        elif isinstance(config, StrategyConfig):
            required_fields = ["name"]

        else:
            print(f"Unknown configuration type: {type(config)}")
            return False

        # Check that all required fields are present
        for field in required_fields:
            if not hasattr(config, field):
                print(f"Missing required field '{field}' in {type(config).__name__}")
                return False

        return True


def generate_ev_fleet_uniform_behavior_history(
    t_a_range=[0, None],
    t_b_range=[0, 1],
    soc_init_range=[0, 1],
    soc_target_range=[0, 1],
    num_episodes=0,
    num_ev_agents=0,
    random_seed=42
):
    """
    Pregenerate synthetic arrival, departure, soc_init and soc_target
    history for the whole fleet over the whole experiment.

    SAFETY GUARANTEES:
    - t_a <= t_b
    - soc_init <= soc_target
    """

    np.random.seed(random_seed)

    if t_a_range[1] is None:
        raise ValueError("t_a_range upper bound cannot be None")

    if t_a_range[1] > t_b_range[1]:
        raise ValueError("t_a_range upper bound must be <= t_b_range upper bound")

    if soc_init_range[0] > soc_target_range[1]:
        raise ValueError("soc_init lower bound must be <= soc_target upper bound")

    t_a_history = np.zeros((num_ev_agents, num_episodes))
    t_b_history = np.zeros((num_ev_agents, num_episodes))
    soc_target_history = np.zeros((num_ev_agents, num_episodes))
    soc_init_history = np.zeros((num_ev_agents, num_episodes))

    for ep in range(num_episodes):
        for ev in range(num_ev_agents):

            t_a = np.random.randint(t_a_range[0], t_a_range[1])
            t_b = np.random.randint(max(t_b_range[0], t_a), t_b_range[1])

            t_a_history[ev, ep] = t_a
            t_b_history[ev, ep] = t_b
            soc_target = np.random.uniform(
                soc_target_range[0],
                soc_target_range[1]
            )
            soc_init = np.random.uniform(soc_init_range[0], min(soc_init_range[1], soc_target))
           

            soc_init_history[ev, ep] = soc_init
            soc_target_history[ev, ep] = soc_target

    return t_a_history, t_b_history, soc_init_history, soc_target_history
