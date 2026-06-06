from dataclasses import dataclass

import numpy as np

from strategies.strategy import Strategy, StrategyConfig
from DRL import DQNAgent, DQNConfig, DDPGAgent,DDPGConfig
@dataclass
class DRLConfig(StrategyConfig):
    """Configuration class for DRL charging Strategy parameters."""

    T:int
    D:int
    name = "EV_DRL"
    drl_config: DQNConfig
    obs_dim: int
    act_dim: int

    @classmethod
    def default_config(cls):
        return DRLConfig(
            T = 96,    
            D= 400,    
            name = "EV_DRL", #
            drl_config = DQNConfig(),  # Configuration
            obs_dim = 6,
            act_dim = 2,
        )

class DRL(Strategy):
    """DRL Charging strategy implementation"""
    def __init__(self, config:StrategyConfig):
        super().__init__(config)

        self.T = config.T
        self.config = config
        self.DRL_Agent = DQNAgent(name =config.name, 
                               cfg= config.drl_config, 
                               obs_dim=config.obs_dim,
                               n_actions=config.act_dim
                               )
        # self.action_map = 
        self.state= None
        self.action = None
        self.reward = None
        self.next_state = None
        self.terminate = None
        
    def update(self, observed_context, reward):
        """Update strategy state given the observed transition."""
        next_state = self._filter_obs(observed_context)
        t = observed_context[0]
        self.next_state = next_state
        self.reward = reward
        print(f'time: {t} : {self.reward}')
        if self.state is not None and self.action is not None:
            # print(self.state, self.action,self.reward, self.next_state)

            self.DRL_Agent.store_transition(self.state, 
                                            self.action, 
                                            self.reward,
                                            self.next_state,
                                            False)
            
        
        if t == 0:
            self.DRL_Agent.train()
    
    def act(self, context_vector)-> float:
        # Convert Observation dataclass → flat np.array
        self.state = self._filter_obs(context_vector=context_vector)
        self.action = self.DRL_Agent.choose_action(self.state)
        # print(self.action)
        return self.action
    
    def store_transition(self, state, action, reward, state_next):
        pass
    
    def save(self, path):
        self.DRL_Agent.save(path)
    
    def load(self, path):
        self.DRL_Agent.load(path)
    
    # def _get_obs_numpy(self, context_vector)->np.array:
    #     state = np.concatenate([
    #         np.array([context_vector.day], dtype=np.float32),
    #         np.array([context_vector.price_t], dtype=np.float32),
    #         np.array([context_vector.congestion_signal_t], dtype=np.float32),
    #         np.array([context_vector.telecommute], dtype=np.float32),
    #         context_vector.price_day.astype(np.float32),
    #         np.array([context_vector.disconnect_t], dtype=np.float32),
    #     ])

    #     return state
    def _filter_obs(self, context_vector) -> np.ndarray:

        state = np.array([
            context_vector[0],   # Current timestep
            context_vector[1],   # Current power
            context_vector[2],   # Current state of charge
            # context_vector[3], # Current availability status
            # context_vector[4], # Arrival time
            context_vector[5],   # Departure time
            # context_vector[6], # Number of instants needed to charge
            context_vector[7],   # Current day
            # context_vector[8], # Current price_t
            context_vector[9],   # Current congestion signal
            # context_vector[10],# Telecommute status
            # context_vector[11],# Price prevision data for the day
            # context_vector[12],# Wiring status
            # context_vector[13],# Current time
        ], dtype=float)

        return state
    def reset(self, observed_context: list):
            """Reset strategy state."""
Strategy.register("EV_DRL", DRL)
