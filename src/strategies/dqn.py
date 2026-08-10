from dataclasses import dataclass, field
import os
import numpy as np
from utils.multi_plotter import MultiLivePlotter
from strategies.strategy import Strategy, StrategyConfig
from DRL import DQNAgent, DQNConfig, DDPGAgent,DDPGConfig

def minute_to_sin_cos(value, resolution):
    """
    Convert a time index (0 .. resolution-1) into sin/cos cyclical encoding.
    
    Parameters
    ----------
    value : int or array-like
        The timestep index (e.g., minute index).
    resolution : int
        Number of steps in one full cycle (e.g., 96 for 15-min steps).
    """
    value = np.asarray(value)
    angle = 2 * np.pi * (value / resolution)
    return np.sin(angle), np.cos(angle)


from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from utils.multi_plotter import MultiLivePlotter
from strategies.strategy import Strategy, StrategyConfig
from DRL import DQNAgent, DQNConfig, ReplayBuffer


Central_buffer = ReplayBuffer(100_000)

@dataclass
class DRLConfig(StrategyConfig):
    """Configuration class for DRL charging strategy parameters."""

    T: int
    D: int
    name: str

    obs_dim: int
    act_dim: int

    # DQN parameters stored flat, not as DQNConfig object
    gamma: float
    lr: float
    buffer_capacity: int
    hidden: Tuple[int, int]
    batch_size: int

    eps_start: float
    eps_end: float
    eps_decay_steps: int

    target_update_every: int
    grad_clip_norm: Optional[float]
    device: str
    seed: Optional[int]
    central_buffer_mode: bool
    replay_buffer: Optional[ReplayBuffer]
    train_mode:str

    @classmethod
    def default_config(cls):
        return cls(
            T=96,
            D=10,
            name="DQN",

            obs_dim=7,
            act_dim=2,

            gamma=0.99,
            lr=1e-4,
            buffer_capacity=100_000,
            hidden=(128, 128),
            batch_size=128,

            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=20_000,

            target_update_every=1000,
            grad_clip_norm=10.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=0,
            # should define the common memory here and 
            # replybuffer = ReplayBuffer(100_000)
            central_buffer_mode = False,
            replay_buffer = Central_buffer,
            train_mode = True
        )

    def to_dqn_config(self):
        return DQNConfig(
            gamma=self.gamma,
            lr=self.lr,
            buffer_capacity=self.buffer_capacity,
            hidden=tuple(self.hidden),
            batch_size=self.batch_size,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_decay_steps=self.eps_decay_steps,
            target_update_every=self.target_update_every,
            grad_clip_norm=self.grad_clip_norm,
            device=self.device,
            seed=self.seed,
            replay_buffer=self.replay_buffer
        )
    

class DRL(Strategy):
    """DRL Charging strategy implementation"""
    def __init__(self, config:StrategyConfig):
        super().__init__(config)

        self.T = config.T
        self.config = config
        
        self.DRL_Agent = DQNAgent(name =config.name, 
                               cfg= config.to_dqn_config(), 
                               obs_dim=config.obs_dim,
                               n_actions=config.act_dim
                               )
        # self.action_map = 
        self.state= None
        self.action = None
        self.reward = None
        self.next_state = None
        self.terminate = None

        # reward evaluation
        self.avg_reward = 0
        self.sum_reward = 0
        self.steps =0
        
        # plotter
        # self.plotter = MultiLivePlotter(n_plots=1) #COMMENTED FOR TESTS (Eymeric)
        self.plotter = None

    def update(self, observed_context, reward):
        """Update strategy state given the observed transition."""
        next_state = self._filter_obs(observed_context)


        t = observed_context[0]
        departure_time = observed_context[5]
        instants_needed = observed_context[6]

        
        self.next_state = next_state
        system_reward = - reward  # to maximized the reward
        
        # Update reward for the SOC 
        # soc excceding 80% we get penalty 
        soc = observed_context[2]
        end_soc_reward = - (0.8-soc)**2

        remaining_time = departure_time -t
        
        if remaining_time > instants_needed:
             slack = instants_needed /remaining_time 
        else: 
            slack = 1
		
        slack = max(0,slack)

        self.reward = slack * end_soc_reward -(1-slack) * system_reward



        # Tracking reward
        self.sum_reward += self.reward
        self.steps +=1
        self.avg_reward = self.sum_reward/self.steps
        
        # self.plotter.update([self.avg_reward])
        # print(f'time: {t} : {self.avg_reward}')
        if self.state is not None and self.action is not None:
            # print(self.state, self.action,self.reward, self.next_state)

            self.DRL_Agent.store_transition(self.state, 
                                            self.action, 
                                            self.reward,
                                            self.next_state,
                                            False)
            
        if self.config.train_mode:
            if t%4 == 0:  # updating every hours
            # if t == 0 : # Update once a day 
                # print(f'updating {t/4} {self.config.train_mode}')
                self.DRL_Agent.train()

        # saving model
        # if observed_context[7] == self.config.D-1:
        #     if t ==95:
        #         self.save('DQN\EV_agent')

    
    def act(self, context_vector)-> float:

        # define Ev connection period: 
        '''
        If EV is connected then enable charging instant and close when EV is disconnected

        '''
        # Convert Observation dataclass → flat np.array
        # print(context_vector)
        self.state = self._filter_obs(context_vector=context_vector)


        # Only take action is ev is connected else not action:
        self.action = self.DRL_Agent.choose_action(self.state)
        # print(self.state[4], self.action)
        # print(self.state)
        return self.action
    
    def save(self,path, filename='model.pth'):
        # create path
        model_path = os.path.join(path, 'Models/')
        os.makedirs(model_path, exist_ok=True)
        file_path = os.path.join(model_path, filename)
        print(self.DRL_Agent.save(file_path))
    
    def load(self, path, filename = 'model.pth'):
        model_path = os.path.join(path, 'Models/')
        file_path = os.path.join(model_path, filename)
        print(self.DRL_Agent.load(file_path))
    
    def _filter_obs(self, context_vector) -> np.ndarray:
        
        # sin_t, cos_t = minute_to_sin_cos(context_vector[0],96)
        # state = np.array([
        #     # context_vector[0],   # Current timestep
        #     sin_t,
        #     cos_t,
        #     context_vector[1],   # Current power
        #     context_vector[2],   # Current state of charge
        #     context_vector[3],   # Current availability status
        #     # context_vector[4], # Arrival time
        #     context_vector[5],   # Departure time in minutes # not remaining
        #     context_vector[6],   # Number of instants needed to charge
        #     # context_vector[7], # Current day
        #     context_vector[8],   # Current price_t
        #     context_vector[9],   # Current congestion signal
        #     # context_vector[10],# Telecommute status
        #     # context_vector[11],# Price prevision data for the day
        #     # context_vector[12],# Wiring status [disconnect status]
        # ], dtype=float)
       
 
        # return state

        t = context_vector[0]          # current timestep index
        sin_t, cos_t = minute_to_sin_cos(t, 96)

        current_power = context_vector[1]
        soc = context_vector[2]
        availability = context_vector[3]
        departure_time = context_vector[5]
        instants_needed = context_vector[6]
        price_t = context_vector[8]
        congestion = context_vector[9]

        remaining_time = departure_time - t
        if remaining_time > instants_needed:
             slack = instants_needed /remaining_time 
        else: 
            slack = 1
		
        slack = max(0,slack)

        # Optional: normalized versions, depending on your preprocessing
        state = np.array([
            sin_t,
            cos_t,
            soc,
            availability,
            slack,
            price_t, # add price for future
            congestion,
            # e.g. target_soc, next_price, min_future_price, etc.
        ], dtype=float)

        return state

    
    def reset(self, observed_context: list):
            """Reset strategy state."""
        # reset the DRL 
        # Reset the memory

Strategy.register("DQN", DRL)
