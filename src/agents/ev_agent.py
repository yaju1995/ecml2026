from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.agent import Agent, AgentConfig, AgentInfoMixin, AgentSize
from physics.ev_physics import EVPhysics, EVPhysicsConfig
from strategies.strategy import Strategy, StrategyConfig


@dataclass
class EVAgentConfig(AgentConfig):
    """Configuration class for Electric Vehicle agent parameters."""

    # Time window configuration
    T: int  # Time horizon (s)
    t_a: int  # Arrival time (s)
    t_b: int  # Departure time (s)

    # Charging targets configuration
    soc_initial: float  # Initial State of Charge [0,1]
    soc_target: float  # Target State of Charge [0,1]

    # EV hardware configuration
    ev_config: EVPhysicsConfig  # EV physics model configuration

    # Strategy configuration
    strategy_config: StrategyConfig  # Charging strategy configuration

    # Schedule configuration
    telecommuting_days: Optional[List[int]]  # Days of the week to telecommute

    # Loss Generator:
    loss_generator: str # Loss signal computation specificition

    @classmethod
    def default_config(cls):
        """Return the default configuration for the EV agent."""
        return EVAgentConfig(
            id=0,
            type="default EV",
            T=24 * 3600,
            t_a=8 * 3600,
            t_b=18 * 3600,
            soc_initial=0.55,
            soc_target=0.8,
            ev_config=EVPhysicsConfig.default_config(),
            strategy_config=StrategyConfig.default_config(),
            telecommuting_days=[],
            loss_generator = "default",
        )
    
    @classmethod
    def congestion_aware_default_config(cls):
        """Return the default configuration for a congestion aware EV agent."""
        return EVAgentConfig(
            id=0,
            type="default EV",
            T=24 * 3600,
            t_a=8 * 3600,
            t_b=18 * 3600,
            soc_initial=0.55,
            soc_target=0.8,
            ev_config=EVPhysicsConfig.default_config(),
            strategy_config=StrategyConfig.default_config(),
            telecommuting_days=[],
            loss_generator = "congestion_aware",
        )

    @classmethod
    def get_config_dict(cls, config):
        """Get configuration dictionary from configuration object."""
        config_dict = super().get_config_dict(config)
        config_dict["ev_config"] = EVPhysicsConfig.get_config_dict(config.ev_config)
        config_dict["strategy_config"] = StrategyConfig.get_config_dict(config.strategy_config)
        return config_dict


@dataclass
class EVState(AgentSize):
    """State class for Electric Vehicle agent."""

    t: int  # Current timestep
    p: float  # Current power
    soc: float  # Current state of charge
    availability: int  # Current availability status
    t_a: int  # Arrival time (s)
    t_b: int  # Departure time (s)
    n_charge: int  # Number of instants needed to charge


@dataclass
class Observation(AgentSize):
    """Observation class for Electric Vehicle agent."""
    
    day: int  # Current day
    price_t: float  # Current price_t
    congestion_signal_t: int  # Current congestion signal
    telecommute: int  # Telecommute status
    price_day: np.array  # Price prevision data for the day
    disconnect_t : int # Wiring status (True if agents been disconnect by the environment)
    time_day: int # Current time


class EVAgent(Agent, AgentInfoMixin):
    """Electric Vehicle agent implementation."""

    def __init__(self, config: EVAgentConfig):
        """Initialize Electric Vehicle agent parameters."""
        super().__init__(config)

        # Store configuration
        self.config = config

        # Initialize models
        self.ev = EVPhysics(config.ev_config)
        print(config.strategy_config.name)
        self.strategy = Strategy.create(config.strategy_config)

        # Convert time parameters to time steps
        self.dt = config.ev_config.dt
        self.t_a = config.t_a // self.dt
        self.t_b = config.t_b // self.dt
        self.T = config.T // self.dt

        # Initialize state
        self.state = EVState(
            t=0,
            p=0,
            soc=config.soc_initial,
            availability=0,
            t_a=self.t_a,
            t_b=self.t_b,
            n_charge=self.compute_n_charge(),
        )

        # Initialize observation
        self.observation = Observation(
            day=0, price_t=0, congestion_signal_t=0, telecommute=0, price_day=np.zeros(self.T), disconnect_t=0,time_day=0,
        )#

        # History tracking
        self.power_history = []
        self.soc_history = []

        # newly added for DRL 
        self.ins_state = None
        self.ins_action = None
        self.ins_reward = None
        self.ins_next_state = None
        self.ins_terminate = None

    def reset(self, obs) -> np.ndarray:
        """Reset agent state."""
        self.observation = obs
        self.power_history = []
        self.soc_history = []
        day = obs.day
        if day % 7 in self.config.telecommuting_days:
            self.ev.reset(soc=self.config.soc_target)
        else:
            self.ev.reset(soc=self.config.soc_initial)

        self.state = EVState(
            t=0,
            p=0,
            soc=self.config.soc_initial,
            availability=0,
            t_a=self.t_a,
            t_b=self.t_b,
            n_charge=self.compute_n_charge(),
        )

        # state and observations concatenated into a LIST
        observed_context = []
        for element in self.get_state():
            observed_context.append(element)
        for element in self.get_observation():
            observed_context.append(element)

        self.strategy.reset(observed_context=observed_context)

        return self.get_state()

    def compute_n_charge(self) -> int:
        """Compute the number of instants needed to charge."""
        soc = self.ev.soc  # Current SOC
        soc_target = self.soc_target  # Target SOC
        e_max = self.ev.e_max  # Maximum energy
        p_max = self.ev.p_max  # Maximum power
        eta_c = self.ev.eta_c  # Charging efficiency
        dt = self.dt  # Time step

        T_charge = 3600 * (
            e_max * (soc_target - soc) / (p_max * eta_c)
        )  # Time needed to charge to soc_b (s)
        n_charge = int(np.ceil(T_charge / dt))  # Number of instants needed for the charge

        self.n_charge = n_charge
        return self.n_charge

    def compute_availability(self, t: int, day: int) -> int:
        """Compute EV availability given the current time step and day."""
        if self.config.telecommuting_days is not None:
            if day % 7 in self.config.telecommuting_days:
                return 0
            elif self.t_a <= t and t <= self.t_b:
                return 1
        return 0

    def act(self, obs: Observation, milp_action=None) -> float:  # dont think MILP is requied already conver in simulation
        """Select an action given the current state following strategy."""
        # Concatenate state and observation into a numpy array
        # observed_context = np.concatenate([self.get_state(), obs])
        observed_context = []
        for element in self.get_state():
            observed_context.append(element)
        for element in self.get_observation():
            observed_context.append(element)

        # self.ins_state:
        self.ins_state = observed_context # but it has more info than required
        if self.config.strategy_config.name == "MILP" or self.config.strategy_config.name == "MILP_Price_Forecast":
            power = self.strategy.act(milp_action) * self.config.ev_config.p_max
        else:
            if self.state.availability == 0:
                power = 0
            else:
                # print(observed_context)
                action = self.strategy.act(observed_context)
                power =  action* self.config.ev_config.p_max
                # print(f'Act: {action, power}')
        self.state.p = power # what values does the strategies return

        return power

    def update(self, obs: np.array) -> tuple[np.array, float]:
        """Update agent state given the observed power consumption."""
        # Get current price_t and congestion signal
        self.observation = obs
        day = obs.day
        price_t = obs.price_t
        congestion_signal_t = obs.congestion_signal_t
        disconnect_signal_t = obs.disconnect_t

        # Get current power
        real_power = self.state.p

        # Cancel charging if congestion signal is high
        if disconnect_signal_t == 1:

            real_power = 0

        # Update EV state
        real_power, e, soc = self.ev.step(real_power)

        reward = 0.0

        if self.loss_generator != "default":

            if self.loss_generator == "congestion_aware":
                # COMPUTE LOSS (CHARGING DURING CONGESTION PENALIZED)
                if congestion_signal_t == 1: # GRID CONGESTION PENALTY
                    reward = +1.0
                else:
                    reward = ((real_power / self.config.ev_config.p_max) * price_t)
                    # Normalized the power 

        else:

            # COMPUTE LOSS (NO CONGESTION PENALTY, PRICE ONLY)
            reward = ((real_power / self.config.ev_config.p_max) * price_t)

        # print(f'Reward : {reward} ={real_power} * {price_t}') 



                
        # Compute reward (BOUNDED Case, REWARD) (MIXTURE REWARD in [0,1])
        # if real_power > 0:
        #     if congestion_signal_t == 1:
        #         reward = 0.0
        #     else:
        #         reward = 0.7 + (1 - ((real_power / self.config.ev_config.p_max) * price_t))*3/10
        # else:
        #     # if disconnect_signal_t == 1:
        #     #     reward = 0.0
        #     # else :
        #         reward = 0.7

        self.state.p = real_power

        # Update strategy
        # observed_context = np.concatenate([self.get_state(), obs])
        observed_context = []
        for element in self.get_state():
            observed_context.append(element)
        for element in self.get_observation():
            observed_context.append(element)

        # print(self.get_observation())
        # if self.strategy.name == 'EV_DRL':
        #     # self.strategy.update(self.ins_state, self.ins_action, self.ins_reward, self.ins_next_state, self.ins_terminate)
        # else:
        self.strategy.update(observed_context, reward)

        # Update state
        self.state.soc = soc
        self.state.t += 1
        self.state.availability = self.compute_availability(self.state.t, day)
        self.state.n_charge = self.compute_n_charge()

        # Record history
        
        self.power_history.append(real_power)
        self.soc_history.append(soc)

        
       
        return (self.get_state(), reward)

    def get_history(self) -> Tuple[np.array, np.array]:
        """Return the recorded power and SOC history."""
        return np.array(self.power_history), np.array(self.soc_history)

    def get_state(self) -> np.array:
        """Return the current agent state as numpy array."""
        return np.array(
            [
                self.state.t,
                self.state.p,
                self.state.soc,
                self.state.availability,
                self.state.t_a,
                self.state.t_b,
                self.state.n_charge, # what they refer to 
            ]
        )

    def get_observation(self) -> list:
        """Return the current observation as numpy array."""
        return [
            self.observation.day,
            self.observation.price_t,
            self.observation.congestion_signal_t,
            self.observation.telecommute,
            self.observation.price_day,
            self.observation.disconnect_t,

        ]

    def get_state_info(self) -> Dict[str, Any]:
        """Get current agent state information."""
        return {
            "t": self.state.t,
            "p": self.state.p,
            "soc": self.state.soc,
            "availability": self.state.availability,
        }

    def get_config_info(self) -> Dict[str, Any]:
        """Get agent configuration information, excluding base fields."""
        config_dict = super().get_config_info()
        # Remove type and id since they're already in core info
        config_dict.pop("type", None)
        config_dict.pop("id", None)
        return config_dict

    def get_observation_info(self) -> Dict[str, Any]:
        """Get current observation information."""
        return {
            "day": self.observation.day,
            "price_t": self.observation.price_t,
            "congestion_signal_t": self.observation.congestion_signal_t,
        }

    def set_state(self, state: np.array):
        """Set the current agent state from numpy array."""
        self.state.t = state[0]
        self.state.p = state[1]
        self.state.soc = state[2]
        self.state.availability = state[3]
