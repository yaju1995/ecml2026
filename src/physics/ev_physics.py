from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EVPhysicsConfig:
    """Configuration class for Electric Vehicle physics parameters."""

    # Agent parameters
    type: str  # EV type
    # Time parameters
    dt: int  # Time step (s)
    # EV parameters
    e_max: float  # Maximum energy (Wh)
    e_min: float  # Minimum energy (Wh)
    p_max: float  # Maximum power (W)
    p_min: float  # Minimum power (W)
    eta_c: float  # Charging efficiency
    eta_d: float  # Discharging efficiency

    @classmethod
    def from_string(cls, config_str):
        return cls(
            **{
                k: (
                    int(v)
                    if k not in ["type", "eta_c", "eta_d"]
                    else (float(v) if k in ["eta_c", "eta_d"] else v.strip("'"))
                )
                for k, v in [
                    item.strip().split("=")
                    for item in config_str.replace("EVPhysicsConfig(", "")
                    .replace(")", "")
                    .split(",")
                ]
            }
        )

    @classmethod
    def default_config(cls):
        """Return the default configuration for the EV physics model."""
        return EVPhysicsConfig(
            type="default",
            dt=15 * 60,
            e_max=52 * 1000,
            e_min=0,
            p_max=7 * 1000,
            p_min=0,
            eta_c=1,
            eta_d=1,
        )

    @classmethod
    def get_config_dict(cls, config):
        """Return the configuration as a dictionary"""
        return config.__dict__


class EVPhysics:
    """Physical EV model implementation."""

    def __init__(self, config: EVPhysicsConfig):
        """Initialize Electric Vehicle physics parameters."""
        self._set_config(config)

        # State variables
        self.t = 0  # Current time step
        self.e = 0  # Current energy (Wh)
        self.soc = 0  # Current SOC [0,1]

        # History tracking
        self.e_history = []  # Energy history (Wh)
        self.p_history = []  # Power history (W)
        self.soc_history = []  # SOC history [0,1]

    def _set_config(self, config: EVPhysicsConfig):
        """Set configuration parameters."""
        for key, value in vars(config).items():
            setattr(self, key, value)

    def reset(self, soc: float, seed: Optional[int] = None):
        """Reset agent state."""
        self.t = 0
        self.e = self.e_max * soc
        self.soc = soc

    def step(self, p: float) -> np.ndarray:
        """Update EV state given the observed power consumption."""
        # Update time step
        self.t += 1

        # Clip with bounds
        if self.e >= self.e_max and p > 0:
            p = 0
        elif self.e <= self.e_min and p < 0:
            p = 0
        elif self.eta_c * p * (self.dt / 3600) > self.e_max - self.e:
            p = (self.e_max - self.e) * 3600 / (self.eta_c * self.dt)

        p = np.clip(p, self.p_min, self.p_max)

        # Update energy
        self.e += self.eta_c * p * self.dt / 3600

        # Clip with bounds
        self.e = np.clip(self.e, self.e_min, self.e_max)

        # Update SOC
        self.soc = self.e / self.e_max

        # Update history
        self.p_history.append(p)
        self.e_history.append(self.e)
        self.soc_history.append(self.soc)

        return np.array([p, self.e, self.soc])
