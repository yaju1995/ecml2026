from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PhysicsConfig:
    """Base configuration class for physics models."""

    type: str  # Model type
    dt: int  # Time step (s)


class FlexibilityPhysics(ABC):
    """Abstract base class for flexibility physics models."""

    def __init__(self, config: PhysicsConfig):
        """Initialize base physics parameters."""
        self._set_config(config)
        self.t = 0  # Current time step

        # History tracking
        self.history = {}

    def _set_config(self, config: PhysicsConfig):
        """Set configuration parameters."""
        for key, value in vars(config).items():
            setattr(self, key, value)

    @abstractmethod
    def reset(self, state: Optional[Any] = None, seed: Optional[int] = None):
        """Reset physics model to initial state."""
        self.t = 0
        self.history = {}

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Update physics model state given an action.

        Args:
            action: The action to apply to the physics model

        Returns:
            Tuple containing:
            - next_state: The updated state after applying the action
            - reward: Any reward associated with the action
            - done: Whether the simulation is complete
            - info: Additional information dictionary
        """
        self.t += 1
        # Child classes should override and implement

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current state as a numpy array."""

    @abstractmethod
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Return the operational bounds of the physics model.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
