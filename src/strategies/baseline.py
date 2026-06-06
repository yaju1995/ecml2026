from dataclasses import dataclass

import numpy as np

from strategies.strategy import Strategy, StrategyConfig


@dataclass
class BaselineConfig(StrategyConfig):
    """Configuration class for Baseline charging strategy parameters."""

    name: str = "GREEDY"

    @classmethod
    def default_config(cls):
        """Return the default configuration for the Baseline strategy."""
        return BaselineConfig(name="GREEDY")


class Baseline(Strategy):
    """Baseline charging strategy implementation."""

    def __init__(self, config: StrategyConfig):
        """Initialize baseline charging strategy parameters."""
        super().__init__(config)

    def reset(self, observed_context: list):
        """Reset strategy state."""

    def act(self, observed_context: np.ndarray) -> int:
        """Select an action given the current context_vector."""
        # print(observed_context)
        n_charge = int(observed_context[6])
        # print(n_charge)
        if n_charge == 0:
            return 0
        else:
            return 1

    def update(self, observed_context: np.array, reward: float):
        """Update strategy context_vector given the observed transition."""


Strategy.register("Baseline", Baseline)
