from dataclasses import dataclass

import numpy as np

from strategies.strategy import Strategy, StrategyConfig


@dataclass
class MilpForecastPriceConfig(StrategyConfig):
    """Configuration class for Baseline charging strategy parameters."""

    name: str = "MILP_Price_ForecastP"

    @classmethod
    def default_config(cls):
        """Return the default configuration for the Baseline strategy."""
        return MilpForecastPriceConfig(name="MILP_Price_Forecast")


class MilpForecastPrice(Strategy):
    """Baseline charging strategy implementation."""

    def __init__(self, config: StrategyConfig):
        """Initialize baseline charging strategy parameters."""
        super().__init__(config)

    def reset(self, observed_context: list):
        """Reset strategy state."""

    def act(self, milp_action: int) -> int:
        """Select an action given the current context_vector."""
        if milp_action == 1:
            return 1
        elif milp_action == 0:
            return 0
        else:
            raise ValueError(f"Invalid action {milp_action} provided.")

    def update(self, observed_context: np.array, reward: float):
        """Update strategy context_vector given the observed transition."""


Strategy.register("MILP_Price_Forecast", MilpForecastPrice)
