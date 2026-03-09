from dataclasses import dataclass

import numpy as np

from strategies.strategy import Strategy, StrategyConfig


@dataclass
class MABTSConfig(StrategyConfig):
    """Configuration class for MAB TS charging strategy parameters."""

    T: int
    # lambda_param: float
    sigma: float  # Adding sigma to config
    name = "MAB TS"

    @classmethod
    def default_config(cls):
        """Return the default configuration for the MAB TS strategy."""
        return MABTSConfig(
            T=int((24 * 3600) / (15 * 60)),
            # lambda_param=1.0,
            sigma=1.00,
            name="Inde-TS",
        )


class MABTS(Strategy):
    """MAB TS charging strategy implementation."""

    def __init__(self, config: StrategyConfig):
        """Initialize baseline charging strategy parameters."""
        super().__init__(config)
        # Store configuration
        self.T = config.T
        self.config = config
        # self.lambda_param = config.lambda_param
        self.sigma = config.sigma

        # Initialize parameters
        self.theta_hat = np.ones(self.T)  # Estimated mean reward
        self.theta_tilde = np.zeros(self.T)
        self.N = np.zeros(self.T)  # Number of times each action was selected
        self.S = np.zeros(self.T)  # Sum of squared rewards for variance estimation
        self.var = np.ones(self.T) * self.sigma**2  # Posterior variance
        self.best_super_arm = np.zeros(self.T)

        self.observed_context_day = None

    def _sample(self):
        """Sample from the posterior distribution."""
        posterior_var = self.var

        self.theta_tilde = np.random.normal(self.theta_hat, np.sqrt(posterior_var))
        return self.theta_tilde

    def _super_arm_space(self, t: int, t_a: int, t_b: int):
        """Define the super arm space."""
        # Define the filter for the super arm space (can't select actions outside
        # the time window, and in the past)
        super_arm_space = np.zeros(self.T)
        super_arm_space[t_a:t_b] = 1
        super_arm_space[t_b:] = 0
        super_arm_space[:t] = 0
        return super_arm_space

    def select_super_arm(self, t: int, t_a: int, t_b: int, n_charge: int):
        """Select the best super arm for minimization.

        Args:
            t: Current time step.
            t_a: Start of allowed charging window.
            t_b: End of allowed charging window.
            n_charge: Number of charging actions to select.

        Returns:
            A binary array of length T indicating selected time steps for charging.
        """
        # Define valid action window
        super_arm_space = self._super_arm_space(t, t_a, t_b)

        # Mask invalid actions with +inf so they are never chosen in minimization
        theta_tilde_filtered = np.where(super_arm_space > 0, self.theta_tilde, np.inf)

        # Select the indices of the n_charge smallest sampled values
        best_super_arm_idx = np.argsort(theta_tilde_filtered)[:n_charge]
        # Build the binary mask of selected actions
        self.best_super_arm = np.zeros(self.T)
        self.best_super_arm[best_super_arm_idx] = 1

        return self.best_super_arm


    def update(self, observed_context: list, reward: float):
        """Update the posterior distribution."""
        t = int(observed_context[0])
        p = observed_context[1]

        if self.best_super_arm[t] > 0:
            # Update the number of times each action was selected
            self.N[t] += 1

            # Update the estimated reward using weighted average
            old_mean = self.theta_hat[t]
            self.theta_hat[t] = ((self.N[t] - 1) * old_mean + reward) / self.N[t]

            # # Update sum of squared rewards for variance estimation
            self.S[t] += reward**2

            # # Update variance estimate
            self.var[t] = (self.sigma**2) / self.N[t]

            # Update variance estimate
            # if self.N[t] > 1:
            #     self.var[t] = (self.S[t] - self.N[t] * self.theta_hat[t]**2) / (self.N[t] - 1)
            # # self.var[t] = max(self.var[t], 0.0)
            # self.var[t] = max(self.var[t], self.sigma**2)  # Ensure variance doesn't
            # go below sigma^2

        # Select super arm for next step
        t_a = int(observed_context[4])
        t_b = int(observed_context[5])
    
        n_charge = int(observed_context[6])
        self.best_super_arm = self.select_super_arm(t, t_a, t_b, n_charge)

    def act(self, observed_context: list) -> int:
        """Select action based on Thompson Sampling."""
        current_t = int(observed_context[0])
        return self.best_super_arm[current_t]

    def reset(self, observed_context: list):
        """Reset for new episode."""
        self.theta_tilde = self._sample()
        self.observed_context_day = observed_context



Strategy.register("MAB TS", MABTS)
