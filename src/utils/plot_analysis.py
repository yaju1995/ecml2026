import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


class PlotAnalysis:
    """Class for plotting and analyzing simulation results"""

    def __init__(self, simulation_results, simulation_config, ev_agents_configs=None):
        """
        Initialize the plotting class with simulation results and configuration.

        Parameters:
            simulation_results: SimulationResults object containing history data
            simulation_config: SimulationConfig object with simulation parameters
            ev_agents_configs: DataFrame containing EV agent configurations
        """
        self.results = simulation_results
        self.config = simulation_config
        self.ev_agents_configs = ev_agents_configs

        # Extract dimensions for convenience
        self.num_episodes = self.config.num_episodes
        self.num_ev_agents = self.config.num_ev_agents
        self.T = self.config.T_episode_seconds // self.config.dt

        # Set figure sizes and styling
        self.figsize_medium = (12, 6)
        self.figsize_large = (14, 8)

        # Set color palettes
        self.palette = sns.color_palette("muted")
        self.telecommuters_color = self.palette[0]  # First color for telecommuters
        self.non_telecommuters_color = self.palette[1]  # Second color for non-telecommuters

        # Set grid style
        plt.rcParams.update(
            {
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
            }
        )

        # Identify telecommuter and non-telecommuter agents
        if ev_agents_configs is not None:
            self.telecommuter_agents = ev_agents_configs[
                ev_agents_configs["telecommuting_days"].apply(
                    lambda x: x is not None and len(x) > 0
                )
            ]["id"].values

            self.non_telecommuter_agents = ev_agents_configs[
                ev_agents_configs["telecommuting_days"].apply(lambda x: x is None or len(x) == 0)
            ]["id"].values
        else:
            # Default assumption based on part_of_telecommuters
            num_telecommuters = int(self.config.num_ev_agents * self.config.part_of_telecommuters)
            self.telecommuter_agents = np.arange(num_telecommuters)
            self.non_telecommuter_agents = np.arange(num_telecommuters, self.config.num_ev_agents)

    # --------------------------
    # Episode Analysis Methods
    # --------------------------

    def plot_episode_power(self, episode=0, figsize=None):
        """
        Plot the total consumed power and the power congestion limit for a selected episode.
        Uses twin axes to separate power values from price values.

        Parameters:
            episode: Episode index to analyze
            figsize: Figure size as (width, height) tuple

        Returns:
            fig, axes: The figure and axes objects (primary and secondary)
        """
        if figsize is None:
            figsize = self.figsize_medium

        fig, ax1 = plt.subplots(figsize=figsize)

        # Calculate time in hours for x-axis
        time_hours = np.arange(0, 24, self.config.dt / 3600)

        # Calculate total consumed power across all agents at each timestep
        total_power = np.sum(self.results.p_history[episode], axis=1) / 1000

        # Create primary axis for power
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Power (kW)", color=self.palette[0])
        ax1.plot(
            time_hours,
            total_power,
            label="Total Consumed Power",
            color=self.palette[0],
            linewidth=2,
        )
        ax1.axhline(
            y=self.config.congestion_limit / 1000,
            color="r",
            linestyle="--",
            label=f"Congestion Limit ({self.config.congestion_limit/1000:.1f} kW)",
        )
        ax1.tick_params(axis="y", labelcolor=self.palette[0])

        # Highlight areas where congestion occurred
        congestion_periods = np.where(total_power > self.config.congestion_limit / 1000)[0]
        if len(congestion_periods) > 0:
            for period in self._group_consecutive_indices(congestion_periods):
                ax1.axvspan(
                    time_hours[period[0]],
                    time_hours[period[-1] + 1],
                    alpha=0.2,
                    color="red",
                    label="Congestion Period" if period is congestion_periods[0] else None,
                )

        # Create secondary axis for price
        ax2 = ax1.twinx()
        ax2.set_ylabel("Price", color=self.palette[2])
        price = self.results.price_history[episode]
        ax2.plot(
            time_hours, price, label="Price", color=self.palette[2], linewidth=1.5, linestyle=":"
        )
        ax2.tick_params(axis="y", labelcolor=self.palette[2])

        # Set x-axis properties
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.set_xlim(0, 24)
        ax1.set_ylim(0, self.config.congestion_limit / 1000 * 1.5)  # Adjusted for better scaling

        # Add minor grid lines
        ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax1.grid(True, alpha=0.3)
        ax1.grid(True, which="minor", alpha=0.1)

        # Title
        ax1.set_title(f"Total Consumed Power and Price - Episode {episode}")

        # Create combined legend with better positioning
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Move legend outside the plot to avoid overlapping with data
        # Option 1: Place in upper left (where there appears to be empty space)
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

        # Alternatively, you could place it outside the plot area:
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_ev_episode_details(self, ev_id=0, episode=0, figsize=None):
        """
        Plot detailed time series data for a selected EV and episode.

        Parameters:
            ev_id: EV agent ID to analyze
            episode: Episode index to analyze
            figsize: Figure size as (width, height) tuple

        Returns:
            fig, ax: The figure and axis objects
        """
        if figsize is None:
            figsize = self.figsize_medium

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate time in hours for x-axis
        time_hours = np.arange(0, 24, self.config.dt / 3600)

        # Extract data for the specified EV and episode
        power = self.results.p_history[episode, :, ev_id]
        soc = self.results.soc_history[episode, :, ev_id]
        price = self.results.price_history[episode]
        reward = self.results.reward_history[episode, :, ev_id]
        congestion = self.results.congestion_history[episode, :, ev_id]

        # Normalize power for better visualization (if power is non-zero)
        max_power = np.max(power) if np.max(power) > 0 else 1
        normalized_power = power / max_power

        # Check if this is a telecommuting agent
        is_telecommuter = ev_id in self.telecommuter_agents

        # Plot normalized power
        ax.step(
            time_hours,
            normalized_power,
            label=f"Power (normalized, max={max_power:.0f}W)",
            where="pre",
            linewidth=2,
            color=self.palette[0],
        )

        # Plot SOC
        ax.plot(time_hours, soc, label="State of Charge (SOC)", linewidth=2, color=self.palette[1])

        # Plot normalized price for reference
        norm_price = price / np.max(price) if np.max(price) > 0 else price
        ax.plot(
            time_hours,
            norm_price,
            label="Price (normalized)",
            linestyle=":",
            linewidth=1.5,
            color=self.palette[2],
        )

        # Highlight congestion periods for this EV
        congestion_periods = np.where(congestion > 0)[0]
        if len(congestion_periods) > 0:
            for period in self._group_consecutive_indices(congestion_periods):
                ax.axvspan(
                    time_hours[period[0] - 1],
                    time_hours[period[-1]],
                    alpha=0.2,
                    color="red",
                    label="Congestion Period" if period is congestion_periods[0] else None,
                )

        # If we have EV agent configs, add arrival/departure markers
        if self.ev_agents_configs is not None:
            agent_config = self.ev_agents_configs[self.ev_agents_configs["id"] == ev_id].iloc[0]

            # Add arrival and departure times if available
            if "t_a" in agent_config:
                # Use t_a from agent_config
                t_a_hours = agent_config["t_a"] / 3600
                # ax.axvline(x=t_a_hours, color='green', linestyle='--',
                #   label='Arrival Time', alpha=0.7)
                ax.axvspan(0, t_a_hours, alpha=0.1, color="gray")
                ax.axvline(x=t_a_hours, color="darkgreen", linestyle="--", alpha=0.5)

            if "t_b" in agent_config:
                # Use t_b from agent_config
                t_b_hours = agent_config["t_b"] / 3600
                # ax.axvline(x=t_b_hours, color='red', linestyle='--',
                #   label='Departure Time', alpha=0.7)
                ax.axvspan(t_b_hours, 24, alpha=0.1, color="gray")
                ax.axvline(x=t_b_hours, color="darkred", linestyle="--", alpha=0.5)

            # Add SOC target if available
            if "soc_target" in agent_config:
                soc_target = agent_config["soc_target"]
                ax.axhline(
                    y=soc_target,
                    color="black",
                    linestyle="--",
                    label=f"SOC Target ({soc_target:.2f})",
                    alpha=0.7,
                )

            """Add text annotations and arrows"""
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)

            # Time labels remain mathematical notation
            y_text_bottom = -0.15
            for time, label, color in [
                (t_a_hours, "t_a", "darkgreen"),
                (t_b_hours, "t_b", "darkred"),
            ]:
                plt.text(
                    time,
                    y_text_bottom,
                    f"${label}$={time:.0f}h",
                    ha="center",
                    va="top",
                    bbox=bbox_props,
                    color=color,
                    fontweight="bold",
                )

            # SOC annotations (mathematical notation stays the same in both languages)
            soc_at_ta = soc[int(agent_config["t_a"] // self.config.dt)]
            soc_at_tb = soc[int(agent_config["t_b"] // self.config.dt)]

            annotations = [
                (t_a_hours, soc_at_ta, "darkgreen", (10, 10), r"$SOC(t_a) = $"),
                (t_b_hours, soc_at_tb, "darkred", (10, -20), r"$SOC(t_b) = $"),
            ]

            for x, y, color, offset, text in annotations:
                plt.annotate(
                    f"{text}{y:.2f}",
                    xy=(x, y),
                    xytext=offset,
                    textcoords="offset points",
                    bbox=bbox_props,
                    color=color,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
                )

        # Add grid, labels, and legend
        ax.grid(True)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Normalized Values")

        telecommuter_status = "Telecommuter" if is_telecommuter else "Non-telecommuter"
        ax.set_title(f"EV {ev_id} ({telecommuter_status}) - Episode {episode}")

        ax.legend(loc="best")

        # Set x-axis ticks to show hours
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlim(0, 24)

        # Set y-axis limits
        ax.set_ylim(-0.05, 1.1)

        # Add minor grid lines
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True, which="minor", alpha=0.1)

        plt.tight_layout()
        return fig, ax

    # --------------------------
    # Training Analysis Methods
    # --------------------------
    def plot_cumulative_reward_by_group(self, figsize=None):
        """
        Plot the average cumulative reward for EVs grouped by telecommuters and non-telecommuters.

        Parameters:
            figsize: Figure size as (width, height) tuple

        Returns:
            fig, ax: The figure and axis objects
        """
        if figsize is None:
            figsize = self.figsize_medium

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate episode-wise cumulative rewards
        # For each episode, sum rewards across all timesteps for each agent
        episode_rewards = np.sum(
            self.results.reward_history, axis=1
        )  # Shape: (num_episodes, num_ev_agents)

        # Separate telecommuters and non-telecommuters
        telecommuter_rewards = episode_rewards[:, self.telecommuter_agents]
        non_telecommuter_rewards = episode_rewards[:, self.non_telecommuter_agents]

        # Calculate mean rewards for each group per episode
        mean_telecommuter_rewards = (
            np.mean(telecommuter_rewards, axis=1)
            if len(self.telecommuter_agents) > 0
            else np.zeros(self.num_episodes)
        )
        mean_non_telecommuter_rewards = (
            np.mean(non_telecommuter_rewards, axis=1)
            if len(self.non_telecommuter_agents) > 0
            else np.zeros(self.num_episodes)
        )

        # Calculate raw cumulative sums
        raw_cum_telecommuter_rewards = np.cumsum(mean_telecommuter_rewards)
        raw_cum_non_telecommuter_rewards = np.cumsum(mean_non_telecommuter_rewards)

        # Calculate average cumulative rewards by dividing by episode number
        episodes = np.arange(1, self.num_episodes + 1)  # Start from 1 to avoid division by zero
        cum_telecommuter_rewards = raw_cum_telecommuter_rewards / episodes
        cum_non_telecommuter_rewards = raw_cum_non_telecommuter_rewards / episodes

        # Plot average cumulative rewards
        episode_indices = np.arange(self.num_episodes)  # 0-based indices for plotting
        ax.plot(
            episode_indices,
            cum_telecommuter_rewards,
            label=f"Telecommuters (n={len(self.telecommuter_agents)})",
            linewidth=2,
            color=self.telecommuters_color,
        )
        ax.plot(
            episode_indices,
            cum_non_telecommuter_rewards,
            label=f"Non-telecommuters (n={len(self.non_telecommuter_agents)})",
            linewidth=2,
            color=self.non_telecommuters_color,
        )

        # Calculate and plot confidence intervals for each group
        if len(self.telecommuter_agents) > 1:
            # Calculate cumulative rewards for each agent
            cum_rewards_by_agent = np.cumsum(telecommuter_rewards, axis=0)

            # Convert to average cumulative rewards
            avg_cum_rewards_by_agent = np.zeros_like(cum_rewards_by_agent)
            for i in range(self.num_episodes):
                avg_cum_rewards_by_agent[i, :] = cum_rewards_by_agent[i, :] / (i + 1)

            # Calculate standard deviation of the averages
            std_cum_rewards = np.std(avg_cum_rewards_by_agent, axis=1)

            ax.fill_between(
                episode_indices,
                cum_telecommuter_rewards - std_cum_rewards,
                cum_telecommuter_rewards + std_cum_rewards,
                alpha=0.2,
                color=self.telecommuters_color,
            )

        if len(self.non_telecommuter_agents) > 1:
            # Calculate cumulative rewards for each agent
            cum_rewards_by_agent = np.cumsum(non_telecommuter_rewards, axis=0)

            # Convert to average cumulative rewards
            avg_cum_rewards_by_agent = np.zeros_like(cum_rewards_by_agent)
            for i in range(self.num_episodes):
                avg_cum_rewards_by_agent[i, :] = cum_rewards_by_agent[i, :] / (i + 1)

            # Calculate standard deviation of the averages
            std_cum_rewards = np.std(avg_cum_rewards_by_agent, axis=1)

            ax.fill_between(
                episode_indices,
                cum_non_telecommuter_rewards - std_cum_rewards,
                cum_non_telecommuter_rewards + std_cum_rewards,
                alpha=0.2,
                color=self.non_telecommuters_color,
            )

        # Add grid, labels, and legend
        ax.grid(True)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Cumulative Reward")
        ax.set_title("Average Cumulative Rewards by Agent Group")
        ax.legend(loc="best")

        # Set x-axis ticks
        ax.set_xticks(np.arange(0, self.num_episodes + 1, max(1, self.num_episodes // 10)))
        ax.set_xlim(0, self.num_episodes - 1)

        plt.tight_layout()
        return fig, ax

    def plot_aggregated_cumulative_reward(self, figsize=None):
        """
        Plot the aggregated (mean) average cumulative reward across all EVs.

        Parameters:
            figsize: Figure size as (width, height) tuple

        Returns:
            fig, ax: The figure and axis objects
        """
        if figsize is None:
            figsize = self.figsize_medium

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate episode-wise rewards, averaged across all agents
        episode_rewards = np.sum(
            self.results.reward_history, axis=1
        )  # (num_episodes, num_ev_agents)
        mean_episode_rewards = np.mean(episode_rewards, axis=1)  # (num_episodes,)

        # Calculate raw cumulative mean rewards
        raw_cum_mean_rewards = np.cumsum(mean_episode_rewards)

        # Calculate average cumulative rewards by dividing by episode number
        episodes = np.arange(1, self.num_episodes + 1)  # Start from 1 to avoid division by zero
        cum_mean_rewards = raw_cum_mean_rewards / episodes

        # Plot average cumulative mean rewards
        episode_indices = np.arange(self.num_episodes)  # 0-based indices for plotting
        ax.plot(
            episode_indices, cum_mean_rewards, label="All Agents", linewidth=2.5, color="darkblue"
        )

        # Calculate and plot confidence interval
        # First get raw cumulative rewards for each agent
        cum_rewards_by_agent = np.cumsum(episode_rewards, axis=0)  # (num_episodes, num_ev_agents)

        # Convert to average cumulative rewards
        avg_cum_rewards_by_agent = np.zeros_like(cum_rewards_by_agent)
        for i in range(self.num_episodes):
            avg_cum_rewards_by_agent[i, :] = cum_rewards_by_agent[i, :] / (i + 1)

        # Calculate standard deviation of the averages
        std_cum_rewards = np.std(avg_cum_rewards_by_agent, axis=1)  # (num_episodes,)

        ax.fill_between(
            episode_indices,
            cum_mean_rewards - std_cum_rewards,
            cum_mean_rewards + std_cum_rewards,
            alpha=0.2,
            color="blue",
            label="±1 Std Dev",
        )

        # Add best and worst agents (based on their average cumulative reward at the end)
        best_agent_idx = np.argmax(avg_cum_rewards_by_agent[-1, :])
        worst_agent_idx = np.argmin(avg_cum_rewards_by_agent[-1, :])

        ax.plot(
            episode_indices,
            avg_cum_rewards_by_agent[:, best_agent_idx],
            label=f"Best Agent (ID: {best_agent_idx})",
            linestyle="--",
            color="green",
            alpha=0.7,
        )
        ax.plot(
            episode_indices,
            avg_cum_rewards_by_agent[:, worst_agent_idx],
            label=f"Worst Agent (ID: {worst_agent_idx})",
            linestyle="--",
            color="red",
            alpha=0.7,
        )

        # Add grid, labels, and legend
        ax.grid(True)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Cumulative Reward")
        ax.set_title("Average Cumulative Reward (Mean Across All Agents)")
        ax.legend(loc="best")

        # Set x-axis ticks
        ax.set_xticks(np.arange(0, self.num_episodes + 1, max(1, self.num_episodes // 10)))
        ax.set_xlim(0, self.num_episodes - 1)

        plt.tight_layout()
        return fig, ax

    def plot_soc_target_difference(self, episode_range=None, figsize=None):
        """
        Plot relevant statistics on the difference between target SOC and actual SOC at episode end.

        Parameters:
            episode_range: Tuple (start, end) for episode range to analyze, or None for all episodes
            figsize: Figure size as (width, height) tuple

        Returns:
            fig, ax: The figure and axis objects
        """
        if figsize is None:
            figsize = self.figsize_large

        if episode_range is None:
            start_episode, end_episode = 0, self.num_episodes
        else:
            start_episode, end_episode = episode_range

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Get SOC at the end of each episode for each agent
        end_soc = self.results.soc_history[
            start_episode:end_episode, -1, :
        ]  # (episode_range, num_ev_agents)

        # Get target SOC for each agent
        if self.ev_agents_configs is not None:
            target_soc = np.array(
                [
                    self.ev_agents_configs[self.ev_agents_configs["id"] == i]["soc_target"].values[
                        0
                    ]
                    for i in range(self.num_ev_agents)
                ]
            )
        else:
            # Default assumption: target SOC is 0.8 for all agents
            target_soc = np.full(self.num_ev_agents, 0.8)

        # Calculate SOC difference (actual-target)
        soc_diff = np.zeros_like(end_soc)
        for i in range(end_soc.shape[0]):
            soc_diff[i, :] = end_soc[i, :] - target_soc

        # Calculate statistics over episodes
        mean_diff = np.mean(soc_diff, axis=1)  # Mean across agents for each episode
        std_diff = np.std(soc_diff, axis=1)  # Std across agents for each episode
        max_diff = np.max(soc_diff, axis=1)  # Max across agents for each episode
        min_diff = np.min(soc_diff, axis=1)  # Min across agents for each episode

        # Plot 1: Mean and std dev of SOC difference over episodes
        episodes = np.arange(start_episode, end_episode)
        ax1.plot(episodes, mean_diff, label="Mean SOC Difference", linewidth=2, color="blue")

        ax1.fill_between(
            episodes, min_diff, max_diff, alpha=0.1, color="gray", label="Min-Max Range"
        )

        # Add zero line for reference
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add grid, labels, and legend for plot 1
        ax1.grid(True)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("SOC Difference (Actual - Target)")
        ax1.set_title("SOC Target vs. Actual Difference Over Episodes")
        ax1.legend(loc="best")

        # Set x-axis ticks for plot 1
        ax1.set_xticks(
            np.arange(start_episode, end_episode + 1, max(1, (end_episode - start_episode) // 10))
        )
        ax1.set_xlim(start_episode, end_episode - 1)
        ax1.set_ylim(-0.1, 0.1)

        # Plot 2: Distribution of SOC differences for the last episode
        last_episode_diff = soc_diff[-1, :]

        # Plot distribution using kernel density estimation
        # Manage if there is no variation in the data
        if np.std(last_episode_diff) == 0:
            # If there is no variation, we can't use kdeplot
            sns.histplot(
                data=last_episode_diff,
                ax=ax2,
                label="All Agents",
                color="darkblue",
                linewidth=2,
                kde=False,
                bins=1000,
            )
        else:
            sns.kdeplot(
                data=last_episode_diff,
                ax=ax2,
                label="All Agents",
                color="darkblue",
                linewidth=2,
                warn_singular=False,
            )

        # Add zero line for reference
        ax2.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        # Add grid, labels, and legend for plot 2
        ax2.grid(True)
        ax2.set_xlabel("SOC Difference (Actual - Target)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Distribution of SOC Differences in Episode {end_episode-1}")
        ax2.set_xlim(-0.1, 0.1)
        ax2.legend(loc="best")

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_reward_differences(self, episode_range=None, figsize=None):
        """
        Plot statistics on differences between agent rewards for a range of episodes.

        Parameters:
            episode_range: Tuple (start, end) for episode range to analyze, or None for all episodes
            figsize: Figure size as (width, height) tuple

        Returns:
            fig, axes: The figure and axis objects
        """
        if figsize is None:
            figsize = self.figsize_large

        if episode_range is None:
            start_episode, end_episode = 0, self.num_episodes
        else:
            start_episode, end_episode = episode_range

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Calculate episode rewards for each agent
        episode_rewards = np.sum(
            self.results.reward_history[start_episode:end_episode], axis=1
        )  # (episode_range, num_ev_agents)

        # Calculate statistics over episodes
        mean_reward = np.mean(episode_rewards, axis=1)  # Mean across agents for each episode
        std_reward = np.std(episode_rewards, axis=1)  # Std across agents for each episode
        max_reward = np.max(episode_rewards, axis=1)  # Max across agents for each episode
        min_reward = np.min(episode_rewards, axis=1)  # Min across agents for each episode

        # Separate telecommuters and non-telecommuters
        telecommuter_rewards = (
            episode_rewards[:, self.telecommuter_agents]
            if len(self.telecommuter_agents) > 0
            else np.array([])
        )
        non_telecommuter_rewards = (
            episode_rewards[:, self.non_telecommuter_agents]
            if len(self.non_telecommuter_agents) > 0
            else np.array([])
        )

        mean_telecommuter_rewards = (
            np.mean(telecommuter_rewards, axis=1)
            if len(self.telecommuter_agents) > 0
            else np.zeros(end_episode - start_episode)
        )
        mean_non_telecommuter_rewards = (
            np.mean(non_telecommuter_rewards, axis=1)
            if len(self.non_telecommuter_agents) > 0
            else np.zeros(end_episode - start_episode)
        )

        # Plot 1: Mean and std dev of rewards over episodes
        episodes = np.arange(start_episode, end_episode)
        ax1.plot(episodes, mean_reward, label="Mean Reward", linewidth=2, color="darkblue")
        ax1.fill_between(
            episodes,
            mean_reward - std_reward,
            mean_reward + std_reward,
            alpha=0.2,
            color="blue",
            label="±1 Std Dev",
        )
        ax1.fill_between(
            episodes, min_reward, max_reward, alpha=0.1, color="gray", label="Min-Max Range"
        )

        # Plot separate means for telecommuters and non-telecommuters
        if len(self.telecommuter_agents) > 0:
            ax1.plot(
                episodes,
                mean_telecommuter_rewards,
                label="Telecommuters Mean",
                linestyle="--",
                color=self.telecommuters_color,
            )

        if len(self.non_telecommuter_agents) > 0:
            ax1.plot(
                episodes,
                mean_non_telecommuter_rewards,
                label="Non-telecommuters Mean",
                linestyle="--",
                color=self.non_telecommuters_color,
            )

        # Add grid, labels, and legend for plot 1
        ax1.grid(True)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Agent Rewards Over Episodes")
        ax1.legend(loc="best")

        # Set x-axis ticks for plot 1
        ax1.set_xticks(
            np.arange(start_episode, end_episode + 1, max(1, (end_episode - start_episode) // 10))
        )
        ax1.set_xlim(start_episode, end_episode - 1)

        # Plot 2: Distribution of rewards for the last episode
        last_episode_rewards = episode_rewards[-1, :]

        # Separate telecommuters and non-telecommuters for the last episode
        last_telecommuter_rewards = (
            last_episode_rewards[self.telecommuter_agents]
            if len(self.telecommuter_agents) > 0
            else np.array([])
        )
        last_non_telecommuter_rewards = (
            last_episode_rewards[self.non_telecommuter_agents]
            if len(self.non_telecommuter_agents) > 0
            else np.array([])
        )

        # Plot distribution using kernel density estimation
        sns.kdeplot(
            data=last_episode_rewards, ax=ax2, label="All Agents", color="darkblue", linewidth=2
        )

        if len(self.telecommuter_agents) > 0:
            sns.kdeplot(
                data=last_telecommuter_rewards,
                ax=ax2,
                label="Telecommuters",
                color=self.telecommuters_color,
            )

        if len(self.non_telecommuter_agents) > 0:
            sns.kdeplot(
                data=last_non_telecommuter_rewards,
                ax=ax2,
                label="Non-telecommuters",
                color=self.non_telecommuters_color,
            )

        # Add grid, labels, and legend for plot 2
        ax2.grid(True)
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Distribution of Rewards in Episode {end_episode-1}")
        ax2.legend(loc="best")

        plt.tight_layout()
        return fig, (ax1, ax2)

    # --------------------------
    # Helper Methods
    # --------------------------

    def _group_consecutive_indices(self, indices):
        """
        Group consecutive indices into sublists.

        Parameters:
            indices: Array of indices to group

        Returns:
            List of lists, where each sublist contains consecutive indices
        """
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]

        if current_group:
            groups.append(current_group)

        return groups
