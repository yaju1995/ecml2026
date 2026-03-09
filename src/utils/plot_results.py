import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Dictionary for text translations
TRANSLATIONS_PLOT_AGENT = {
    "en": {
        "time": "Time (hours)",
        "norm_values": "Normalized values",
        "title": "Normalized power and $SOC$ for $EV_{id}$ charging simulation for day ${day}$",
        "power": "Power",
        "soc_target": "SOC target",
    },
    "fr": {
        "time": "Temps (heures)",
        "norm_values": "Valeurs normalisées",
        "title": "Puissance normalisée et $SOC$ pour la simulation de charge $EV_{id}$ au jour ${day}$",
        "power": "Puissance",
        "soc_target": "SOC cible",
    },
}

TRANSLATIONS_PLOT_EV_PHYSICS = {
    "en": {
        "time": "Time (hours)",
        "norm_values": "Normalized values",
        "title": "Normalized power and $SOC$ for EV charging simulation",
        "power": "Power",
        "soc_target": "SOC target",
    },
    "fr": {
        "time": "Temps (heures)",
        "norm_values": "Valeurs normalisées",
        "title": "Puissance normalisée et $SOC$ pour la simulation de charge de VE",
        "power": "Puissance",
        "soc_target": "SOC cible",
    },
}


SIZE = (12, 5)


def create_ev_physics_charging_plot(power_history, soc_history, config, ev, language="en"):
    """
    Create a visualization of EV charging data.

    Parameters:
        power_history (np.array): Array of power values over time
        soc_history (np.array): Array of State of Charge values over time
        config: Configuration object containing t_a, t_b, soc_b, dt
        ev: EV object containing T parameter
        language (str): Language for labels ('en' for English, 'fr' for French)
    """
    # Validate language selection
    if language not in TRANSLATIONS_PLOT_EV_PHYSICS:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are: {list(TRANSLATIONS_PLOT_EV_PHYSICS.keys())}"
        )

    texts = TRANSLATIONS_PLOT_EV_PHYSICS[language]

    # Setup
    fig, ax = plt.subplots(figsize=SIZE)
    time = np.arange(0, 24, config.dt / 3600)

    max_power = np.max(power_history) if np.max(power_history) > 0 else 1

    def plot_main_curves():
        """Plot the main power and SOC curves"""
        plt.step(time, power_history / max_power, label=texts["power"], where="pre")
        plt.plot(time, soc_history, label="SOC")

    def setup_grid_and_labels():
        """Setup grid, ticks, labels, and legend"""
        plt.xticks(np.arange(0, 24 + 1, 2))
        ax.xaxis.set_minor_locator(MultipleLocator(1 / 4))
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.2)

        plt.xlabel(texts["time"])
        plt.ylabel(texts["norm_values"])
        plt.title(texts["title"])
        plt.legend()

        plt.tight_layout()

    # Execute plotting functions
    plot_main_curves()
    setup_grid_and_labels()

    return fig, ax


def analyze_charging_periods(power_history):
    """Find and return charging periods"""
    return np.where(power_history != 0)[0]


def create_ev_agent_charging_plot(
    power_history, soc_history, ev_agent_config, day=0, language="en", t_a = [], t_b = [], price=None, reward=None
):
    """
    Create a visualization of EV charging data.

    Parameters:
        power_history (np.array): Array of power values over time
        soc_history (np.array): Array of State of Charge values over time
        config: Configuration object containing t_a, t_b, soc_b, dt
        ev_agent: EV object containing T parameter
        language (str): Language for labels ('en' for English, 'fr' for French)
    """
    # Validate language selection
    if language not in TRANSLATIONS_PLOT_AGENT:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are: {list(TRANSLATIONS_PLOT_AGENT.keys())}"
        )

    texts = TRANSLATIONS_PLOT_AGENT[language]

    # Variables
    config = ev_agent_config
    T = config.T
    dt = config.ev_config.dt

    print(T)
    print(dt)

    soc_target = config.soc_target

    # Setup
    fig, ax = plt.subplots(figsize=SIZE)
    time = np.arange(0, 24, (dt / 3600))
    # Filter data for the selected day
    power_history = power_history[day]
    soc_history = soc_history[day]
    if price is not None:
        price = price[day]
    if reward is not None:
        reward = reward[day]

    max_power = np.max(power_history) if np.max(power_history) > 0 else 1

    # Convert times to hours
    t_a_hours = float(t_a[day] * float(dt)) / 3600.0
    t_b_hours = float((t_b[day] * float(dt))) / 3600.0
    def plot_main_curves():
        """Plot the main power and SOC curves"""
        plt.step(time, power_history / max_power, label=texts["power"], where="pre")
        plt.plot(time, soc_history, label="SOC")
        plt.plot(
            time,
            soc_target * np.ones(time.shape[0]),
            label=texts["soc_target"],
            linestyle="dashed",
            color="black",
        )

    def add_time_regions():
        """Add shaded regions and time markers"""
        plt.axvspan(0, t_a_hours, alpha=0.1, color="gray")
        plt.axvspan(t_b_hours, 24, alpha=0.1, color="gray")

        plt.axvline(x=t_a_hours, color="darkgreen", linestyle="--", alpha=0.5)
        plt.axvline(x=t_b_hours, color="darkred", linestyle="--", alpha=0.5)

    def add_annotations():
        """Add text annotations and arrows"""
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)

        # Time labels remain mathematical notation
        y_text_bottom = -0.15
        for time, label, color in [(t_a_hours, "t_a", "darkgreen"), (t_b_hours, "t_b", "darkred")]:
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
        # print(config.t_a)
        # print(dt)
        # print(config.t_a // dt)
        soc_at_ta = soc_history[int(t_a[day])]
        soc_at_tb = soc_history[int(t_b[day])]

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

    def setup_grid_and_labels():
        """Setup grid, ticks, labels, and legend"""
        plt.xticks(np.arange(0, 24 + 1, 2))
        ax.xaxis.set_minor_locator(MultipleLocator(dt / 3600))
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.2)

        plt.xlabel(texts["time"])
        plt.ylabel(texts["norm_values"])
        plt.title(texts["title"].replace("{id}", str(config.id)).replace("{day}", str(day)))
        plt.legend()

        plt.tight_layout()

    def setup_limits():
        """Setup limits for the plot"""
        plt.ylim(-1, 1.1)
        plt.xlim(0, 24)

    def plot_price_curve(price):
        """Plot the price curve"""
        plt.plot(time, price, label="Price", color="red")

    def plot_reward_curve(reward):
        """Plot the reward curve"""
        plt.plot(time, reward, label="Reward", color="blue")

    # Execute plotting functions
    plot_main_curves()
    if price is not None:
        plot_price_curve(price)

    if reward is not None:
        plot_reward_curve(reward)

    add_time_regions()
    add_annotations()
    setup_grid_and_labels()
    setup_limits()

    return fig, ax


# Plot reward
def create_ev_agent_reward_plot(rewards, config, n_episodes, language="en"):
    """
    Create a visualization of the rewards obtained by an EV agent.

    Parameters:
        rewards (np.array): Array of rewards obtained by the EV agent
        config: Configuration object containing t_a, t_b, soc_b, dt
        n_episodes (int): Number of episodes
        language (str): Language for labels ('en' for English, 'fr' for French)
    """
    # Validate language selection
    if language not in TRANSLATIONS:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are: {list(TRANSLATIONS.keys())}"
        )

    texts = TRANSLATIONS[language]

    # Setup
    fig, ax = plt.subplots(figsize=SIZE)
    time = np.arange(0, n_episodes, 1)

    def plot_main_curve():
        """Plot the main reward curve"""
        plt.plot(time, rewards, label="Reward", color="blue")

    def setup_grid_and_labels():
        """Setup grid, ticks, labels, and legend"""
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.2)

        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Reward obtained by agent $EV_{config.id}$")
        plt.legend()

        plt.tight_layout()

    def setup_ticks():
        """Setup ticks and limits for the plot"""
        plt.xticks(np.arange(0, n_episodes // 10 + 1) * 10)
        plt.xlim(0, n_episodes - 1)

    # Execute plotting functions
    plot_main_curve()
    setup_grid_and_labels()
    setup_ticks()

    return fig, ax


# Plot cumulative reward
def create_ev_agent_cumulative_reward_plot(cum_rewards, config, n_episodes, language="en"):
    """
    Create a visualization of the cumulative rewards obtained by an EV agent.

    Parameters:
        cum_rewards (np.array): Array of cumulative rewards obtained by the EV agent
        config: Configuration object containing t_a, t_b, soc_b, dt
        n_episodes (int): Number of episodes
        language (str): Language for labels ('en' for English, 'fr' for French)
    """
    # Validate language selection
    if language not in TRANSLATIONS_PLOT_AGENT:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are: {list(TRANSLATIONS_PLOT_AGENT.keys())}"
        )

    texts = TRANSLATIONS_PLOT_AGENT[language]

    # Setup
    fig, ax = plt.subplots(figsize=SIZE)
    time = np.arange(0, n_episodes, 1)

    def plot_main_curve():
        """Plot the main cumulative reward curve"""
        plt.plot(time, cum_rewards, label="Cumulative reward", color="red")

    def setup_grid_and_labels():
        """Setup grid, ticks, labels, and legend"""
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.2)

        plt.xlabel("Episodes")
        plt.ylabel("Cumulative reward")
        plt.title(f"Cumulative reward obtained by agent $EV_{config.id}$")
        plt.legend()

        plt.tight_layout()

    def setup_ticks():
        """Setup ticks and limits for the plot"""
        # plt.xticks(np.arange(0, n_episodes))
        plt.xlim(0, n_episodes - 1)

    # Execute plotting functions
    plot_main_curve()
    setup_grid_and_labels()
    setup_ticks()

    return fig, ax


def plot_data(data, title="Data", xlabel="Time", ylabel="Value", legend=["A", "B", "C"]):
    """
    Plot data with a given title, labels, and legend.

    Parameters:
        data (np.array): Data to plot
        title (str): Title of the plot
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        legend (list): List of labels for the legend
    """
    fig, ax = plt.subplots(figsize=SIZE)

    for i in range(data.shape[0]):
        plt.plot(data[i, :], label=legend[i] if i < len(legend) else f"Data {i+1}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig, ax


# Dictionary for text translations
TRANSLATIONS_PLOT_AGENT_V2 = {
    "en": {
        "time": "Time (hours)",
        "norm_values": "Normalized values",
        "title": "Normalized power and $SOC$ for $EV_{id}$ charging simulation for day ${day}$\n on sample ${n_sample}$ with strategy ${strategy}$",
        "power": "Power",
        "soc_target": "SOC target",
    },
    "fr": {
        "time": "Temps (heures)",
        "norm_values": "Valeurs normalisées",
        "title": "Puissance normalisée et $SOC$ pour la simulation de charge $EV_{id}$ au jour ${day}$\n sur l'échantillon ${n_sample}$ avec la stratégie ${strategy}$",
        "power": "Puissance",
        "soc_target": "SOC cible",
    },
}


def create_ev_agent_charging_plot_v2(
    power_history,
    soc_history,
    ev_agent_config,
    day=0,
    n_sample=0,
    language="en",
    price=None,
    reward=None,
):
    """
    Create a visualization of EV charging data.

    Parameters:
        power_history (np.array): Array of power values over time
        soc_history (np.array): Array of State of Charge values over time
        config: Configuration object containing t_a, t_b, soc_b, dt
        ev_agent: EV object containing T parameter
        language (str): Language for labels ('en' for English, 'fr' for French)
    """
    # Validate language selection
    if language not in TRANSLATIONS_PLOT_AGENT_V2:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are: {list(TRANSLATIONS_PLOT_AGENT_V2.keys())}"
        )

    texts = TRANSLATIONS_PLOT_AGENT_V2[language]

    # Variables
    config = ev_agent_config
    T = config.T
    dt = config.ev_config.dt
    t_a = config.t_a
    t_b = config.t_b
    soc_target = config.soc_target

    # Setup
    fig, ax = plt.subplots(figsize=SIZE)
    time = np.arange(0, 24, (dt / 3600))
    # Filter data for the selected day
    power_history = power_history[day]
    soc_history = soc_history[day]
    if price is not None:
        price = price[day]
    if reward is not None:
        reward = reward[day]

    max_power = np.max(power_history) if np.max(power_history) > 0 else 1

    # Convert times to hours
    t_a_hours = t_a // 3600
    t_b_hours = t_b // 3600

    def plot_main_curves():
        """Plot the main power and SOC curves"""
        plt.step(time, power_history / max_power, label=texts["power"], where="pre")
        plt.plot(time, soc_history, label="SOC")
        plt.plot(
            time,
            soc_target * np.ones(time.shape[0]),
            label=texts["soc_target"],
            linestyle="dashed",
            color="black",
        )

    def add_time_regions():
        """Add shaded regions and time markers"""
        plt.axvspan(0, t_a_hours, alpha=0.1, color="gray")
        plt.axvspan(t_b_hours, 24, alpha=0.1, color="gray")

        plt.axvline(x=t_a_hours, color="darkgreen", linestyle="--", alpha=0.5)
        plt.axvline(x=t_b_hours, color="darkred", linestyle="--", alpha=0.5)

    def add_annotations():
        """Add text annotations and arrows"""
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)

        # Time labels remain mathematical notation
        y_text_bottom = -0.15
        for time, label, color in [(t_a_hours, "t_a", "darkgreen"), (t_b_hours, "t_b", "darkred")]:
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
        # print(config.t_a)
        # print(dt)
        # print(config.t_a // dt)
        soc_at_ta = soc_history[int(config.t_a // dt)]
        soc_at_tb = soc_history[int(config.t_b // dt)]

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

    def setup_grid_and_labels():
        """Setup grid, ticks, labels, and legend"""
        plt.xticks(np.arange(0, 24 + 1, 2))
        ax.xaxis.set_minor_locator(MultipleLocator(dt / 3600))
        plt.grid(True, which="major", alpha=0.5)
        plt.grid(True, which="minor", alpha=0.2)

        plt.xlabel(texts["time"])
        plt.ylabel(texts["norm_values"])
        plt.title(
            texts["title"]
            .replace("{id}", str(config.id))
            .replace("{day}", str(day))
            .replace("{n_sample}", str(n_sample))
            .replace("{strategy}", ev_agent_config.strategy_config)
        )
        plt.legend()

        plt.tight_layout()

    def setup_limits():
        """Setup limits for the plot"""
        plt.ylim(0, 1.1)
        plt.xlim(0, 24)

    def plot_price_curve(price):
        """Plot the price curve"""
        plt.plot(time, price, label="Price", color="red")

    def plot_reward_curve(reward):
        """Plot the reward curve"""
        plt.plot(time, reward, label="Reward", color="blue")

    # Execute plotting functions
    plot_main_curves()
    if price is not None:
        plot_price_curve(price)

    if reward is not None:
        plot_reward_curve(reward)

    add_time_regions()
    add_annotations()
    setup_grid_and_labels()
    setup_limits()

    return fig, ax
