import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MultipleLocator
import matplotlib
import os
from math import erf, sqrt

from utils.plot_analysis import PlotAnalysis
from utils.plot_results import *
from agents.ev_agent import EVAgentConfig
from physics.ev_physics import EVPhysicsConfig
from strategies.strategy import StrategyConfig
from simulation.simulation import *

import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import pickle



class StrategyComparison:
    """
    Compare strategies between each others.
    """

    matplotlib.rcParams["font.size"] = 14

    def __init__(
        self,
        path: str,
        n_agents: int,
        verbose: bool = True,
        strategy_order=None,
        strategy_palette=None,
        line_styles=None,
        markers=None,
    ):
        """
        Parameters
        ----------
        path : str
            Root folder that contains EVs_{n_agents}/
        n_agents : int
            Number of agents, used to build EVs_{n_agents}/
        verbose : bool
            Whether to print strategies/samples found.
        strategy_order, strategy_palette, line_styles, markers:
            Optional overrides. If None, defaults match your StrategyComparison.
        """

        self.path = path
        self.n_agents = int(n_agents)

        # Build correct base path exactly like your original code
        self.base_path = os.path.join(self.path, f"EVs_{self.n_agents}/")
        if verbose:
            print(self.base_path)

        # Discover strategies + sample counts from disk
        self.strategies_and_samples_dict = find_strategies_and_samples(
            self.base_path, verbose=verbose
        )

        self.strategies = list(self.strategies_and_samples_dict.keys())
        self.num_strategies = len(self.strategies)

        # --- Fixed, stable order (matches your tables) ---
        self.strategy_order = strategy_order or [
            "FTPL-IRS-EXP",
            "FTPL-IRS-EXP-CA",
            "FTPL-IRS-F2",
            "FTPL-IRS-F2-CA",
            "GREEDY",                 # Naive internal key
            "Inde-TS",
            "Inde-TS-CA",
            "MILP_Price_Forecast",
            "DRL-DQN",
        ]

        self.markers = markers or [
            "o",   # circle
            "s",   # square
            "^",   # triangle up
            "D",   # diamond
            "v",   # triangle down
            "P",   # plus (filled)
            "X",   # X (filled)
            "*",   # star
        ]

        # --- Palette (stable) ---
        self.strategy_palette = strategy_palette or [
            "#1F1F1F",  # 1) FTPL-IRS-EXP
            "#F02E48",  # 2) FTPL-IRS-EXP-CA
            "#1B998B",  # 3) FTPL-IRS-F2
            "#6A4C93", # 4) FTPL-IRS-F2-CA
            "#F46036",  # 5) Naive (GREEDY)
            "#FF00BF",# 6) Inde-TS
            "#66c000",  # 7) Inde-TS-CA
            "#2E86AB", # 8) MILP_Price_Forecast
        ]

        # --- Line styles (10 available; stable mapping for first 8) ---
        self.line_styles = line_styles or [
            "-",                      # 1
            "--",                     # 2
            ":",                      # 3
            "-.",                     # 4
            (0, (3, 1, 1, 1)),        # 5
            (0, (5, 1)),              # 6
            (0, (1, 1)),              # 7
            (0, (5, 2, 1, 2)),        # 8
            (0, (2, 2)),              # 9
            (0, (8, 2, 2, 2)),        # 10
        ]

        # Build stable maps for canonical strategies
        base_colors = dict(zip(self.strategy_order, self.strategy_palette))
        base_styles = dict(zip(self.strategy_order, self.line_styles))
        base_markers = dict(zip(self.strategy_order, self.markers))

        # Extras beyond canonical
        extras = [s for s in self.strategies if s not in base_colors]
        extra_palette = ["#7F7F7F", "#BCBD22", "#17BECF", "#E377C2", "#8C564B", "#9467BD"]
        extra_styles = self.line_styles[len(self.strategy_order):] + self.line_styles  # safe wrap
        extra_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

        for i, s in enumerate(extras):
            base_colors[s] = extra_palette[i % len(extra_palette)]
            base_styles[s] = extra_styles[i % len(extra_styles)]
            base_markers[s] = extra_markers[i % len(extra_markers)]

        self.strategy_colors = base_colors
        self.strategy_linestyle = base_styles
        self.strategy_markers = base_markers

    def compute_time_table_from_disk(self):
        """
        Compute computing-time table by streaming results from disk (RAM-safe).
        """

        metrics = {
            "Strategy": [],
            "Mean Computing Time (s)": [],
            "Std Computing Time (s)": [],
            "Min Computing Time (s)": [],
            "Max Computing Time (s)": [],
            "Avg Final Reward": [],
        }

        for strategy, samples in tqdm(
            self.strategies_and_samples_dict.items(),
            leave=True,
            desc=f"Strategies (N={self.n_agents})",
        ):
            times = []
            final_rewards = []

            for sample in tqdm(range(samples), leave=False, desc="Samples"):
                simulation_path = os.path.join(self.base_path, strategy, str(sample))
                if not os.path.exists(simulation_path):
                    continue

                simulation_config = load_simulation_config(simulation_path)
                simulation_config.path = simulation_path

                simulation = Simulation(simulation_config)
                results = simulation.load_results(
                    # interpolate_telecommuter_reward=True
                )

                if hasattr(results, "computing_time"):
                    times.append(float(results.computing_time))

                if hasattr(results, "reward_history"):
                    final_episode_rewards = np.sum(results.reward_history[-1], axis=0)
                    final_rewards.append(float(np.mean(final_episode_rewards)))

                # Free memory immediately
                del results
                del simulation
                del simulation_config
                gc.collect()

            if len(times) > 0:
                metrics["Strategy"].append(strategy)
                metrics["Mean Computing Time (s)"].append(float(np.mean(times)))
                metrics["Std Computing Time (s)"].append(float(np.std(times)))
                metrics["Min Computing Time (s)"].append(float(np.min(times)))
                metrics["Max Computing Time (s)"].append(float(np.max(times)))
                metrics["Avg Final Reward"].append(float(np.mean(final_rewards)) if final_rewards else np.nan)

        return pd.DataFrame(metrics)

    @staticmethod
    def load_computation_table_pickle(pkl_path: str):
        """Load the legacy computation_table = [Ns, [df_per_N,...]] pickle."""
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def computation_table_to_long_df(computation_table):
        """
        Convert computation_table = [Ns, [df1, df2, ...]] into one long DataFrame with N_agents column.
        """
        Ns = computation_table[0]
        dfs = computation_table[1]
        full_df = pd.concat(
            [df.assign(N_agents=int(N)) for N, df in zip(Ns, dfs)],
            ignore_index=True
        )
        return full_df.sort_values(["Strategy", "N_agents"])
    
    def plot_mean_computation_time_vs_N_from_pickle(
    self,
    pkl_path: str,
    figsize=(9, 6),
    capsize=5,
    elinewidth=1.5,
    linewidth=1.5,
    use_correct_names: bool = False,
    loglog: bool = False,
    save_path: str = None,
    legend_fontsize: int = 14,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 14,
    show_title: bool = False,
    title: str = None,  # only used if show_title=True
):
        """
        Plot mean computation time vs N with min–max error bars using a saved computation_table.pkl.
        """
        import matplotlib.ticker as ticker

        computation_table = self.load_computation_table_pickle(pkl_path)
        full_df = self.computation_table_to_long_df(computation_table)

        # ---- Display-name remap (legend only; styles use raw keys)
        label_map = {
            "MILP_Price_Forecast": "ILP-PF",
            "GREEDY": "Naive",
        }

        full_df = full_df.copy()

        # Optional mapping via correct_strategy_name 
        if use_correct_names and "correct_strategy_name" in globals():
            full_df["Strategy_label"] = full_df["Strategy"].apply(
                lambda s: correct_strategy_name(
                    s,
                    getattr(self, "simulation_configs_dict", {}),
                    getattr(self, "results_dict", {}),
                    getattr(self, "ev_agents_configs_dict", {}),
                )
            ).replace(label_map)
        else:
            full_df["Strategy_label"] = full_df["Strategy"].replace(label_map)

        fig, ax = plt.subplots(figsize=figsize)

        # IMPORTANT: plot order and style keys are RAW strategy keys
        plotted_strategies = list(full_df["Strategy"].unique())

        # Keep stable plotting order if possible (based on raw keys)
        if hasattr(self, "strategy_order"):
            plotted_strategies = (
                [s for s in self.strategy_order if s in plotted_strategies]
                + [s for s in plotted_strategies if s not in getattr(self, "strategy_order", [])]
            )

        for strat in plotted_strategies:
            sub = full_df[full_df["Strategy"] == strat].sort_values("N_agents")
            if sub.empty:
                continue

            # legend label (display name)
            label = sub["Strategy_label"].iloc[0]

            x = sub["N_agents"].to_numpy()
            y = sub["Mean Computing Time (s)"].to_numpy()
            y_min = sub["Min Computing Time (s)"].to_numpy()
            y_max = sub["Max Computing Time (s)"].to_numpy()

            # asymmetric error bars: lower = mean - min, upper = max - mean
            yerr = np.vstack([y - y_min, y_max - y])

            # Stable style maps based on RAW strategy keys
            color = getattr(self, "strategy_colors", {}).get(strat, None)
            ls = getattr(self, "strategy_linestyle", {}).get(strat, "-")
            mk = getattr(self, "strategy_markers", {}).get(strat, "o")

            # ---- Plot MEAN curve explicitly (so it's always the mean)
            line, = ax.plot(
                x,
                y,
                label=label,
                color=color,
                linewidth=linewidth,
                marker=mk,
                markersize= 7
            )
            # Apply linestyle including tuple dash patterns
            line.set_linestyle(ls)

            # ---- Draw min–max error bars WITHOUT drawing another line
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                capsize=capsize,
                elinewidth=elinewidth,
                color=color,
            )

        # ---- Axis labels + tick sizes
        ax.set_xlabel("Number of agents (N)", fontsize=axis_label_fontsize)
        ax.set_ylabel("Computation time (s)", fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.get_offset_text().set_fontsize(28)

        # ---- Title handling (removed by default)
        if show_title:
            if title is None:
                title = "Mean computation time vs N (min–max error bars)"
            ax.set_title(title, fontsize=axis_label_fontsize)

        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.grid(True, which="both" if loglog else "major", alpha=0.3)

        # ---- Legend size parameter
        ax.legend(fontsize=legend_fontsize)

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        plt.show()
        return full_df
    
    def plot_congestion_frequency_from_disk(
    self,
    sample_indices=None,
    day_start: int = 0,
    day_end=None,
    window_size: int = 3,
    figsize=(10, 6),
    save_path="results_ECML2026_bis/images",
    show: bool = True,
    # ---- NEW formatting params (same as your RAM version)
    legend_fontsize: int = 14,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 14,
    show_title: bool = False,
    title: str = None,  # only used if show_title=True
):
        """
        Disk-based congestion frequency plot.
        """

        import os
        import gc
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        # -----------------------------
        # Helpers
        # -----------------------------
        def _samples_to_iter(strategy_key: str):
            if sample_indices is not None:
                return list(sample_indices)
            return list(range(int(self.strategies_and_samples_dict.get(strategy_key, 0))))

        def _safe_strategy_label(strategy_key: str) -> str:
            # Minimal fallback mapping; keep disk-safe
            if strategy_key == "GREEDY":
                return "Naive"
            if strategy_key == "MILP_Price_Forecast":
                return "ILP-PF"
            return strategy_key


        freq_by_strategy = {s: [] for s in self.strategies}
        max_freq = 0
                


        for strategy in tqdm(self.strategies, desc=f"Strategies (N={self.n_agents})", leave=True):
            for sample_idx in tqdm(_samples_to_iter(strategy), desc="Samples", leave=False):
                sim_path = os.path.join(self.base_path, strategy, str(sample_idx))
                if not os.path.isdir(sim_path):
                    continue

                try:
                    cfg = load_simulation_config(sim_path)
                    cfg.path = sim_path
                    sim = Simulation(cfg)
                    results = sim.load_results()

                    if not hasattr(results, "congestion_history"):
                        del results, sim, cfg
                        gc.collect()
                        continue

                    congested_slots = np.sum(results.congestion_history, axis=2)  # (D, T)
                    congestion_per_day = np.count_nonzero(congested_slots, axis=1)  # (D,)

                    D = len(congestion_per_day)
                    d0 = max(0, int(day_start))
                    d1 = D if day_end is None else min(D, int(day_end))
                    if d0 >= d1:
                        del results, sim, cfg
                        gc.collect()
                        continue

                    vec = congestion_per_day[d0:d1].astype(float)

                    if vec.size > 0:
                        freq_by_strategy[strategy].append(vec)
                        max_freq = max(max_freq, int(np.max(vec)))
                    del results, sim, cfg
                    gc.collect()

                except Exception:
                    try:
                        del sim
                    except Exception:
                        pass
                    try:
                        del cfg
                    except Exception:
                        pass
                    gc.collect()
                    continue

        plotted_any = any(len(v) > 0 for v in freq_by_strategy.values())
        if not plotted_any:
            print("No congestion frequency data available for the requested window.")
            return None

        # -----------------------------
        # Plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=figsize)

        strategies_plot_order = [s for s in self.strategy_order if s in self.strategies] + \
                                [s for s in self.strategies if s not in self.strategy_order]

        displayed_end = day_end if day_end is not None else None
        minD_global = None

        for strategy in strategies_plot_order:
            per_sample = freq_by_strategy.get(strategy, [])
            if len(per_sample) == 0:
                continue

            minD = min(len(x) for x in per_sample)
            minD_global = minD if minD_global is None else min(minD_global, minD)

            mat = np.stack([x[:minD] for x in per_sample], axis=0)  # (S, minD)

            mean_curve = mat.mean(axis=0)
            std_curve = mat.std(axis=0)

            days = np.arange(int(day_start), int(day_start) + minD)

            smoothed_mean = (
                pd.Series(mean_curve)
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

            low = mean_curve - std_curve
            high = mean_curve + std_curve

            color = self.strategy_colors.get(strategy, None)
            ls = self.strategy_linestyle.get(strategy, "-")

            label = _safe_strategy_label(strategy)

            ax.fill_between(days, low, high, alpha=0.2, color=color, linewidth=0)
            line, = ax.plot(
                days,
                smoothed_mean,
                linewidth=3,
                color=color,
                label=label,
            )
            line.set_linestyle(ls)

        if displayed_end is None and minD_global is not None:
            displayed_end = int(day_start) + int(minD_global)

        # ---- Title handling (removed by default)
        if show_title:
            if title is None:
                title = (
                    "Congestion frequency (mean ± std across samples)\n"
                    f"N={self.n_agents} agents — Days {day_start}–{displayed_end}"
                )
            ax.set_title(title, fontsize=axis_label_fontsize)

        # ---- Axis labels + tick sizes
        ax.set_xlabel("episode", fontsize=axis_label_fontsize)
        ax.set_ylabel("congestion frequency", fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

        ax.set_ylim(0, max_freq * 1.1 if max_freq > 0 else 1)
        ax.grid(True, alpha=0.3)

        # ---- Legend font size
        ax.legend(fontsize=legend_fontsize,ncol=2)

        fig.tight_layout()

        # Save
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filename = (
                f"congestion_frequency_N{self.n_agents}_days{day_start}_{displayed_end}_"
                f"roll{window_size}_stdband.pdf"
            )
            fullpath = os.path.join(save_path, filename)
            fig.savefig(fullpath, format="pdf", bbox_inches="tight")
            print(f"Saved: {fullpath}")

        if show:
            plt.show()

        return fig, ax
    
    def _load_triplet_from_disk(self, strategy: str, sample_idx: int):
        """
        Returns (results, sim_cfg, ev_df) or (None, None, None) if missing.
        """
        sim_path = os.path.join(self.base_path, strategy, str(sample_idx))
        if not os.path.isdir(sim_path):
            return None, None, None

        # 1) simulation config
        sim_cfg = load_simulation_config(sim_path)
        sim_cfg.path = sim_path

        # 2) results
        sim = Simulation(sim_cfg)
        results = sim.load_results()

        # 3) EV agents config DF
        ev_df = None


        if hasattr(sim, "load_ev_agents_configs"):
            ev_df = sim.load_ev_agents_configs()

        if ev_df is None:
            candidates = [
                os.path.join(sim_path, "ev_agents_configs.pkl"),
                os.path.join(sim_path, "ev_agents_config.pkl"),
                os.path.join(sim_path, "ev_agents_configs.csv"),
                os.path.join(sim_path, "ev_agents_config.csv"),
            ]
            for fp in candidates:
                if os.path.exists(fp):
                    if fp.endswith(".pkl"):
                        ev_df = pd.read_pickle(fp)
                    else:
                        ev_df = pd.read_csv(fp)
                    break

        del sim
        gc.collect()

        return results, sim_cfg, ev_df

    def plot_congestion_severity_cdf_from_disk(
        self,
        sample_indices=None,
        episode_start=1050,
        episode_end=1250,
        figsize=(7, 5),
        x_grid_size=300,
        band="std",          # "std", "minmax", or None
        save_path="results_ECML2026_bis/images",
        debug=False,
    ):
        import os
        import gc
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        try:
            from tqdm.notebook import tqdm
        except Exception:
            from tqdm import tqdm


        def _make_step_cdf(x_grid, y_cdf):
            x0 = x_grid[0]
            x_plot = np.concatenate(([x0, x0], x_grid[1:]))
            y_plot = np.concatenate(([0.0, y_cdf[0]], y_cdf[1:]))
            return x_plot, y_plot


        class _EvCfg:
            def __init__(self, ev_p_max: np.ndarray):
                self.ev_p_max = ev_p_max


        sev_by_strategy = {}
        pooled_by_strategy = {}
        inferred_N = None

        # Debug counters (optional)
        skipped_missing_files = 0
        skipped_load_error = 0
        skipped_missing_attrs = 0
        skipped_missing_ev_p_max = 0
        skipped_bad_window = 0

        for strategy in tqdm(self.strategies, desc="Strategies"):
            n_samples = int(self.strategies_and_samples_dict.get(strategy, 0))
            samples = list(range(n_samples)) if sample_indices is None else list(sample_indices)

            sev_list = []

            for sample_idx in tqdm(samples, desc=f"{strategy} samples", leave=False):
                simulation_path = os.path.join(self.base_path, strategy, str(sample_idx))

                if not (
                    os.path.exists(os.path.join(simulation_path, "simulation_parameters.h5"))
                    and os.path.exists(os.path.join(simulation_path, "simulation_results.h5"))
                    and os.path.exists(os.path.join(simulation_path, "agent_configs.h5"))
                ):
                    skipped_missing_files += 1
                    continue

                try:
                    sim_cfg = load_simulation_config(simulation_path, verbose=False)
                    sim_cfg.path = simulation_path
                    sim = Simulation(sim_cfg)

                    results = sim.load_results()
                    agents_df = sim.load_agents_configs_df()

                except Exception as e:
                    skipped_load_error += 1
                    if debug:
                        print("DEBUG load error:", simulation_path, "->", repr(e))
                    try:
                        del sim
                    except Exception:
                        pass
                    gc.collect()
                    continue

  
                if not all(hasattr(results, attr) for attr in ["congestion_history", "n_flexible_power"]):
                    skipped_missing_attrs += 1
                    del results, agents_df, sim, sim_cfg
                    gc.collect()
                    continue


                if agents_df is None or "ev_p_max" not in agents_df.columns:
                    skipped_missing_ev_p_max += 1
                    if debug:
                        cols = [] if agents_df is None else list(agents_df.columns)
                        print("DEBUG missing ev_p_max column in agents_df:", simulation_path, "cols:", cols[:20])
                    del results, agents_df, sim, sim_cfg
                    gc.collect()
                    continue

                ev_p_max = pd.to_numeric(agents_df["ev_p_max"], errors="coerce").to_numpy()
                ev_p_max = ev_p_max[np.isfinite(ev_p_max)]
                if ev_p_max.size == 0:
                    skipped_missing_ev_p_max += 1
                    if debug:
                        print("DEBUG ev_p_max exists but empty/NaN:", simulation_path)
                    del results, agents_df, sim, sim_cfg
                    gc.collect()
                    continue

                ev_cfg = _EvCfg(ev_p_max=ev_p_max)

                if inferred_N is None:
                    inferred_N = int(len(ev_cfg.ev_p_max))

                congestion_history = results.congestion_history
                number_of_charging_agent_per_congestion = np.sum(congestion_history, axis=2)

                sim_config = sim_cfg  
                nf_consumption = results.n_flexible_power[:congestion_history.shape[0], :]
                p_max = float(np.mean(ev_cfg.ev_p_max)) 

                nbr_of_agent_congestion_capacity = np.floor(
                    (np.full_like(number_of_charging_agent_per_congestion,
                                sim_config.congestion_limit,
                                dtype=float) - nf_consumption) / p_max
                )

                nbr_to_disconnect = np.clip(
                    number_of_charging_agent_per_congestion - nbr_of_agent_congestion_capacity,
                    0, None
                )

                congestion_occurence_per_day = np.count_nonzero(
                    number_of_charging_agent_per_congestion, axis=1
                )

                congestion_severity = np.zeros_like(congestion_occurence_per_day, dtype=float)
                non_zero_days = congestion_occurence_per_day > 0
                congestion_severity[non_zero_days] = (
                    np.sum(nbr_to_disconnect[non_zero_days], axis=1)
                    / congestion_occurence_per_day[non_zero_days]
                )

                end_idx = min(int(episode_end), len(congestion_severity))
                if int(episode_start) >= end_idx:
                    skipped_bad_window += 1
                    del results, agents_df, sim, sim_cfg, ev_cfg
                    gc.collect()
                    continue

                sev = congestion_severity[int(episode_start):end_idx]
                sev_list.append(sev)

                del results, agents_df, sim, sim_cfg, ev_cfg
                gc.collect()

            if len(sev_list) == 0:
                continue

            sev_by_strategy[strategy] = sev_list
            pooled_by_strategy[strategy] = np.concatenate(sev_list, axis=0)

        if debug:
            print(
                "DEBUG:",
                "skipped_missing_files=", skipped_missing_files,
                "| skipped_load_error=", skipped_load_error,
                "| skipped_missing_attrs=", skipped_missing_attrs,
                "| skipped_missing_ev_p_max=", skipped_missing_ev_p_max,
                "| skipped_bad_window=", skipped_bad_window,
            )

        if len(pooled_by_strategy) == 0:
            print("No congestion severity data available for the requested window.")
            return

        if inferred_N is None:
            inferred_N = -1  

        global_xmin = 0.0
        global_xmax = max(np.max(v) for v in pooled_by_strategy.values())
        if global_xmax <= global_xmin:
            global_xmax = global_xmin + 1e-6

        x_grid = np.linspace(global_xmin, global_xmax, int(x_grid_size))

        fig, ax = plt.subplots(figsize=figsize)
        colors = self.strategy_colors

        for strategy in self.strategies:
            if strategy not in sev_by_strategy:
                continue

      
            rep_sample_idx = 0


            n_samples = int(self.strategies_and_samples_dict.get(strategy, 0))
            candidate_samples = list(range(n_samples)) if sample_indices is None else list(sample_indices)
            rep_found = None
            for sidx in candidate_samples:
                sp = os.path.join(self.base_path, strategy, str(sidx), "simulation_parameters.h5")
                ap = os.path.join(self.base_path, strategy, str(sidx), "agent_configs.h5")
                rp = os.path.join(self.base_path, strategy, str(sidx), "simulation_results.h5")
                if os.path.exists(sp) and os.path.exists(ap) and os.path.exists(rp):
                    rep_found = sidx
                    break
            if rep_found is None:
                strategy_name = "Naive" if strategy == "GREEDY" else strategy
            else:
                rep_path = os.path.join(self.base_path, strategy, str(rep_found))
                rep_cfg = load_simulation_config(rep_path, verbose=False)
                rep_cfg.path = rep_path
                rep_sim = Simulation(rep_cfg)
                rep_agents_df = rep_sim.load_agents_configs_df()

                if rep_agents_df is not None and "ev_p_max" in rep_agents_df.columns:
                    rep_ev_p_max = pd.to_numeric(rep_agents_df["ev_p_max"], errors="coerce").to_numpy()
                    rep_ev_p_max = rep_ev_p_max[np.isfinite(rep_ev_p_max)]
                else:
                    rep_ev_p_max = np.array([], dtype=float)

                rep_ev_cfg = _EvCfg(ev_p_max=rep_ev_p_max)

                sim_cfgs_dict = {strategy: {0: rep_cfg}}
                results_dict_dummy = {strategy: {0: None}}  # correct_strategy_name doesn't use results
                ev_dict = {strategy: {0: rep_ev_cfg}}

                strategy_name = correct_strategy_name(strategy, sim_cfgs_dict, results_dict_dummy, ev_dict)

                del rep_sim, rep_agents_df, rep_cfg, rep_ev_cfg
                gc.collect()

            if band is None:
                sev = np.sort(pooled_by_strategy[strategy])
                y = np.searchsorted(sev, x_grid, side="right") / len(sev)
                x_plot, y_plot = _make_step_cdf(x_grid, y)
                ax.step(
                    x_plot, y_plot, where="post", linewidth=2,
                    label=strategy_name, color=colors[strategy]
                )
            else:
                cdfs = []
                for sev in sev_by_strategy[strategy]:
                    sev_sorted = np.sort(sev)
                    cdf_vals = np.searchsorted(sev_sorted, x_grid, side="right") / len(sev_sorted)
                    cdfs.append(cdf_vals)

                mat = np.stack(cdfs, axis=0)
                mean_cdf = mat.mean(axis=0)

                if band == "std":
                    spread = mat.std(axis=0)
                    low = mean_cdf - spread
                    high = mean_cdf + spread
                    band_label = "std"
                elif band == "minmax":
                    low = mat.min(axis=0)
                    high = mat.max(axis=0)
                    band_label = "minmax"
                else:
                    raise ValueError("band must be 'std', 'minmax', or None")

                low = np.clip(low, 0, 1)
                high = np.clip(high, 0, 1)

                x_plot, mean_plot = _make_step_cdf(x_grid, mean_cdf)
                _, low_plot = _make_step_cdf(x_grid, low)
                _, high_plot = _make_step_cdf(x_grid, high)

                ax.step(
                    x_plot, mean_plot, where="post", linewidth=2,
                    label=strategy_name, color=colors[strategy]
                )
                ax.fill_between(
                    x_plot, low_plot, high_plot, step="post", alpha=0.2,
                    color=colors[strategy], linewidth=0
                )

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Congestion Severity")
        ax.set_ylabel("CDF")

        title_band = "" if band is None else f" (mean ± {band} across samples)"
        ax.set_title(
            f"CDF of Congestion Severity{title_band}\n"
            f"N={inferred_N} agents — Episodes {episode_start}–{episode_end}"
        )

        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            band_tag = "pooled" if band is None else band_label
            filename = f"cdf_congestion_severity_N{inferred_N}_ep{episode_start}_{episode_end}_{band_tag}.pdf"
            fullpath = os.path.join(save_path, filename)
            fig.savefig(fullpath, format="pdf", bbox_inches="tight")
            print(f"Saved: {fullpath}")

        plt.show()
    
    def plot_agent_cumulative_loss_from_disk(
    self,
    ev_id: int,
    sample_indices=None,
    episode_start: int = 0,
    episode_end: int = 1000,
    cumulative_mode: str = "mean",   # "mean" or "sum"
    band: str = "std",               # "std" or "minmax"
    figsize=(12, 4),
    save_path="results_ECML2026_bis/images",
    verbose: bool = False,
    # ---- NEW formatting params (same pattern as others)
    legend_fontsize: int = 14,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 14,
    show_title: bool = False,
    title: str = None,  # only used if show_title=True
    ylim = 12,
):
        """
        Loads one sample at a time, computes per-sample cumulative curve for one agent, aggregates across samples.

        """
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        # -----------------------------
        # helpers
        # -----------------------------
        def _samples_to_iter(strategy_key: str):
            if sample_indices is not None:
                return list(sample_indices)
            return list(range(int(self.strategies_and_samples_dict.get(strategy_key, 0))))

        def _safe_strategy_label(strategy_key: str) -> str:
            # Minimal label mapping without needing RAM dicts
            name = "Naive" if strategy_key == "GREEDY" else strategy_key
            if strategy_key == "MILP_Price_Forecast":
                name = "ILP-PF"

            # Try to infer "-CA" from a config (first existing sample)
            try:
                for s in _samples_to_iter(strategy_key):
                    sim_path = os.path.join(self.base_path, strategy_key, str(s))
                    if not os.path.isdir(sim_path):
                        continue
                    cfg = load_simulation_config(sim_path)
                    if getattr(cfg, "ev_agent_type", None) == "congestion_aware":
                        if name in ["FTPL-IRS", "Inde-TS", "FTPL-IRS-F2", "FTPL-IRS-EXP"]:
                            name = name + "-CA"
                    break
            except Exception:
                pass

            return name


        inferred_N = int(getattr(self, "n_agents", -1))

        fig, ax = plt.subplots(figsize=figsize)
        colors = self.strategy_colors

        # stable plotting order if possible
        strategies_plot_order = [s for s in self.strategy_order if s in self.strategies] + \
                                [s for s in self.strategies if s not in self.strategy_order]

        band_tag = band  # for filename fallback if nothing plotted
      
        for strategy in tqdm(strategies_plot_order, desc=f"Strategies (N={inferred_N})", leave=True):
            all_samples_cumulative = []

            for sample_idx in tqdm(_samples_to_iter(strategy), desc=f"{strategy} samples", leave=False):
                sim_path = os.path.join(self.base_path, strategy, str(sample_idx))

                # require the 3 canonical files (matches your other disk methods)
                if not (
                    os.path.exists(os.path.join(sim_path, "simulation_parameters.h5")) and
                    os.path.exists(os.path.join(sim_path, "simulation_results.h5")) and
                    os.path.exists(os.path.join(sim_path, "agent_configs.h5"))
                ):
                    continue

                try:
                    sim_cfg = load_simulation_config(sim_path, verbose=False)
                    sim_cfg.path = sim_path
                    sim = Simulation(sim_cfg)
                    results = sim.load_results()

                    if not hasattr(results, "reward_history"):
                        del results, sim, sim_cfg
                        gc.collect()
                        continue

                    reward_history = results.reward_history  # (E, T, A)
                    E, T, A = reward_history.shape

                    if ev_id < 0 or ev_id >= A:
                        del results, sim, sim_cfg
                        gc.collect()
                        continue

                    e0 = max(0, int(episode_start))
                    e1 = E if episode_end is None else min(E, int(episode_end))
                    if e0 >= e1:
                        del results, sim, sim_cfg
                        gc.collect()
                        continue

                    # Episode reward for this agent: (E,)
                    episode_reward = reward_history.sum(axis=1)[:, ev_id]

                    # Cumulative (computed on full E then sliced)
                    cum = np.cumsum(episode_reward)
                    if cumulative_mode == "mean":
                        cum = cum / np.arange(1, E + 1)

                    all_samples_cumulative.append(cum[e0:e1])

                    del results, sim, sim_cfg
                    gc.collect()

                except Exception as e:
                    if verbose:
                        print(f"[WARN] skip {strategy}/{sample_idx} due to {repr(e)}")
                    try:
                        del results
                    except Exception:
                        pass
                    try:
                        del sim
                    except Exception:
                        pass
                    try:
                        del sim_cfg
                    except Exception:
                        pass
                    gc.collect()
                    continue

            if len(all_samples_cumulative) == 0:
                continue

            # Align lengths across samples
            minE = min(len(c) for c in all_samples_cumulative)
            mat = np.stack([c[:minE] for c in all_samples_cumulative], axis=0)  # (S, minE)

            mean_cum = mat.mean(axis=0)

            if band == "std":
                spread = mat.std(axis=0)
                low = mean_cum - 2*spread
                high = mean_cum + 2*spread
                band_tag = "std"
            elif band == "minmax":
                low = mat.min(axis=0)
                high = mat.max(axis=0)
                band_tag = "minmax"
            else:
                raise ValueError("band must be 'std' or 'minmax'")

            episodes = np.arange(int(episode_start), int(episode_start) + int(minE))

            strategy_name = _safe_strategy_label(strategy)

            color = colors.get(strategy, None)
            ls = self.strategy_linestyle.get(strategy, "-")
            mk = self.strategy_markers.get(strategy, "o")

            line, = ax.plot(
                episodes,
                mean_cum,
                label=strategy_name,
                linewidth=2,
                color=color,
                # marker=mk,
                # markevery=100,
            )
            line.set_linestyle(ls)
            ax.fill_between(episodes, low, high, alpha=0.2, color=color, linewidth=0)

        # ---- Axis labels + tick sizes
        ax.set_xlabel("Episode", fontsize=axis_label_fontsize)
        ylabel = "Cumulative loss"
        if cumulative_mode == "mean":
            ylabel = "Running-avg loss"
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0,ymax=ylim)

        # ---- Title handling (removed by default)
        if show_title:
            if title is None:
                title = (
                    f"Agent {ev_id} rolling cumulative loss — mean ± {band} across samples\n"
                    f"N={inferred_N} agents — Episodes {episode_start}–{episode_end}"
                )
            ax.set_title(title, fontsize=axis_label_fontsize)

        # ---- Legend size
        ax.legend(fontsize=legend_fontsize,framealpha = 0.7,ncol=1)

        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filename = (
                f"agent{ev_id}_cumloss_N{inferred_N}_ep{episode_start}_{episode_end}_"
                f"{cumulative_mode}_{band_tag}.pdf"
            )
            fullpath = os.path.join(save_path, filename)
            fig.savefig(fullpath, format="pdf", bbox_inches="tight")
            print(f"Saved: {fullpath}")

        plt.show()
        return fig, ax

    @staticmethod
    def _mean_std_n(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n == 0:
            return np.nan, np.nan, 0
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if n > 1 else 0.0
        return mean, std, n

    @staticmethod
    def _paired_diff(a: dict, b: dict):
        ks = sorted(set(a.keys()).intersection(b.keys()))
        if not ks:
            return np.array([], dtype=float)
        d = np.array([a[k] - b[k] for k in ks], dtype=float)
        return d[np.isfinite(d)]

    # -----------------------------
    # Disk: normalized price final value
    # -----------------------------
    @staticmethod
    def _normalized_price_final_value_from_results(results, day_start=0, day_end=None, eps=1e-6):
        if not (
            hasattr(results, "price_history")
            and hasattr(results, "soc_history")
            and hasattr(results, "soc_init_history")
        ):
            return None

        price_history = results.price_history          # (D, T, A)
        soc_history = results.soc_history              # (D, T, A)
        soc_init_history = results.soc_init_history    # (A, D)

        D, T, A = price_history.shape
        d0 = max(0, int(day_start))
        d1 = D if day_end is None else min(D, int(day_end))
        if d0 >= d1:
            return None

        price_w = price_history[d0:d1, :, :]
        soc_w = soc_history[d0:d1, :, :]

        # (A,D) -> (D,A)
        soc_init = soc_init_history.T
        soc_init_w = soc_init[d0:d1, :]

        final_soc = soc_w[:, -1, :]               # (Dw, A)
        daily_total_price = np.sum(price_w, axis=1)  # (Dw, A)

        delta_soc = np.maximum(final_soc - soc_init_w, eps)
        normalized_total_price = daily_total_price / delta_soc  # (Dw, A)

        cum = np.cumsum(normalized_total_price, axis=0)
        running_avg = cum / (np.arange(1, cum.shape[0] + 1)[:, None])
        fleet_curve = running_avg.mean(axis=1)

        return float(fleet_curve[-1])


    @staticmethod
    def _congestion_metrics_old_from_triplet(results, sim_cfg, agents_df, day_start=0, day_end=None, clip_capacity_zero=True):
        """
        Returns:
          - total_congestive_instants
          - avg_disconnections_per_agent
        Uses the SAME definitions as your RAM version.
        """
        if not all(hasattr(results, a) for a in ["congestion_history", "n_flexible_power"]):
            return None

        if agents_df is None or "ev_p_max" not in agents_df.columns:
            return None

        cong = results.congestion_history  # (D,T,A)
        D = cong.shape[0]
        d0 = max(0, int(day_start))
        d1 = D if day_end is None else min(D, int(day_end))
        if d0 >= d1:
            return None

        cong_w = cong[d0:d1, :, :]
        charging_agents = np.sum(cong_w, axis=2)  # (Dw,T)

        total_congestive_instants = float(np.count_nonzero(charging_agents))

        nf = results.n_flexible_power[:D, :][d0:d1, :]  # (Dw,T)

        ev_p_max = pd.to_numeric(agents_df["ev_p_max"], errors="coerce").to_numpy()
        ev_p_max = ev_p_max[np.isfinite(ev_p_max)]
        if ev_p_max.size == 0:
            return None

        p_max = float(np.mean(ev_p_max))
        N = int(ev_p_max.size)

        capacity_agents = np.floor(
            (np.full_like(charging_agents, sim_cfg.congestion_limit, dtype=float) - nf) / p_max
        )
        if clip_capacity_zero:
            capacity_agents = np.maximum(capacity_agents, 0)

        to_disconnect = np.clip(charging_agents - capacity_agents, 0, None)
        total_disconnections = float(np.sum(to_disconnect))
        avg_disconnections_per_agent = total_disconnections / max(N, 1)

        return {
            "total_congestive_instants": total_congestive_instants,
            "avg_disconnections_per_agent": avg_disconnections_per_agent,
        }


    def _disk_strategy_label(self, strategy_key: str, sample_indices=None):
        """
        Returns a label consistent with your RAM correct_strategy_name,
        but disk-safe: only peeks at ONE existing sample to detect CA and FTPL perturbation.
        """
        if strategy_key == "GREEDY":
            base = "Naive"
        elif strategy_key == "MILP_Price_Forecast":
            base = "ILP-PF"
        else:
            base = strategy_key

        n_samples = int(self.strategies_and_samples_dict.get(strategy_key, 0))
        candidates = list(range(n_samples)) if sample_indices is None else list(sample_indices)

        rep = None
        for s in candidates:
            sp = os.path.join(self.base_path, strategy_key, str(s), "simulation_parameters.h5")
            rp = os.path.join(self.base_path, strategy_key, str(s), "simulation_results.h5")
            ap = os.path.join(self.base_path, strategy_key, str(s), "agent_configs.h5")
            if os.path.exists(sp) and os.path.exists(rp) and os.path.exists(ap):
                rep = s
                break

        if rep is None:
            return base

        rep_path = os.path.join(self.base_path, strategy_key, str(rep))
        try:
            cfg = load_simulation_config(rep_path, verbose=False)
            cfg.path = rep_path
            sim = Simulation(cfg)
            agents_df = sim.load_agents_configs_df()
        except Exception:
            try:
                del sim
            except Exception:
                pass
            gc.collect()
            return base

        # FTPL-IRS perturbation (if you still use it)
        if strategy_key == "FTPL-IRS" and agents_df is not None:
            try:
                pert = agents_df.iloc[0].get("strategy_perturbation", "default")
                base = f"{base}-{pert}"
            except Exception:
                pass

        # congestion-aware suffix
        if getattr(cfg, "ev_agent_type", None) == "congestion_aware":
            if base in ["FTPL-IRS", "Inde-TS", "FTPL-IRS-F2", "FTPL-IRS-EXP"]:
                base = base + "-CA"

        del sim, cfg, agents_df
        gc.collect()
        return base

    def compute_old_metrics_scientific_table_from_disk(
        self,
        naive_strategy="GREEDY",
        optimal_strategy="MILP_Price_Forecast",
        price_day_start=0,
        price_day_end=None,
        cong_day_start=1050,
        cong_day_end=1200,
        eps=1e-6,
        sample_indices=None,
        save_dir="results_ECML2026_bis/tables",
        filename="disk_old_metrics_scientific.pkl",
        verbose=False,
    ):
        """
        streams samples, returns one dataframe with the 5 metrics :
          - Price mean/std
          - Total congestive instants mean/std
          - Avg disconnections per agent mean/std
          - Regret vs MILP (paired diff) mean/std
          - Regret vs Naive (paired diff) mean/std

        Saves the dataframe as pickle if save_dir is not None.
        """

        opt_aliases = {
            "ILP-Forecast": "MILP_Price_Forecast",
            "ILP_Forecast": "MILP_Price_Forecast",
            "MILP_Forecast": "MILP_Price_Forecast",
        }
        optimal_strategy = opt_aliases.get(optimal_strategy, optimal_strategy)

        # per strategy: dict(seed->scalar)
        price_by_strategy = {s: {} for s in self.strategies}
        cong_inst_by_strategy = {s: {} for s in self.strategies}
        disc_pa_by_strategy = {s: {} for s in self.strategies}

        def _samples_to_iter(strategy_key: str):
            if sample_indices is not None:
                return list(sample_indices)
            return list(range(int(self.strategies_and_samples_dict.get(strategy_key, 0))))

        for strategy in tqdm(self.strategies, desc=f"Strategies (N={self.n_agents})", leave=True):
            for sample_idx in tqdm(_samples_to_iter(strategy), desc=f"{strategy} samples", leave=False):
                sim_path = os.path.join(self.base_path, strategy, str(sample_idx))

                # require canonical files 
                if not (
                    os.path.exists(os.path.join(sim_path, "simulation_parameters.h5")) and
                    os.path.exists(os.path.join(sim_path, "simulation_results.h5")) and
                    os.path.exists(os.path.join(sim_path, "agent_configs.h5"))
                ):
                    continue

                try:
                    sim_cfg = load_simulation_config(sim_path, verbose=False)
                    sim_cfg.path = sim_path
                    sim = Simulation(sim_cfg)

                    results = sim.load_results()
                    agents_df = sim.load_agents_configs_df()

                    # price scalar
                    pv = self._normalized_price_final_value_from_results(
                        results, day_start=price_day_start, day_end=price_day_end, eps=eps
                    )
                    if pv is not None and np.isfinite(pv):
                        price_by_strategy[strategy][sample_idx] = float(pv)

                    # congestion scalars
                    cm = self._congestion_metrics_old_from_triplet(
                        results, sim_cfg, agents_df,
                        day_start=cong_day_start, day_end=cong_day_end
                    )
                    if cm is not None:
                        ti = cm.get("total_congestive_instants", np.nan)
                        da = cm.get("avg_disconnections_per_agent", np.nan)
                        if np.isfinite(ti):
                            cong_inst_by_strategy[strategy][sample_idx] = float(ti)
                        if np.isfinite(da):
                            disc_pa_by_strategy[strategy][sample_idx] = float(da)

                    # cleanup
                    del results, agents_df, sim, sim_cfg
                    gc.collect()

                except Exception as e:
                    if verbose:
                        print(f"[WARN] skip {strategy}/{sample_idx}: {repr(e)}")
                    try:
                        del results
                    except Exception:
                        pass
                    try:
                        del agents_df
                    except Exception:
                        pass
                    try:
                        del sim
                    except Exception:
                        pass
                    try:
                        del sim_cfg
                    except Exception:
                        pass
                    gc.collect()
                    continue

        # ---- build table
        rows = []
        naive_label = self._disk_strategy_label(naive_strategy, sample_indices=sample_indices)
        milp_label = self._disk_strategy_label(optimal_strategy, sample_indices=sample_indices)

        for strategy in self.strategies:
            label = self._disk_strategy_label(strategy, sample_indices=sample_indices)

            # price stats
            price_vals = np.array(list(price_by_strategy[strategy].values()), dtype=float)
            p_mean, p_std, p_n = self._mean_std_n(price_vals)

            # congestion instants stats
            inst_vals = np.array(list(cong_inst_by_strategy[strategy].values()), dtype=float)
            inst_mean, inst_std, inst_n = self._mean_std_n(inst_vals)

            # disconnections/agent stats
            dis_vals = np.array(list(disc_pa_by_strategy[strategy].values()), dtype=float)
            dis_mean, dis_std, dis_n = self._mean_std_n(dis_vals)

            # regrets (paired diffs)
            reg_vs_naive = self._paired_diff(price_by_strategy[strategy], price_by_strategy.get(naive_strategy, {}))
            rn_mean, rn_std, rn_n = self._mean_std_n(reg_vs_naive)

            reg_vs_milp = self._paired_diff(price_by_strategy[strategy], price_by_strategy.get(optimal_strategy, {}))
            rm_mean, rm_std, rm_n = self._mean_std_n(reg_vs_milp)

            rows.append({
                "Strategy": label,
                "N_agents": int(self.n_agents),

                "Price_final_norm (mean)": p_mean,
                "Price_final_norm (std)": p_std,
                "Price_n_seeds": p_n,

                "Total_congestive_instants (mean)": inst_mean,
                "Total_congestive_instants (std)": inst_std,
                "Congestion_n_seeds": inst_n,

                "Avg_disconnections_per_agent (mean)": dis_mean,
                "Avg_disconnections_per_agent (std)": dis_std,
                "Disconnections_n_seeds": dis_n,

                f"Regret_price_vs_{naive_label} (mean)": rn_mean,
                f"Regret_price_vs_{naive_label} (std)": rn_std,
                f"Regret_price_vs_{naive_label} (n)": rn_n,

                f"Regret_price_vs_{milp_label} (mean)": rm_mean,
                f"Regret_price_vs_{milp_label} (std)": rm_std,
                f"Regret_price_vs_{milp_label} (n)": rm_n,
            })

        df = pd.DataFrame(rows)

        # keep stable order if possible (optional)
        if hasattr(self, "strategy_order"):
            order_labels = [self._disk_strategy_label(s, sample_indices=sample_indices) for s in self.strategy_order]
            df["__ord"] = df["Strategy"].apply(lambda x: order_labels.index(x) if x in order_labels else 999)
            df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

        # save
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, filename)
            df.to_pickle(outpath)
            print(f"Saved: {outpath}")

        return df

def find_strategies_and_samples(path, verbose=True):
    """ Return a dictionary with strategies as keys and number of samples as values """
    strategies = {}
    for folder in os.listdir(path):
        if os.path.isdir(path + folder):
            strategies[folder] = len(os.listdir(path + folder))

    strategies = dict(sorted(strategies.items()))
    if verbose:
        # Print the strategies and samples
        print("Strategies and samples found:")
        for strategy, samples in strategies.items():
            print(f"  - {strategy}: {samples}")

    return strategies



def ev_series_to_agent_config(ev_series: pd.Series) -> EVAgentConfig:
    """
    Convert a pandas Series row into an EVAgentConfig instance.
    """

    # --- EV physics config ---
    ev_config = EVPhysicsConfig(
        type=ev_series.get("ev_type", "default"),  # required
        dt=ev_series.get("ev_dt", 3600),
        e_max=ev_series.get("ev_e_max", 0),
        e_min=ev_series.get("ev_e_min", 0),
        p_max=ev_series.get("ev_p_max", 0),
        p_min=ev_series.get("ev_p_min", 0),
        eta_c=ev_series.get("ev_eta_c", 1.0),
        eta_d=ev_series.get("ev_eta_d", 1.0),
    )

    # --- Strategy config ---
    strategy_config = StrategyConfig(
        name=ev_series.get("strategy_name", "default")
    )

    # --- EV agent config ---
    agent_config = EVAgentConfig(
        id=int(ev_series.get("id", 0)),
        type=str(ev_series.get("type", "default EV")),  # EV agent type
        T=int(ev_series.get("T", 24 * 3600)),
        t_a=int(ev_series.get("t_a", 8 * 3600)),
        t_b=int(ev_series.get("t_b", 18 * 3600)),
        soc_initial=float(ev_series.get("soc_initial", 0.55)),
        soc_target=float(ev_series.get("soc_target", 0.8)),
        ev_config=ev_config,
        strategy_config=strategy_config,
        telecommuting_days=ev_series.get("telecommuting_days", []),
        loss_generator = ev_series.get("loss_generator", "default"),
    )

    return agent_config

def correct_strategy_name(strategy, simulation_configs_dict, results_dicts, ev_agents_dict):
        strategy_name = strategy
        if strategy == "GREEDY":
            strategy_name = "Naive"
        elif strategy == "MILP_Price_Forecast":
            strategy_name = "ILP-PF"
        if strategy == "FTPL-IRS":
            strategy_name+= "-" + ev_agents_dict[strategy][0].iloc[0].get("strategy_perturbation", "default")
        if simulation_configs_dict[strategy][0].ev_agent_type == "congestion_aware":
            if strategy in ["FTPL-IRS", "Inde-TS","FTPL-IRS-F2","FTPL-IRS-EXP"]:
                strategy_name += "-CA"
        
        
        return strategy_name
