import matplotlib.pyplot as plt
import csv
import numpy as np

class MultiLivePlotter:
    def __init__(self, n_plots=4, titles=None, xlabels=None, ylabels=None, figsize=(10, 8)):
        """
        Generic multi‑subplot live plotter.
        - n_plots: number of subplots
        """

        self.n = n_plots

        # Defaults
        if titles is None:
            titles = [f"Plot {i+1}" for i in range(n_plots)]
        if xlabels is None:
            xlabels = [f"X{i+1}" for i in range(n_plots)]
        if ylabels is None:
            ylabels = [f"Y{i+1}" for i in range(n_plots)]

        # Data buffers
        self.x_data = [[] for _ in range(n_plots)]
        self.y_data = [[] for _ in range(n_plots)]

        # Layout: auto square grid
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))

        self.fig, self.axs = plt.subplots(rows, cols, figsize=figsize)
        self.axs = np.array(self.axs).reshape(-1)  # flatten

        # Create line objects
        self.lines = []
        for i in range(n_plots):
            line, = self.axs[i].plot([], [], '-')
            self.axs[i].set_title(titles[i])
            self.axs[i].set_xlabel(xlabels[i])
            self.axs[i].set_ylabel(ylabels[i])
            self.lines.append(line)

        plt.tight_layout()

    def _append(self, idx, y, x):
        """Internal helper to append data to subplot idx."""
        if x is None:
            # Auto-generate x
            if isinstance(y, (int, float)):
                self.y_data[idx].append(y)
                self.x_data[idx].append(len(self.y_data[idx]) - 1)
            else:
                n = len(y)
                start = len(self.x_data[idx])
                self.y_data[idx].extend(y)
                self.x_data[idx].extend(range(start, start + n))
        else:
            # x provided
            if isinstance(y, (int, float)):
                self.x_data[idx].append(x)
                self.y_data[idx].append(y)
            else:
                self.x_data[idx].extend(x)
                self.y_data[idx].extend(y)

    def update(self, y_list, x_list=None):
        """
        Update all subplots.
        - y_list: list of length n_plots
        - x_list: list of length n_plots or None
        """
        if x_list is None:
            x_list = [None] * self.n

        for i in range(self.n):
            self._append(i, y_list[i], x_list[i])

            # Update line
            self.lines[i].set_xdata(self.x_data[i])
            self.lines[i].set_ydata(self.y_data[i])

            # Rescale
            ax = self.axs[i]
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    def save_csv(self, prefix="plot"):
        """Save each subplot's data to CSV."""
        for i in range(self.n):
            filename = f"{prefix}_{i}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for x, y in zip(self.x_data[i], self.y_data[i]):
                    writer.writerow([x, y])
            print(f"Saved {filename}")
