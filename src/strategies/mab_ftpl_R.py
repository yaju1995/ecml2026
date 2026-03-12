from dataclasses import dataclass
import numpy as np
from strategies.strategy import Strategy, StrategyConfig
from collections import defaultdict


@dataclass
class MABFTPLConfig(StrategyConfig):
    T: int
    D: int
    M: int
    name: str
    perturbation: str

    @classmethod
    def default_config(cls):
        return MABFTPLConfig(
            T=96,
            D=1460,
            name="FTPL-IRS",
            perturbation="EXP",
            M=150,
        )

    @classmethod
    def F2_perturbed_config(cls):
        return MABFTPLConfig(
            T=96,
            D=1460,
            name="FTPL-IRS",
            perturbation="F2",
            M=150,
        )


class MABFTPL(Strategy):

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        self.T = config.T # episode dimension {0,1}^T
        self.D = config.D # Number of episode
        self.M = config.M # Geometric Resampling truncation index
        self.perturbation = config.perturbation # FTPL perturbation

        self.gamma = 0 #Exploration factor
        self.n_obj = 0 #Discrete stochastic demand

        self.L_hat = np.zeros(self.T) # Initial selection loss estimator
        self.L_hat_RS = np.zeros(self.T) # Rescheduling loss estimator
        self.I = np.zeros(self.T) # Importance weight Initial selection
        self.I_RS = np.zeros(self.T) # Importance weight Rescheduling
        self.N = np.zeros(self.T) # Number of selection tracker
        self.last_loss = np.zeros(self.T) # Loss suffered at the last episode
        self.o = np.zeros(self.T) # Observed disconnections

        self.S_mask = np.zeros(self.T, dtype=bool)

        self.A_0 = set() # Initialy selected schedule
        self.A_t = set() # Executed schedule
        self.play_order = []
        self.observed_context_day = None

    # -------------------------------------------------------
    # RESET
    # -------------------------------------------------------
    def reset(self, observed_context: list):

        self.observed_context_day = observed_context
        day = int(observed_context[7])
        # Observe feasibility window
        t_a, t_b = int(observed_context[4]), int(observed_context[5])

        if day == 0:
            self.n_obj = int(observed_context[6])

            if self.perturbation == "EXP":
                self.gamma = (
                    np.sqrt(self.T)
                    * (np.log(self.T) + 1)
                    / (2 * self.D * self.T)
                ) ** (2 / 3)



        S_indices = np.flatnonzero(self.S_mask)
        S_indices_rev = S_indices[::-1]

        r = self.A_0.copy()
        r_RS = self.A_t.copy()

        for i in range(1, self.M + 1):

            Z = self.sample_perturbation()
            
            # -----------------------------------------
            # Initial selection Geometric resampling
            # -----------------------------------------
            temp_est = self.L_hat[S_indices] - Z[S_indices] / self.gamma
            order = np.argsort(temp_est)
            selected = S_indices[order[: self.n_obj]]

            V_ = set(selected.tolist())

            if r:
                intersect = r.intersection(V_)
                for t in intersect:
                    self.I[t] = i
                r -= intersect

            # -----------------------------------------
            # Rescheduling selection Geometric resampling
            # -----------------------------------------
            Z_RS = Z

            # -------- Initial selection resampling ---------- 
            temp_est_RS = self.L_hat_RS[S_indices] - Z_RS[S_indices] / self.gamma
            order_RS = np.argsort(temp_est_RS)
            sorted_RS = S_indices[order_RS]

            # -------- Rescheduling resampling ----------
            for t in S_indices_rev:
                if t in V_ and self.o[t] == 1:
                    mask = (sorted_RS > t) & (sorted_RS < t_b)
                    future = sorted_RS[mask]
                    if len(future != 0):
                        V_.add(future[0])

            if r_RS:
                intersect_RS = r_RS.intersection(V_)
                for t in intersect_RS:
                    self.I_RS[t] = i
                r_RS -= intersect_RS

            if not r and not r_RS:
                break

        # Fallback
        if r:
            self.I[list(r)] = float(self.M)
        if r_RS:
            self.I_RS[list(r_RS)] = float(self.M)

        # -------------------------------------
        # LOSS UPDATE
        # -------------------------------------
        all_indices = np.arange(self.T)

        outside = np.concatenate((all_indices[:t_a], all_indices[t_b:]))

        to_add_hat = np.union1d(
            np.fromiter(self.A_0, dtype=int),
            outside
        )
        self.L_hat[to_add_hat] += self.last_loss[to_add_hat] * self.I[to_add_hat]
        self.L_hat_RS += self.last_loss * self.I_RS

        # -------------------------------------
        # RESET DECISION SET
        # -------------------------------------
        self.o.fill(0)
        self.S_mask.fill(False)
        self.S_mask[t_a:t_b] = True
        self.last_loss[t_a:t_b] = 0.0

        # Observe stochastic demand
        self.n_obj = int(observed_context[6])

        # Update exploration factor if needed.
        if self.perturbation == "F2":
            self.gamma = 1.0 / np.sqrt(float(day) + 1.0)

        # -------------------------------------
        # Initial FTPL selection
        # -------------------------------------

        # Perturbation Sampling
        S_indices = np.flatnonzero(self.S_mask)
        Z = self.sample_perturbation()

        # Optimization oracle
        temp_est = self.L_hat[S_indices] - Z[S_indices] / self.gamma
        order = np.argsort(temp_est)
        sorted_t = S_indices[order]

        self.A_0 = set(sorted_t[: self.n_obj])
        self.A_t = self.A_0.copy()

        # Precomputation of Reschuling perturbed loss.
        Z_RS = Z

        temp_est_RS = self.L_hat_RS[S_indices] - Z_RS[S_indices] / self.gamma
        order_RS = np.argsort(temp_est_RS)
        sorted_RS = S_indices[order_RS]

        # Rescheduling priority order computation
        self.play_order = list(zip(sorted_RS, temp_est_RS[order_RS]))

              
    

    # -------------------------------------------------------
    # ACT
    # -------------------------------------------------------
    def act(self, observed_context: list) -> int:
        t = int(observed_context[0])
        return t in self.A_t

    # -------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------
    def update(self, observed_context: list, reward: float):

        t = int(observed_context[0])
        t_b = int(observed_context[5])

        if t in self.A_t:

            self.last_loss[t] = reward
            self.N[t] += 1

            disconnect_t = int(observed_context[12])
            self.o[t] = disconnect_t

            # Rescheduling
            if disconnect_t == 1:

                future_candidates = [
                    idx for idx, _ in self.play_order
                    if idx not in self.A_t and t < idx < t_b
                ]

                if future_candidates:
                    self.A_t.add(future_candidates[0])

    # -------------------------------------------------------
    # PERTURBATIONS
    # -------------------------------------------------------
    def sample_perturbation(self):

        Z = np.zeros(self.T)

        S_indices = np.where(self.S_mask)[0]

        if self.perturbation == "EXP":
            Z[S_indices] = np.random.exponential(
                scale=1.0, size=len(S_indices)
            )

        elif self.perturbation == "F2":
            U = np.random.uniform(0, 1, size=len(S_indices))
            Z[S_indices] = (-np.log(U)) ** (-1 / 2)

        return Z


Strategy.register("MAB FTPL", MABFTPL)
