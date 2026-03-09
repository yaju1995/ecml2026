from dataclasses import dataclass
import numpy as np
from strategies.strategy import Strategy, StrategyConfig


@dataclass
class PriorOptConfig(StrategyConfig):
    T: int
    D: int
    name = "PRIOR OPT"

    @classmethod
    def default_config(cls):
        return PriorOptConfig(
            T=96,
            D = 400,
            name="PRIOR OPT"
        )


class PriorOpt(Strategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.moves_ordering = []
        self.T = config.T
        self.D = config.D 
        self.reward_list = []
        self.n_obj = 0
        self.N = dict((0,i) for i in range(self.D))  # Number of plays for each instant

    def init_reward_list(self, reward_list: list):
        """Initialize the reward list for the strategy."""
        self.reward_list = reward_list # [T,D] shape
        # for each D index, compute the mean of the rewards
        self.reward_list = np.mean(self.reward_list, axis=0)  # [T,]

        # Sort the indices of the rewards in descending order
        self.moves_ordering = np.argsort(self.reward_list)  # [

        
    def reset(self, observed_context: list):
        self.observed_context_day = observed_context
        day = int(observed_context[7])
        t_a, t_b = int(observed_context[4]), int(observed_context[5])
        self.n_obj = int(observed_context[6])
        # Select n_obj best future instants available based on the reward list
        self.A_t = set()
        for t in self.moves_ordering:
            if t_a <= t <= t_b and len(self.A_t) < self.n_obj:
                self.A_t.add(t)



        return
        

    def act(self, observed_context: list) -> int:
        t = int(observed_context[0])
        self.N[t] = self.N.get(t, 0) + int(t in self.A_t)
        return t in self.A_t

    def update(self, observed_context: list, reward: float):
        t = int(observed_context[0])
        t_b = int(observed_context[5])
        t_a = int(observed_context[4])
        n = int(observed_context[6])

        # On stocke la perte observée (1 - reward)
        if not hasattr(self, 'last_loss'):
            self.last_loss = dict()
        if t_a <= t and t <= t_b:
            if t in self.A_t:
                
                unwired_t = int(observed_context[12])

                if unwired_t == 1: # Reselect a future instant to play if unwired by DSO
                    # Ajouter une action supplémentaire à A_t
                    # Filtrer les instants déjà présents dans A_t
                    future_candidates = [i for i in self.moves_ordering if i not in self.A_t and t <= i <= t_b]
                    
                    if future_candidates:
                        # Sélectionner le plus grand instant parmi les candidats restants
                        new_action = future_candidates[0]
                        self.A_t.add(new_action)  # Ajouter cet instant à A_t


Strategy.register("PRIOR OPT", PriorOpt)
