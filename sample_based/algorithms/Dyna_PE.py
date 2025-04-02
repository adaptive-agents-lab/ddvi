import sys
import numpy as np
from rl_utilities import get_optimal_policy_mdp, sample_one_from_mdp, get_policy_value
from tqdm import tqdm
from model.LocallySmoothedModel import LocallySmoothedModel

class Dyna_PE:

    def __init__(self, mdp, policy, model):
        self.mdp = mdp
        self.policy = policy
        self.model = model

    def train(self, num_iteration):
        discount = self.mdp.discount()
        num_states = self.mdp.num_states()
        self.V_trace = np.zeros((num_iteration, num_states))
        V = np.zeros(num_states)
        with tqdm(iter(range(num_iteration)),desc="Dyna", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                self.V_trace[k, :] = V
                (s, a, r, next_s) = sample_one_from_mdp(self.mdp, self.policy)
                self.model.update(np.array([(s, a, r, next_s)]))
                P_hat = self.model.get_P_hat()
                r_hat = self.model.get_r_hat()
                V = get_policy_value(P_hat, r_hat, discount, self.policy)

    def run(self, num_iteration, output_filename,save_to_file=True):
        self.train(num_iteration)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace
