import numpy as np
from rl_utilities import  sample_one_from_mdp
from tqdm import tqdm

class TDLearning_PE:

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def train(self, num_iteration, lr_scheduler):
        discount = self.mdp.discount()
        num_actions = self.mdp.num_actions()
        num_states = self.mdp.num_states()
        Q = np.zeros((num_actions, num_states))
        self.V_trace = np.zeros((num_iteration, num_states))
        with tqdm(iter(range(num_iteration)), desc="TD Learning", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                (s,a,r,next_s) = sample_one_from_mdp(self.mdp, self.policy)
                next_a = self.policy[next_s]
                target = r + discount * Q[next_a, next_s]
                Q[a, s] = Q[a, s] + lr_scheduler.get_lr(current_iter=k) * (target - Q[a, s])
                self.V_trace[k, :] = Q[self.policy, np.arange(num_states)]

    def run(self, num_iteration, lr_scheduler, output_filename, save_to_file=True):
        self.train(num_iteration, lr_scheduler)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace


