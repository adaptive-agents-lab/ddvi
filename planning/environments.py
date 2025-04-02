import numpy as np

class AbstractMDP:

    def __init__(self, P, R, discount, initial_dist):
        self._P = P
        self._R = R
        self._initial_dist = initial_dist
        self._discount = discount

        self._num_actions = R.shape[0]
        self._num_states = R.shape[1]


    def P(self):
        return self._P

    def R(self):
        return self._R

    def discount(self):
        return self._discount

    def num_states(self):
        return self._num_states

    def num_actions(self):
        return self._num_actions

    def initial_dist(self):
        return self._initial_dist

    def sample_step(self, state, action):
        transition_vector = self._P[action, state, :]
        next_state = np.random.choice(np.arange(0, self.num_states()), p=transition_vector)
        reward = self._R[action, state]
        return next_state, reward


# In[3]:


import numpy as np

class AbstractProblem:

    def P(self):
        return self._P

    def R(self):
        return self._R

    def num_states(self):
        return self._num_states

    def num_actions(self):
        return self._num_actions


# In[4]:


class Maze55(AbstractMDP):
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ENV_NAME = "Maze55"

    def __init__(self, success_prob, discount, single_step_prob = 1, double_step_prob = 0, have_walls=True):

        self.n_columns = 5
        self.n_rows = 5

        goal_state = 20
        terminal_states = []
        initial_states = [0]
        self.walls = []
        if have_walls:
            self.walls = [	{0,	1},
                            {5,	6},
                            {10,11},
                            {15, 20},
                            {16, 21},
                            {16, 17},
                            {11, 12},
                            {6, 7},
                            {2, 7},
                            {3, 8},
                            {9, 8},
                            {14, 19},
                            {13, 18},
                            {13, 12},
                            {17, 22},
                            {18, 23}]

        P = self.get_transition_matrix(success_prob, single_step_prob, double_step_prob, terminal_states)
        R = self.get_reward_function(goal_state)

        self._num_states = self.n_columns * self.n_rows
        self._num_actions = 4
        initial_dist = np.zeros((self._num_states))
        initial_dist = np.zeros((self._num_states))
        for s in initial_states:
            initial_dist[s] = 1 / len(initial_states)

        terminal_states = set(terminal_states)

        super().__init__(P, R, discount, initial_dist)

    def get_transition_matrix(self, success_prob, single_step_prob, double_step_prob, terminal_states):
        n_states = self.n_columns * self.n_rows
        stay_prob = 1 - single_step_prob - double_step_prob
        unif_prob = (1 - success_prob) / 3
        P = np.zeros((4, n_states, n_states))

        for r in range(self.n_rows):
            for c in range(self.n_columns):
                state = r * self.n_columns + c
                if state in terminal_states:
                    P[:, state, state] = 1
                else:
                    for a in range(4):
                        for dir in range(4):
                            target1 = self.get_target(state, dir)
                            target2 = self.get_target(target1, dir)
                            if dir == a:
                                P[a, state, state] += success_prob * stay_prob
                                P[a, state, target1] += success_prob * single_step_prob
                                P[a, state, target2] += success_prob * double_step_prob
                            else:
                                P[a, state, state] += unif_prob * stay_prob
                                P[a, state, target1] += unif_prob * single_step_prob
                                P[a, state, target2] += unif_prob * double_step_prob

        return P

    def get_target(self, state, action):
        column = state % self.n_columns
        row = int((state - column) / self.n_columns)

        if action == Maze55.ACTION_UP:
            top_c = column
            top_r = max(row - 1, 0)
            target = top_r * self.n_columns + top_c
        elif action == Maze55.ACTION_RIGHT:
            right_c = min(column + 1, self.n_columns - 1)
            right_r = row
            target = right_r * self.n_columns + right_c
        elif action == Maze55.ACTION_DOWN:
            bottom_c = column
            bottom_r = min(row + 1, self.n_rows - 1)
            target = bottom_r * self.n_columns + bottom_c
        elif action == Maze55.ACTION_LEFT:
            left_c = max(column - 1, 0)
            left_r = row
            target = left_r * self.n_columns + left_c
        else:
            raise Exception("Illegal action")

        if {state, target} in self.walls:
            target = state

        return target

    def get_reward_function(self, goal_state):
        n_states = self.n_columns * self.n_rows

        R = np.ones((4, n_states)) * -1
        R[:, goal_state] = 10

        return R


# In[77]:


import os

import numpy as np
import yaml
from scipy.stats import rv_discrete




class CliffWalkmodified(AbstractMDP):
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    ENV_NAME = "CliffWalkmodified"

    def __init__(self, success_prob, discount):

        self.n_columns = 7
        self.n_rows = 3

        terminal_states = []
        goal_state = self.n_columns-1
        terminal_states.append(goal_state)

        for state in range(self.n_columns * self.n_rows):
            if state // self.n_columns in [0] and state % self.n_columns in range(1, self.n_columns-1):
                terminal_states.append(state)

        initial_states = [0]
        self.walls = []
        self.terminal_states = terminal_states

        P = self.get_transition_matrix(success_prob, terminal_states)
        R = self.get_reward_function(goal_state)

        self._num_states = self.n_columns * self.n_rows
        self._num_actions = 4
        initial_dist = np.zeros((self._num_states))
        for s in initial_states:
            initial_dist[s] = 1 / len(initial_states)

        super().__init__(P, R, discount, initial_dist)

    def get_transition_matrix(self, success_prob, terminal_states):
        n_states = self.n_columns * self.n_rows
        P = np.zeros((4, n_states, n_states))
        unif_prob = (1 - success_prob) / 3
        for r in range(self.n_rows):
            for c in range(self.n_columns):
                state = r * self.n_columns + c
                if state in terminal_states:
                    P[:, state, state] = 1
                else:
                    for a in range(4):
                        for dir in range(4):
                            target = self.get_target(state, dir)
                            if dir == a:
                                P[a, state, target] += success_prob
                            else:
                                P[a, state, target] += unif_prob


        return P

    def get_target(self, state, action):
        column = state % self.n_columns
        row = int((state - column) / self.n_columns)

        if action == CliffWalkmodified.ACTION_UP:
            top_c = column
            top_r = max(row - 1, 0)
            target = top_r * self.n_columns + top_c
        elif action == CliffWalkmodified.ACTION_RIGHT:
            right_c = min(column + 1, self.n_columns - 1)
            right_r = row
            target = right_r * self.n_columns + right_c
        elif action == CliffWalkmodified.ACTION_DOWN:
            bottom_c = column
            bottom_r = min(row + 1, self.n_rows - 1)
            target = bottom_r * self.n_columns + bottom_c
        elif action == CliffWalkmodified.ACTION_LEFT:
            left_c = max(column - 1, 0)
            left_r = row
            target = left_r * self.n_columns + left_c
        else:
            raise Exception("Illegal action")

        if {state, target} in self.walls:
            target = state

        return target

    def get_reward_function(self, goal_state):
        n_states = self.n_columns * self.n_rows

        R = np.zeros((4, n_states))

        for state in range(n_states):
            if state in self.terminal_states:
              if state == goal_state:
                R[:, state] = 10
                # if state % self.n_columns in range(1, self.n_columns-1):
                #     if state // self.n_columns == 0:
              else:
                R[:, state] = -10
                    # if state // self.n_columns == 2:
                    #     R[:, state] = -16
                    # if state // self.n_columns == 4:
                    #     R[:, state] = -8
                    # if state // self.n_columns == 6:
                    #     R[:, state] = -4
                    # if state // self.n_columns == 8:
                    #     R[:, state] = -2
                    # if state // self.n_columns == 10:
                    #     R[:, state] = -1
            else:
                R[:, state] = -1
        return R


# In[6]:


class ChainWalk(AbstractProblem):

    ENV_NAME = "ChainWalk"

    def __init__(self, num_states):
        self._num_states = num_states
        self._num_actions = 2


        self._P = [np.zeros((num_states, num_states)) for _ in range(self._num_actions)] #List of |A| many |S| x |S| transition matrices
        self._R = [np.zeros(num_states)  for _ in range(self._num_actions)]  #np.array of shape |S|x1

        self._populate_P()
        self._populate_R()

        super

    def name(self):
        return "Random Walk"


    def _populate_P(self):
        for a in range(self._num_actions):
            for s in range(self._num_states):
                pVec = np.zeros(self._num_states)
                if a == 0:
                    pVec[(s+1) % self._num_states] = 0.7  # Walking to the right!
                    pVec[s] = 0.2
                    pVec[(s-1) % self._num_states] = 0.1
                else:
                    pVec[(s-1) % self._num_states] = 0.7  # Walking to the left!
                    pVec[s] = 0.2
                    pVec[(s+1) % self._num_states] = 0.1
                self._P[a][s, :] = pVec


    #Set up the reward vector
    def _populate_R(self):
        reward_position = int(self._num_states / 5)
        self._R[0][reward_position]=-1
        self._R[0][self._num_states-1-reward_position]=1
        self._R[1] = self._R[0]


# In[7]:


class Garnet(AbstractProblem):

    ENV_NAME = "Garnet"

    def __init__(self, discount, num_states=10, num_actions=1, b_P=5, b_R=5):
        self._num_states = num_states
        self._num_actions = num_actions
        self.b_P = b_P  # branching factor: number of possible next states for each (s,a) pair
        self.b_R = b_R  # number of non-zero rewards

        self._P = [np.zeros((num_states, num_states)) for _ in range(num_actions)] #List of |A| many |S| x |S| transition matrices
        self._R = [np.zeros(num_states)  for _ in range(num_actions)] #np.array of shape |S|x1

        self._populate_P()
        self._populate_R()

        P = np.zeros((num_actions, num_states, num_states))
        R = np.zeros((num_actions, num_states))
        for action in range(num_actions):
            P[action, :, :] = self._P[action]
            R[action, : ] = self._R[action]

        initial_dist = np.zeros((self._num_states))
        initial_dist[0] = 1
        
        


    def get_initial_state_dist(self):
        initial_dist = np.zeros((self._num_states))
        initial_dist[0] = 1
        return initial_dist


    # Setup up the transition probability matrix. Garnet-like (not exact implementation).
    def _populate_P(self):
        for a in range(self._num_actions):
            for s in range(self._num_states):
                p_row = np.zeros(self._num_states)
                indices = np.random.choice(self._num_states, self.b_P, replace=False)
                p_row[indices] = self._generate_stochastic_row(length=self.b_P) #Insert the non-zero transition probabilities in to P
                self._P[a][s, :] = p_row

    def _generate_stochastic_row(self, length):
        p_vec = np.append(np.random.uniform(0, 1, length- 1), [0, 1])
        return np.diff(np.sort(p_vec))  # np.array of length b_P

    #Set up the reward vector
    def _populate_R(self):
      for a in range(self._num_actions):
        self._R[a][np.random.choice(self._num_states, self.b_R, replace=False)] = np.random.uniform(0, 1, self.b_R)
