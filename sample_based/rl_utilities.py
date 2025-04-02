import numpy as np


def get_policy_value_mdp(mdp, policy, err=1e-10, max_iteration=100000):
    return get_policy_value(mdp.P(), mdp.R(), mdp.discount(), policy, err, max_iteration)


def get_policy_value(P, R, discount, policy, err=1e-10, max_iteration=100000):
    num_states = P.shape[2]
    r_pi = R[policy, np.arange(num_states)]
    P_pi = P[policy, np.arange(num_states), :]
    
    # V = np.zeros((num_states))
    # for k in range(max_iteration):
    #     new_V = r_pi + discount * P_pi @ V
    #     if np.max(np.abs(new_V - V)) < err:
    #         return new_V
    #     V = new_V 
    V = np.linalg.inv(np.identity(num_states) - discount * P_pi) @ r_pi
    return V

def get_policy_value_count_iters(P, R, discount, policy, err=1e-10, max_iteration=100000):
    num_states = P.shape[2]
    r_pi = R[policy, np.arange(num_states)]
    P_pi = P[policy, np.arange(num_states), :]
    
    V = np.zeros((num_states))
    iters = 0
    while iters < max_iteration:
        new_V = r_pi + discount * P_pi @ V
        if np.max(np.abs(new_V - V)) < err:
            return new_V, iters
        V = new_V 
        iters += 1

    return V, iters

def get_policy_value(P, R, discount, policy, err=1e-10, max_iteration=100000):
    num_states = P.shape[2]
    r_pi = R[policy, np.arange(num_states)]
    P_pi = P[policy, np.arange(num_states), :]
    
    # V = np.zeros((num_states))
    # for k in range(max_iteration):
    #     new_V = r_pi + discount * P_pi @ V
    #     if np.max(np.abs(new_V - V)) < err:
    #         return new_V
    #     V = new_V 
    V = np.linalg.inv(np.identity(num_states) - discount * P_pi) @ r_pi
    return V

def get_optimal_policy_mdp(mdp):
    return get_optimal_policy(mdp.P(), mdp.R(), mdp.discount(), mdp.num_states(), mdp.num_actions(), err=1e-10, max_iterations=100000)


def get_optimal_policy(P, R, discount, num_states, num_actions, err=1e-10, max_iterations=100000):
    V = value_iteration(P, R.reshape((num_states * num_actions)), discount, err=err, max_iteration=max_iterations)
    optimal_policy = (R.reshape((num_states * num_actions)) + discount * P.reshape((-1, num_states)) @ V).reshape((-1, num_states)).argmax(axis=0)
    return optimal_policy


def sample_from_mdp(num_samples, mdp, policy):
    num_states = mdp.num_states()
    samples = []
    for i in range(num_samples):
        s = int(np.random.choice(np.arange(num_states), p=np.ones((num_states)) / num_states))
        a = policy[s]
        r = mdp.R()[a, s]
        next_s = int(np.random.choice(np.arange(num_states), p=mdp.P()[a, s, :]))
        samples.append(np.array([s, a, r, next_s]))
    return samples


def sample_one_from_mdp(mdp, policy):
    (s,a,r,next_s) = sample_from_mdp(1, mdp, policy)[0]
    return int(s),int(a), r, int(next_s)


def sample_uniform_from_mdp(num_samples, mdp):
    num_states, num_actions = mdp.num_states(), mdp.num_actions()
    samples = []
    for i in range(num_samples):
        s = int(np.random.choice(np.arange(num_states), p=np.ones((num_states)) / num_states))
        a = int(np.random.choice(np.arange(num_actions), p=np.ones((num_actions)) / num_actions))
        r = mdp.R()[a, s]
        next_s = int(np.random.choice(np.arange(num_states), p=mdp.P()[a, s, :]))
        samples.append(np.array([s, a, r, next_s]))
    return samples


def sample_one_uniform_from_mdp(mdp):
    (s, a, r, next_s) = sample_uniform_from_mdp(1, mdp)[0]
    return int(s),int(a), r, int(next_s)


def value_iteration(P, R, discount, err=1e-6, max_iteration=10000):
    num_states = P.shape[2]
    V = np.zeros(num_states)
    for l in range(max_iteration):
        new_V = (R + discount * P.reshape((-1, num_states)) @ V).reshape((-1, num_states)).max(axis=0)
        if np.max(np.abs(new_V-V)) < err:
            return new_V
        V = new_V
    return V


# def get_lr(current_iter, start_lr, total_iter, delay):
#     if current_iter <= delay:
#         return start_lr
#     # return start_lr * (total_iter-current_iter) / (total_iter-delay)
#     return start_lr * 1 / (current_iter - delay)