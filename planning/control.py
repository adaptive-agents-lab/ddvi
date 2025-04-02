import numpy as np
import matplotlib.pyplot as plt
from environments import *
from algorithms.ddvi import *
from algorithms.vi import *

def get_v_optimal(Q):
    return Q.max(axis=0)

def get_v_policy(Q, policy):
    return np.choose(policy, Q)

def get_greedy_policy(Q):
    return Q.argmax(axis=0)

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

def get_Q_policy(policy, P, R, gamma, n_states, n_actions, err=1e-6, max_iteration=10_000):

    V = get_policy_value(P, R, gamma, policy)
    Q = np.zeros((n_actions, n_states))
    for a in range(n_actions):
        Q[a, :] = R[a, :] + gamma * P[a, :, :] @ V
    return Q    

def get_Q_optimal(P, R, gamma, n_states, n_actions, err=1e-6, max_iteration=10_000):
    Q = np.zeros((n_actions, n_states))
    for k in range(max_iteration):
        V = get_v_optimal(Q)
        new_Q = np.zeros((n_actions, n_states))
        for a in range(n_actions):
            new_Q[a, :] = R[a, :] + gamma * P[a, :, :] @ V
        if np.max(np.abs(Q - new_Q)) < err:
            return new_Q
        Q = new_Q
    return Q




def value_iteration(P, R, discount, err=1e-6, max_iteration=10000):
    num_states = P.shape[2]
    V = np.zeros(num_states)
    for l in range(max_iteration):
        new_V = (R + discount * P.reshape((-1, num_states)) @ V).reshape((-1, num_states)).max(axis=0)
        if np.max(np.abs(new_V-V)) < err:
            return new_V
        V = new_V
    return V

def get_optimal_policy(P, R, discount, num_states, num_actions):
    pi = np.zeros((n_states), dtype=int)
    new_pi = np.ones((n_states), dtype=int)
    while not np.array_equal(pi, new_pi):
        pi = new_pi
        Q = get_Q_policy(pi, P, R, discount, num_states, num_actions)
        new_pi = get_greedy_policy(Q)
    return pi



########### Environment ###########

mdp = ChainWalk(2000)
# mdp = CliffWalkmodified(0.9, 0.995)
gamma = 0.999
n_states = mdp.num_states()
n_actions = mdp.num_actions()
P = np.array(mdp.P())
R = np.array(mdp.R())



# n = 100  #Each part of chain
# gamma = 0.9999
# g = 1e30
# eps = 1e-4

# n_states = 2 * n + 1
# n_actions = 2

# P = np.zeros((n_actions, n_states, n_states))
# P[0, 0:n, 0:n] = np.identity(n)
# P[0, n:2*n-1, n+1:2*n] = np.identity(n-1)
# P[0, 2*n-1, 2*n-1] = 1
# P[0, 2*n, 2*n] = 1
# P[1, :, 2*n] = 1


# Phat = np.zeros((n_actions, n_states, n_states))
# Phat[0, 0:2*n-1, 1:2*n] = np.identity(2*n-1)
# Phat[0, 2*n-1, 2*n-1] = 1
# Phat[0, 2*n, 2*n] = 1
# Phat[1, :, 2*n] = 1

# R = np.zeros((n_actions, n_states))
# R[:, 2*n-1] = g * (1 - gamma)
# R[:, 2*n] = eps * (1 - gamma)

# n = 100  #Each part of chain
# gamma = 0.99
# g = 1e30
# eps = 1e-4

# n_states = 2 * n + 1
# n_actions = 2

# P = np.zeros((n_actions, n_states, n_states))
# P[0, 0:n, 0:n] = np.identity(n)
# P[0, n:2*n-1, n+1:2*n] = np.identity(n-1)
# P[0, 2*n-1, 2*n-1] = 1
# P[0, 2*n, 2*n] = 1
# P[1, :, 2*n] = 1


# R = np.zeros((n_actions, n_states))
# R[:, 2*n-1] = g * (1 - gamma)
# R[:, 2*n] = eps * (1 - gamma)



########### Algorithms ########### 

max_iters = 50
load_file = False

# VI
# pi = np.ones((n_states), dtype=int)
# Q = np.zeros((n_actions, n_states))
# vi_policy_trace = np.ones((max_iters, n_states), dtype=int)
# for k in range(1, max_iters):
#     V = get_v_optimal(Q)
#     new_Q = np.zeros((n_actions, n_states))
#     for a in range(n_actions):
#         new_Q[a, :] = R[a, :] + gamma * P[a, :, :] @ V
#     Q = new_Q
#     pi = get_greedy_policy(Q)
#     vi_policy_trace[k, :] = pi
# vi_return_trace = np.zeros((max_iters))
# for k in range(max_iters):
#     V = get_policy_value(P, R, gamma, vi_policy_trace[k])
#     vi_return_trace[k] = np.sum(V)/n_states

eval_ops_vals = [30]
num_eval_ops_vals = len(eval_ops_vals)


if load_file:
    policy_trace = np.load(f"planning/output/Control_{mdp.ENV_NAME}_policies.npy")
else:
    policy_trace = np.zeros((2, num_eval_ops_vals, max_iters, n_states), dtype=int)

    for i in range(len(eval_ops_vals)):
        eval_ops = eval_ops_vals[i]

        # MPI
        print("Rnning MPI")
        pi = np.ones((n_states), dtype=int)
        Q = np.zeros((n_actions, n_states))
        for k in range(1, max_iters):
            print(f"Rnning MPI - iter {k}")
            for _ in range(eval_ops):
                V = get_v_optimal(Q)
                new_Q = np.zeros((n_actions, n_states))
                for a in range(n_actions):
                    new_Q[a, :] = R[a, :] + gamma * P[a, :, :] @ V
                Q = new_Q
            pi = get_greedy_policy(Q)
            policy_trace[0, i, k, :] = pi

        # DDMPI
        pi = np.ones((n_states), dtype=int)
        Q = np.zeros((n_actions, n_states))
        for k in range(1, max_iters):
            print(f"Rnning DDMPI - iter {k}")
            V = Q[pi, np.arange(n_states)]
            V_trace, time_trace = ddvi_AutoQR(P, R, gamma, V, eval_ops, pi, 20, 0.99)
            # V_trace, time_trace = ddvi_qr(P, R, gamma, V, pi, eval_ops, 1, 0, 0.99)
            # V_trace, time_trace = value_iteration_pe(P, R, gamma, V, eval_ops, pi)

            V = V_trace[-1, :]
            for a in range(n_actions):
                Q[a, :] = R[a, :] + gamma * P[a, :, :] @ V
            pi = get_greedy_policy(Q)
            policy_trace[1, i, k, :] = pi

    np.save(f"planning/output/Control_{mdp.ENV_NAME}_policies.npy", policy_trace)

print("Evaluating Policies")
return_trace = np.zeros((2, num_eval_ops_vals, max_iters))
for alg_id in range(2):
    for i in range(num_eval_ops_vals):
        for k in range(max_iters):
            V = get_policy_value(P, R, gamma, policy_trace[alg_id, i, k])
            return_trace[alg_id, i, k] = np.sum(V)/n_states
# PI
# pi = np.ones((n_states), dtype=int)
# pi_policy_trace = np.ones((max_iters, n_states), dtype=int)
# for k in range(1, max_iters):
#     Q = get_Q_policy(pi, P, R, gamma, n_states, n_actions)
#     pi = get_greedy_policy(Q)
#     pi_policy_trace[k, :] = pi
# pi_return_trace = np.zeros((max_iters))
# for k in range(max_iters):
#     V = get_policy_value(P, R, gamma, pi_policy_trace[k])
#     pi_return_trace[k] = np.sum(V)/n_states

plt.plot(return_trace[0, 0], label="MPI (VI)")
plt.plot(return_trace[1, 0], label="MPI (DDVI)")
plt.legend()
plt.savefig("control.png")


