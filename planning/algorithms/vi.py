import numpy as np
from time import time
from algorithms.util import transition_matrix_policy, reward_policy

def value_iteration_pe(A, b, r, V, num_iterations, policy):
    num_actions = A.shape[0]
    num_states = A.shape[2]

    P_pi = transition_matrix_policy(A, policy)
    R_pi = reward_policy(b, policy)

    V = V.reshape(-1)
    V_trace = np.zeros((num_iterations, num_states), dtype=complex)
    time_trace = np.zeros((num_iterations))
    V_trace[0, :] = V.reshape(-1)
    time_trace[0] = 0
    time0 = time()
    for l in range(1, int(num_iterations)):
        V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
        V_trace[l, :] = V.reshape(-1)
        time_trace[l] = time() - time0
    return V_trace, time_trace


def value_iteration_control(A,b, r, V, num_iterations: int):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  
  V_trace = np.zeros((num_iterations, num_states), dtype=complex)
  time_trace = np.zeros((num_iterations))
  V = V.reshape(-1)
  V_trace[0, :] = V
  time_trace[0] = 0
  time0 = time()

  Policies=[]
  for l in range(1, num_iterations):
      policy = np.argmax((b.reshape(num_actions* num_states,-1) + r * A.reshape((-1, num_states)) @ (V.reshape(num_states,1))).reshape((-1, num_states)), axis=0)
      P_pi = transition_matrix_policy(A, policy)
      R_pi = reward_policy(b, policy)
      V = r*np.dot(P_pi,(V.reshape(num_states,1)))+R_pi.reshape(num_states,1)
      V_trace[l, :] = V.reshape(-1)
  return V_trace, None