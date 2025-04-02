import numpy as np
from time import time
from algorithms.util import transition_matrix_policy, reward_policy

def nesterov_VI_pe(A, b, r, V, num_iteration: int, policy):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  
  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V_trace[0, :] = V.reshape(-1)
  time_trace[0] = 0
  time0 = time()

  alpha = 1/(1+r)
  gamma = (1 - np.sqrt(1-r**2)) / r
  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(b, policy)
  r_hat = (1+r)/2
  for num_iter in range(1, int(num_iteration)):
    if num_iter == 1:
      V_prev = V
      V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      V_trace[num_iter, :] = V.reshape(-1)
      time_trace[num_iter] = time() - time0
      init_cond = V-V_prev
    else:
      h = V+ gamma*(V-V_prev)
      T_h = r*np.dot(P_pi,h.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      V_s = h-alpha*(h-T_h)
      T_V_s = r*np.dot(P_pi,V_s.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      if np.linalg.norm(T_V_s - V_s, ord = np.inf) <= np.linalg.norm(init_cond, ord = np.inf)*r_hat**(num_iter+1):
        V_prev = V
        V = V_s
      else:
        V_prev = V
        V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      V_trace[num_iter, :] = V.reshape(-1)
      time_trace[num_iter] = time() - time0
  return V_trace, time_trace

def nesterov_MVI_pe(A, b, r, V, num_iteration: int, policy):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  
  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  alpha = 2/(1+np.sqrt(1-r**2))
  beta= (1-np.sqrt(1-r**2))/ (1+np.sqrt(1-r**2))
  # alpha, beta = 1.1, 0.1
  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(b, policy)
  V = V.reshape(-1)
  time_trace[0] = 0
  V_trace[0, :] = V_prev = V
  time0 = time()
  V_trace[1, :] = V = R_pi + r * P_pi @ V
  for num_iter in range(2, int(num_iteration)):
    new_V = V - alpha * (V - R_pi - r * P_pi @ V) + beta * (V - V_prev)
    V_trace[num_iter, :] = new_V.reshape(-1)
    time_trace[num_iter] = time() - time0
    V_prev = V
    V = new_V
  return V_trace, time_trace