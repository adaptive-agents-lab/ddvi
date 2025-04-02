import numpy as np
from time import time
from algorithms.util import transition_matrix_policy, reward_policy
from scipy.optimize import minimize

def anderson_VI_pe(P, R, r, V0, num_iteration: int, policy, m):
  num_actions = P.shape[0]
  num_states = P.shape[2]
  P_pi = transition_matrix_policy(P, policy)
  R_pi = reward_policy(R, policy)

  V = np.zeros((num_iteration, num_states))
  time_trace = np.zeros((num_iteration))
  TV = np.zeros((num_iteration, num_states))
  V[0] = V0.reshape(-1)
  time_trace[0] = 0
  time0 = time()
  TV[0] = V[1] = R_pi + r * P_pi @ V[0]
  TV[1] = R_pi + r * P_pi @ V[1]
  time_trace[1] = time() - time0
  for k in range(1, num_iteration-1):
    m_k = min(m, k)
    Delta_k = TV[k-m_k:k+1].T - V[k-m_k:k+1].T
    ones_vec = np.ones(m_k+1)
    # numerator = np.linalg.inv(Delta_k.T @ Delta_k) @ ones_vec 
    numerator = np.linalg.pinv(Delta_k.T @ Delta_k) @ ones_vec  # for Maze
    alpha_kp1 = numerator / (ones_vec.T @ numerator)
    V[k+1] = TV[k-m_k:k+1].T @ alpha_kp1
    TV[k+1] = R_pi + r * P_pi @ V[k+1]
    time_trace[k+1] = time() - time0
  
  return V, time_trace

  

def objective_function(x, A):
    return np.linalg.norm(np.dot(A, x))

def equality_constraint(x):
    return np.sum(x) - 1
    
def anderson_VI_pe_old(A, b, r, V, num_iteration: int, policy):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  
  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(b, policy)
  
  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V_trace[0, :] = V.reshape(-1)
  time_trace[0] = 0
  time0 = time()
  
  for iter_num in range(1, int(num_iteration)):
    if iter_num == 1:
      V_4 = V
      T_V_4 = r*np.dot(P_pi,V_4.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      V_trace[iter_num, :] = V_4.reshape(-1)
      time_trace[iter_num] = time() - time0
      V_3 = T_V_4

    elif iter_num == 2:
      T_V_3 = r*np.dot(P_pi,V_3.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      merged_delta = np.concatenate(((T_V_4-V_4).flatten(),(T_V_3-V_3).flatten()))
      delta = merged_delta.reshape(2, P_pi.shape[1])
      alpha = np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(2))/ np.dot(np.ones(2),np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(2)))
      V_2 = alpha[0]*T_V_4+ alpha[1]*T_V_3
      V_trace[iter_num, :] = V_2.reshape(-1)
      time_trace[iter_num] = time() - time0

    elif iter_num == 3:
      T_V_2 = r*np.dot(P_pi,V_2.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      merged_delta = np.concatenate(((T_V_4-V_4).flatten(),(T_V_3-V_3).flatten(), (T_V_2-V_2).flatten()))
      delta = merged_delta.reshape(3, P_pi.shape[1])
      alpha = np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(3))/ np.dot(np.ones(3),np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(3)))
      V_1 = alpha[0]*T_V_4+ alpha[1]*T_V_3 + alpha[2]*T_V_2
      V_trace[iter_num, :] = V_1.reshape(-1)
      time_trace[iter_num] = time() - time0

    elif iter_num == 4:
      T_V_1 = r*np.dot(P_pi,V_1.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      merged_delta  = np.concatenate(((T_V_4-V_4).flatten(),(T_V_3-V_3).flatten(), (T_V_2-V_2).flatten(), (T_V_1-V_1).flatten()))
      delta = merged_delta.reshape(4, P_pi.shape[1])
      delta = delta.transpose()
      x_initial = np.zeros(4)
      # bounds = [(0, None) for _ in range(4)]
      opt_result = minimize(objective_function, x_initial, args=(delta,), constraints={'type': 'eq', 'fun': equality_constraint}, method='SLSQP')
      alpha = opt_result.x
      # alpha = np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(4))/ np.dot(np.ones(4),np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(4)))
      V_0 = alpha[0]*T_V_4+ alpha[1]*T_V_3 + alpha[2]*T_V_2 + alpha[3]*T_V_1
      V_trace[iter_num, :] = V_0.reshape(-1)
      time_trace[iter_num] = time() - time0

    else:
      T_V_0 = r*np.dot(P_pi,V_0.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      merged_delta = np.concatenate(((T_V_4-V_4).flatten(),(T_V_3-V_3).flatten(), (T_V_2-V_2).flatten(), (T_V_1-V_1).flatten(), (T_V_0-V_0).flatten()))
      delta = merged_delta.reshape(5, P_pi.shape[1])
      delta = delta.transpose()
      x_initial = np.zeros(5)
      # bounds = [(0, None) for _ in range(5)]
      opt_result = minimize(objective_function, x_initial, args=(delta,), constraints={'type': 'eq', 'fun': equality_constraint}, method='SLSQP')
      alpha = opt_result.x
      # print(alpha)
      # alpha = np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(5))/ np.dot(np.ones(5),np.dot(np.linalg.inv(delta @ delta.transpose()), np.ones(5)))
      V_r = alpha[0]*T_V_4+ alpha[1]*T_V_3 + alpha[2]*T_V_2 + alpha[3]*T_V_1 + alpha[4]*T_V_0
      V_trace[iter_num, :] = V_r.reshape(-1)
      time_trace[iter_num] = time() - time0

      V_1 = V_0
      T_V_1 = T_V_0
      V_2 = V_1
      T_V_2 = T_V_1
      V_3 = V_2
      T_V_3 = T_V_2
      V_4 = V_3
      T_V_4 = T_V_3
      V_0 = V_r

    
  return V_trace, time_trace