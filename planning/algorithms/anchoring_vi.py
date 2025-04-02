import numpy as np
from time import time
from algorithms.util import transition_matrix_policy, reward_policy


def anc_VI_pe(A, b, r, V0, num_iteration: int, policy):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  
  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(b, policy)

  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V_trace[0, :] = V = V0.reshape(-1)
  time_trace[0] = 0
  time0 = time()

  if r==1:
    avg_fac=[1/(1+l) for l in range(1,num_iteration+1)]
  else:
    avg_fac=[(1-r**(-2))/((1-r**(-2*l-2))) for l in range(1,num_iteration+1)]
  for l in range(int(num_iteration - 1)):
    T_V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
    V=(1-avg_fac[l])*T_V+(avg_fac[l])*V0
    V_trace[l + 1, :] = V.reshape(-1)
    time_trace[l + 1] = time() - time0
  return V_trace, time_trace