import numpy as np
from time import time
from algorithms.util import transition_matrix_policy, reward_policy
import numpy as np
import warnings

def pid_VI_pe(P, R, gamma, V0, num_iteration: int, policy, eta=0.05, epsilon = 10**(-20)):
  num_actions = P.shape[0]
  num_states = P.shape[2]
  P_pi = transition_matrix_policy(P, policy)
  R_pi = reward_policy(R, policy)

  V = np.zeros((num_iteration, num_states))
  z = np.zeros((num_iteration, num_states))
  br = np.zeros((num_iteration, num_states))
  gains = np.zeros((num_iteration, 3))
  gains_deriv = np.zeros((num_iteration, 3, num_states))


  time_trace = np.zeros((num_iteration))
  V[0] = V0.reshape(-1)
  z[0] = 0
  gains[:,0], gains[:,1], gains[:,2]= 1, 0, 0,   #kp, ki, kd
  alpha, beta = 0.05, 0.95
  time_trace[0] = 0
  time0 = time()

  for k in range(num_iteration - 1):

    br[k] = R_pi + gamma * P_pi @ V[k] - V[k]
    z[k+1] = beta * z[k] + alpha * br[k]
    V[k+1] = V[k] + gains[k, 0] * br[k] + gains[k, 1] * z[k+1] + gains[k, 2] * (V[k] - V[k-1])
    time_trace[k+1] = time() - time0

    
    if k >= 2:
      gains_deriv[k, 0] = - br[k-1] + gamma * np.dot(P_pi, br[k-1])
      gains_deriv[k, 1] = - z[k] + gamma * np.dot(P_pi, z[k]) 
      gains_deriv[k, 2] = - (V[k-1] - V[k-2]) + gamma * np.dot(P_pi, V[k-1] - V[k-2])

      gains[k+1] = gains[k] - eta * np.dot(gains_deriv[k], br[k]) / (np.dot(br[k-1], br[k-1]) + epsilon)
  
  return V, time_trace
    
    
    
    

def pid_VI_pe_old(A, B, r, V, num_iteration: int, policy):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  a = 1
  b = 0
  c = 0
  beta = 0.95
  alpha = 0.05
  eta = 0.05
  epsilon = 10**(-20)
  z=0
  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(B, policy)
  
  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V_trace[0, :] = V.reshape(-1)
  time_trace[0] = 0
  time0 = time()

  for num_iter in range(1, int(num_iteration)):

    if num_iter == 1:
      V_prev = V
      T_V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      Br = T_V-V
      z =  beta*z + alpha*Br
      V = T_V
      V_trace[num_iter, :] =V.reshape(-1)
      time_trace[num_iter] = time() - time0

    elif num_iter == 2:
      V_pprev = V_prev
      V_prev = V
      T_V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      Br = T_V-V
      z = beta*z + alpha*Br
      V = T_V
      V_trace[num_iter, :] =V.reshape(-1)
      time_trace[num_iter] = time() - time0

    elif num_iter == 3:
      T_V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      Br = T_V-V
      z_cur = beta*z + alpha*Br
      V_trace[num_iter, :] =V.reshape(-1)
      time_trace[num_iter] = time() - time0

      T_V_prev = r*np.dot(P_pi,V_prev.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      Br_prev = T_V_prev-V_prev

      a_d = -np.dot(np.identity(P_pi.shape[1]) - r*P_pi, Br_prev)
      a = a - eta*np.dot(Br.flatten(), a_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

      b_d = -np.dot(np.identity(P_pi.shape[1]) - r*P_pi, V_prev - V_pprev)
      b = b - eta*np.dot(Br.flatten(), b_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

      c_d = -np.dot(np.identity(P_pi.shape[1]) - r*P_pi, z)
      c = c - eta*np.dot(Br.flatten(), c_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

      V_pprev = V_prev
      V_prev = V
      V = T_V
      z=z_cur


    else:

      T_V = r*np.dot(P_pi,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      Br = T_V - V

      z_cur = beta*z + alpha*Br
      V_cur = (1-a)*V + a*T_V + c*z_cur + b*(V-V_prev)
      V_trace[num_iter, :] =V_cur.reshape(-1)
      time_trace[num_iter] = time() - time0

      T_V_prev = r*np.dot(P_pi,V_prev.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      Br_prev = T_V_prev-V_prev

      a_d = -np.dot(np.identity(P_pi.shape[1]) - r*P_pi, Br_prev)
      a = a - eta*np.dot(Br.flatten(), a_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

      b_d = -np.dot(np.identity(P_pi.shape[1]) - r*P_pi, V_prev - V_pprev)
      b = b - eta*np.dot(Br.flatten(), b_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

      c_d = -np.dot(np.identity(P_pi.shape[1]) - r*P_pi, z)
      c = c - eta*np.dot(Br.flatten(), c_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)


      V_pprev = V_prev
      V_prev = V
      V= V_cur
      z=z_cur

      #alpha_d = -c*np.dot(np.identity(P_pi.shape[1]) - r*P_pi, Br_prev)
      #alpha = a - eta*np.dot(Br.flatten(), alpha_d.flatten())/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

      #beta_d = -c*np.dot(np.identity(P_pi.shape[1]) - r*P_pi,z_prev)
      #beta = a - eta*np.dot(Br.flatten(),  beta_d.flatten() )/(np.linalg.norm(Br_prev, ord = 2)**2 + epsilon)

  return V_trace, time_trace