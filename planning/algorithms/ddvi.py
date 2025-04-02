from algorithms.util import *
import numpy as np
from time import time
import scipy

def qr_iteration(Ppi, rank, num_iterations):
    
    # Initialize a random orthogonal matrix
    # B = np.random.rand(Ppi.shape[0], rank) + np.random.rand(Ppi.shape[0], rank) * 1j
    B = np.random.rand(Ppi.shape[0], rank)

    # Set first vector to all-one vector since it is known
    b= np.ones(Ppi.shape[0])
    b= b/np.linalg.norm(b, ord=2)
    B[:, 0] = b
    Q, _ = np.linalg.qr(B)

    for i in range(num_iterations):
        Z = np.dot(Ppi, Q)
        Q, _ = np.linalg.qr(Z)

    eigenvalues = np.diag(np.dot(Q.conj().T, np.dot(Ppi, Q)))
    schurvectors = Q

    return eigenvalues, schurvectors


def ddvi_qr(A, b, r, V, policy, num_iterations, rank, num_qr_iterations, alpha, method="Arnoldi"):
    num_actions = A.shape[0]
    num_states = A.shape[2]
    P_pi = transition_matrix_policy(A, policy)
    R_pi = reward_policy(b, policy)

    V_trace = np.zeros((num_iterations, num_states), dtype=complex)
    time_trace = np.zeros((num_iterations))
    V = V.reshape(-1)
    V_trace[0, :] = V
    time_trace[0] = 0
    time0 = time()
    if rank > 1:  
      if method == "QR":
        eig_vals, schur_vecs = qr_iteration(P_pi, rank, num_qr_iterations)
        E = schur_vecs @ np.diag(eig_vals) @ schur_vecs.T
        E_inv = np.identity(num_states) + schur_vecs @ np.diag(alpha * eig_vals * r / (1 - alpha * r * eig_vals)) @ schur_vecs.T
      elif method == "Full_eig":
        eig_vals, left_eig_vecs, right_eig_vecs = scipy.linalg.eig(P_pi, left=True, right=True)
        idx = np.argsort(np.abs(eig_vals))[::-1][:rank]
        eig_vals, left_eig_vecs, right_eig_vecs = eig_vals[idx], left_eig_vecs[:, idx], right_eig_vecs[:, idx]
        right_eig_vecs = right_eig_vecs / np.diagonal(left_eig_vecs.conj().T @ right_eig_vecs).reshape(1, -1)
        E = right_eig_vecs @ np.diag(eig_vals) @ left_eig_vecs.conj().T
        E_inv = np.identity(num_states) + right_eig_vecs @ np.diag(alpha * eig_vals * r / (1 - alpha * r * eig_vals)) @ left_eig_vecs.conj().T
      elif method=="Arnoldi":
        eig_vals, right_eig_vecs = scipy.sparse.linalg.eigs(P_pi, k=rank, which='LM')
        schur_vecs = np.linalg.qr(right_eig_vecs)[0]
        E = schur_vecs @ np.diag(eig_vals) @ schur_vecs.conj().T
        E_inv = np.identity(num_states) + schur_vecs @ np.diag(alpha * eig_vals * r / (1 - alpha * r * eig_vals)) @ schur_vecs.conj().T
    else:
      E = rank_one_matrix(A)
      E_inv = np.identity(num_states) + alpha * r/(1 - alpha * r) * E
    for k in range(1, num_iterations):
        W = (1 - alpha) * V + alpha * (R_pi + r * (P_pi - E) @ V)
        V = E_inv @ W
        V_trace[k, :] = V
        time_trace[k] = time() - time0

    return V_trace, time_trace

    

def ddvi_AutoPI(A, b, r, V, num_iteration, policy, lower_bound, alpha):
  num_actions = A.shape[0]
  num_states = A.shape[2]
  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(b, policy)
  E = rank_one_matrix(P_pi)

  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V = V.reshape(-1)
  V_trace[0, :] = V
  time_trace[0] = 0
  time0 = time()
  err = 10**(-4)
  V_pprev = V
  V_prev = V
  first= True
  ad = False
  for l in range(1, int(num_iteration)):
    if l> lower_bound:
      power_err = (V_pprev -V_prev)/np.linalg.norm((V_pprev -V_prev))- (V_prev -V)/np.linalg.norm(V_prev -V)
      if np.linalg.norm(np.linalg.norm(power_err)) < err:
        ad = True
        if first:
          first=False
          eigvec_2, eigval_2 = recover_eigvec_2_new(V, V_prev, V_pprev , P_pi-E, r)
          u_1, u_2, E_2, E_2_inv = rank_two_matrix_with(eigvec_2, eigval_2 , P_pi, r, alpha)
          W= (np.identity(P_pi.shape[1])+r/(1-r)*E)@V.reshape(P_pi.shape[1],1)
      if ad:
        W = np.dot((1-alpha)*np.identity(P_pi.shape[1])+ alpha*r* (P_pi-E_2),W.reshape(P_pi.shape[1],1))+alpha*R_pi.reshape(P_pi.shape[1],1)
        W = E_2_inv@W.reshape(P_pi.shape[1],1)
        V_trace[l, :] = W.reshape(-1)
        time_trace[l] = time() - time0
      else:
        V_pprev = V_prev
        V_prev = V
        V = r*np.dot(P_pi-E,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
        V_trace[l, :] =( (np.identity(P_pi.shape[1])+r/(1-r)*E)@V.reshape(P_pi.shape[1],1) ).reshape(-1)
        time_trace[l] = time() - time0
    else:
      V_pprev = V_prev
      V_prev = V
      V = r*np.dot(P_pi-E,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      V_trace[l, :] = ( (np.identity(P_pi.shape[1])+r/(1-r)*E)@V.reshape(P_pi.shape[1],1) ).reshape(-1)
      time_trace[l] = time() - time0
  return V_trace, time_trace




def ddvi_AutoQR(A, b, r, V, num_iteration, policy, lower_bound, alpha):
  num_actions = A.shape[0]
  num_states = A.shape[2]

  P_pi = transition_matrix_policy(A, policy)
  R_pi = reward_policy(b, policy)
  E = rank_one_matrix(P_pi)
  
  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V = V.reshape(-1)
  V_trace[0, :] = V
  time_trace[0] = 0
  time0 = time()

  err = 10**(-4)
  V_pprev = V
  V_prev = V
  first= True
  ad = False
  first2 =True
  ad2=False
  for l in range(1, int(num_iteration)):
    if l> lower_bound:
      power_err = (V_pprev -V_prev)/np.linalg.norm((V_pprev -V_prev))- (V_prev -V)/np.linalg.norm(V_prev -V)
      if np.linalg.norm(np.linalg.norm(power_err)) < err:
        ad = True
        if first:
          first=False
          count=l
          psuedo_eigvec_2 = (V_prev -V)/np.linalg.norm(V_prev -V)
          psuedo_eigvec_2 = psuedo_eigvec_2.reshape( P_pi.shape[0],1)
          eigval_2 = psuedo_eigvec_2.transpose() @  (P_pi-E) @ psuedo_eigvec_2
          eigval_1=1
          schur_1 = np.ones(P_pi.shape[1]).reshape(A.shape[1],1)
          schur_1=schur_1/np.linalg.norm(schur_1)
          schur_1, schur_2= orthogonalize(schur_1, psuedo_eigvec_2)
          E_2 = eigval_1*(schur_1@ schur_1.transpose())+eigval_2*(schur_2@ schur_2.transpose())
          E_2_inv = np.identity(A.shape[1])+r*alpha*eigval_1/(1-r*alpha*eigval_1)*(schur_1@ schur_1.transpose())+r*alpha*eigval_2/(1-r*alpha*eigval_2)*(schur_2@ schur_2.transpose())
          W= (np.identity(P_pi.shape[1])+r/(1-r)*E)@V.reshape(P_pi.shape[1],1)
          W_pprev=W
          W_prev= W
      if ad:
        if l> count+lower_bound:
          power_err = (W_pprev -W_prev)/np.linalg.norm((W_pprev -W_prev))- (W_prev -W)/np.linalg.norm(W_prev -W)
          if np.linalg.norm(np.linalg.norm(power_err)) < err:
            ad2 = True
            if first2:
              first2=False
              w_pprev = (W_prev -W_pprev).reshape( P_pi.shape[0],1)
              w_prev = (W-W_prev).reshape( P_pi.shape[0],1)
              psuedo_eigval_3= w_pprev.transpose()@w_prev/(w_pprev.transpose()@w_pprev)
              eigval_3= psuedo_eigval_3
              schur_1, schur_2, schur_3= orthogonalize_3(schur_1, schur_2, w_prev)
              E_3 = eigval_1*(schur_1@ schur_1.transpose())+eigval_2*(schur_2@ schur_2.transpose())+eigval_3*(schur_3@ schur_3.transpose())
              E_3_inv = np.identity(A.shape[1])+r*alpha*eigval_1/(1-r*alpha*eigval_1)*(schur_1@ schur_1.transpose())+r*alpha*eigval_2/(1-r*alpha*eigval_2)*(schur_2@ schur_2.transpose())+r*alpha*eigval_3/(1-r*alpha*eigval_3)*(schur_3@ schur_3.transpose())
              Z=W.reshape(P_pi.shape[1],1)
          if ad2:
            Z = np.dot((1-alpha)*np.identity(P_pi.shape[1])+ alpha*r* (P_pi-E_3),Z.reshape(P_pi.shape[1],1))+alpha*R_pi.reshape(P_pi.shape[1],1)
            Z = E_3_inv@Z.reshape(P_pi.shape[1],1)
            V_trace[l, :] = Z.reshape(-1)
            time_trace[l] = time() - time0
          else:
            W_pprev = W_prev
            W_prev = W
            W = np.dot((1-alpha)*np.identity(P_pi.shape[1])+ alpha*r* (P_pi-E_2),W.reshape(P_pi.shape[1],1))+alpha*R_pi.reshape(P_pi.shape[1],1)
            W = E_2_inv@W.reshape(P_pi.shape[1],1)
            V_trace[l, :] = W.reshape(-1)
            time_trace[l] = time() - time0
        else:
          W_pprev = W_prev
          W_prev = W
          W = np.dot((1-alpha)*np.identity(P_pi.shape[1])+ alpha*r* (P_pi-E_2),W.reshape(P_pi.shape[1],1))+alpha*R_pi.reshape(P_pi.shape[1],1)
          W = E_2_inv@W.reshape(P_pi.shape[1],1)
          V_trace[l, :] = W.reshape(-1)
          time_trace[l] = time() - time0
      else:
        V_pprev = V_prev
        V_prev = V
        V = r*np.dot(P_pi-E,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
        V_trace[l, :] =( (np.identity(P_pi.shape[1])+r/(1-r)*E)@V.reshape(P_pi.shape[1],1) ).reshape(-1)
        time_trace[l] = time() - time0
    else:
      V_pprev = V_prev
      V_prev = V
      V = r*np.dot(P_pi-E,V.reshape(P_pi.shape[1],1))+R_pi.reshape(P_pi.shape[1],1)
      V_trace[l, :] = ( (np.identity(P_pi.shape[1])+r/(1-r)*E)@V.reshape(P_pi.shape[1],1) ).reshape(-1)
      time_trace[l] = time() - time0
  return V_trace, time_trace

def ddvi_for_control(A,b, r, V, num_iteration: int):
  num_actions = A.shape[0]
  num_states = A.shape[2]

  V_trace = np.zeros((num_iteration, num_states), dtype=complex)
  time_trace = np.zeros((num_iteration))
  V = V.reshape(-1)
  V_trace[0, :] = V
  time_trace[0] = 0
  time0 = time()

  Policies=[]
  E = rank_one_matrix(A)
  for l in range(1, num_iteration):
      policy = np.argmax((b.reshape(num_actions* num_states,-1) + r * A.reshape((-1, num_states)) @ (V.reshape(num_states,1))).reshape((-1, num_states)), axis=0)
      Policies.append(policy)
      P_pi = transition_matrix_policy(A, policy)
      R_pi = reward_policy(b, policy)
      V = r*np.dot(P_pi-E,(V.reshape(num_states,1)))+R_pi.reshape(num_states,1)
      V_trace[l, :] = ( np.dot(np.identity(A.shape[2])+r/(1-r)*E, V.reshape(num_states,1)) ).reshape(-1)
  return V_trace, None