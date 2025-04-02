
import numpy as np
import matplotlib.pyplot as plt

def get_error_by_time(error_trace, time_trace, time_points):
  error_points = np.zeros(error_trace.shape[:-1] + time_points.shape)
  for trace_idx in np.ndindex(error_trace.shape[:-1]):
    idx = np.searchsorted(time_trace[trace_idx], time_points, side='right') - 1
    error_points[trace_idx] = error_trace[trace_idx][idx]
  return error_points


def get_time_by_error(error_trace, time_trace, error_points):
  n = time_trace.shape[-1]
  time_points = np.zeros(error_trace.shape[:-1] + error_points.shape)
  for trace_idx in np.ndindex(error_trace.shape[:-1]):
    idx = n - np.searchsorted(np.minimum.accumulate(error_trace[trace_idx])[::-1], error_points, side='right')
    time_points[trace_idx] = np.where(idx < n, time_trace[trace_idx][np.minimum(idx, n-1)], np.inf)
  return time_points

def plot_with_shades(ax, x, y, yerr, label, color, linestyle):
  # ax.plot(x, y, label=label, linestyle=linestyle)
  # ax.fill_between(x, y - yerr, y + yerr, alpha=0.3)

  ax.plot(x, y, color=color, label=label, linestyle=linestyle)
  ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.3)
  # if np.any(np.isinf(y)):
  #   inf_index = np.where(np.isinf(y))[0][0]
  #   y_start = y[inf_index - 1]
  #   y_end = ax.get_ylim()[1]
  #   x_start = x_end = x[inf_index - 1]
  #   ax.plot([x_start, x_end], [y_start, y_end], color=color, linestyle=linestyle)



def value_function(A, b, r, max_iteration=10000):
    num_actions = A.shape[0]
    num_states = A.shape[2]
    V = np.zeros(num_states)
    for l in range(max_iteration):
        V = (b.reshape(num_actions* num_states,-1) + r * A.reshape((-1, num_states)) @ V.reshape(num_states,1)).reshape((-1, num_states)).max(axis=0)
    return V


# In[16]:


def optimal_policy_old(A, b, r):
    V = value_function(A, b, r,  max_iteration=10000)
    num_actions = A.shape[0]
    num_states = A.shape[2]
    opt_policy = np.argmax((b.reshape(num_actions* num_states,-1) + r * A.reshape((-1, num_states)) @ (V.reshape(num_states,1))).reshape((-1, num_states)), axis=0)
    return opt_policy

def get_policy_value(P, R, discount, policy, err=1e-10, max_iteration=100000):
    num_states = P.shape[2]
    r_pi = R[policy, np.arange(num_states)]
    P_pi = P[policy, np.arange(num_states), :]
    V = np.linalg.inv(np.identity(num_states) - discount * P_pi) @ r_pi
    return V



def optimal_policy(P, R, discount, err=1e-10, max_iterations=1000):
    num_actions = R.shape[0]
    num_states = P.shape[2]
    policy = np.zeros((num_states), dtype=int)
    for i in range(max_iterations):
        V = get_policy_value(P, R, discount, policy)
        improved_policy = (R.reshape((num_states * num_actions)) + discount * P.reshape((-1, num_states)) @ V).reshape((-1, num_states)).argmax(axis=0)
        if np.all(improved_policy == policy):
            break
        policy = improved_policy
    return policy



def value_function_policy(A,b,r, policy):
  A=np.array([A[c, i, :] for i, c in enumerate(policy)])
  b=np.array([b[c, i] for i, c in enumerate(policy)])
  V = np.dot(np.linalg.inv(np.identity(A.shape[1])-r*A ), b)
  return V


# In[18]:


def transition_matrix_policy(A, policy):
  P = np.array([A[c, i, :] for i, c in enumerate(policy)])
  return P

def reward_policy(b, policy):
  R = np.array([b[c, i] for i, c in enumerate(policy)])
  return R



def proj(u, v):
  u = u-u@v.conj().T*v/np.linalg.norm(v, ord=2)**2
  return u


# In[9]:

def rank_two_matrix_with(eigvec_2,eigval_2, A, r, alpha):
  er_prev = 10**(-12)
  eigvec_1=np.ones(A.shape[1]).reshape(A.shape[1],1)
  eigval_1=1

  u_1=np.random.randn(A.shape[1],1)
  u_1=u_1-u_1.conj().T@eigvec_2*eigvec_2/np.linalg.norm(eigvec_2, ord=2)**2
  u_1=u_1/(u_1.conj().T@eigvec_1+er_prev)
  u_2=np.random.randn(A.shape[1],1)
  u_2=u_2-(u_2.conj().T@eigvec_1)*eigvec_1/(np.linalg.norm(eigvec_1, ord=2))**2
  u_2=u_2/(u_2.conj().T@eigvec_2+er_prev)

  E_2 = eigval_1*(eigvec_1@ u_1.conj().T)+eigval_2*(eigvec_2@ u_2.conj().T)
  E_2_inv = np.identity(A.shape[1])+r*alpha*eigval_1/(1-r*alpha*eigval_1)*(eigvec_1@ u_1.conj().T)+r*alpha*eigval_2/(1-r*alpha*eigval_2)*(eigvec_2@ u_2.conj().T)
  return u_1, u_2, E_2, E_2_inv




def recover_eigvec_2(V, V_prev, A, r):
  er_prev = 10**(-12)
  eigvec_1 = np.ones(A.shape[1]).reshape(A.shape[1],1)
  eigvec_1 = eigvec_1/np.linalg.norm(eigvec_1)
  eigval_1 = 1
  psuedo_eigvec_2 = (V-V_prev)/np.linalg.norm(V-V_prev)
  psuedo_eigvec_2 = psuedo_eigvec_2.reshape(A.shape[1],1)
  eigval_2 = psuedo_eigvec_2.conj().T @ A @ psuedo_eigvec_2
  eigvec_2 = psuedo_eigvec_2 -eigval_1*eigvec_1.conj().T@psuedo_eigvec_2/(eigval_1-eigval_2+er_prev)*eigvec_1
  eigvec_2 = eigvec_2/np.linalg.norm(eigvec_2)
  return eigvec_2, eigval_2


# In[10]:


def recover_eigvec_2_new(V, V_prev, V_pprev, A, r):
  er_prev = 10**(-12)
  eigvec_1 = np.ones(A.shape[1]).reshape(A.shape[1],1)
  eigvec_1 = eigvec_1/np.linalg.norm(eigvec_1)
  eigval_1 = 1
  w_prev = (V-V_prev).reshape(A.shape[1],1)
  w_pprev = (V_prev-V_pprev).reshape(A.shape[1],1)
  psuedo_eigvec_2 = (V-V_prev)
  # /np.linalg.norm(V-V_prev)
  psuedo_eigvec_2 = psuedo_eigvec_2.reshape(A.shape[1],1)
  eigval_2 = w_pprev.conj().T @ w_prev/(w_pprev.conj().T @ w_pprev) /r
  eigvec_2 = psuedo_eigvec_2 -eigval_1*eigvec_1.conj().T@psuedo_eigvec_2/(eigval_1-eigval_2+er_prev)*eigvec_1
  eigvec_2 = eigvec_2/np.linalg.norm(eigvec_2)
  return eigvec_2, eigval_2




def orthogonalize(vec_1, vec_2):
  er_prev = 10**(-12)
  vec_2=vec_2-vec_2.conj().T@vec_1*vec_1/np.linalg.norm(vec_1, ord=2)**2
  vec_2=vec_2/np.linalg.norm(vec_2, ord=2)
  vec_1=vec_1/np.linalg.norm(vec_1, ord=2)
  return vec_1, vec_2




def orthogonalize_3(vec_1, vec_2, vec_3):
  er_prev = 10**(-12)
  vec_3=vec_3-vec_3.conj().T@vec_1*vec_1/np.linalg.norm(vec_1, ord=2)**2
  vec_3=vec_3-vec_3.conj().T@vec_2*vec_2/np.linalg.norm(vec_2, ord=2)**2
  vec_3=vec_3/np.linalg.norm(vec_3, ord=2)
  return vec_1, vec_2, vec_3




def orthogonalize_4(vec_1, vec_2, vec_3, vec_4):
  er_prev = 10**(-12)
  vec_4=vec_4-vec_4.conj().T@vec_1*vec_1/np.linalg.norm(vec_1, ord=2)**2
  vec_4=vec_4-vec_4.conj().T@vec_2*vec_2/np.linalg.norm(vec_2, ord=2)**2
  vec_4=vec_4-vec_4.conj().T@vec_3*vec_3/np.linalg.norm(vec_3, ord=2)**2
  vec_4=vec_4/np.linalg.norm(vec_4, ord=2)
  return vec_1, vec_2, vec_3, vec_4




def recover_eigvec_3(V, V_prev, A, r, u_1, u_2, eigvec_2, eigval_2):
  er_prev = 10**(-12)
  eigvec_1 = np.ones(A.shape[1]).reshape(A.shape[1],1)
  eigvec_1 = eigvec_1/np.linalg.norm(eigvec_1)
  eigval_1 = 1
  psuedo_eigvec_3 = (V-V_prev)/np.linalg.norm(V-V_prev)
  psuedo_eigvec_3 = psuedo_eigvec_3.reshape(A.shape[1],1)
  eigval_3 = psuedo_eigvec_3.conj().T @ A @ psuedo_eigvec_3
  eigvec_3 = psuedo_eigvec_3 -eigval_1*u_1.conj().T@psuedo_eigvec_3/(eigval_1-eigval_3+er_prev)*eigvec_1-eigval_2*u_2.conj().T@psuedo_eigvec_3/(eigval_2-eigval_3+er_prev)*eigvec_2
  eigvec_3 = eigvec_3/np.linalg.norm(eigvec_3)
  return eigvec_3, eigval_3


def rank_one_matrix (A):
  u_1=np.ones(A.shape[1]).reshape(A.shape[1],1)
  eigvec_1=np.ones(A.shape[1]).reshape(A.shape[1],1)
  eigval_1=1
  u_1=u_1/(u_1.conj().T@eigvec_1)
  E_1=eigval_1*(eigvec_1@ u_1.conj().T)
  return E_1