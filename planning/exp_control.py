import matplotlib.pyplot as plt
from environments import *
from tqdm import tqdm
from algorithms.ddvi import ddvi_for_control
from algorithms.vi import value_iteration_control
from algorithms.util import value_function, optimal_policy, value_function_policy

np.random.seed(0)

r=0.995

num_runs =1000
num_states = 100
num_actions = 8
b_P = 6
b_R = 10
mdp = Garnet(0.9, num_states, num_actions, b_P, b_R)
exp_name = f"Garnet-{num_states}-{b_P}-{b_R}-{r}"

# num_runs =1
# mdp = ChainWalk(50)
# exp_name = f"ChainWalk"







num_iter =100
alpha = 0.99

P=np.array(mdp.P())
R=np.array(mdp.R())



V_traces = np.zeros((num_runs, 2, num_iter, mdp.num_states()))
v_errors = np.zeros((num_runs, 2, num_iter))

for run_i in tqdm(range(num_runs)):
    if mdp.ENV_NAME == "Garnet":
        mdp = Garnet(0.9, num_states, num_actions, b_P, b_R)
        P=np.array(mdp.P())
        R=np.array(mdp.R())
    V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

    V_traces[run_i, 0, :], _ = value_iteration_control(P, R, r, V, num_iter)
    V_traces[run_i, 1, :], _ = ddvi_for_control(P, R, r, V, num_iter)

    opt_policy = optimal_policy(P, R, r)
    opt = value_function_policy(P, R, r, opt_policy).reshape(-1)
    v_errors[run_i] = np.linalg.norm(V_traces[run_i] - opt, ord=1, axis=-1) / np.linalg.norm(opt, ord=1)
v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)



plt.figure()
x  = np.arange(0,num_iter)
plt.plot(x, v_errors_mean[0, :], color='b', label = "VI", linestyle='-')
plt.plot(x, v_errors_mean[1, :], color='r', label = "rank-$1$ DDVI for control", linestyle='-')
if num_runs > 1:
  plt.fill_between(x, v_errors_mean[0, :] - v_errors_se[0, :], v_errors_mean[0, :] + v_errors_se[0, :], color='b', alpha=0.3)
  plt.fill_between(x, v_errors_mean[1, :] - v_errors_se[1, :], v_errors_mean[1, :] + v_errors_se[1, :], color='r', alpha=0.3)



# naming the x axis
plt.xlabel('Iterations (k)', fontsize=13)
plt.yscale('log')
# naming the y axis
plt.ylabel('Normalized $\|V^{k} - V^{\star}\|_{1}$', fontsize=12)
plt.legend(loc='lower left')
plt.grid(alpha=0.3)

plt.savefig(f'planning/output/DDVI_control_{exp_name}.png')
plt.savefig(f'planning/output/DDVI_control_{exp_name}.pdf')