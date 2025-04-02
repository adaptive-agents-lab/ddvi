import matplotlib.pyplot as plt
from environments import *
from tqdm import tqdm
from algorithms.ddvi import ddvi_AutoPI, ddvi_AutoQR
from algorithms.vi import value_iteration_pe
from algorithms.util import value_function_policy, optimal_policy


np.random.seed(0)

# mdp=Maze55(0.9, 0.9)
mdp = CliffWalkmodified(0.9, 0.995)
# mdp=ChainWalk(50)

P=np.array(mdp.P())
R=np.array(mdp.R())

r=0.995
num_iter =1000
power_iter = 400
num_runs=1
alpha = 0.99

policy = optimal_policy(P,R, r)
print("g")
if mdp.ENV_NAME == "Maze55":
    policy = np.array([2, 2, 3, 0, 3, 0, 2, 1, 3, 2, 2, 2, 3, 3, 1, 0, 3, 0, 3, 3, 2, 2, 1, 1, 0])
elif mdp.ENV_NAME == "CliffWalkmodified":
    policy = np.array([0, 0, 2, 1, 2, 1, 2, 1, 0, 2, 0, 3, 2, 0, 3, 1, 0, 1, 1, 1, 3])



V_traces = np.zeros((num_runs, 3, num_iter, mdp.num_states()))
v_errors = np.zeros((num_runs, 3, num_iter))

for run_i in tqdm(range(num_runs)):
    if mdp.ENV_NAME == "Garnet":
        mdp = Garnet(0.9, 100, 8, 6, 10)
    V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

    print("VI")
    V_traces[run_i, 0, :], _ = value_iteration_pe(P, R, r, V, num_iter, policy)
    V_traces[run_i, 1, :], _ = ddvi_AutoQR(P, R, r, V, num_iter, policy, 20, alpha)
    V_traces[run_i, 2, :], _ = ddvi_AutoPI(P, R, r, V, num_iter, policy, 20, alpha)

    opt= value_function_policy(P,R,r, policy).reshape(-1)
    v_errors[run_i] = np.linalg.norm(V_traces[run_i] - opt, ord=1, axis=-1) / np.linalg.norm(opt, ord=1)
v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)



plt.figure()
x  = np.arange(0,num_iter)
plt.plot(x, v_errors_mean[0, :], color='b', label = "VI", linestyle='-')
plt.plot(x, v_errors_mean[1, :], color='r', label = "DDVI with AutoQR", linestyle='-')

# naming the x axis
plt.xlabel('Iterations (k)', fontsize=13)
plt.yscale('log')
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', fontsize=12)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/DDVI_AutoQR_{mdp.ENV_NAME}")


plt.figure()
x  = np.arange(0,num_iter)
plt.plot(x, v_errors_mean[0, :], color='b', label = "VI", linestyle='-')
plt.plot(x, v_errors_mean[2, :], color='r', label = "DDVI with AutoPI", linestyle='-')

# naming the x axis
plt.xlabel('Iterations (k)', fontsize=13)
plt.yscale('log')
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', fontsize=12)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/DDVI_AutoPI_{mdp.ENV_NAME}")
