import matplotlib.pyplot as plt
from environments import *
from tqdm import tqdm
from algorithms.ddvi import ddvi_qr
from algorithms.vi import value_iteration_pe
from algorithms.util import value_function_policy, optimal_policy, get_error_by_time, plot_with_shades, get_time_by_error


np.random.seed(0)

def get_mdp():
    # mdp = Garnet(0.9, 100, 1, 100, 1)
    # mdp=Maze55(0.9, 0.9)
    # mdp = CliffWalkmodified(0.9, 0.995)
    mdp=ChainWalk(50)
    return mdp
mdp = get_mdp()
P=np.array(mdp.P())
R=np.array(mdp.R())

r=0.999
num_iter = 10000
power_iter = 400
num_runs= 20
alpha = 0.99
rank_vals = [1, 2, 3, 4, 5]
num_ranks = len(rank_vals)

policy = optimal_policy(P,R, r)
if mdp.ENV_NAME == "Maze55":
    policy = np.array([2, 2, 3, 0, 3, 0, 2, 1, 3, 2, 2, 2, 3, 3, 1, 0, 3, 0, 3, 3, 2, 2, 1, 1, 0])
elif mdp.ENV_NAME == "ChainWalk":
    # policy = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    n = mdp.num_states()
    # policy = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1])
    # policy = np.concatenate([np.zeros(n//2), np.ones(n//2)]).astype(int)
    policy = np.array([0, 1, 1, 0, 1, 1] + [0] * int(n/2-6) + [1] * int(n/2-6) + [1, 0, 0, 1, 0, 0]).astype(int)
    # policy = np.ones((n)).astype(int)



run_V_traces = np.zeros((num_ranks + 1, num_iter, mdp.num_states()))
v_errors = np.zeros((num_runs, num_ranks + 1, num_iter))
time_traces = np.zeros((num_runs, num_ranks + 1, num_iter))

for run_i in tqdm(range(num_runs)):
    if mdp.ENV_NAME == "Garnet":
        mdp = get_mdp()
        P=np.array(mdp.P())
        R=np.array(mdp.R())
        policy = optimal_policy(P,R, r)
    V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

    run_V_traces[0, :], time_traces[run_i, 0, :] = value_iteration_pe(P, R, r, V, num_iter, policy)
    for rank_i in range(num_ranks):
        run_V_traces[1 + rank_i, :], time_traces[run_i, 1+rank_i, :] = ddvi_qr(P, R, r, V, policy, num_iter, rank_vals[rank_i], power_iter, alpha)

    opt= value_function_policy(P,R,r, policy).reshape(-1)
    v_errors[run_i] = np.linalg.norm(run_V_traces - opt, ord=1, axis=-1) / np.linalg.norm(opt, ord=1)


v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)

fig, ax = plt.subplots()

x  = np.arange(0, 1000)
plot_with_shades(ax, x, v_errors_mean[0, x], v_errors_se[0, x], color='b', label = "VI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[1, x], v_errors_se[1, x], color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[2, x], v_errors_se[2, x], color='g', label = "rank-$2$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[3, x], v_errors_se[3, x], color='y', label = "rank-$3$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[4, x], v_errors_se[4, x], color='orange', label = "rank-$4$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[5, x], v_errors_se[5, x], color='purple', label = "rank-$5$ DDVI", linestyle='-')

# naming the x axis
plt.xlabel('Iterations (k)', fontsize=13)
plt.yscale('log')
plt.ylim(bottom=1e-10)
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', fontsize=12)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/DDVI_QR_{mdp.ENV_NAME}")
################################################################################
time_points = np.linspace(0, 70e-3, 100)
error_points = get_error_by_time(v_errors, time_traces, time_points)

error_points_mean = np.mean(error_points, axis = 0)
error_points_se = np.std(error_points, axis = 0) / np.sqrt(num_runs)

plt.figure()

fig, ax = plt.subplots()

plot_with_shades(ax, time_points * 1000, error_points_mean[0, :], error_points_se[0, :], color='b', label = "VI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[1, :], error_points_se[1, :], color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[2, :], error_points_se[2, :], color='g', label = "rank-$2$ DDVI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[3, :], error_points_se[2, :], color='y', label = "rank-$3$ DDVI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[4, :], error_points_se[4, :], color='orange', label = "rank-$4$ DDVI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[5, :], error_points_se[5, :], color='purple', label = "rank-$5$ DDVI", linestyle='-')

# naming the x axis
plt.xlabel('Wall Clock (ms)', fontsize=13)
plt.yscale('log')
plt.ylim(bottom=1e-10)
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', fontsize=12)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/DDVI_QR_{mdp.ENV_NAME}_time")

#### ####################################################################################
error_points = np.logspace(-10, 0, 21, endpoint=True)
time_points = get_time_by_error(v_errors, time_traces, error_points)

time_points_mean = np.mean(time_points, axis = 0) * 1000
time_points_se = np.std(time_points, axis = 0) / np.sqrt(num_runs) * 1000


fig, ax = plt.subplots()

plot_with_shades(ax, error_points, time_points_mean[0, :], time_points_se[0, :], color='b', label = "VI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[1, :], time_points_se[1, :], color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[2, :], time_points_se[2, :], color='g', label = "rank-$2$ DDVI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[3, :], time_points_se[2, :], color='y', label = "rank-$3$ DDVI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[4, :], time_points_se[4, :], color='orange', label = "rank-$4$ DDVI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[5, :], time_points_se[5, :], color='purple', label = "rank-$5$ DDVI", linestyle='-')

# naming the x axis
plt.ylabel('Wall Clock (ms)', fontsize=13)
plt.xscale('log')
# plt.ylim(top=50)
plt.xlim(1e-10, 1)
# naming the y axis
plt.xlabel(' Normalized Error', fontsize=12)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
ax.invert_xaxis()
plt.savefig(f"planning/output/DDVI_QR_{mdp.ENV_NAME}_error")



