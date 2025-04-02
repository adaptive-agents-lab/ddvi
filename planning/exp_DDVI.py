import matplotlib.pyplot as plt
from environments import *
from tqdm import tqdm
from algorithms.util import *
from algorithms.vi import value_iteration_pe
from algorithms.anderson_vi import anderson_VI_pe
from algorithms.nesterov_vi import nesterov_VI_pe
from algorithms.ddvi import ddvi_qr, ddvi_AutoQR, ddvi_AutoPI
from algorithms.pid_vi import pid_VI_pe, pid_VI_pe_old
from algorithms.anchoring_vi import anc_VI_pe


load_file = False
np.random.seed(0)

def get_mdp(num_states):
    # mdp=Maze55(0.9, 0.9)
    # mdp = CliffWalkmodified(0.9, 0.995)
    mdp = ChainWalk(50)
    # mdp = Garnet(0.9, num_states, 1, 2, int(num_states * 0.1))
    return mdp


r=0.95
num_iter = 2000
num_runs= 1
ddvi_qr_alpha = [0.99, 0.99, 0.99, 0.99, 0.99]
ddvi_autopi_alpha = 0.99
rank_vals = [1, 2, 3, 4]
power_iter = [100, 100, 100, 100]

num_states = 500
num_ranks = len(rank_vals)
mdp = get_mdp(num_states)
P=np.array(mdp.P())
R=np.array(mdp.R())
policy = optimal_policy(P,R, r)
if mdp.ENV_NAME == "Maze55":
    policy = np.array([2, 2, 3, 0, 3, 0, 2, 1, 3, 2, 2, 2, 3, 3, 1, 0, 3, 0, 3, 3, 2, 2, 1, 1, 0])
elif mdp.ENV_NAME == "ChainWalk":
    # policy = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    policy = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1])

if load_file:
    time_traces = np.load(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_times.npy")
    v_errors = np.load(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_errors.npy")
else:
    run_V_traces = np.zeros((num_ranks + 3, num_iter, mdp.num_states()))
    v_errors = np.zeros((num_runs, num_ranks + 3, num_iter))
    time_traces = np.zeros((num_runs, num_ranks + 3, num_iter))

    for run_i in tqdm(range(num_runs)):
        if mdp.ENV_NAME == "Garnet":
            mdp = get_mdp(num_states)
            P=np.array(mdp.P())
            R=np.array(mdp.R())
            policy = optimal_policy(P,R, r)
        opt= value_function_policy(P,R,r, policy).reshape(-1)
        V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

        run_V_traces[0, :], time_traces[run_i, 0, :] = value_iteration_pe(P, R, r, V, num_iter, policy)
        print("VI")
        run_V_traces[1, :], time_traces[run_i, 1, :] = ddvi_AutoQR(P, R, r, V, num_iter, policy, 20, ddvi_autopi_alpha)
        print("DDVI-AutoQR")
        run_V_traces[2, :], time_traces[run_i, 2, :] = ddvi_AutoPI(P, R, r, V, num_iter, policy, 20, ddvi_autopi_alpha)
        print("DDVI-AutoPI")

        for rank_i in range(num_ranks):
            run_V_traces[3 + rank_i, :], time_traces[run_i, 3 + rank_i, :] = ddvi_qr(P, R, r, V, policy, num_iter, rank_vals[rank_i], power_iter[rank_i], ddvi_qr_alpha[rank_i], method="QR")
            print(f"DDVI-rank{rank_vals[rank_i]}")

        v_errors[run_i] = np.linalg.norm(run_V_traces - opt, ord=1, axis=-1) / np.linalg.norm(opt, ord=1)
    np.save(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_times.npy", time_traces)
    np.save(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_errors.npy", v_errors)

v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)

################################################

rank_colors = [None, 'r', 'g', 'y', 'orange', 'purple'] 
rank_shifts = [4, 0, 0, 0, 0]
# rank_shifts = [0, 1, 0, 0, 0]
auto_pi_shift = 0
auto_pi_color = 'grey'
auto_qr_color = 'black'
x  = np.arange(0,500)
fig, ax = plt.subplots()
params = {'font.size': 13,
            'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10, "axes.labelpad":2,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 1, 'axes.linewidth': 1}
plt.rcParams.update(params)

fig.set_size_inches(5, 3)
plt.rc('legend')
plot_with_shades(ax, x, v_errors_mean[1, x], v_errors_se[1, x], color=auto_qr_color, label = "DDVI with AutoQR", linestyle='-')
plot_with_shades(ax, x + auto_pi_shift, v_errors_mean[2, x], v_errors_se[2, x], color=auto_pi_color, label = "DDVI with AutoPI", linestyle='-')
for rank_i in range(num_ranks):
    plot_with_shades(ax, x + rank_shifts[rank_i], v_errors_mean[3 + rank_i, x], v_errors_se[3 + rank_i, x], color=rank_colors[rank_vals[rank_i]], label = f"rank-${rank_vals[rank_i]}$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[0, x], v_errors_se[0, x], color='b', label = "VI", linestyle='--')

# naming the x axis
plt.xlabel('Iterations (k)')
plt.yscale('log')
plt.ylim(1e-8, 1.1)
plt.xlim(0, 500)
plt.ylabel("Normalized $||V^{k} - V^{\pi}||_{1}$", labelpad=0)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.subplots_adjust(left=0.15,
                        bottom=0.15,
                        right=0.99,
                        top=0.98)
plt.savefig(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}.png")
plt.savefig(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}.pdf")


################################################

error_points = np.logspace(-10, 0, 21, endpoint=True)
time_points = get_time_by_error(v_errors, time_traces, error_points)

time_points_mean = np.mean(time_points, axis = 0) * 1000
time_points_se = np.std(time_points, axis = 0) / np.sqrt(num_runs) * 1000


fig, ax = plt.subplots()
fig.set_size_inches(5, 3)

plot_with_shades(ax, error_points, time_points_mean[1, :], time_points_se[1, :], color=auto_qr_color, label = "DDVI with AutoQR", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[2, :], time_points_se[2, :], color=auto_pi_color, label = "DDVI with AutoPI", linestyle='-')
for rank_i in range(num_ranks):
    plot_with_shades(ax, error_points, time_points_mean[3 + rank_i, :], time_points_se[3 + rank_i, :], color=rank_colors[rank_vals[rank_i]], label = f"rank-${rank_vals[rank_i]}$ DDVI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[0, :], time_points_se[0, :], color='b', label = "VI", linestyle='--')


# naming the x axis
plt.ylabel('Wall Clock (ms)')
plt.xscale('log')
plt.ylim(0, 250)
plt.xlim(1e-10, 1)
# naming the y axis
plt.xlabel(' Normalized Error')
plt.legend()

# function to show the plot
plt.grid(alpha=0.3)
ax.invert_xaxis()
plt.savefig(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_error.png") 
plt.savefig(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_error.pdf") 

################################################

time_points = np.linspace(0, 600e-3, 100)
error_points = get_error_by_time(v_errors, time_traces, time_points)

error_points_mean = np.mean(error_points, axis = 0)
error_points_se = np.std(error_points, axis = 0) / np.sqrt(num_runs)

plt.figure()

fig, ax = plt.subplots()

plot_with_shades(ax, time_points * 100, error_points_mean[1, :], error_points_se[1, :], color=auto_qr_color, label = "DDVI with AutoQR", linestyle='-')
plot_with_shades(ax, time_points * 100, error_points_mean[2, :], error_points_se[2, :], color=auto_pi_color, label = "DDVI with AutoPI", linestyle='-')
for rank_i in range(num_ranks):
    plot_with_shades(ax, time_points * 100, error_points_mean[3 + rank_i, :], error_points_se[3 + rank_i, :], color=rank_colors[rank_vals[rank_i]], label = f"rank-${rank_vals[rank_i]}$ DDVI", linestyle='-')
plot_with_shades(ax, time_points * 100, error_points_mean[0, :], error_points_se[0, :], color='b', label = "VI", linestyle='--')

# naming the x axis
plt.xlabel('Wall Clock (ms)')
plt.yscale('log')
plt.ylim(bottom=1e-10)
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', labelpad=0)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_time.png")
plt.savefig(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_time.pdf")