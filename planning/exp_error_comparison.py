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


# Figure 2
# num_states = 200
# r=0.995
# num_iter = 5000
# power_iter = 600
# num_runs= 20
# rank_vals = [1, 2]
# b_P = 2
# b_R = int(num_states * 0.1)
# mdp = Garnet(0.9, num_states, 1, b_P, b_R)
# exp_name = f"Garnet-{num_states}-{b_P}-{b_R}-{r}"
# P=np.array(mdp.P())
# R=np.array(mdp.R())
# policy = optimal_policy(P,R, r)

# Figure 10
# num_states = 50
# r=0.995
# num_iter = 5000
# power_iter = 600
# num_runs= 20
# rank_vals = [1, 2]
# b_P = 40
# b_R = 50
# mdp = Garnet(0.9, num_states, 1, b_P, b_R)
# exp_name = f"Garnet-{num_states}-{b_P}-{b_R}-{r}"
# P=np.array(mdp.P())
# R=np.array(mdp.R())
# policy = optimal_policy(P,R, r)

# Figure 11
# mdp=Maze55(0.9, 0.9)
# r=0.995
# num_iter = 5000
# num_runs= 1
# rank_vals = [1, 2]
# exp_name = f"Maze55"
# P=np.array(mdp.P())
# R=np.array(mdp.R())
# policy = np.array([2, 2, 3, 0, 3, 0, 2, 1, 3, 2, 2, 2, 3, 3, 1, 0, 3, 0, 3, 3, 2, 2, 1, 1, 0])

# Figure 12
# mdp = ChainWalk(50)
# r=0.995
# num_iter = 5000
# num_runs= 1
# exp_name = f"ChainWalk"
# P=np.array(mdp.P())
# R=np.array(mdp.R())
# policy = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1])

# Figure 13
mdp = CliffWalkmodified(0.9, 0.995)
r=0.995
num_iter = 5000
num_runs= 1
exp_name = f"CliffWalkmodified"
P=np.array(mdp.P())
R=np.array(mdp.R())
policy = optimal_policy(P,R, r)







def get_garnet_mdp(num_states, b_P, b_R):
    mdp = Garnet(0.9, num_states, 1, b_P, b_R)
    return mdp



# Hyperparams
power_iter = 600
rank_vals = [1, 2]
ddvi_qr_alpha = [0.99, 0.99, 0.99]
ddvi_autopi_alpha = 0.99
pid_eta, pid_eps = 0.05, 10**(-10)
anderson_m = 5

num_ranks = len(rank_vals)

if load_file:
    time_traces = np.load(f"planning/output/Error_Comparison_{exp_name}_times.npy")
    v_errors = np.load(f"planning/output/Error_Comparison_{exp_name}_errors.npy")
else:
    run_V_traces = np.zeros((num_ranks + 6, num_iter, mdp.num_states()))
    v_errors = np.zeros((num_runs, num_ranks + 6, num_iter))
    time_traces = np.zeros((num_runs, num_ranks + 6, num_iter))

    for run_i in tqdm(range(num_runs)):
        if mdp.ENV_NAME == "Garnet":
            mdp = get_garnet_mdp(num_states, b_P, b_R)
            P=np.array(mdp.P())
            R=np.array(mdp.R())
            policy = optimal_policy(P,R, r)
        opt= value_function_policy(P,R,r, policy).reshape(-1)
        V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

        run_V_traces[0, :], time_traces[run_i, 0, :] = value_iteration_pe(P, R, r, V, num_iter, policy)
        print("VI")
        run_V_traces[1, :], time_traces[run_i, 1, :] = anderson_VI_pe(P, R, r, V, num_iter, policy, anderson_m)
        print("Anderson VI")
        run_V_traces[2, :], time_traces[run_i, 2, :] = nesterov_VI_pe(P, R, r, V, num_iter, policy)
        print("Nestrov VI", )
        run_V_traces[3, :], time_traces[run_i, 3, :] = pid_VI_pe(P, R, r, V, num_iter, policy, pid_eta, pid_eps)
        print("PID VI")
        run_V_traces[4, :], time_traces[run_i, 4, :] = anc_VI_pe(P, R, r, V, num_iter, policy)
        print("Anc VI")
        run_V_traces[5, :], time_traces[run_i, 5, :] = ddvi_AutoQR(P, R, r, V, num_iter, policy, 20, ddvi_autopi_alpha)
        print("DDVI-AutoQR")

        for rank_i in range(num_ranks):
            run_V_traces[6 + rank_i, :], time_traces[run_i, 6 + rank_i, :] = ddvi_qr(P, R, r, V, policy, num_iter, rank_vals[rank_i], power_iter, ddvi_qr_alpha[rank_i])
            print(f"DDVI-rank{rank_vals[rank_i]}")

        v_errors[run_i] = np.linalg.norm(run_V_traces - opt, ord=1, axis=-1) / np.linalg.norm(opt, ord=1)
    np.save(f"planning/output/Error_Comparison_{exp_name}_times.npy", time_traces)
    np.save(f"planning/output/Error_Comparison_{exp_name}_errors.npy", v_errors)

v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)

################################################

rank_colors = ['b', 'r', 'g', 'y', 'orange', 'purple'] 
auto_pi_color = 'magenta'
x  = np.arange(0,5000)
fig, ax = plt.subplots()
plt.rc('legend', fontsize=11)
plot_with_shades(ax, x, v_errors_mean[1, x], v_errors_se[1, x], color='orange', label = "Anderson VI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[2, x], v_errors_se[2, x], color='y', label = "Nesterov VI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[3, x], v_errors_se[3, x], color='grey', label = "PID VI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[4, x], v_errors_se[4, x], color='purple', label = "Anc VI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[5, x], v_errors_se[5, x], color='k', label = "DDVI with AutoQR", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[6, x], v_errors_se[6, x], color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[7, x], v_errors_se[7, x], color='purple', label = "rank-$2$ DDVI", linestyle='-')
# plot_with_shades(ax, x, v_errors_mean[8, x], v_errors_se[8, x], color='magenta', label = "rank-$5$ DDVI", linestyle='-')
plot_with_shades(ax, x, v_errors_mean[0, x], v_errors_se[0, x], color='b', label = "VI", linestyle='--')
# plt.plot(x, v_errors_mean[8, x], color='g', label = "rank-$3$ DDVI", linestyle='-')
# plt.plot(x, v_errors_mean[9, x], color='pink', label = "rank-$4$ DDVI", linestyle='-')
# plt.plot(x, v_errors_mean[10, x], color='olive', label = "rank-$5$ DDVI", linestyle='-')
# plt.plot(x, v_errors_mean[11, x], color='g', label = "rank-$6$ DDVI", linestyle='-')

# naming the x axis
plt.xlabel('Iterations (k)', fontsize=15)
plt.yscale('log')
plt.ylim(bottom=1e-10)
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', fontsize=15)
# giving a title to my graph
#plt.title('Discount= 0.9, States=50, Actions=4')
# plt.title('Discount= 0.995, States=100, Actions=10, power=13')
# plt.title('States=800, Actions=64')
# show a legend on the plot
plt.legend()

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/Error_Comparison_{exp_name}.pdf")
plt.savefig(f"planning/output/Error_Comparison_{exp_name}.png")

################################################

error_points = np.logspace(-10, 0, 21, endpoint=True)
time_points = get_time_by_error(v_errors, time_traces, error_points)

time_points_mean = np.mean(time_points, axis = 0) * 1000
time_points_se = np.std(time_points, axis = 0) / np.sqrt(num_runs) * 1000


fig, ax = plt.subplots()

plot_with_shades(ax, error_points, time_points_mean[0, :], time_points_se[0, :], color='b', label = "VI", linestyle='--')
plot_with_shades(ax, error_points, time_points_mean[1, :], time_points_se[1, :], color='orange', label = "Anderson VI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[2, :], time_points_se[2, :], color='y', label = "Nesterov VI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[3, :], time_points_se[2, :], color='grey', label = "PID VI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[4, :], time_points_se[4, :], color='purple', label = "Anc VI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[5, :], time_points_se[5, :], color='k', label = "DDVI with AutoPI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[6, :], time_points_se[6, :], color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, error_points, time_points_mean[7, :], time_points_se[7, :], color='g', label = "rank-$2$ DDVI", linestyle='-')
# plot_with_shades(ax, error_points, time_points_mean[8, :], time_points_se[8, :], color='purple', label = "rank-$8$ DDVI", linestyle='-')
# plot_with_shades(ax, error_points, time_points_mean[9, :], time_points_se[9, :], color='pink', label = "rank-$4$ DDVI", linestyle='-')
# plot_with_shades(ax, error_points, time_points_mean[10, :], time_points_se[10, :], color='g', label = "rank-$5$ DDVI", linestyle='-')
# plot_with_shades(ax, error_points, time_points_mean[11, :], time_points_se[11, :], color='purple', label = "rank-$6$ DDVI", linestyle='-')


# naming the x axis
plt.ylabel('Wall Clock (ms)', fontsize=13)
plt.xscale('log')
plt.ylim(0, 250)
plt.xlim(1e-10, 1)
# naming the y axis
plt.xlabel(' Normalized Error', fontsize=12)
plt.legend()

# function to show the plot
plt.grid(alpha=0.3)
ax.invert_xaxis()
plt.savefig(f"planning/output/Error_Comparison_{exp_name}_error.pdf") 
plt.savefig(f"planning/output/Error_Comparison_{exp_name}_error.png") 

################################################

time_points = np.linspace(0, 2000e-3, 100)
error_points = get_error_by_time(v_errors, time_traces, time_points)

error_points_mean = np.mean(error_points, axis = 0)
error_points_se = np.std(error_points, axis = 0) / np.sqrt(num_runs)

plt.figure()

fig, ax = plt.subplots()

plot_with_shades(ax, time_points * 1000, error_points_mean[0, :], error_points_se[0, :],  color='b', label = "VI", linestyle='--')
plot_with_shades(ax, time_points * 1000, error_points_mean[1, :], error_points_se[1, :],  color='orange', label = "Anderson VI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[2, :], error_points_se[2, :],  color='y', label = "Nesterov VI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[3, :], error_points_se[2, :],  color='grey', label = "PID VI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[4, :], error_points_se[4, :],  color='purple', label = "Anc VI", linestyle='-')
# plot_with_shades(ax, time_points * 1000, error_points_mean[5, :], error_points_se[5, :],  color='k', label = "DDVI with AutoPI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[6, :], error_points_se[6, :],  color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, time_points * 1000, error_points_mean[7, :], error_points_se[7, :],  color='g', label = "rank-$2$ DDVI", linestyle='-')
# plot_with_shades(ax, time_points * 1000, error_points_mean[8, :], error_points_se[8, :],  color='magenta', label = "rank-$5$ DDVI", linestyle='-')
# plot_with_shades(ax, time_points * 1000, error_points_mean[9, :], error_points_se[9, :],  color='pink', label = "rank-$4$ DDVI", linestyle='-')
# plot_with_shades(ax, time_points * 1000, error_points_mean[10, :], error_points_se[10, :],color='g', label = "rank-$5$ DDVI", linestyle='-')
# plot_with_shades(ax, time_points * 1000, error_points_mean[11, :], error_points_se[11, :], color='purple', label = "rank-$6$ DDVI", linestyle='-')

# naming the x axis
plt.xlabel('Wall Clock (ms)', fontsize=13)
plt.yscale('log')
plt.ylim(bottom=1e-10)
# naming the y axis
plt.ylabel(' Normalized $\|V^{k} - V^{\pi}\|_{1}$', fontsize=12)
plt.legend(loc='lower left')

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/Error_Comparison_{exp_name}_time.pdf")
plt.savefig(f"planning/output/Error_Comparison_{exp_name}_time.png")