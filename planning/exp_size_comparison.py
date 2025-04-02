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



np.random.seed(0)
load_file = False

def get_mdp(num_states):
    # mdp=Maze55(0.9, 0.9)
    # mdp = CliffWalkmodified(0.9, 0.995)
    # mdp = ChainWalk(num_states)
    mdp = Garnet(0.9, num_states, 1, 2, int(num_states * 0.1))
    return mdp

def get_eign_vals(mdp):
    P=np.array(mdp.P())
    R=np.array(mdp.R())
    r=0.995
    policy = optimal_policy(P,R, r)
    # policy = np.zeros(P.shape[1], dtype=int)
    P_pi = transition_matrix_policy(P, policy)
    eigvals, _ = np.linalg.eig(P_pi)
    return np.sort(np.abs(eigvals))[-10:]

def get_stopping_time(run_result, target_error, opt):
    v_trace, time_trace = run_result
    v_errors = np.linalg.norm(v_trace - opt, ord=1, axis=-1) / np.linalg.norm(opt, ord=1)
    time_point = get_time_by_error(v_errors, time_trace, np.array([target_error]))
    return time_point[0]


size_vals = np.array([200, 400, 600, 800, 1000])
# size_vals = np.array([50, 100])
num_sizes = size_vals.shape[0]
power_iter = 600
num_runs=1
r = 0.995
rank_vals = [1, 2, 3, 4, 5]
num_ranks = len(rank_vals)
target_error = 1e-4

# Hyperparams
ddvi_qr_alpha = [0.99, 0.99, 0.99, 0.99, 0.99]
ddvi_autopi_alpha = 0.99
pid_eta, pid_eps = 0.05, 10**(-10)
anderson_m = 5

mdp = get_mdp(size_vals[0])
num_iter=[10000, #VI
        2000, #Anderson
        10000, #Nesterov
        4000, #PID
        10000, #ANC
        2000, #AutoPI
        ] + [2000] * 10 #DDVI


if load_file:
    time_traces = np.load(f"planning/output/Size_Comparison_{mdp.ENV_NAME}.npy")
else:
    time_traces = np.zeros((num_runs, num_sizes, num_ranks + 6))
    for run_i in tqdm(range(num_runs)):
        for size_i in range(num_sizes):
            mdp = get_mdp(size_vals[size_i])
            P=np.array(mdp.P())
            R=np.array(mdp.R())
            policy = optimal_policy(P,R, r)
            opt= value_function_policy(P,R,r, policy).reshape(-1)
            V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

            # time_traces[run_i, size_i, 0] = get_stopping_time(value_iteration_pe(P, R, r, V, num_iter[0], policy), target_error, opt)
            # print("VI", time_traces[run_i, size_i, 0])
            # time_traces[run_i, size_i, 1] = get_stopping_time(anderson_VI_pe(P, R, r, V, num_iter[1], policy, anderson_m), target_error, opt)
            # print("Anderson VI", time_traces[run_i, size_i, 1])
            # time_traces[run_i, size_i, 2] = get_stopping_time(nesterov_VI_pe(P, R, r, V, num_iter[2], policy), target_error, opt)
            # print("Nestrov VI", time_traces[run_i, size_i, 2])
            # time_traces[run_i, size_i, 3] = get_stopping_time(pid_VI_pe(P, R, r, V, num_iter[3], policy, pid_eta, pid_eps), target_error, opt)
            # print("PID VI", time_traces[run_i, size_i, 3])
            # time_traces[run_i, size_i, 4] = get_stopping_time(anc_VI_pe(P, R, r, V, num_iter[4], policy), target_error, opt)
            # print("Anc VI", time_traces[run_i, size_i, 4])
            # time_traces[run_i, size_i, 5] = get_stopping_time(ddvi_AutoQR(P, R, r, V, num_iter[5], policy, 20, ddvi_autopi_alpha), target_error, opt)
            # print("DDVI-AutoQR", time_traces[run_i, size_i, 5])

            for rank_i in range(num_ranks):
                time_traces[run_i, size_i, 6 + rank_i] = get_stopping_time(ddvi_qr(P, R, r, V, policy, num_iter[6+rank_i], rank_vals[rank_i], power_iter, ddvi_qr_alpha[rank_i]), target_error, opt)
                print(f"DDVI-rank{rank_vals[rank_i]}", time_traces[run_i, size_i, 6 + rank_i])
    np.save(f"planning/output/Size_Comparison_{mdp.ENV_NAME}.npy", time_traces)


time_mean = np.mean(time_traces, axis = 0) * 1000
time_se = np.std(time_traces, axis = 0) / np.sqrt(num_runs) * 1000

fig, ax = plt.subplots()

rank_colors = [None, 'r', 'g', 'y', 'orange', 'purple'] 
alg_colors = ['b', #VI
              'r', #Anderson
              'orange', #Nesterov
              'g', #PID
              'purple', #ANC
              'k', #AutoPI
              ]
auto_pi_color = 'magenta'

plt.rc('legend', fontsize=11)

plot_with_shades(ax, size_vals, time_mean[:, 1], time_se[:, 1], color=alg_colors[1], label = "Anderson VI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 2], time_se[:, 2], color=alg_colors[2], label = "Nesterov VI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 3], time_se[:, 3], color=alg_colors[3], label = "PID VI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 4], time_se[:, 4], color=alg_colors[4], label = "Anc VI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 5], time_se[:, 5], color=alg_colors[5], label = "DDVI with AutoPI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 6], time_se[:, 6], color=rank_colors[1], label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 7], time_se[:, 7], color=rank_colors[2], label = "rank-$2$ DDVI", linestyle='-')
plot_with_shades(ax, size_vals, time_mean[:, 0], time_se[:, 0], color=alg_colors[0], label = "VI", linestyle='dotted')

# naming the x axis
plt.xlabel('Number of States', fontsize=15)
# plt.yscale('log')
# plt.ylim(0, 2500)
# naming the y axis
plt.ylabel('Wall Clock (ms)', fontsize=15)
# giving a title to my graph
#plt.title('Discount= 0.9, States=50, Actions=4')
# plt.title('Discount= 0.995, States=100, Actions=10, power=13')
# plt.title('States=800, Actions=64')
# show a legend on the plot
plt.legend()

# function to show the plot
plt.grid(alpha=0.3)
plt.savefig(f"planning/output/Size_Comparison_{mdp.ENV_NAME}")
