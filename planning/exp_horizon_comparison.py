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





num_states = 200
power_iter = 600
num_runs=20
rank_vals = [1, 2]
num_ranks = len(rank_vals)
target_error = 1e-4

# Hyperparams
ddvi_qr_alpha = [0.99, 0.99, 0.99]
ddvi_autopi_alpha = 0.99
pid_eta, pid_eps = 0.05, 10**(-10)
anderson_m = 5

mdp = get_mdp(num_states)
num_iter=[10000, #VI
        10000, #Anderson
        10000, #Nesterov
        10000, #PID
        10000, #ANC
        2000, #AutoPI
        ] + [2000] * 10 #DDVI
horizon_vals = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
num_horizons = horizon_vals.shape[0]

if load_file:
    time_traces = np.load(f"planning/output/Horizon_Comparison_{mdp.ENV_NAME}.npy")
else:
    time_traces = np.zeros((num_runs, num_horizons, num_ranks + 6))
    for run_i in tqdm(range(num_runs)):
        for horizon_i in range(num_horizons):
            mdp = get_mdp(num_states)
            P=np.array(mdp.P())
            R=np.array(mdp.R())
            r = 1 - 1/horizon_vals[horizon_i]
            policy = optimal_policy(P,R, r)
            opt= value_function_policy(P,R,r, policy).reshape(-1)
            V=np.zeros(P.shape[1]).reshape(P.shape[1],1)

            time_traces[run_i, horizon_i, 0] = get_stopping_time(value_iteration_pe(P, R, r, V, num_iter[0], policy), target_error, opt)
            print("VI", time_traces[run_i, horizon_i, 0])
            time_traces[run_i, horizon_i, 1] = get_stopping_time(anderson_VI_pe(P, R, r, V, num_iter[1], policy, anderson_m), target_error, opt)
            print("Anderson VI", time_traces[run_i, horizon_i, 1])
            time_traces[run_i, horizon_i, 2] = get_stopping_time(nesterov_VI_pe(P, R, r, V, num_iter[2], policy), target_error, opt)
            print("Nestrov VI", time_traces[run_i, horizon_i, 2])
            time_traces[run_i, horizon_i, 3] = get_stopping_time(pid_VI_pe(P, R, r, V, num_iter[3], policy, pid_eta, pid_eps), target_error, opt)
            print("PID VI", time_traces[run_i, horizon_i, 3])
            time_traces[run_i, horizon_i, 4] = get_stopping_time(anc_VI_pe(P, R, r, V, num_iter[4], policy), target_error, opt)
            print("Anc VI", time_traces[run_i, horizon_i, 4])
            time_traces[run_i, horizon_i, 5] = get_stopping_time(ddvi_AutoPI(P, R, r, V, num_iter[5], policy, 20, ddvi_autopi_alpha), target_error, opt)
            print("DDVI-AutoPI", time_traces[run_i, horizon_i, 5])

            for rank_i in range(num_ranks):
                time_traces[run_i, horizon_i, 6 + rank_i] = get_stopping_time(ddvi_qr(P, R, r, V, policy, num_iter[6+rank_i], rank_vals[rank_i], power_iter, ddvi_qr_alpha[rank_i]), target_error, opt)
                print(f"DDVI-rank{rank_vals[rank_i]}", time_traces[run_i, horizon_i, 6 + rank_i])
    np.save(f"planning/output/Horizon_Comparison_{mdp.ENV_NAME}.npy", time_traces)


time_mean = np.mean(time_traces, axis = 0) * 1000
time_se = np.std(time_traces, axis = 0) / np.sqrt(num_runs) * 1000

fig, ax = plt.subplots()
plt.rc('legend', fontsize=11)
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 0], time_se[np.arange(num_horizons), 0], color='b', label = "VI", linestyle='--')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 1], time_se[np.arange(num_horizons), 1], color='orange', label = "Anderson VI", linestyle='-')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 2], time_se[np.arange(num_horizons), 2], color='y', label = "Nesterov VI", linestyle='-')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 3], time_se[np.arange(num_horizons), 3], color='grey', label = "PID VI", linestyle='-')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 4], time_se[np.arange(num_horizons), 4], color='purple', label = "Anc VI", linestyle='-')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 5], time_se[np.arange(num_horizons), 5], color='k', label = "DDVI with AutoPI", linestyle='-')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 6, ], time_se[np.arange(num_horizons), 6, ], color='r', label = "rank-$1$ DDVI", linestyle='-')
plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 7, ], time_se[np.arange(num_horizons), 7, ], color='purple', label = "rank-$2$ DDVI", linestyle='-')
# plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 8, ], time_se[np.arange(num_horizons), 8, ], color='g', label = "rank-$3$ DDVI", linestyle='-')
# plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 9, ], time_se[np.arange(num_horizons), 9, ], color='pink', label = "rank-$4$ DDVI", linestyle='-')
# plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 10],time_se[np.arange(num_horizons), 10],  color='olive', label = "rank-$5$ DDVI", linestyle='-')
# plot_with_shades(ax, horizon_vals, time_mean[np.arange(num_horizons), 11],time_se[np.arange(num_horizons), 11],  color='g', label = "rank-$6$ DDVI", linestyle='-')

# naming the x axis
plt.xlabel(r'Horizon ($1/(1-\gamma)$)', fontsize=15)
# plt.yscale('log')
plt.ylim(0, 400)
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
plt.savefig(f"planning/output/Horizon_Comparison_{mdp.ENV_NAME}")
