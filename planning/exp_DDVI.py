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


# Config
# mdp = ChainWalk(50)
mdp=Maze55(0.9, 0.9)

# r=0.95
# r = 0.99
r=0.995

num_iter = 2000
num_runs= 1
ddvi_qr_alpha = [0.99, 0.99, 0.99, 0.99, 0.99]
ddvi_autopi_alpha = 0.99
rank_vals = [1, 2, 3, 4]
power_iter = [100, 100, 100, 100]

num_states = 500
num_ranks = len(rank_vals)
P=np.array(mdp.P())
R=np.array(mdp.R())
policy = optimal_policy(P,R, r)
if mdp.ENV_NAME == "Maze55":
    policy = np.array([2, 2, 3, 0, 3, 0, 2, 1, 3, 2, 2, 2, 3, 3, 1, 0, 3, 0, 3, 3, 2, 2, 1, 1, 0])
elif mdp.ENV_NAME == "ChainWalk":
    policy = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1])

if load_file:
    time_traces = np.load(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_times.npy")
    v_errors = np.load(f"planning/output/DDVI_Comparison_{mdp.ENV_NAME}_{r}_errors.npy")
else:
    run_V_traces = np.zeros((num_ranks + 3, num_iter, mdp.num_states()))
    v_errors = np.zeros((num_runs, num_ranks + 3, num_iter))
    time_traces = np.zeros((num_runs, num_ranks + 3, num_iter))

    for run_i in tqdm(range(num_runs)):
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