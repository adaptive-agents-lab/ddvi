import matplotlib.pyplot as plt
import numpy as np
from algorithms.util import *

# r = 0.95
# r = 0.99
r = 0.995

power_iter = np.array([100, 100, 100, 100])
qr_cost = power_iter * np.array([0, 2, 3, 4])

rank_colors = [None, 'r', 'g', 'y', 'orange', 'purple'] 
auto_pi_color = 'grey'
auto_qr_color = 'black'

fig = plt.figure()
fig.set_size_inches(10, 3)
rank_shifts = qr_cost
# rank_shifts = [4, 0, 0, 0, 0]
# rank_shifts = [0, 1, 0, 0, 0]
auto_pi_shift = 0
x  = np.arange(0,1000)
params = {'font.size': 13,
            'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10, "axes.labelpad":2,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 2, 'axes.linewidth': 1}
plt.rcParams.update(params)

##########################
time_traces = np.load(f"planning/output/DDVI_Comparison_Maze55_{r}_times.npy")
v_errors = np.load(f"planning/output/DDVI_Comparison_Maze55_{r}_errors.npy")
num_runs = v_errors.shape[0]
rank_vals = [1, 2, 3, 4]
num_ranks = len(rank_vals)
v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)

ax1 = fig.add_subplot(121)
plot_with_shades(ax1, x, v_errors_mean[1, x], v_errors_se[1, x], color=auto_qr_color, label = "DDVI (AutoQR)", linestyle='-')
plot_with_shades(ax1, x + auto_pi_shift, v_errors_mean[2, x], v_errors_se[2, x], color=auto_pi_color, label = "DDVI (AutoPI)", linestyle='-')
for rank_i in range(num_ranks):
    plot_with_shades(ax1, x + rank_shifts[rank_i], v_errors_mean[3 + rank_i, x], v_errors_se[3 + rank_i, x], color=rank_colors[rank_vals[rank_i]], label = f"DDVI (rank-${rank_vals[rank_i]}$)", linestyle='-')
plot_with_shades(ax1, x, v_errors_mean[0, x], v_errors_se[0, x], color='b', label = "VI", linestyle='--')

ax1.set_xlabel('Iterations (k)')
ax1.set_yscale('log')
ax1.set_ylim(1e-6, 1.1)
ax1.set_xlim(0, 1000)
ax1.set_ylabel("Normalized $||V^{k} - V^{\pi}||_{1}$", labelpad=0)
ax1.grid(alpha=0.3)

##########################

time_traces = np.load(f"planning/output/DDVI_Comparison_ChainWalk_{r}_times.npy")
v_errors = np.load(f"planning/output/DDVI_Comparison_ChainWalk_{r}_errors.npy")
num_runs = v_errors.shape[0]
rank_vals = [1, 2, 3, 4]
num_ranks = len(rank_vals)
v_errors_mean = np.mean(v_errors, axis = 0)
v_errors_se = np.std(v_errors, axis = 0) / np.sqrt(num_runs)

ax2 = fig.add_subplot(122)
plot_with_shades(ax2, x, v_errors_mean[1, x], v_errors_se[1, x], color=auto_qr_color, label = "DDVI (AutoQR)", linestyle='-')
plot_with_shades(ax2, x + auto_pi_shift, v_errors_mean[2, x], v_errors_se[2, x], color=auto_pi_color, label = "DDVI (AutoPI)", linestyle='-')
for rank_i in range(num_ranks):
    plot_with_shades(ax2, x + rank_shifts[rank_i], v_errors_mean[3 + rank_i, x], v_errors_se[3 + rank_i, x], color=rank_colors[rank_vals[rank_i]], label = f"DDVI (rank-${rank_vals[rank_i]}$)", linestyle='-')
plot_with_shades(ax2, x, v_errors_mean[0, x], v_errors_se[0, x], color='b', label = "VI", linestyle='--')

ax2.set_xlabel('Iterations (k)')
ax2.set_yscale('log')
ax2.set_ylim(1e-6, 1.1)
ax2.set_xlim(0, 1000)
ax2.set_ylabel("Normalized $||V^{k} - V^{\pi}||_{1}$", labelpad=0)
ax2.grid(alpha=0.3)

##########################
handles, labels = ax1.get_legend_handles_labels()
order = [i for i in range(len(labels))]
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="center right", ncol=1)


plt.subplots_adjust(left=0.08,
                        bottom=0.15,
                        right=0.8,
                        top=0.98,
                        wspace=0.3,
                        hspace=0.25)
plt.savefig(f"planning/output/DDVI_Comparison_Maze(Left)ChainWalk(Right)_{r}.png")
plt.savefig(f"planning/output/DDVI_Comparison_Maze(Left)ChainWalk(Right)_{r}.pdf")
