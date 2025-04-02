import matplotlib.pyplot as plt
from algorithms.util import *

# error_time_traces = np.load(f"planning/output/Error_Comparison_Garnet_times.npy")
# error_v_errors = np.load(f"planning/output/Error_Comparison_Garnet_errors.npy")
horizon_time_traces = np.load(f"planning/output/Horizon_Comparison_Garnet.npy")
size_time_traces = np.load(f"planning/output/Size_Comparison_Garnet.npy")

# error_time_traces = np.load(f"planning/output/Error_Comparison_Maze55_times.npy")
# error_v_errors = np.load(f"planning/output/Error_Comparison_Maze55_errors.npy")

# error_time_traces = np.load(f"planning/output/Error_Comparison_CliffWalkmodified_times.npy")
# error_v_errors = np.load(f"planning/output/Error_Comparison_CliffWalkmodified_errors.npy")

error_time_traces = np.load(f"planning/output/Error_Comparison_ChainWalk_times.npy")
error_v_errors = np.load(f"planning/output/Error_Comparison_ChainWalk_errors.npy")



rank_colors = [None, 'r', 'g'] 
alg_colors = ['b', #VI
              'grey', #Anderson
              'orange', #Nesterov
              'y', #PID
              'purple', #ANC
              'k', #AutoPI
              ]
# cmap = plt.get_cmap("tab20")
# rank_colors = [None, cmap(2), cmap(4), cmap(6), cmap(8)] 
# alg_colors = [cmap(0), #VI
#               cmap(3), #Anderson
#               cmap(5), #Nesterov
#               cmap(7), #PID
#               cmap(9), #ANC
#               cmap(10), #AutoPI
#               ]



fig = plt.figure()
fig.set_size_inches(10, 6)
params = {'font.size': 13,
            'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10, "axes.labelpad":2,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 2, 'axes.linewidth': 1}
plt.rcParams.update(params)

error_v_errors_mean = np.mean(error_v_errors, axis = 0)
error_v_errors_se = np.std(error_v_errors, axis = 0) / np.sqrt(error_v_errors.shape[0])
####################
ax1 = fig.add_subplot(221)
x  = np.arange(0,2000)
plot_with_shades(ax1, x, error_v_errors_mean[0, x], error_v_errors_se[0, x], color=alg_colors[0], label = "VI", linestyle='--')
plot_with_shades(ax1, x, error_v_errors_mean[1, x], error_v_errors_se[1, x], color=alg_colors[1], label = "Anderson VI", linestyle='dashdot')
plot_with_shades(ax1, x, error_v_errors_mean[2, x], error_v_errors_se[2, x], color=alg_colors[2], label = "S-AVI", linestyle='dashdot')
plot_with_shades(ax1, x, error_v_errors_mean[3, x], error_v_errors_se[3, x], color=alg_colors[3], label = "PID VI", linestyle='dashdot')
plot_with_shades(ax1, x, error_v_errors_mean[4, x], error_v_errors_se[4, x], color=alg_colors[4], label = "Anc VI", linestyle='dashdot')
plot_with_shades(ax1, x, error_v_errors_mean[5, x], error_v_errors_se[5, x], color=alg_colors[5], label = "DDVI (AutoQR)", linestyle='-')
plot_with_shades(ax1, x, error_v_errors_mean[6, x], error_v_errors_se[6, x], color=rank_colors[1], label = "DDVI (rank-$1$)", linestyle='-')
plot_with_shades(ax1, x, error_v_errors_mean[7, x], error_v_errors_se[7, x], color=rank_colors[2], label = "DDVI (rank-$2$)", linestyle='-')

ax1.set_xlabel('Iterations (k)')
ax1.set_yscale('log')
ax1.set_ylim(1e-10, 1.1)
ax1.set_xlim(0, 2000)
ax1.set_ylabel(' Normalized $\|\|V^{k} - V^{\pi}\|\|_{1}$', labelpad=0)
ax1.grid(alpha=0.3)


#################### 
ax2 = fig.add_subplot(222)
time_points = np.linspace(0, 200e-3, 100)
error_points = get_error_by_time(error_v_errors, error_time_traces, time_points)

error_points_mean = np.mean(error_points, axis = 0)
error_points_se = np.std(error_points, axis = 0) / np.sqrt(error_v_errors.shape[0])

plot_with_shades(ax2, time_points * 1000, error_points_mean[0, :], error_points_se[0, :],  color=alg_colors[0], label = "VI", linestyle='--')
plot_with_shades(ax2, time_points * 1000, error_points_mean[1, :], error_points_se[1, :],  color=alg_colors[1], label = "Anderson VI", linestyle='dashdot')
plot_with_shades(ax2, time_points * 1000, error_points_mean[2, :], error_points_se[2, :],  color=alg_colors[2], label = "S-AVI", linestyle='dashdot')
plot_with_shades(ax2, time_points * 1000, error_points_mean[3, :], error_points_se[2, :],  color=alg_colors[3], label = "PID VI", linestyle='dashdot')
plot_with_shades(ax2, time_points * 1000, error_points_mean[4, :], error_points_se[4, :],  color=alg_colors[4], label = "Anc VI", linestyle='dashdot')
plot_with_shades(ax2, time_points * 1000, error_points_mean[5, :], error_points_se[5, :],  color=alg_colors[5], label = "DDVI with AutoPI", linestyle='-')
plot_with_shades(ax2, time_points * 1000, error_points_mean[6, :], error_points_se[6, :],  color=rank_colors[1], label = "DDVI (rank-$1$)", linestyle='-')
plot_with_shades(ax2, time_points * 1000, error_points_mean[7, :], error_points_se[7, :],  color=rank_colors[2], label = "DDVI (rank-$2$)", linestyle='-')

# naming the x axis
ax2.set_xlabel('Wall Clock (ms)')
ax2.set_yscale('log')
ax2.set_ylim(1e-10, 1.1)
ax2.set_xlim(0, 100)
ax2.set_ylabel(' Normalized $\|\|V^{k} - V^{\pi}\|\|_{1}$', labelpad=0)
# ax2.set_yticklabels([])
ax2.grid(alpha=0.3)

#################### 
ax3 = fig.add_subplot(223)
size_vals = np.array([200, 400, 600, 800, 1000])
num_sizes = size_vals.shape[0]
time_mean = np.mean(size_time_traces, axis = 0) * 1000
time_se = np.std(size_time_traces, axis = 0) / np.sqrt(size_time_traces.shape[0]) * 1000

plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 0], time_se[np.arange(num_sizes), 0], color=alg_colors[0], label = "VI", linestyle='--')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 1], time_se[np.arange(num_sizes), 1], color=alg_colors[1], label = "Anderson VI", linestyle='dashdot')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 2], time_se[np.arange(num_sizes), 2], color=alg_colors[2], label = "S-AVI", linestyle='dashdot')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 3], time_se[np.arange(num_sizes), 3], color=alg_colors[3], label = "PID VI", linestyle='dashdot')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 4], time_se[np.arange(num_sizes), 4], color=alg_colors[4], label = "Anc VI", linestyle='dashdot')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 5], time_se[np.arange(num_sizes), 5], color=alg_colors[5], label = "DDVI (AutoQR)", linestyle='-')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 6, ], time_se[np.arange(num_sizes), 6, ], color=rank_colors[1], label = "DDVI (rank-$1$)", linestyle='-')
plot_with_shades(ax3, size_vals, time_mean[np.arange(num_sizes), 7, ], time_se[np.arange(num_sizes), 7, ], color=rank_colors[2], label = "DDVI (rank-$2$)", linestyle='-')
ax3.set_xlabel('Number of States')
ax3.grid(alpha=0.3)
ax3.set_ylim(0, 1000)
ax3.set_xlim(200, 1000)
ax3.set_yticks([0, 200, 400, 600, 800, 1000])
ax3.set_ylabel('Wall Clock (ms)', labelpad=3)

#################### 
horizon_vals = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
num_horizons = horizon_vals.shape[0]


ax4 = fig.add_subplot(224)
time_mean = np.mean(horizon_time_traces, axis = 0) * 1000
time_se = np.std(horizon_time_traces, axis = 0) / np.sqrt(horizon_time_traces.shape[0]) * 1000

plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 0], time_se[np.arange(num_horizons), 0], color=alg_colors[0], label = "VI", linestyle='--')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 1], time_se[np.arange(num_horizons), 1], color=alg_colors[1], label = "Anderson VI", linestyle='dashdot')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 2], time_se[np.arange(num_horizons), 2], color=alg_colors[2], label = "S-AVI", linestyle='dashdot')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 3], time_se[np.arange(num_horizons), 3], color=alg_colors[3], label = "PID VI", linestyle='dashdot')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 4], time_se[np.arange(num_horizons), 4], color=alg_colors[4], label = "Anc VI", linestyle='dashdot')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 5], time_se[np.arange(num_horizons), 5], color=alg_colors[5], label = "DDVI with AutoPI", linestyle='-')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 6], time_se[np.arange(num_horizons), 6, ], color=rank_colors[1], label = "DDVI (rank-$1$)", linestyle='-')
plot_with_shades(ax4, horizon_vals, time_mean[np.arange(num_horizons), 7], time_se[np.arange(num_horizons), 7, ], color=rank_colors[2], label = "DDVI (rank-$2$)", linestyle='-')


ax4.set_xlabel(r'Horizon ($1/(1-\gamma)$)')
ax4.set_ylim(0, 250)
ax4.set_yticks([0, 50, 100, 150, 200, 250])
ax4.set_xlim(100, 1000)
ax4.set_ylabel('Wall Clock (ms)', labelpad=8)
ax4.grid(alpha=0.3)


handles, labels = ax1.get_legend_handles_labels()
order = [i for i in range(len(labels))]
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right", ncol=1)


plt.subplots_adjust(left=0.08,
                        bottom=0.095,
                        right=0.80,
                        top=0.99,
                        wspace=0.3,
                        hspace=0.25)
plt.savefig(f"planning/output/Comparison.pdf", tight_layout=True)
plt.savefig(f"planning/output/Comparison.png", tight_layout=True)