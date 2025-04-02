from matplotlib import pyplot as plt, ticker
import seaborn as sns
import numpy as np
import yaml

from rl_utilities import get_optimal_policy_mdp, get_policy_value_mdp
from utilities import setup_problem

num_trials = 20
num_iterations = 20000
plot_every = 100




def plot_alg(ax, alg_dir, label, color, linestyle):
    value_per_trial = []
    for trial in range(num_trials):  
        with open(f'{alg_dir}/config.yaml', 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        mdp, policy = setup_problem(config)
        true_V = get_policy_value_mdp(mdp, policy)
        true_trace = np.zeros((num_iterations, mdp.num_states()))
        true_trace[np.arange(num_iterations), :] = true_V

        with open(f'{alg_dir}/V_trace_{trial}.npy', 'rb') as f:
            v_trace = np.load(f)
            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / np.linalg.norm(true_V, ord=1)
            value_per_trial.append(value_errors)

    mean = np.array(value_per_trial).mean(axis=0)
    stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

    ax.plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=label, linestyle=linestyle, color=color)
    ax.fill_between(x=np.arange(num_iterations)[::plot_every],
                        y1=(mean - stderr)[::plot_every],
                        y2=(mean + stderr)[::plot_every],
                        alpha=0.1,
                        color=color
                        )

rank_colors = [None, 'r', 'g', 'y', 'orange'] 
td_color = 'b'
dyna_color = 'purple'

fig = plt.figure()
fig.set_size_inches(10, 3)
params = {'font.size': 13,
            'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10, "axes.labelpad":2,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 2, 'axes.linewidth': 1}
plt.rcParams.update(params)

##########################
exp_dir = f"sample_based/output/final_7996_chainwalk/exp_pe_sample"
alpha = 0.3


cmap = plt.get_cmap("tab10")
ax1 = fig.add_subplot(121)
plot_alg(ax1, f"{exp_dir}/dyna_pe_{alpha}", "Dyna", dyna_color, "dotted")
plot_alg(ax1, f"{exp_dir}/tdlearning_pe", "TD Learning", td_color, "dashed")
plot_alg(ax1, f"{exp_dir}/ddtd1_pe_{alpha}", rf"DDTD (rank-$1$)", rank_colors[1], "solid")
plot_alg(ax1, f"{exp_dir}/ddtd2_pe_{alpha}", rf"DDTD (rank-$2$)", rank_colors[2], "solid")
plot_alg(ax1, f"{exp_dir}/ddtd3_pe_{alpha}", rf"DDTD (rank-$3$)", rank_colors[3], "solid")
plot_alg(ax1, f"{exp_dir}/ddtd4_pe_{alpha}", rf"DDTD (rank-$4$)", rank_colors[4], "solid")


# plt.yscale("log")
# plt.ylim((5e-2, 5))
# plt.legend()
ax1.set_xlabel("Environment Samples (t)")
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
ax1.grid(alpha=0.3)
ax1.set_ylabel(r'Normalized $||V_t-V^\pi||_1$')
# plt.title(f"TTD(PE) - {exp_name}")

##########################
exp_dir = f"sample_based/output/final_8a69_maze55/exp_pe_sample"
alpha = 0.3


ax2 = fig.add_subplot(122)
plot_alg(ax2, f"{exp_dir}/dyna_pe_{alpha}", "Dyna", dyna_color, "dotted")
plot_alg(ax2, f"{exp_dir}/tdlearning_pe", "TD Learning", td_color, "dashed")
plot_alg(ax2, f"{exp_dir}/ddtd1_pe_{alpha}", rf"DDTD (rank-$1$)", rank_colors[1], "solid")
plot_alg(ax2, f"{exp_dir}/ddtd2_pe_{alpha}", rf"DDTD (rank-$2$)", rank_colors[2], "solid")
plot_alg(ax2, f"{exp_dir}/ddtd3_pe_{alpha}", rf"DDTD (rank-$3$)", rank_colors[3], "solid")
plot_alg(ax2, f"{exp_dir}/ddtd4_pe_{alpha}", rf"DDTD (rank-$4$)", rank_colors[4], "solid")


# plt.yscale("log")
# plt.ylim((5e-2, 5))
# plt.legend()
ax2.set_xlabel("Environment Samples (t)")
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
ax2.grid(alpha=0.3)
ax2.set_ylabel(r'Normalized $||V_t-V^\pi||_1$')

handles, labels = ax1.get_legend_handles_labels()
order = [i for i in range(len(labels))]
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="center right", ncol=1)


plt.subplots_adjust(left=0.08,
                        bottom=0.15,
                        right=0.8,
                        top=0.98,
                        wspace=0.3,
                        hspace=0.25)

plt.savefig(f"sample_based/output/main_pe.png", bbox_inches="tight")
plt.savefig(f"sample_based/output/main_pe.pdf", bbox_inches="tight")