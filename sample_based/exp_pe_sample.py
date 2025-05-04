import argparse
import numpy as np
import yaml
import shutil
import random
from algorithms.DDTD_PE import DDTD_PE
from model.LocallySmoothedModel import LocallySmoothedModel
from multiprocessing import Pool
from algorithms.Dyna_PE import Dyna_PE
from algorithms.TDLearning_PE import TDLearning_PE
from rl_utilities import get_optimal_policy_mdp
from utilities import setup_problem, setup_alg_output_dir, get_exp_dir, get_default_alg_output_dir
from LearningRate import LearningRate

ROOT_OUTPUT_DIR = "./output"

def run_td_learning(inputs):
    mdp, policy, config, config_path, num_iterations, trial, exp_dir, model_class = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    tdlearning_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "tdlearning_pe", smoothing_param=None,
                                              exp_dir=exp_dir)
    tdlearning_pe = TDLearning_PE(mdp, policy)
    lr_scheduler = LearningRate(config["exp_pe_lr_type"],
                                config["exp_pe_td_learning_pe_lr"],
                                config["exp_pe_td_learning_pe_delay"],
                                config["exp_pe_td_gamma"])
    shutil.copyfile(src=config_path, dst=f"{tdlearning_out_dir}/config.yaml")
    tdlearning_pe.run(num_iterations, lr_scheduler=lr_scheduler,
                      output_filename=f"{tdlearning_out_dir}/V_trace_{trial}.npy")

def run_dyna_pe(inputs):
    mdp, policy, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        dyna_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "dyna_pe", smoothing_param=alpha, exp_dir=exp_dir)
        dyna_pe = Dyna_PE(mdp, policy, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        shutil.copyfile(src=config_path, dst=f"{dyna_out_dir}/config.yaml")
        dyna_pe.run(num_iterations, output_filename=f"{dyna_out_dir}/V_trace_{trial}.npy")

def run_ddtd_pe(inputs):
    rank, mdp, policy, config, config_path, alpha_vals, alpha_indices, num_iterations, trial, exp_dir, model_class  = inputs["rank"], inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["alpha_indices"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i in range(len(alpha_indices)):
        alpha_index = alpha_indices[i]
        alpha = alpha_vals[alpha_index]
        splitting_alpha = config[f"exp_pe_ddtd{rank}_alpha"][i]
        update_Ehat_interval = config[f"exp_pe_ddtd{rank}_update_Ehat_interval"]
        ddtd_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", f"ddtd{rank}_pe", smoothing_param=alpha, exp_dir=exp_dir)
        ddtd_pe = DDTD_PE(mdp, policy, rank, splitting_alpha, model_class(mdp.num_states(), mdp.num_actions(), alpha), update_Ehat_interval, use_eigen_E=False)
        lr_scheduler = LearningRate(config["exp_pe_lr_type"],
                                config[f"exp_pe_ddtd{rank}_lr"][i],
                                config[f"exp_pe_ddtd{rank}_delay"],
                                config[f"exp_pe_ddtd{rank}_gamma"][i])
        shutil.copyfile(src=config_path, dst=f"{ddtd_out_dir}/config.yaml")
        ddtd_pe.run(num_iterations, lr_scheduler=lr_scheduler,
                      output_filename=f"{ddtd_out_dir}/V_trace_{trial}.npy")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path  of config file')
    parser.add_argument('alg_name', help='Algorithm to run. "ALL" to run all, "None" to just plot')
    parser.add_argument('--num_trials', default=1, help='Number of trials to run')
    parser.add_argument('--alpha_index', default=-1, type=int, help='Index of alpha value to run. -1 for all.')
    parser.add_argument('--first_trial', default=0, type=int, help='Index of alpha value to run. -1 for all.')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_pe_sample"
    exp_dir = get_exp_dir(config, "exp_pe_sample", exp_dir=args.exp_dir)
    mdp, pe_policy = setup_problem(config)
    num_trials = int(args.num_trials)
    num_iterations = config["exp_pe_sample_num_iterations"]
    alpha_vals = config["exp_pe_sample_alphas"]
    if args.alpha_index == -1:
        alpha_indices = [i for i in range(len(alpha_vals))]
    else:
        alpha_indices = [args.alpha_index]

    if config["exp_pe_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    else:
        assert False

    # Running Algorithms
    if args.alg_name in ["ALL", "tdlearning_pe"]:
        tdlearning_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "tdlearning_pe", smoothing_param=None,
                                                  exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                                "policy": pe_policy,
                                "config": config,
                                "config_path": args.config,
                                "num_iterations": num_iterations,
                                "trial": trial,
                                "exp_dir": args.exp_dir,
                                "model_class": model_class,
                                })
            p.map(run_td_learning, inputs)

    if args.alg_name in ["ALL", "dyna_pe"]:
        for alpha in alpha_vals:
            dyna_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "dyna_pe", smoothing_param=alpha,
                                                exp_dir=args.exp_dir)
        with Pool(24) as p:
            inputs = []
            for trial in range(args.first_trial, args.first_trial + num_trials):
                inputs.append({"mdp": mdp,
                               "policy": pe_policy,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "alpha_indices": alpha_indices,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_dyna_pe, inputs)


    for rank in [1, 2, 3, 4]:
        if args.alg_name in ["ALL", "ALL_DDTD", f"ddtd{rank}_pe"]:
            for i in range(len(alpha_indices)):
                alpha_index = alpha_indices[i]
                alpha = alpha_vals[alpha_index]
                ddtd_out_dir = setup_alg_output_dir(config, "exp_pe_sample", f"ddtd{rank}_pe", smoothing_param=alpha, exp_dir=args.exp_dir)
            print(ddtd_out_dir)
            with Pool(24) as p:

                inputs = []
                for trial in range(args.first_trial, args.first_trial + num_trials):
                    inputs.append({
                                "rank": rank,
                                "mdp": mdp,
                                "policy": pe_policy,
                                "config": config,
                                "config_path": args.config,
                                "alpha_vals": alpha_vals,
                                "alpha_indices": alpha_indices,
                                "num_iterations": num_iterations,
                                "trial": trial,
                                "exp_dir": args.exp_dir,
                                "model_class": model_class,
                                })
                p.map(run_ddtd_pe, inputs)

