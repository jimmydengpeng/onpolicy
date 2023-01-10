#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn
from onpolicy import logger
from onpolicy.config import get_all_args_and_config
# from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.metadrive.MetaDrive_env import getMetaDriveEnv
from onpolicy.envs.metadrive_vec_env import SubprocVecEnv, DummyVecEnv, ShareVecEnv
from onpolicy.utils.utils import LogLevel, debug_msg, debug_print


def make_train_env(all_args) -> ShareVecEnv:
    env_config = {}
    env_config['delay_done'] = all_args.delay_done
    def get_env_fn(rank):
        def init_env():
            assert "MetaDrive" in all_args.env_name, (logger.error("Can not support the " + all_args.env_name + " environment.")) 
            return getMetaDriveEnv(all_args, rank, env_config)
        return init_env

    if all_args.n_rollout_threads == 1:
        return SubprocVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args) -> ShareVecEnv:
    def get_env_fn(rank):
        def init_env():
            if "MetaDrive" in all_args.env_name:
                env = getMetaDriveEnv(all_args, rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        debug_msg("all_args.n_eval_rollout_threads == 1")
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

# TODO
def check_args(all_args):
    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert all_args.desc is not None, "give a desc!"

# TODO: eval & render mode
def main(all_args, runner_config):
    check_args(all_args) 

    # logger.debug("all_args", vars(all_args))

    logger.info(">>> all_args.share_policy:", all_args.share_policy, True)
    logger.info(">>> all_args.algorithm_name:", all_args.algorithm_name, True)
    logger.info(">>> all_args.use_recurrent_policy:", all_args.use_recurrent_policy, True)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logger.warning("choose to use:", "gpu", True)
        device = torch.device("cuda:0")
        if all_args.cuda_deterministic: # default by True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logger.warning("choose to use:", "cpu", True)
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # run dir
    # TODO : folder by desc
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
                / str(all_args.env_name+all_args.scenario_name) / all_args.algorithm_name \
                / all_args.experiment_name / all_args.desc
                # / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args, #type: ignore
                         project=all_args.env_name+all_args.scenario_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" + \
                              str(all_args.experiment_name) + \
                              "_seed" + str(all_args.seed),
                         group=all_args.desc,
                         dir=str(run_dir),
                         job_type="training",
                         mode=all_args.wandb_mode,
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    runner_config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.metadrive_runner import MetaDriveRunner as Runner
    else:
        from onpolicy.runner.separated.metadrive_runner import MetaDriveRunner as Runner

    runner = Runner(runner_config)
    runner.run() # start to train
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close() #type: ignore

    if all_args.use_wandb:
        run.finish() #type: ignore
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


def log_exp_info(all_args):
    print('\n')
    logger.success(f"{'='*10} Start training {'=='*10}")
    logger.info(" <description>", all_args.desc, True)
    logger.info(" <env_name>", all_args.env_name, True)
    logger.info(" <scn_name>", all_args.scenario_name, True)
    logger.info(" <alg_name>", all_args.algorithm_name, True)
    logger.info(" <exp_name>", all_args.experiment_name, True)
    print('\n')

if __name__ == "__main__":
    cli_args = sys.argv[1:]
    all_args, config = get_all_args_and_config(cli_args)
    seed_max = all_args.seed_max
    logger.debug(all_args)
    logger.debug(config)
    log_exp_info(all_args)

    for seed in range(seed_max):
        logger.success(" Running begin! >>> seed:", f"{seed+1}/{seed_max}", True)
        all_args.seed = seed+1
        main(all_args, config)
        logger.success(f" Running done! >>> seed:", f"{seed+1}/{seed_max}", True)
        print('\n')