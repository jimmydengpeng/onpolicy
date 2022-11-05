import os, sys
import socket
from pathlib import Path
import wandb
import torch
from onpolicy.config import get_config
from onpolicy.envs.metadrive.MetaDrive_env import getMetaDriveEnv
from envs.metadrive_vec_env import SubprocVecEnv, DummyVecEnv, ShareVecEnv
from onpolicy.utils.utils import LogLevel, debug_msg, debug_print


def make_render_env(all_args) -> ShareVecEnv:
    def get_env_fn(rank):
        def init_env():
            if "MetaDrive" in all_args.env_name:
                env = getMetaDriveEnv(all_args, rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    if all_args.n_render_rollout_threads == 1:
        return  SubprocVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_render_rollout_threads)])

def main(args):
    parser = get_config()
    parser.add_argument("--actor_dir", type=str, default="latest-run")
    all_args = parser.parse_known_args(args)[0]
    debug_print(">>> all_args.share_policy:", all_args.share_policy, level=LogLevel.INFO, inline=True)
    debug_print(">>> all_args.algorithm_name:", all_args.algorithm_name, level=LogLevel.INFO, inline=True)
    debug_print(">>> all_args.use_recurrent_policy:", all_args.use_recurrent_policy, level=LogLevel.INFO, inline=True)
    
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        debug_msg("choose to use gpu...", level=LogLevel.SUCCESS)
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic: # default by True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        debug_msg("choose to use cpu...", level=LogLevel.INFO)
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)


    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
                / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if all_args.use_wandb:
        all_args.model_dir = run_dir / 'wandb' / all_args.actor_dir / 'files'
    else:
        raise NotImplementedError

    debug_print("run_dir", run_dir)
    debug_print("all_args.model_dir", all_args.model_dir)
    assert run_dir.exists()
    assert all_args.model_dir.exists()

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args, #type: ignore
                         project=all_args.env_name+all_args.scenario_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" + \
                              str(all_args.experiment_name) + \
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="rendering",
                         mode="offline",
                         reinit=True)
    else:
        raise NotImplementedError

    exit()
    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    debug_print(">>> Using share_policy:", all_args.share_policy, inline=True, level=LogLevel.INFO)
    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.metadrive_runner import MetaDriveRunner as Runner
    else:
        from onpolicy.runner.separated.metadrive_runner import MetaDriveRunner as Runner

    runner = Runner(config)
    print(runner.trainer.policy)
    runner.render()



    envs.close()

    if all_args.use_wandb:
        run.finish() #type: ignore

if __name__ == "__main__":
    main(sys.argv[1:])
