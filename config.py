import argparse
from typing import Dict
import yaml


'''参数、设置的优先级：
    cli -> yaml -> parser default
'''

# TODO 将通用的args与只和特定环境如metadrive环境相关的args分离
def get_parser():
    """
    The configuration parser for common hyperparameters of all environment. 
    """

    parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", type=str, default="metadrive", help="yaml config file name")

    # prepare parameters
    parser.add_argument("--algorithm_name",             type=str, default='mappo', choices=["rmappo", "mappo"])
    parser.add_argument("--experiment_name",            type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed",                       type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--seed_max",                   type=int, default=1, help="Max seeds for seed")
    parser.add_argument("--cuda",                       action='store_false', default=True, 
                                                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",         action='store_false', default=True, 
                                                        help="by default True, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads",         type=int, default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads",          type=int, default=32, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads",     type=int, default=1, help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads",   type=int, default=1, help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps",              type=int, default=10e6, help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name",                  type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb",                  action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument("--wandb_mode",                 type=str, default='online', help="whether or not to use online mode in wandb")

    # env parameters
    parser.add_argument("--env_name",                   type=str, default='MetaDrive', help="specify the name of environment")
    parser.add_argument("--scenario_name",              type=str, default='Intersection', help="Which scenario to run on")
    parser.add_argument('--num_agents',                 type=int, default=2, help="number of agents")
    parser.add_argument("--use_obs_instead_of_state",   action='store_true', default=False, help="Whether to use global state or concatenated obs")
    
    # MetaDrive #TODO
    parser.add_argument("--delay_done", type=int, default=25, help="")

    # replay buffer parameters
    parser.add_argument("--episode_length",             type=int, default=1000, help="Max length for any episode") # horizon in env config

    # network parameters
    parser.add_argument("--share_policy",               action='store_false', default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V",          action='store_false', default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames",             type=int, default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames",         action='store_true', default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size",                type=int, default=64, help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N",                    type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU",                   action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart",                 action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm",              action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization",  action='store_false', default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal",             action='store_false', default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain",                       type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    # for DEBUG & grouping in wanb
    parser.add_argument("--desc", type=str, default=None, help="by default None. description of this debug test")

    return parser


ALL_YAML_CONFIGS = {
    "metadrive": "metadrive.yaml",
    # "bridge": "configs/bridge.yaml"
}


def make_config(cfg_name) -> Dict:
    with open(ALL_YAML_CONFIGS[cfg_name]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def merge_commone_cfg(cli_args, all_args, config):
    ''' 遍历 config(yaml) 里 commone 的每个选项, 如果 cli_args 里没有设置该项,
        就将 config 里的该值添加到 all_args.
        注意: Parser 中未添加 yaml 中的一些设置, 如 policy, environment, 这样就可以从 config 中单独获取这些设置.
    :param cli_args: 终端传入的所有参数
    :param all_args: parser 解析的所有已知参数
    :param config:   从 yaml 文件读取的所有设置
    '''
    KEY_COMMON = 'common'
    for k, v in config.get(KEY_COMMON, {}).items():
        if f"--{k}" not in cli_args:
            setattr(all_args, k, v)
        else:
            from onpolicy import logger
            logger.warning(f"Overwritten arguments:", f"{k} = {getattr(all_args, k)}.", True)


""" API """

def get_all_args_and_config(cli_args):
    parser = get_parser()
    all_args = parser.parse_known_args(cli_args)[0] # returning a two item tuple containing known & unknown agrs
    config = make_config(all_args.config)
    merge_commone_cfg(cli_args, all_args, config)
    return all_args, config