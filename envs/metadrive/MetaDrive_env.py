# from .environment import MultiAgentEnv
# from .scenarios import load
import random
from onpolicy.utils.utils import debug_print

from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)

envs = dict(
    Roundabout=MultiAgentRoundaboutEnv,
    Intersection=MultiAgentIntersectionEnv,
    Tollgate=MultiAgentTollgateEnv, # can't run
    Bottleneck=MultiAgentBottleneckEnv,
    Parkinglot=MultiAgentParkingLotEnv, # can't run
    PGMA=MultiAgentMetaDrive
)

def MetaDriveEnv(args, rank):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    debug_print(">>> building MetaDriveEnv, seed:", rank, inline=True)
    ENV_NAME = args.scenario_name # "Roundabout"

    ''' set args '''
    METADRIVE_ENV_CONFIG = dict(
        use_render=False,
        start_seed=args.seed + rank * 1000,
        environment_num=100,
        num_agents=args.num_agents,
        manual_control=False,
        crash_done=False,
        horizon=1000, # default by env
    )

    env_func = envs[ENV_NAME]
    env = env_func(METADRIVE_ENV_CONFIG)
    setattr(env, "share_observation_space", env.observation_space)

    return env
