from typing import Tuple
import numpy as np
import gym
import gym.spaces
from gym.spaces import Box 
from gym.spaces import Discrete
from gym.spaces import Dict

''' gym space '''
def get_num_agents(env_space) -> int:
    """ rerturn the number of agents according to a env's space
    Args:
        space: a env's obs or action space.
    Returns:
        num of agent in the env.
    """
    if isinstance(env_space, Box) or isinstance(env_space, Discrete):
        return 1
    elif isinstance(env_space, Dict): # multi-agent
        return len(env_space) #type: ignore
    else:
        raise NotImplementedError 

def _check_homogeneous_agents(env_space):
    assert isinstance(env_space, Dict)
    spaces = [np.array(get_space_shape(env_space[key])) for key in env_space]  #type: ignore
    shape = spaces[0]
    spaces = [(s - shape).all() for s in spaces]
    if isinstance(shape, np.ndarray):
        return not any(spaces)
    else:
        raise NotImplementedError 

''' for env space '''
def get_single_agent_space(env_space):
    """ wrapper of space for both single- & multi- agent env
    """
    if env_space.__class__.__name__ == 'Box':
        return env_space
    elif env_space.__class__.__name__ == 'Discrete':
        return  env_space
    elif env_space.__class__.__name__ == 'Dict':
        assert _check_homogeneous_agents(env_space)
        return get_single_agent_space(env_space[next(env_space.__iter__())])
    else:
        raise NotImplementedError


''' for single agent space '''
def get_space_width(space) -> int:
    """ return a shape's first size, e.g. a int of Box.shape[0] or Discrete.n
        for use of the size of first nueron layer in mlp
    """
    if isinstance(space, Box):
        return space.shape[0] #type: ignore
    elif isinstance(space, Discrete):
        return space.n #type: ignore
    elif isinstance(space, Dict): # multi-agent
        assert _check_homogeneous_agents(space)
        return get_space_width(space[next(space.__iter__())]) #type: ignore
    else:
        raise NotImplementedError

def get_space_shape(space) -> Tuple[int]:
    if space.__class__.__name__ == 'Box':
        return space.shape
    elif space.__class__.__name__ == 'Discrete':
        return  space.n #type: ignore
    else:
        raise NotImplementedError

def dict_to_list(ary_dict) -> list:
    ''' transfer a gym.spaces.Dict space to list of space '''
    ary_list = []
    if isinstance(ary_dict, gym.spaces.Dict):
        for k in ary_dict: #type: ignore
            ary_list.append(ary_dict[k]) #type: ignore
    elif isinstance(ary_dict, dict):
        for k in ary_dict.keys():
            ary_list.append(ary_dict[k])
    else:
        raise NotImplementedError

    return ary_list 

def get_metadrive_env_obs_shape(env):
    return get_space_width(env.observation_space)

def get_metadrive_env_action_shape(env):
    return get_space_width(env.action_space)



''' tests '''

def test_metadrive(env):
    print(get_metadrive_env_obs_shape(env))
    print(get_metadrive_env_action_shape(env))

def test_get_num_agents(space):
    print(get_num_agents(space))

def test_get_space_shape(space):
    print(get_space_shape(space))

if __name__ == "__main__":

    from metadrive import MultiAgentRoundaboutEnv
    METADRIVE_ENV_CONFIG = dict(
        use_render=False,
        start_seed=42,
        environment_num=100,
        num_agents=16,
        manual_control=False,
        crash_done=False,
        delay_done=25,
        horizon=1000, # default in env
    )

    env = MultiAgentRoundaboutEnv(METADRIVE_ENV_CONFIG)

    # print(_check_homogeneous_agents(env.observation_space))
    print(get_num_agents() (get_single_agent_space(env.observation_space)))

    # for k in env.observation_space:
    #     space = env.observation_space[k]
    #     print(type(space))
    #     print(space.shape)
    #     print(space.shape[0])
    #     # test_get_space_shape(space)
    #     exit()
    # test_get_num_agents(env.observation_space)
    # test_get_num_agents(env.action_space)
