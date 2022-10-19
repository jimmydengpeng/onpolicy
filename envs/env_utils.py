from inspect import getmembers
import gym
import gym.spaces


''' gym space '''

def get_space_shape(space) -> int:
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]  # type: ignore
    elif isinstance(space, gym.spaces.Discrete):
        return space.n  # type: ignore
    elif isinstance(space, gym.spaces.Dict): # multi-agent
        spaces = [get_space_shape(space[key]) for key in space] # type: ignore
        dim = spaces[0]
        spaces = [s - dim for s in spaces]
        assert not any(spaces)
        return dim
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

def get_metadrive_obs_shape(env):
    return get_space_shape(env.observation_space)

def get_metadrive_action_shape(env):
    return get_space_shape(env.action_space)


''' tests '''

def test_metadrive():
    from metadrive import MultiAgentRoundaboutEnv
    METADRIVE_ENV_CONFIG = dict(
        use_render=False,
        start_seed=42,
        environment_num=100,
        num_agents=16,
        manual_control=False,
        crash_done=False,
        horizon=1000, # default in env
    )

    env = MultiAgentRoundaboutEnv(METADRIVE_ENV_CONFIG)
    print(get_metadrive_obs_shape(env))
    print(get_metadrive_action_shape(env))


if __name__ == "__main__":
    test_metadrive()