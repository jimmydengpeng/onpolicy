import math, time
import numpy as np
from collections import defaultdict
from gym import Wrapper
from onpolicy.utils.utils import LogLevel, debug_print, debug_msg, pretty_print
from onpolicy.utils.env_utils import get_metadrive_env_obs_shape, get_metadrive_env_action_shape
from colorlog import logger

from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)


def norm(a, b):
    return math.sqrt(a**2 + b**2)

class DistanceMap:
    def __init__(self):
        self.distance_map = None
        self.clear()

    def find_in_range(self, v_id, distance):
        if distance <= 0:
            return []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret

    def update_distance_map(self, vehicles):
        self.distance_map.clear()
        keys = list(vehicles.keys())
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = vehicles[k1].position
                p2 = vehicles[k2].position
                distance = norm(p1[0] - p2[0], p1[1] - p2[1])
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance

    def clear(self):
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def get_rewards(self, reward_dict, distance):
        own_r_ret = {}
        nei_r_ret = {}
        num_neighbours = {}
        for k, own_r in reward_dict.items():
            other_rewards = []
            # neighbours = self._find_k_nearest(k, K)
            neighbours = self.find_in_range(k, distance)
            for other_k in neighbours:
                if other_k is None:
                    break
                else:
                    other_rewards.append(reward_dict[other_k])
            if len(other_rewards) == 0:
                other_reward = own_r
            else:
                other_reward = np.mean(other_rewards)

            own_r_ret[k] = own_r
            nei_r_ret[k] = other_reward
            num_neighbours[k] = len(neighbours)
        return own_r_ret, nei_r_ret, num_neighbours


class MetaDriveEvalEnv(Wrapper):
    """ Providing evaluating info across one episode for every agent spawn:
        (Note that one episode maybe less than the max length.)
        1. Total rewards of all agents
        2. Average reward
        3. Episodic success rate
    """
    _default_eval_config = dict(neighbours_distance=20)

    EPISODE_END = -1

    def __init__(self, env) -> None:
        super(MetaDriveEvalEnv, self).__init__(env)
        eval_config = self._default_eval_config
        self.eval_config = eval_config

    def reset(self):
        o = super(MetaDriveEvalEnv, self).reset()
        self.episode_step = 0
        return o

    def step(self, *args, **kwargs):
        o, r, d, i = super(MetaDriveEvalEnv, self).step(*args, **kwargs)
        self.on_episode_step(o, r, d, i)
        for kkk, ddd in d.items():
            if kkk != "__all__" and ddd:
                self.on_episode_end(kkk, o, r, d, i)
        return o, r, d, i
    
    def on_episode_start(self):
        self.user_data = defaultdict(lambda: defaultdict(dict))
        self.step_active_agents = {}
        self.episode_step = 0
        self.distance_map = DistanceMap()

    def on_episode_step(self, o, r, d, i):
        if self.episode_step == 0:
            self.on_episode_start()
            self.distance_map.clear()

        # Deal with SVO estimation
        distance = self.eval_config['neighbours_distance']  # TODO(Anonymous) not determined yet!
        self.distance_map.update_distance_map(self.unwrapped.vehicles)
        own_rewards, nei_rewards, num_neighbours = self.distance_map.get_rewards(r, distance=distance)
        for kkk in own_rewards.keys():
            self.user_data["own_reward"][self.episode_step][kkk] = own_rewards[kkk]
            self.user_data["num_neighbours"][self.episode_step][kkk] = num_neighbours[kkk]
            self.user_data["nei_reward"][self.episode_step][kkk] = nei_rewards[kkk]

        # Active agents
        self.step_active_agents[self.episode_step] = set(r.keys())

        # Ordinary stats
        for kkk in r.keys():
            info = i[kkk]
            if info:
                self.user_data["velocity"][self.episode_step][kkk] = info["velocity"]
                self.user_data["steering"][self.episode_step][kkk] = info["steering"]
                self.user_data["step_reward"][self.episode_step][kkk] = info["step_reward"]
                self.user_data["acceleration"][self.episode_step][kkk] = info["acceleration"]
                self.user_data["cost"][self.episode_step][kkk] = info["cost"]
                self.user_data["episode_length"][self.episode_step][kkk] = info["episode_length"]
                self.user_data["episode_reward"][self.episode_step][kkk] = info["episode_reward"]
                self.user_data["energy"][self.episode_step][kkk] = info["step_energy"]
                self.user_data["raw_action0_l2"][self.episode_step][kkk] = (info["raw_action"][0])**2
                self.user_data["raw_action1_l2"][self.episode_step][kkk] = (info["raw_action"][1])**2

        # Count
        self.episode_step += 1

    def on_episode_end(self, kkk, o, r, d, i):
        info = i[kkk]
        arrive_dest = info.get("arrive_dest", False)
        crash = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        max_step_rate = not (arrive_dest or crash or out_of_road)
        self.user_data["success"][self.EPISODE_END][kkk] = arrive_dest
        self.user_data["crash"][self.EPISODE_END][kkk] = crash
        self.user_data["max_step"][self.EPISODE_END][kkk] = max_step_rate
        self.user_data["out"][self.EPISODE_END][kkk] = out_of_road
        self.user_data["episode_energy"][self.EPISODE_END][kkk] = info["episode_energy"]


    ''' Interface Funtions'''

    def get_step_result(self):
        ret = {}
        step, active_agents = list(self.step_active_agents.items())[-1]
        for stat_name in self.user_data.keys():
            agent_step_data = []
            for kkk in active_agents:
                agent_step_value = self.user_data[stat_name][step].get(kkk, None)
                if agent_step_value is not None:
                    agent_step_data.append(agent_step_value)
            if len(agent_step_data) > 0:
                ret[stat_name] = np.mean(agent_step_data)

        # Group 2.2: Episode reward (return)
        agent_reward_list = list(self.user_data["episode_reward"].values())[-1]
        ret["episode_reward_mean"] = np.mean(list(agent_reward_list.values()))

        # Group 2.3: Episode cost
        agent_cost_list = defaultdict(float)
        for step, active_agents in self.step_active_agents.items():
            for kkk in active_agents:
                agent_step_value = self.user_data["cost"][step].get(kkk, None)
                if agent_step_value is not None:
                    agent_cost_list[kkk] += agent_step_value
        ret["episode_cost_mean"] = np.mean(list(agent_cost_list.values()))
        ret["episode_cost_sum"] = np.sum(list(agent_cost_list.values()))
        return ret

    def get_episode_result(self):
        ret = {}
        # ===== Group 1: Step-level stats =====
        # STAT_STEP_MEAN_EP_MIN/MEAN/MAX:
        #     average across agent, min/mean/max across steps.

        # Group 1.1: velocity
        step_means = []
        for step, active_agents in self.step_active_agents.items():
            agent_step_data = []
            for kkk in active_agents:
                agent_step_value = self.user_data["velocity"][step].get(kkk, None)
                if agent_step_value is not None:
                    agent_step_data.append(agent_step_value)
            if len(agent_step_data) > 0:
                step_means.append(np.mean(agent_step_data))
        ret["velocity_step_mean_episode_min"] = np.min(step_means)
        ret["velocity_step_mean_episode_mean"] = np.mean(step_means)
        ret["velocity_step_mean_episode_max"] = np.max(step_means)

        # Group 1.2: energy
        step_means = []
        for step, active_agents in self.step_active_agents.items():
            agent_step_data = []
            for kkk in active_agents:
                agent_step_value = self.user_data["energy"][step].get(kkk, None)
                if agent_step_value is not None:
                    agent_step_data.append(agent_step_value)
            if len(agent_step_data) > 0:
                step_means.append(np.mean(agent_step_data))
        ret["energy_step_mean_episode_min"] = np.min(step_means)
        ret["energy_step_mean_episode_mean"] = np.mean(step_means)
        ret["energy_step_mean_episode_max"] = np.max(step_means)

        # Group 1.4: Num neighbours
        step_means = []
        for step, active_agents in self.step_active_agents.items():
            agent_step_data = []
            for kkk in active_agents:
                agent_step_value = self.user_data["num_neighbours"][step].get(kkk, None)
                if agent_step_value is not None:
                    agent_step_data.append(agent_step_value)
            if len(agent_step_data) > 0:
                step_means.append(np.mean(agent_step_data))
        ret["num_neighbours_mean_episode_mean"] = np.mean(step_means)
        ret["num_neighbours_mean_episode_max"] = np.max(step_means)

        # ===== Group 2: Episode-level stats =====
        # STAT_EP_MIN/MEAN/MAX:
        #     min/mean/max across agent episodes, sum across steps.

        # Group 2.1: Success, number of active agents
        env_episode_len = len(self.step_active_agents)
        agent_success_list = list(self.user_data["success"][self.EPISODE_END].values())
        agent_crash_list = list(self.user_data["crash"][self.EPISODE_END].values())
        num_agents = len(agent_success_list)
        ret["num_agents_total"] = num_agents
        ret["num_agents_total_per_300_steps"] = num_agents / env_episode_len * 300
        ret["success_rate"] = sum(agent_success_list) / num_agents
        ret["num_agents_success"] = sum(agent_success_list)
        ret["num_agents_success_per_300_steps"] = sum(agent_success_list) / env_episode_len * 300
        ret["num_agents_failed_per_300_steps"] = sum(agent_crash_list) / env_episode_len * 300

        # Group 2.2: Episode reward (return)
        agent_reward_list = defaultdict(float)
        for step in sorted(self.step_active_agents): # sorted keys(steps:int) -> 0,1,2,...,999
            active_agents = self.step_active_agents[step]
            if step == self.EPISODE_END:
                continue
            for kkk in active_agents:
                agent_reward_list[kkk] = self.user_data["episode_reward"][step].get(kkk, 0)
        ret["episode_reward_mean"] = np.mean(list(agent_reward_list.values()))
        ret["episode_reward_min"] = np.min(list(agent_reward_list.values()))
        ret["episode_reward_max"] = np.max(list(agent_reward_list.values()))
        ret["episode_reward_sum"] = np.sum(list(agent_reward_list.values()))

        # Group 2.3: Episode cost
        agent_cost_list = defaultdict(float)
        for step, active_agents in self.step_active_agents.items():
            for kkk in active_agents:
                agent_step_value = self.user_data["cost"][step].get(kkk, None)
                if agent_step_value is not None:
                    agent_cost_list[kkk] += agent_step_value
        ret["episode_cost_mean"] = np.mean(list(agent_cost_list.values()))
        ret["episode_cost_min"] = np.min(list(agent_cost_list.values()))
        ret["episode_cost_max"] = np.max(list(agent_cost_list.values()))
        ret["episode_cost_sum"] = np.sum(list(agent_cost_list.values()))

        # Group 2.4: Crash, number of crash agents
        agent_crash_list = list(self.user_data["crash"][self.EPISODE_END].values())
        ret["crash_rate"] = sum(agent_crash_list) / num_agents
        ret["num_agents_crash"] = sum(agent_crash_list)

        # Group 2.5: Out, number of out agents
        agent_out_list = list(self.user_data["out"][self.EPISODE_END].values())
        ret["out_rate"] = sum(agent_out_list) / num_agents
        ret["num_agents_out"] = sum(agent_out_list)

        # Group 2.6: Episode length
        agent_len_list = defaultdict(float)
        for step in sorted(self.step_active_agents):
            active_agents = self.step_active_agents[step]
            if step == self.EPISODE_END:
                continue
            for kkk in active_agents:
                agent_len_list[kkk] = self.user_data["episode_length"][step].get(kkk, 0)
        ret["episode_length_mean"] = np.mean(list(agent_len_list.values()))

        # flag = False
        # for kkk, vvv in agent_len_list.items():
        #     if kkk not in self.user_data["success"][self.EPISODE_END]:
        #         debug_msg(f"{kkk} not in, ", level=LogLevel.ERROR)
        #         flag = True
        # if flag:
        #     debug_print("", agent_len_list)

        success_ep_len_mean = []
        for kkk, vvv in agent_len_list.items():
            if kkk in self.user_data["success"][self.EPISODE_END]:
                success_ep_len_mean.append(vvv)

        # success_ep_len_mean = [
        #     vvv for kkk, vvv in agent_len_list.items() if self.user_data["success"][self.EPISODE_END][kkk]
        # ]

        ret["success_episode_length_mean"] = np.mean(success_ep_len_mean) if success_ep_len_mean else 0

        # ===== Group 3: Meta stats =====
        # STAT:
        #     scalar
        # Group 3.1: Estimated SVO
        # all_own_reward = 0
        # for step_reward_dict in self.user_data["own_reward"].values():
        #     all_own_reward += sum(step_reward_dict.values())
        # all_own_reward /= num_agents
        # all_nei_reward = 0
        # for step_reward_dict in self.user_data["nei_reward"].values():
        #     all_nei_reward += sum(step_reward_dict.values())
        # all_nei_reward /= num_agents
        # alpha = np.rad2deg(math.atan2(all_nei_reward, all_own_reward))
        # multiplier = norm(all_nei_reward, all_own_reward)
        # svo = min(max(0, alpha), 90)
        # ret["svo_estimate_v0_deg"] = svo
        # ret["svo_estimate_v0_rad"] = np.deg2rad(svo)
        # ret["svo_reward_v0"] = multiplier * math.cos(np.deg2rad(svo) - np.deg2rad(alpha))

        # Group 3.2: Estimated SVO in step-level
        # all_own_reward = []
        # for step_reward_dict in self.user_data["own_reward"].values():
        #     all_own_reward += list(step_reward_dict.values())
        # all_own_reward = np.asarray(all_own_reward)
        # all_nei_reward = []
        # for step_reward_dict in self.user_data["nei_reward"].values():
        #     all_nei_reward += list(step_reward_dict.values())
        # all_nei_reward = np.asarray(all_nei_reward)
        # alpha = np.arctan2(all_nei_reward, all_own_reward)
        # multiplier = np.sqrt(all_nei_reward ** 2 + all_own_reward ** 2)
        # alpha_deg = np.rad2deg(alpha)
        # ret["svo_estimate_v2_deg_mean"] = np.mean(alpha_deg)
        # ret["svo_estimate_v2_deg_min"] = np.min(alpha_deg)
        # ret["svo_estimate_v2_deg_max"] = np.max(alpha_deg)
        # rewards = np.cos(alpha) * all_own_reward + np.sin(alpha) * all_nei_reward
        # ret["svo_reward_v2"] = np.sum(rewards) / num_agents

        # Group 3.3: Estimated SVO in agent-level
        all_own_reward = defaultdict(float)
        for _, step_reward_dict in self.user_data["own_reward"].items():
            for kkk, value in step_reward_dict.items():
                all_own_reward[kkk] += value
        all_nei_reward = defaultdict(float)
        for step_reward_dict in self.user_data["nei_reward"].values():
            for kkk, value in step_reward_dict.items():
                all_nei_reward[kkk] += value
        svos = []
        svo_rewards = []
        for kkk, own_reward in all_own_reward.items():
            nei_reward = all_nei_reward[kkk]
            alpha = np.rad2deg(math.atan2(nei_reward, own_reward))
            multiplier = norm(nei_reward, own_reward)
            svo = min(max(0, alpha), 90)
            svos.append(svo)
            svo_rewards.append(multiplier * math.cos(np.deg2rad(svo) - np.deg2rad(alpha)))
        ret["svo_estimate_deg_mean"] = np.mean(svos)
        ret["svo_estimate_deg_min"] = np.min(svos)
        ret["svo_estimate_deg_max"] = np.max(svos)
        ret["svo_reward"] = np.sum(svo_rewards) / num_agents

        return ret

    def render(self, mode='human', **kwargs):
        if mode == "top_down":
            return super(MetaDriveEvalEnv, self).render(mode, **kwargs)
        else:
            return self.unwrapped.render(text={k: "{:.3f}".format(v) for k, v in self.get_step_result().items()})


def getMetaDriveEnv(args, rank, env_config=None):
    '''
    Creates & return a MultiAgentEnv object as env.

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    logger.info("building MetaDriveEnv, rank:", rank, inline=True)

    envs = dict(
        Roundabout=MultiAgentRoundaboutEnv,
        Intersection=MultiAgentIntersectionEnv,
        Tollgate=MultiAgentTollgateEnv, # can't run
        Bottleneck=MultiAgentBottleneckEnv,
        Parkinglot=MultiAgentParkingLotEnv, # can't run
        PGMA=MultiAgentMetaDrive
    )

    ENV_NAME = args.scenario_name # "Roundabout"

    ''' set args '''
    METADRIVE_ENV_CONFIG = {
        "use_render": False,
        "start_seed": args.seed + rank * 1000, # * 1000
        "environment_num": 100,
        "num_agents": args.num_agents,
        "manual_control": False,
        "crash_done": True, # default True by env
        "delay_done": 0, # default 25 by env
        "horizon": args.episode_length, # default 1000 by env
    }
    
    if env_config:
        METADRIVE_ENV_CONFIG.update(env_config)
        logger.info("building MetaDriveEnv, delay_done:",METADRIVE_ENV_CONFIG['delay_done'], inline=True)
    env = envs[ENV_NAME](METADRIVE_ENV_CONFIG)
    setattr(env, "share_observation_space", env.observation_space) #TODO
    
    return MetaDriveEvalEnv(env)


def _detect_new_spawn(infos):
    flag = False
    for a in infos:
        if infos[a] == {}:
            debug_msg(f">>> step {i}, new: {a}")
            flag = True
    return flag
                    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', type=str, default='Intersection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_agents', type=int, default=8)
    parser.add_argument('--episode_length', type=int, default=1000)
    args = parser.parse_args()

    from metadrive.policy.idm_policy import ManualControllableIDMPolicy
    env_config = {
        # "use_render": True if not args.pygame_render else False,
        "use_render": False,
        "manual_control": False,
        "crash_done": True,
        "agent_policy": ManualControllableIDMPolicy,
        "num_agents": 4,
        "horizon": 1000, # default by env
        "delay_done": 1000
        # "allow_respawn": False
    }

    extra_args = dict(mode="top_down", film_size=(1000, 1000))

    env = getMetaDriveEnv(args, 0, env_config) 
  
    ''' eval '''

    obs = env.reset()
    start = time.time()
    if env.current_track_vehicle:
        print(env.current_track_vehicle)
        env.current_track_vehicle.expert_takeover = True
    
    # relative_step = 2
    # relative_flag = False
    # all_agents = set(obs.keys())


    episodes = 0
    epi_mean_rews = []
    epi_mean_length = []
    max_steps = 10000
    for i in range(0, max_steps):
        oall_agentsall_agentsbs, rews, dones, infos = env.step({agent_id: [0, 1] for agent_id in env.vehicles.keys()})

        env.render(**extra_args)
        time.sleep(0.02)


        if (i + 1) % 100 == 0:
            print(
                f"Finish {i+1}/{max_steps} simulation steps. Time elapse: {(time.time() - start):.4f}. \
                    Average FPS: {(i + 1) / (time.time() - start):.4f}")

 
        if dones["__all__"]:
            debug_print(" >>> reset() @ step i+1", i+1, inline=True)
            debug_print("     env step ", env.episode_steps, inline=True)
            
            res = env.get_episode_result()
            epi_mean_rews.append(res['episode_reward_mean'])
            epi_mean_length.append(res['episode_length_mean'])

            debug_print("episode_reward_mean", res['episode_reward_mean'], inline=True)
            debug_print("episode_length_mean", res['episode_length_mean'], inline=True)
            episodes += 1
            env.reset()
            # break

    debug_msg(f"epi_mean_rews {np.mean(epi_mean_rews)} / {episodes} episodes")
    debug_msg(f"epi_length_rews {np.mean(epi_mean_length)} / {episodes} episodes")
    # debug_print("all_agents", len(all_agents))

    # res = env.get_episode_result()
    env.close()
    print('\n')
    # print(pretty_print(res))


""" episode_result dict:

-->   crash_rate: 0.5252525252525253
      energy_step_mean_episode_max: 0.03562079709131
      energy_step_mean_episode_mean: 0.026447600511791205
      energy_step_mean_episode_min: 0.0
      episode_cost_max: 1.0
      episode_cost_mean: 0.45217391304347826
      episode_cost_min: 0.0
      episode_cost_sum: 52.0
-->   episode_length_mean: 123.80869565217391
      episode_reward_max: 159.29317931327068
-->   episode_reward_mean: 75.10874539724108  
      episode_reward_min: 3.4851252684743126
~~>   episode_reward_sum   # <~~ add
      num_agents_crash: 52
      num_agents_failed_per_300_steps: 15.615615615615615
      num_agents_out: 17
      num_agents_success: 30
      num_agents_success_per_300_steps: 9.00900900900901
-->   num_agents_total: 99 
      num_agents_total_per_300_steps: 29.72972972972973
      num_neighbours_mean_episode_max: 2.625
      num_neighbours_mean_episode_mean: 1.4821634610850296
-->   out_rate: 0.1717171717171717
      success_episode_length_mean: 129.91919191919192
-->   success_rate: 0.30303030303030304
      svo_estimate_deg_max: 76.64344350211869
      svo_estimate_deg_mean: 45.67520000243315
      svo_estimate_deg_min: 22.99266581453143
      svo_reward: 124.82216134715392
      velocity_step_mean_episode_max: 29.41912722558177
      velocity_step_mean_episode_mean: 22.494677042339884
      velocity_step_mean_episode_min: 0.0
"""