from collections import defaultdict
from email.policy import default
import time
import numpy as np
from copy import copy
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
from onpolicy.utils.utils import LogLevel, debug_msg, debug_print, time_str
from onpolicy.utils.env_utils import get_single_agent_space
from colorlog import logger

def _t2n(x):
    return x.detach().cpu().numpy()

def log_title():
    print('#' * 108 + '\n'
          f"{'seed':>4}{'Step':>10}{'totalStep':>12}{'episode':>10}  |"
          f"{'avgRew':>8}{'sumRew':>10}{'numAgt':>8}  |"
          f"{'succ':>7}{'crash':>7}{'out':>7}  |"
          f"{'FPS':>6}{'ETA':>10}"
    )
    '''
    ################################################################################
    seed     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    '''

class MetaDriveRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MetaDriveRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        log_title()
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                #                      ^--> list of dict: [ {'agent0': r, ... * n_agents}, ... * n_envs ]
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                # insert data into buffer
                self.insert(data)


            # compute return and update network
            self.compute()

            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                # print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                #         .format(self.all_args.scenario_name,
                #                 self.algorithm_name,
                #                 self.experiment_name,
                #                 episode,
                #                 episodes,
                #                 total_num_steps,
                #                 self.num_env_steps,
                #                 int(total_num_steps / (end - start))))
                
                # env_infos = {}
                # for agent_id in range(self.num_agents):
                #     idv_rews = []
                #     for info in infos: #FIXME
                #         if 'step_reward' in info[agent_id].keys():
                #             idv_rews.append(info[agent_id]['step_reward'])
                #     agent_k = 'agent%i/individual_rewards' % agent_id
                #     env_infos[agent_k] = idv_rews

                '''
                1. individual rewards in one episode (环境的车辆数量不是每一时刻都是严格的最大车辆数)
                2. global total/sum rewards of every agents in one episode
                '''
                episode_log_infos = {}
                episode_results = infos #type: ignore # tuple of dict(every env's result) 
                res_keys = ('episode_length_mean',
                            'episode_reward_mean',
                            'episode_reward_sum',
                            'num_agents_total',
                            'success_rate',
                            'crash_rate',
                            'out_rate')
                skipped_keys = defaultdict(int)
                for k in res_keys:
                    # episode_log_infos[k] = np.mean([res[k] for res in episode_results if k in res])
                    _data = []
                    for res in episode_results:
                        if k in res:
                            _data.append(res[k])
                        else:
                            skipped_keys[k] += 1
                            if res == {}:
                                debug_msg(f"find 1 empty env res!!! @ key {k}")
                    episode_log_infos[k] = np.mean(_data)
                     
                # if result from less env than normal
                # episode_log_infos['<envs'] = skipped_keys if len(skipped_keys) > 0 else ""
                if len(skipped_keys) > 0:
                    debug_print(skipped_keys, level=LogLevel.ERROR)
                    # debug_print(infos, level=LogLevel.ERROR)
                    # from onpolicy.utils.utils import pretty_print
        

                self.log_train(episode_log_infos, total_num_steps)

                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                time_used = (end - start)
                fps = int(total_num_steps / time_used)
                train_infos['fps'] = fps
                eta = time_str((int((self.num_env_steps - total_num_steps) / fps)), simple=True)
                print(f"{self.all_args.seed:>4}"
                      f"{total_num_steps:>10}{self.num_env_steps:>12}{str(episode+1)+'/'+str(episodes):>10}  |" 
                      f"{episode_log_infos['episode_reward_mean']:>8.2f}{episode_log_infos['episode_reward_sum']:>10.2f}{str(int(episode_log_infos['num_agents_total']))+'/'+str(self.num_agents):>8}  |"
                      f"{episode_log_infos['success_rate']*100:>7.2f}{episode_log_infos['crash_rate']*100:>7.2f}{episode_log_infos['out_rate']*100:>7.2f}  |"
                      f"{fps:>6}{eta:>10}"
                    #   f"  | {episode_log_infos['<envs']}"
                )
                self.log_train(train_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset() # obs.shape: (n_rollout_threads, num_agents, obs_dim)
        # debug_print("reset obs shape", obs.shape, inline=True)
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1) # shape: (n_rollout_threads, num_agents * obs_dim)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1) #FIXME in MPE, global state is a concatenation of all agents' local obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        assert get_single_agent_space(self.envs.action_space).__class__.__name__ == 'Box'
        actions_env = actions

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32) # filter & reassign value: 将 dones 里为 True 的位置对应在mask里的位置上的元素反转成0，注意mask的维度比dones多1

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render_old(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    @torch.no_grad()
    def render(self):
        envs = self.envs

        extra_args = dict(mode="top_down", film_size=(1000, 1000))

        succ_rates = []
        crash_rates = []
        out_rates = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()

            rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_render_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_render_rollout_threads))

                assert get_single_agent_space(self.envs.action_space).__class__.__name__ == 'Box'

                actions_env = actions

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)

                envs.render(**extra_args)
                # time.sleep(0.015)

            debug_msg(f"epi: {episode+1}/{self.all_args.render_episodes} finished!")
            info = infos[0]
            succ_rates.append(info['success_rate'])
            crash_rates.append(info['crash_rate'])
            out_rates.append(info['out_rate'])
        
        print('')
        debug_msg(f'average succ rate in {self.all_args.render_episodes} episodes:   {np.mean(succ_rates)*100:.2f}%', LogLevel.SUCCESS)
        debug_msg(f'average crash rate in {self.all_args.render_episodes} episodes:    {np.mean(crash_rates)*100:.2f}%', LogLevel.ERROR)
        debug_msg(f'average out rate in {self.all_args.render_episodes} episodes:    {np.mean(out_rates)*100:.2f}%', LogLevel.WARNING)
        print('')

            # print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
