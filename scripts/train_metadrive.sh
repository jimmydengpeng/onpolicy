#!/bin/sh

TEST_MODE=true

# env="MetaDriveIntersection"
# env="MetaDriveRoundaboutEval"
env="MetaDrive"
config="metadrive"
# scenario="Bottleneck"
# scenario="Roundabout"
scenario="Intersection"
num_agents=8
# algo="rmappo"

num_torch_threads=4
num_rollout_env=5
# num_rollout_env=30
# num_env_steps=100000
num_env_steps=10000000
delay_done=25

seed_max=1
horizon=1000

desc="delay=${delay_done}&envs=${num_rollout_env}"

exp="share_${num_agents}a${num_torch_threads}t${num_rollout_env}p"

if ${TEST_MODE}; then wandb_mode='disabled'; else wandb_mode='online'; fi


CUDA_VISIBLE_DEVICES=0 python train/train_metadrive.py \
--env_name ${env} \
--config ${config} \
--experiment_name ${exp} \
--scenario_name ${scenario} \
--num_agents ${num_agents} \
--seed_max ${seed_max} \
--n_training_threads ${num_torch_threads} \
--n_rollout_threads ${num_rollout_env} \
--episode_length ${horizon} \
--num_env_steps ${num_env_steps} \
--desc ${desc} \
--delay_done ${delay_done} \
--user_name "jimmydeng" \
--wandb_mode ${wandb_mode}
