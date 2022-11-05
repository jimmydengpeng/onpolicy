#!/bin/sh
# env="MetaDriveIntersection"
# env="MetaDriveRoundaboutEval"
env="MetaDrive"
scenario="Bottleneck"
# scenario="Roundabout"
# scenario="Intersection"
desc="share_buffer_debug"
num_agents=4
# algo="rmappo"
algo="mappo" # set --use_recurrent_policy (will be False, default by True)
num_torch_threads=8
num_rollout_env=2
# num_rollout_env=30
num_env_steps=100000
# num_env_steps=10000000
log_interval=1
seed_max=1
horizon=1000
# SHARE=true # <~~ Change Here!

exp="share_${num_agents}a${num_torch_threads}t${num_rollout_env}p"
echo "== Start training =="
echo "  <description> ${desc}"
echo "  <env> ${env}, <scenario> ${scenario}, <algo> ${algo}, <exp> ${exp}, <max seed> ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "  >>> [seed] ${seed}"
    CUDA_VISIBLE_DEVICES=0 python train/train_metadrive.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --seed ${seed} \
    --n_training_threads ${num_torch_threads} \
    --n_rollout_threads ${num_rollout_env} \
    --num_mini_batch 1 \
    --episode_length ${horizon} \
    --num_env_steps ${num_env_steps} \
    --log_interval ${log_interval} \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "jimmydeng" \
    --use_centralized_V \
    --debug_desc ${desc} \
    --use_recurrent_policy \
    --use_wandb
done