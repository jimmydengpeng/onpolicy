#!/bin/sh
env="MetaDriveRoundabout"
scenario="Roundabout"
num_agents=8
algo="rmappo"
# algo="mappo" # set --use_recurrent_policy (will be False, default by True)
num_torch_threads=8
num_rollout_env=8
num_env_steps=10000000
# num_env_steps=10000
seed_max=5
horizon=1000
SHARE=true # <~~ Change Here!

if $SHARE
then
    exp="share_${num_agents}a${num_torch_threads}t${num_rollout_env}p"
    echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
    for seed in `seq ${seed_max}`;
    do
        echo "seed is ${seed}:"
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
        --ppo_epoch 10 \
        --use_ReLU \
        --gain 0.01 \
        --lr 7e-4 \
        --critic_lr 7e-4 \
        --user_name "jimmydeng" \
        --use_centralized_V \
        --log_interval 1
        # --use_recurrent_policy \
        # --use_wandb
    done
else
    exp="${scenario}_separate_${num_torch_threads}t${num_rollout_env}p"
    echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
    for seed in `seq ${seed_max}`;
    do
        echo "seed is ${seed}:"
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
        --ppo_epoch 10 \
        --use_ReLU \
        --gain 0.01 \
        --lr 7e-4 \
        --critic_lr 7e-4 \
        --user_name "jimmydeng" \
        --use_recurrent_policy \
        --share_policy
    done
fi
