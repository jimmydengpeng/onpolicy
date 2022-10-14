#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_spread  # simple_reference
num_landmarks=3
num_agents=3
algo="rmappo"
# algo="mappo"
num_torch_threads=8
num_rollout_env=2
exp="simple_spread_share_8t${num_rollout_env}p"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads ${num_torch_threads} \
    --n_rollout_threads ${num_rollout_env} \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 20000000 \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "jimmydeng" \
    # --use_recurrent_policy \
    # --share_policy \
    #--wandb_name "zoeyuchao" 
done
