#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="rmappo"
exp="simple_spread_share"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py \
    --save_gifs \
    --share_policy \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --use_render \
    --episode_length 25 \
    --render_episodes 5 \
    --model_dir '/Users/jimmy/Projects/RL/MAPPO_codebase_on_policy-main/onpolicy/scripts/results/MPE/simple_spread/rmappo/simple_spread_share/wandb/latest-run/files' 
done
