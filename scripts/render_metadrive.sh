#!/bin/sh

# env="MetaDriveBottleneck"
env="MetaDriveIntersection"
# env="MetaDriveRoundaboutEval"
# env="MetaDrive"
# scenario="Bottleneck"
# scenario="Roundabout"
scenario="Intersection"
# algo="rmappo"
algo="mappo" # set --use_recurrent_policy (will be False, default by True)

num_agents=4
num_torch_threads=8
num_rollout_env=20

render_episodes=10
log_interval=1
horizon=1000
# actor_dir="latest-run"
# actor_dir="run-20221031_012011-30xyy89r"
# actor_dir="run-20221031_101306-2wkz9dlh"
actor_dir="run-20221029_024422-b0l6gtb8"
# actor_dir="run-20221029_205940-31601k7w"
seed=3

exp="share_${num_agents}a${num_torch_threads}t${num_rollout_env}p"

    CUDA_VISIBLE_DEVICES=0 python train/render_metadrive.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --seed ${seed} \
    --episode_length ${horizon} \
    --render_episodes ${render_episodes} \
    --log_interval ${log_interval} \
    --actor_dir ${actor_dir} \
    --use_ReLU \
    --user_name "jimmydeng" \
    --use_recurrent_policy