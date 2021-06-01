
import json
import gym
import ray
import arcade
import os, sys
import wandb

from gym_car.envs.gameEnv import CarEnv
from ray import tune

import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac
import ray.rllib.agents.pg as pg


list_of_algorithems = ('dqn', 'ppo', 'sac', 'pg', 'a2c', 'a3c') # '','ddpg'



def main(alg=0):
    if alg in list_of_algorithems:
        algorithem = alg
    else:
        algorithem = list_of_algorithems[alg]

    wandb.init(project='RL', entity='henrik96', name=algorithem)


    N_ITER = 5
    EPOCH = 400

    checkpoint_root = "cp/a3c"


    #env = gym.make('gym_car:CarEnv-v1')



    results = []
    episode_data = []
    episode_json = []

    n = 0
    r = []
    t = []


    if algorithem is 'a3c':
        config = a3c.DEFAULT_CONFIG.copy()
        agent = ppo.PPOTrainer(env=CarEnv, config=config)
    elif algorithem is 'ddpg':
        config = ddpg.DEFAULT_CONFIG.copy()
        agent = ddpg.DDPGTrainer(env=CarEnv, config=config)
    elif algorithem is 'dqn':
        config = dqn.DEFAULT_CONFIG.copy()
        agent = dqn.DQNTrainer(env=CarEnv, config=config)
    elif algorithem is 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        agent = ppo.PPOTrainer(env=CarEnv, config=config)
    elif algorithem is 'sac':
        config = sac.DEFAULT_CONFIG.copy()
        agent = sac.SACTrainer(env=CarEnv, config=config)
    elif algorithem is 'td3':
        config = ddpg.DEFAULT_CONFIG.copy()
        agent = ddpg.TD3Trainer(env=CarEnv, config=config)
    elif algorithem is 'pg':
        config = pg.DEFAULT_CONFIG.copy()
        agent = pg.PGTrainer(env=CarEnv, config=config)
    elif algorithem is 'a2c':
        config = a3c.DEFAULT_CONFIG.copy()
        agent = a3c.A2CTrainer(env=CarEnv, config=config)
    else:
        raise Exception("No valide algorithem: "+algorithem)


    for n in range(EPOCH):
        result = agent.train()

        wandb.log({"episode_reward_min": result["episode_reward_min"],
                   "episode_reward_max": result["episode_reward_max"],
                   "episode_reward_mean": result["episode_reward_mean"],
                   "episodes_this_iter": result["episodes_this_iter"],
                   "episode_len_mean": result["episode_len_mean"]}
                  )

        results.append(result)

        episode = {'n': n,
                   'episode_reward_min': result['episode_reward_min'],
                   'episode_reward_mean': result['episode_reward_mean'],
                   'episode_reward_max': result['episode_reward_max'],
                   'episode_len_mean': result['episode_len_mean']
                   }

        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        file_name = agent.save(checkpoint_root)

        print(f'{n + 1:3d}/{EPOCH}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')
        t.append(n)
        r.append(result["episode_reward_mean"])

    wandb.finish()

if __name__ == '__main__':
    ray.init(ignore_reinit_error=False, include_dashboard=True)

    for alg in list_of_algorithems:
        main(alg)

