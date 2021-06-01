import gym, ray
import numpy as np
from game import *
import arcade

MAX_STEP_COUNT = 300


class CarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None):

        self.Game = MyGame(render=True, AI_mode=True)
        self.Game.setup()
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=np.array([-10, -10, -10, -10, -10, -10, -10, 0, -2 * np.pi]),
                                                high=np.array([600, 600, 600, 600, 600, 600, 600, 8, 2 * np.pi]))
        #arcade.run()
        #arcade.get_window()
        #arcade.start_render()

        self.step_count = 0


    def step(self, action):
        ob, score, done = self.Game.take_action(action)
        self.step_count += 1

        if self.step_count >= MAX_STEP_COUNT:
            done = True

        #self.Game.on_draw()
        return np.asarray(ob), score, done, {}

    def reset(self):
        self.Game.reset()
        ob, _, __ = self.Game.take_action(0)
        self.step_count = 0
        return np.asarray(ob)

    def render(self, mode='human'):
        self.Game.do_rendering = True



import ray.rllib.agents.a3c as a3c
#import ray.rllib.agents.ddpg as ddpg
#import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
#import ray.rllib.agents.sac as sac