import numpy as np
import gym, random
from tqdm import tqdm
from gym_car.envs.gameEnv import CarEnv
from matplotlib import pyplot as plt
import wandb


class q():
    def __init__(self, env, epsilon=.1, gamma=.9, epsilon_deecay=0.9, epsilon_interval=100):
        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_deecay
        self.epsilon_runs = epsilon_interval

        self.gamma = gamma

        self.run = 1

        self.q_table = None

        self.init_q_table()

    def init_q_table(self):

        self.q_table = np.random.randint(0, 1, [10, 10, 10, 10, 10, 10, 10, 9], dtype=np.int16)

    def train(self):

        ## Training
        self.done = False
        self.ob = None

        step = 0
        state_action = []
        reward = []
        if self.run % self.epsilon_runs == 0:
            self.epsilon *= self.epsilon_decay
        while not self.done:

            p = np.random.random()

            if p > 1 - self.epsilon or self.ob is None:

                action = self.env.action_space.sample()
            else:

                ob = np.floor(self.ob / 60).astype(int)
                actions = self.q_table[tuple(ob)].astype(np.float)
                actions /= np.sum(actions.reshape(-1))

                p = np.random.random()

                action = 0
                val = 0
                for i, a in enumerate(actions):
                    if p > (a + val):
                        val += a
                    else:
                        action = i
                        break

            if not self.ob is None:
                ob = np.floor(self.ob / 60).astype(int)
                state_action.append([ob, action])

            self.ob, score, self.done, _ = self.env.step(action)
            reward.append(score)

            step += 1

        ## Bellman-update
        run_return = 0
        state_action.reverse()
        reward.reverse()

        for i, R in enumerate(reward):
            run_return += R * np.power(self.gamma, i)

        last = None
        for i, S_A in enumerate(state_action):
            if last is None:
                self.q_table[tuple(S_A)] = run_return
            else:
                self.q_table[tuple(S_A)] = reward[i] + last * self.gamma
            last = self.q_table[tuple(S_A)]

        self.run += 1
        self.env.reset()

        wandb.log({"episode_reward_min": min(reward),
                   "episode_reward_max": max(reward),
                   "episode_reward_mean": run_return,
                   "episodes_this_iter": step,
                   "episode_len_mean": step,
                   "epsilon": self.epsilon}
                  )

        return run_return, step


if __name__ == '__main__':
    wandb.init(project='RL', entity='henrik96', name='Q_learning')
    env = CarEnv()

    q = q(env, epsilon=.9, gamma=.95, epsilon_interval=40)

    Return = []
    Runn = []

    for _ in tqdm(range(1000)):
        G, N = q.train()
        Runn.append(N)
        Return.append(G)

    wandb.finish()

    plt.figure()

    plt.plot(range(len(Return)), Return)

    plt.show()
