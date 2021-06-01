from gym.envs.registration import register

register(
    id='CarEnv-v1',
    entry_point='gym_car.envs:CarEnv',
)