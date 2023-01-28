from gym.envs.registration import register

register(
    id='balletenv-v0',
    entry_point='gym_examples.envs:ballet_environment',
)