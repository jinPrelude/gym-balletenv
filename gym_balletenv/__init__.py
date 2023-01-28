from gym.envs.registration import register

register(
    id='gym_balletenv/balletenv-v0',
    entry_point='gym_balletenv.envs:ballet_environment',
)