from gymnasium.envs.registration import register

register(
    id='gym_balletenv/balletenv-v0',
    entry_point='gym_balletenv.envs.ballet_environment:BalletEnvironment',
    kwargs={"level_name": "2_delay16", "max_steps": 500},
)