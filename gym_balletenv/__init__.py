from gymnasium.envs.registration import register

register(
    id='gym_balletenv/balletenv-v0',
    entry_point='gym_balletenv.envs.ballet_environment:BalletEnvironment',
    kwargs={"level_name": "2_delay16"},
)

register(
    id='gym_balletenv/BalletEnvironment-v1',
    entry_point='gym_balletenv.envs.ballet_environment:BalletEnvironment',
)

register(
    id='gym_balletenv/balletenv-symbolic-v0',
    entry_point='gym_balletenv.envs.ballet_environment:BalletEnvironment',
    kwargs={"level_name": "2_delay16", "symbolic": True},
)