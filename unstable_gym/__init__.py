from gymnasium.envs.registration import register

register(
    id="unstable_gym/UnstablePendulum-v0",
    entry_point="unstable_gym:UnstablePendulumEnv",
)