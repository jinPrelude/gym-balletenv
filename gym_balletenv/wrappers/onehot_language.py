"""One-hot encode language observations"""

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Tuple, MultiBinary, Discrete

class OnehotLanguage(gym.ObservationWrapper):
    """One-hot encode language observations from (N,) to (N, vocab_size)"""

    def __init__(self, env: gym.Env):
        """One-hot encode language observations from (N,) to (N, vocab_size)

        Args:
            env (Env): The environment to apply the wrappe
        """
        super().__init__(env)

        assert (
            isinstance(self.observation_space[1], Discrete)
            and self.observation_space[1].n == 14
        )

        self.observation_space = Tuple(
            (self.observation_space[0], MultiBinary(14))
        )

    def observation(self, observation):
        image_obs, lang_obs = observation
        lang_one_hot = np.zeros(14, dtype=np.uint8)
        lang_one_hot[lang_obs] = 1
        return (image_obs, lang_one_hot)