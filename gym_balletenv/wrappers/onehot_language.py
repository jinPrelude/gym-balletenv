"""One-hot encode language observations"""

import numpy as np

import gym
from gym.spaces import Box, Tuple, MultiBinary

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
            and self.observation_space[1].n == 1
        )

        self.observation_space = Tuple(
            (self.observation_space[0], MultiBinary(14))
        )

    def observation(self, observation):
        image_obs, lang_obs = observation
        lang_one_hot = np.zeros(14, dtype=np.uint8)
        lang_one_hot[lang_obs] = 1
        return (lang_one_hot, lang_one_hot)