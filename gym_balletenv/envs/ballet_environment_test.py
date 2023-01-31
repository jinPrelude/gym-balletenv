# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for pycolab_ballet.ballet_environment_wrapper."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from gym_balletenv.envs import ballet_environment
from gym_balletenv.envs import ballet_environment_core


class BalletEnvironmentTest(parameterized.TestCase):

  def test_full_wrapper(self):
    env = ballet_environment.BalletEnvironment(
      "1_delay16",
      max_steps=200
    )
    observation, _ = env.reset(seed=0)
    level_size = ballet_environment_core.ROOM_SIZE
    upsample_size = ballet_environment.UPSAMPLE_SIZE
    # wait for dance to complete
    for i in range(30):
      observation = env.step(0)[0]
      self.assertEqual(observation[0].shape,
                       (level_size[0] * upsample_size,
                        level_size[1] * upsample_size,
                        3))
      self.assertEqual(observation[1], 0) # index 0 equal to "watch"
    for i in [1, 1, 1, 1]:  # first gets eaten before agent can move
      observation, reward, done, _, info = env.step(i)
      self.assertEqual(observation[0].shape,
                       (level_size[0] * upsample_size,
                        level_size[1] * upsample_size,
                        3))
      self.assertEqual(observation[1], 3) # index 3 equal to "up_and_down"
    self.assertEqual(reward, 1.)
    # check egocentric scrolling is working, by checking object is in center
    np.testing.assert_array_almost_equal(
        observation[0][45:54, 45:54],
        ballet_environment._generate_template("orange plus"))


if __name__ == "__main__":
  absltest.main()