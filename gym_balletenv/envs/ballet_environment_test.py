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
    self.assertEqual(env._easy_mode, False)
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
    
    env = ballet_environment.BalletEnvironment(
      "2_delay2_easy",
      max_steps=200
    )
    self.assertEqual(env._easy_mode, True)

  def test_easy_mode_observation_space(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2_easy", max_steps=200)
    img_space = env.observation_space[0]
    self.assertEqual(img_space.shape, (99, 99, 1))
    self.assertEqual(env.observation_space[1].n, 14)

  def test_easy_mode_end_to_end(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2_easy", max_steps=200)
    obs, info = env.reset(seed=42)
    self.assertEqual(obs[0].shape, (99, 99, 1))
    self.assertIn(obs[1], range(14))
    for _ in range(20):
      obs, reward, terminated, truncated, info = env.step(0)
      self.assertEqual(obs[0].shape, (99, 99, 1))
      if terminated or truncated:
        break

  def test_generate_template_bw(self):
    template = ballet_environment._generate_template(
        "red triangle", easy_mode=True)
    self.assertEqual(template.shape, (9, 9, 1))
    self.assertEqual(template.dtype, np.uint8)
    unique_vals = set(np.unique(template))
    self.assertTrue(unique_vals.issubset({0, 255}))

  def test_generate_template_rgb_unchanged(self):
    template = ballet_environment._generate_template("red triangle")
    self.assertEqual(template.shape, (9, 9, 3))

  def test_easy_mode_agent_template_single_channel(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2_easy", max_steps=200)
    env.reset(seed=0)
    agent_template = env._char_to_template[
        ballet_environment_core.AGENT_CHAR]
    self.assertEqual(agent_template.shape, (9, 9, 1))

  def test_char_to_template_base_not_mutated(self):
    original_shape = ballet_environment._CHAR_TO_TEMPLATE_BASE[
        ballet_environment_core.AGENT_CHAR].shape
    env = ballet_environment.BalletEnvironment(
        "2_delay2_easy", max_steps=200)
    env.reset(seed=0)
    after_shape = ballet_environment._CHAR_TO_TEMPLATE_BASE[
        ballet_environment_core.AGENT_CHAR].shape
    self.assertEqual(original_shape, (9, 9, 3))
    self.assertEqual(after_shape, (9, 9, 3))

  def test_easy_mode_observation_single_channel(self):
    env = ballet_environment.BalletEnvironment(
        "1_delay2_easy", max_steps=200)
    obs, info = env.reset(seed=0)
    img = obs[0]
    self.assertEqual(img.shape, (99, 99, 1))
    self.assertEqual(img.dtype, np.uint8)

  def test_easy_mode_shapes_unified(self):
    env = ballet_environment.BalletEnvironment(
        "4_delay2_easy", max_steps=200)
    env.reset(seed=0)
    char_to_color_shape = env._current_game.the_plot[
        "char_to_color_shape"]
    shapes = [cs[1].split()[1] for cs in char_to_color_shape]
    self.assertTrue(all(s == "triangle" for s in shapes))

  def test_easy_mode_render(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2_easy", max_steps=200)
    env.reset(seed=0)
    rendered = env.render()
    self.assertEqual(rendered.shape, (99, 200, 3))
    self.assertEqual(rendered.dtype, np.uint8)


if __name__ == "__main__":
  absltest.main()