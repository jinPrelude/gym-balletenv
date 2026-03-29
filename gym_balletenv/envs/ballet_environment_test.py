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

"""Tests for ballet_environment."""
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
    # check agent template is at its actual board position (no scrolling)
    ar, ac = env._current_game.agent_row, env._current_game.agent_col
    np.testing.assert_array_almost_equal(
        observation[0][ar*9:(ar+1)*9, ac*9:(ac+1)*9],
        ballet_environment._CHAR_TO_TEMPLATE_BASE[
            ballet_environment_core.AGENT_CHAR])

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
    char_to_color_shape = env._current_game.char_to_color_shape
    shapes = [cs[1].split()[1] for cs in char_to_color_shape]
    self.assertTrue(all(s == "triangle" for s in shapes))

  def test_easy_mode_render(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2_easy", max_steps=200)
    env.reset(seed=0)
    rendered = env.render()
    self.assertEqual(rendered.shape, (99, 200, 3))
    self.assertEqual(rendered.dtype, np.uint8)


  # --- Symbolic mode tests ---

  def test_symbolic_observation_space(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2", max_steps=200, symbolic=True)
    obs_box = env.observation_space[0]
    self.assertEqual(obs_box.shape, (121,))
    self.assertEqual(obs_box.dtype, np.float32)
    np.testing.assert_array_equal(obs_box.low, np.zeros(121, dtype=np.float32))
    np.testing.assert_array_equal(obs_box.high, np.ones(121, dtype=np.float32))
    self.assertEqual(env.observation_space[1].n, 14)

  def test_symbolic_reset_shape(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2", max_steps=200, symbolic=True)
    obs, info = env.reset(seed=0)
    self.assertEqual(obs[0].shape, (121,))
    self.assertEqual(obs[0].dtype, np.float32)
    self.assertIn(obs[1], range(14))

  def test_symbolic_values_in_range(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2", max_steps=200, symbolic=True)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
      self.assertTrue(np.all(obs[0] >= 0.0))
      self.assertTrue(np.all(obs[0] <= 1.0))
      obs, _, terminated, truncated, _ = env.step(0)
      if terminated or truncated:
        break

  def test_symbolic_board_encoding(self):
    """Verify board cells map to expected categorical values."""
    env = ballet_environment.BalletEnvironment(
        "2_delay2", max_steps=200, symbolic=True)
    obs, _ = env.reset(seed=0)
    board = obs[0].reshape(11, 11)
    # Walls: top row should all be wall category (1/10 = 0.1)
    wall_val = 1.0 / (ballet_environment._NUM_CATEGORIES - 1)
    for val in board[0, :]:
      self.assertAlmostEqual(val, wall_val)
    # Floor: empty cells should be 0.0
    self.assertAlmostEqual(board[1, 1], 0.0)  # interior corner is floor
    # Agent: center (5, 5) should be agent category (2/10 = 0.2)
    agent_val = 2.0 / (ballet_environment._NUM_CATEGORIES - 1)
    self.assertAlmostEqual(board[5, 5], agent_val)

  def test_symbolic_dancer_movement_changes_board(self):
    """Board state should change when a dancer moves during its solo."""
    env = ballet_environment.BalletEnvironment(
        "1_delay2", max_steps=200, symbolic=True)
    obs, _ = env.reset(seed=0)
    prev_board = obs[0].copy()
    board_changed = False
    for _ in range(30):
      obs, _, terminated, truncated, _ = env.step(0)
      if not np.array_equal(obs[0], prev_board):
        board_changed = True
        break
      prev_board = obs[0].copy()
      if terminated or truncated:
        break
    self.assertTrue(board_changed,
                    "Board should change as dancer moves during solo")

  def test_symbolic_end_to_end(self):
    """Navigate to the single dancer using board positions."""
    env = ballet_environment.BalletEnvironment(
        "1_delay2", max_steps=200, symbolic=True)
    obs, _ = env.reset(seed=0)
    agent_val = 2.0 / (ballet_environment._NUM_CATEGORIES - 1)
    dancer_val = 3.0 / (ballet_environment._NUM_CATEGORIES - 1)  # dancer_0
    got_reward = False
    for _ in range(200):
      board = obs[0].reshape(11, 11)
      # Find agent and dancer positions from the board
      agent_pos = np.argwhere(np.isclose(board, agent_val))
      dancer_pos = np.argwhere(np.isclose(board, dancer_val))
      if len(agent_pos) > 0 and len(dancer_pos) > 0:
        dr = dancer_pos[0, 0] - agent_pos[0, 0]
        dc = dancer_pos[0, 1] - agent_pos[0, 1]
        if dr < 0 and dc == 0: action = 0
        elif dr < 0 and dc > 0: action = 1
        elif dr == 0 and dc > 0: action = 2
        elif dr > 0 and dc > 0: action = 3
        elif dr > 0 and dc == 0: action = 4
        elif dr > 0 and dc < 0: action = 5
        elif dr == 0 and dc < 0: action = 6
        elif dr < 0 and dc < 0: action = 7
        else: action = 0
      else:
        action = 0
      obs, reward, terminated, truncated, info = env.step(action)
      if reward == 1.0:
        got_reward = True
        break
      if terminated or truncated:
        break
    self.assertTrue(got_reward, "Agent should reach the single dancer")

  def test_symbolic_render(self):
    env = ballet_environment.BalletEnvironment(
        "2_delay2", max_steps=200, symbolic=True)
    env.reset(seed=0)
    rendered = env.render()
    self.assertEqual(rendered.shape, (99, 200, 3))
    self.assertEqual(rendered.dtype, np.uint8)


if __name__ == "__main__":
  absltest.main()
