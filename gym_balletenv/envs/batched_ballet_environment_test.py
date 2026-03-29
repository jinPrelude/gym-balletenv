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

"""Tests for batched_ballet_environment."""
from absl.testing import absltest
import numpy as np
from gymnasium.spaces import Tuple, Box, Discrete

from gym_balletenv.envs.batched_ballet_environment import BatchedBalletEnv, _AGENT_CODE
from gym_balletenv.envs import ballet_environment_core as ballet_core


class BatchedBalletEnvConstructorTest(absltest.TestCase):

    def test_constructor_rgb(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16")
        self.assertEqual(env.num_envs, 4)
        self.assertIsInstance(env.observation_space, Tuple)
        img_space = env.observation_space[0]
        self.assertEqual(img_space.shape, (99, 99, 3))
        self.assertEqual(img_space.dtype, np.uint8)
        self.assertEqual(env.observation_space[1].n, 14)
        self.assertIsInstance(env.action_space, Discrete)
        self.assertEqual(env.action_space.n, 8)
        env.close()

    def test_constructor_easy_bw(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16_easy")
        img_space = env.observation_space[0]
        self.assertEqual(img_space.shape, (99, 99, 1))
        env.close()

    def test_constructor_symbolic(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16", symbolic=True)
        obs_space = env.observation_space[0]
        self.assertEqual(obs_space.shape, (121,))
        self.assertEqual(obs_space.dtype, np.float32)
        env.close()

    def test_max_steps_auto(self):
        env = BatchedBalletEnv(num_envs=2, level_name="2_delay16")
        self.assertEqual(env._max_steps, 320)
        env.close()
        env = BatchedBalletEnv(num_envs=2, level_name="2_delay48")
        self.assertEqual(env._max_steps, 1024)
        env.close()


class BatchedBalletEnvResetTest(absltest.TestCase):

    def test_reset_symbolic_shapes(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16", symbolic=True)
        obs, info = env.reset()
        self.assertEqual(obs[0].shape, (4, 121))
        self.assertEqual(obs[0].dtype, np.float32)
        self.assertEqual(obs[1].shape, (4,))
        self.assertIn("instruction_string", info)
        env.close()

    def test_reset_rgb_shapes(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16")
        obs, info = env.reset()
        self.assertEqual(obs[0].shape, (4, 99, 99, 3))
        self.assertEqual(obs[0].dtype, np.uint8)
        self.assertEqual(obs[1].shape, (4,))
        env.close()

    def test_reset_easy_bw_shapes(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16_easy")
        obs, info = env.reset()
        self.assertEqual(obs[0].shape, (4, 99, 99, 1))
        env.close()

    def test_reset_deterministic(self):
        """Same seed -> same initial observation."""
        env1 = BatchedBalletEnv(num_envs=4, level_name="2_delay16", symbolic=True)
        obs1, _ = env1.reset(seed=42)
        env1.close()
        env2 = BatchedBalletEnv(num_envs=4, level_name="2_delay16", symbolic=True)
        obs2, _ = env2.reset(seed=42)
        env2.close()
        np.testing.assert_array_equal(obs1[0], obs2[0])
        np.testing.assert_array_equal(obs1[1], obs2[1])

    def test_reset_boards_have_entities(self):
        """After reset, each board has agent + dancers placed."""
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16", symbolic=True)
        env.reset(seed=0)
        for i in range(4):
            board = env._boards[i]
            # Agent should be present
            self.assertTrue(np.any(board == _AGENT_CODE))
            # At least 2 dancers (chars B, C, D, ... for dancer 0, 1, ...)
            dancer_count = 0
            for ch in ballet_core.POSSIBLE_DANCER_CHARS[:2]:
                if np.any(board == ord(ch)):
                    dancer_count += 1
            self.assertEqual(dancer_count, 2)
        env.close()

    def test_reset_initial_phase_is_dance(self):
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16", symbolic=True)
        env.reset(seed=0)
        np.testing.assert_array_equal(env._task_phase, np.zeros(4, dtype=np.uint8))
        env.close()


class BatchedBalletEnvDancePhaseTest(absltest.TestCase):

    def test_dance_phase_dancer_moves(self):
        """During dance phase, a dancer should move on the board."""
        env = BatchedBalletEnv(num_envs=1, level_name="1_delay2", symbolic=True)
        obs, _ = env.reset(seed=0)
        board_before = obs[0][0].copy()
        # Step a few times (still in dance phase)
        obs, rew, term, trunc, info = env.step(np.array([0]))
        obs, rew, term, trunc, info = env.step(np.array([0]))
        obs, rew, term, trunc, info = env.step(np.array([0]))
        board_after = obs[0][0]
        # Board should have changed due to dancer movement
        self.assertFalse(np.array_equal(board_before, board_after),
                         "Board should change as dancer moves")
        env.close()

    def test_dance_phase_agent_frozen(self):
        """Agent should not move during dance phase."""
        env = BatchedBalletEnv(num_envs=1, level_name="1_delay16", symbolic=True)
        env.reset(seed=0)
        agent_row_before = env._agent_rows[0].copy()
        agent_col_before = env._agent_cols[0].copy()
        # Try to move agent (action=2 = East) during dance phase
        env.step(np.array([2]))
        self.assertEqual(env._agent_rows[0], agent_row_before)
        self.assertEqual(env._agent_cols[0], agent_col_before)
        env.close()

    def test_transitions_to_choice_phase(self):
        """After all dances complete, should transition to choice phase."""
        env = BatchedBalletEnv(num_envs=1, level_name="1_delay2", symbolic=True)
        env.reset(seed=0)
        # 1 dancer, delay 2: dance = 16 steps + initial delay = ~19 steps
        for _ in range(50):
            env.step(np.array([0]))
            if env._task_phase[0] == 1:
                break
        self.assertEqual(env._task_phase[0], 1,
                         "Should have transitioned to choice phase")
        env.close()


if __name__ == "__main__":
    absltest.main()
