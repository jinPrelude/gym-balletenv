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
from gym_balletenv.envs.ballet_environment import BalletEnvironment


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
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16", easy_mode=True)
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
        env = BatchedBalletEnv(num_envs=4, level_name="2_delay16", easy_mode=True)
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


class BatchedBalletEnvChoicePhaseTest(absltest.TestCase):

    def test_agent_moves_in_choice_phase(self):
        """Agent should move when in choice phase."""
        env = BatchedBalletEnv(num_envs=1, level_name="1_delay2", symbolic=True)
        env.reset(seed=0)
        for _ in range(50):
            env.step(np.array([0]))
            if env._task_phase[0] == 1:
                break
        agent_row_before = env._agent_rows[0].copy()
        env.step(np.array([4]))  # South
        self.assertEqual(env._agent_rows[0], agent_row_before + 1)
        env.close()

    def test_agent_blocked_by_wall(self):
        """Agent should not move into walls."""
        env = BatchedBalletEnv(num_envs=1, level_name="1_delay2", symbolic=True)
        env.reset(seed=0)
        for _ in range(50):
            env.step(np.array([0]))
            if env._task_phase[0] == 1:
                break
        prev_row = env._agent_rows[0].copy()
        for _ in range(20):
            env.step(np.array([0]))  # North
            if env._agent_rows[0] == prev_row:
                break
            prev_row = env._agent_rows[0].copy()
        self.assertEqual(env._agent_rows[0], 1)
        env.close()

    def test_collision_gives_reward(self):
        """Agent reaching target dancer should give reward=1.0."""
        env = BatchedBalletEnv(num_envs=1, level_name="1_delay2", symbolic=True)
        env.reset(seed=0)
        for _ in range(50):
            env.step(np.array([0]))
            if env._task_phase[0] == 1:
                break
        got_reward = False
        for _ in range(200):
            agent_r, agent_c = env._agent_rows[0], env._agent_cols[0]
            dancer_r, dancer_c = env._dancer_rows[0, 0], env._dancer_cols[0, 0]
            dr = int(np.sign(dancer_r - agent_r))
            dc = int(np.sign(dancer_c - agent_c))
            dir_map = {(-1,0):0, (-1,1):1, (0,1):2, (1,1):3,
                       (1,0):4, (1,-1):5, (0,-1):6, (-1,-1):7, (0,0):0}
            action = dir_map.get((dr, dc), 0)
            _, rew, term, trunc, _ = env.step(np.array([action]))
            if rew[0] == 1.0:
                got_reward = True
                break
            if term[0] or trunc[0]:
                break
        self.assertTrue(got_reward, "Agent should reach dancer and get reward")
        env.close()


class BatchedBalletEnvAutoResetTest(absltest.TestCase):

    def test_auto_reset_continues_after_termination(self):
        """After a terminated env auto-resets, stepping continues."""
        env = BatchedBalletEnv(num_envs=2, level_name="1_delay2", symbolic=True)
        env.reset(seed=0)
        terminated_count = 0
        for _ in range(500):
            actions = np.array([env.rng.integers(8) for _ in range(2)])
            _, _, term, trunc, info = env.step(actions)
            if term.any() or trunc.any():
                terminated_count += 1
                done_idx = np.where(term | trunc)[0]
                for i in done_idx:
                    self.assertEqual(env._task_phase[i], 0,
                                     "Reset env should be in dance phase")
            if terminated_count >= 2:
                break
        self.assertGreaterEqual(terminated_count, 2,
                                "Should see at least 2 terminations")
        env.close()

    def test_auto_reset_final_observation(self):
        """info['final_observation'] should contain terminal obs for done envs."""
        env = BatchedBalletEnv(num_envs=4, level_name="1_delay2", symbolic=True)
        env.reset(seed=0)
        for _ in range(500):
            actions = np.array([env.rng.integers(8) for _ in range(4)])
            obs, _, term, trunc, info = env.step(actions)
            done = term | trunc
            if done.any():
                for i in range(4):
                    if done[i]:
                        self.assertIsNotNone(info["final_observation"][i])
                        final_obs = info["final_observation"][i]
                        self.assertEqual(final_obs[0].shape, (121,))
                    else:
                        self.assertIsNone(info["final_observation"][i])
                break
        env.close()

    def test_auto_reset_does_not_affect_other_envs(self):
        """Resetting one env should not change another env's state."""
        from gym_balletenv.envs.batched_ballet_environment import _BASE_BOARD
        env = BatchedBalletEnv(num_envs=2, level_name="1_delay2", symbolic=True)
        env.reset(seed=42)
        for _ in range(500):
            actions = np.array([env.rng.integers(8) for _ in range(2)])
            _, _, term, trunc, _ = env.step(actions)
            done = term | trunc
            if done[0] and not done[1]:
                self.assertFalse(
                    np.array_equal(env._boards[1], _BASE_BOARD),
                    "Env 1 should not have been reset")
                break
            if done[1] and not done[0]:
                self.assertFalse(
                    np.array_equal(env._boards[0], _BASE_BOARD),
                    "Env 0 should not have been reset")
                break
        env.close()


class BatchedBalletEnvEquivalenceTest(absltest.TestCase):
    """Verify BatchedBalletEnv(num_envs=1) matches BalletEnvironment exactly."""

    def _run_equivalence(self, level_name, symbolic, num_steps=100, easy_mode=False):
        """Run both envs with same seed and actions, compare outputs."""
        ref = BalletEnvironment(level_name=level_name, symbolic=symbolic, easy_mode=easy_mode)
        ref_obs, ref_info = ref.reset(seed=42)

        batched = BatchedBalletEnv(
            num_envs=1, level_name=level_name, symbolic=symbolic, easy_mode=easy_mode)
        bat_obs, bat_info = batched.reset(seed=42)

        np.testing.assert_array_almost_equal(
            ref_obs[0], bat_obs[0][0], err_msg="Initial obs mismatch")
        self.assertEqual(ref_obs[1], bat_obs[1][0],
                         "Initial instruction mismatch")

        rng = np.random.default_rng(123)
        for step_i in range(num_steps):
            action = int(rng.integers(8))
            ref_result = ref.step(action)
            ref_obs_s, ref_rew, ref_term, ref_trunc, ref_info_s = ref_result

            bat_result = batched.step(np.array([action]))
            bat_obs_s, bat_rew, bat_term, bat_trunc, bat_info_s = bat_result

            if ref_term or ref_trunc:
                final_obs = bat_info_s["final_observation"][0]
                self.assertIsNotNone(final_obs,
                    f"Step {step_i}: ref done but no final_observation")
                np.testing.assert_array_almost_equal(
                    ref_obs_s[0], final_obs[0],
                    err_msg=f"Step {step_i}: terminal obs mismatch")
                self.assertEqual(ref_obs_s[1], final_obs[1],
                    f"Step {step_i}: terminal instruction mismatch")
                np.testing.assert_almost_equal(
                    ref_rew, bat_rew[0],
                    err_msg=f"Step {step_i}: reward mismatch")
                break

            np.testing.assert_array_almost_equal(
                ref_obs_s[0], bat_obs_s[0][0],
                err_msg=f"Step {step_i}: obs mismatch")
            self.assertEqual(ref_obs_s[1], bat_obs_s[1][0],
                f"Step {step_i}: instruction mismatch")
            np.testing.assert_almost_equal(
                ref_rew, bat_rew[0],
                err_msg=f"Step {step_i}: reward mismatch")
            self.assertEqual(ref_term, bat_term[0],
                f"Step {step_i}: terminated mismatch")
            self.assertEqual(ref_trunc, bat_trunc[0],
                f"Step {step_i}: truncated mismatch")

        ref.close()
        batched.close()

    def test_equivalence_symbolic(self):
        self._run_equivalence("2_delay16", symbolic=True, num_steps=200)

    def test_equivalence_rgb(self):
        self._run_equivalence("2_delay16", symbolic=False, num_steps=200)

    def test_equivalence_easy_bw(self):
        self._run_equivalence("2_delay16", symbolic=False, num_steps=200, easy_mode=True)

    def test_equivalence_1_dancer(self):
        self._run_equivalence("1_delay2", symbolic=True, num_steps=100)

    def test_equivalence_4_dancers(self):
        self._run_equivalence("4_delay16", symbolic=True, num_steps=200)

    def test_equivalence_8_dancers(self):
        self._run_equivalence("8_delay48", symbolic=True, num_steps=300)


import time


class BatchedBalletEnvBenchmarkTest(absltest.TestCase):

    def test_throughput_symbolic(self):
        """Benchmark symbolic mode throughput at N=512."""
        N = 512
        env = BatchedBalletEnv(num_envs=N, level_name="4_delay16", symbolic=True)
        env.reset(seed=42)
        for _ in range(100):
            env.step(np.random.randint(0, 8, size=N))
        steps = 2000
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(np.random.randint(0, 8, size=N))
        dt = time.perf_counter() - t0
        total = steps * N
        throughput = total / dt
        print(f"\nBatched symbolic (N={N}): {throughput:,.0f} steps/s "
              f"({dt/steps*1e3:.2f} ms/batch)")
        self.assertGreater(throughput, 500_000,
                           f"Symbolic throughput {throughput:.0f} < 500K target")
        env.close()

    def test_throughput_rgb(self):
        """Benchmark RGB mode throughput at N=512."""
        N = 512
        env = BatchedBalletEnv(num_envs=N, level_name="4_delay16")
        env.reset(seed=42)
        for _ in range(50):
            env.step(np.random.randint(0, 8, size=N))
        steps = 500
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(np.random.randint(0, 8, size=N))
        dt = time.perf_counter() - t0
        total = steps * N
        throughput = total / dt
        print(f"\nBatched RGB (N={N}): {throughput:,.0f} steps/s "
              f"({dt/steps*1e3:.2f} ms/batch)")
        # RGB rendering is memory-bandwidth limited; target parity with SyncVectorEnv
        self.assertGreater(throughput, 30_000,
                           f"RGB throughput {throughput:.0f} < 30K target")
        env.close()

    def test_throughput_easy_bw(self):
        """Benchmark Easy B&W mode throughput at N=512."""
        N = 512
        env = BatchedBalletEnv(num_envs=N, level_name="4_delay16", easy_mode=True)
        env.reset(seed=42)
        for _ in range(50):
            env.step(np.random.randint(0, 8, size=N))
        steps = 500
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(np.random.randint(0, 8, size=N))
        dt = time.perf_counter() - t0
        total = steps * N
        throughput = total / dt
        print(f"\nBatched Easy B&W (N={N}): {throughput:,.0f} steps/s "
              f"({dt/steps*1e3:.2f} ms/batch)")
        self.assertGreater(throughput, 50_000,
                           f"Easy B&W throughput {throughput:.0f} < 50K target")
        env.close()


if __name__ == "__main__":
    absltest.main()
