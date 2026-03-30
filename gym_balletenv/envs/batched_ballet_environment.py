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

"""Native batched ballet environment for high-throughput RL training.

Processes N environments in single numpy operations, eliminating the
per-env Python loop overhead of SyncVectorEnv.
"""
import numpy as np
from gymnasium.spaces import Tuple, Box, Discrete

from gym_balletenv.envs import ballet_environment_core as ballet_core
from gym_balletenv.envs.ballet_environment import (
    UPSAMPLE_SIZE, BOARD_ROWS, BOARD_COLS, SCROLL_CROP_SIZE,
    SYMBOLIC_OBS_SIZE, MAX_DANCERS, LANG_DICT, COLORS, DANCER_SHAPES,
    _BOARD_CHAR_MAP, _CHAR_TO_TEMPLATE_BASE, _generate_template,
)

_FLOOR_CODE = ord(ballet_core.FLOOR_CHAR)
_WALL_CODE = ord(ballet_core.WALL_CHAR)
_AGENT_CODE = ord(ballet_core.AGENT_CHAR)

# Direction deltas: indexed by action (0-7) -> (dr, dc)
_DIRECTION_DELTAS = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    dtype=np.int16)

# Pre-computed base board (walls + floor)
_BASE_BOARD = np.full(ballet_core.ROOM_SIZE, _FLOOR_CODE, dtype=np.uint8)
_BASE_BOARD[0, :] = _WALL_CODE
_BASE_BOARD[-1, :] = _WALL_CODE
_BASE_BOARD[:, 0] = _WALL_CODE
_BASE_BOARD[:, -1] = _WALL_CODE

# Number of compact LUT entries: floor + wall + agent + 8 max dancers
_NUM_COMPACT_ENTRIES = 1 + 1 + 1 + MAX_DANCERS  # 11

# Pre-convert all dance sequences to delta arrays: {name: (16, 2) int16}
_DANCE_DELTAS = {}
for _name, _seq in ballet_core.DANCE_SEQUENCES.items():
    _DANCE_DELTAS[_name] = np.array(
        [_DIRECTION_DELTAS[d] for d in _seq], dtype=np.int16)
_DANCE_NAMES = list(ballet_core.DANCE_SEQUENCES.keys())

# Motion name -> LANG_DICT index
_MOTION_TO_LANG = {name: LANG_DICT[name] for name in _DANCE_NAMES}


class BatchedBalletEnv:
    """Batched ballet environment: N envs processed in single numpy ops.

    All environments in one batch share the same num_dancers and dance_delay.
    """

    def __init__(self, num_envs, level_name, symbolic=False, seed=None, easy_mode=False):
        self.num_envs = num_envs
        self._symbolic = symbolic
        self._easy_mode = easy_mode

        # Parse level_name: "{num_dancers}_delay{delay}"
        num_dancers, dance_delay_str = level_name.split("_")

        self._num_dancers = int(num_dancers)
        self._dance_delay = int(dance_delay_str[5:])  # "delay16" -> 16
        assert 1 <= self._num_dancers <= 8

        # Max steps
        self._max_steps = 320 if self._dance_delay <= 16 else 1024

        # Observation / action spaces (single-env, not batched)
        if self._symbolic:
            obs_box = Box(low=0.0, high=1.0, shape=(SYMBOLIC_OBS_SIZE,),
                          dtype=np.float32)
        else:
            channels = 1 if self._easy_mode else 3
            self._channels = channels
            img_size = (SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                        SCROLL_CROP_SIZE * UPSAMPLE_SIZE, channels)
            obs_box = Box(low=0, high=255, shape=img_size, dtype=np.uint8)
        self.observation_space = Tuple((obs_box, Discrete(14)))
        self.action_space = Discrete(8)

        # RNG
        self.rng = np.random.default_rng(seed)

        # State arrays will be allocated in reset()
        self._initialized = False

    def close(self):
        pass

    def _allocate_state(self):
        """Allocate all state arrays. Called once on first reset."""
        N = self.num_envs
        D = self._num_dancers

        # Board state
        self._boards = np.zeros((N, 11, 11), dtype=np.uint8)

        # Agent positions
        self._agent_rows = np.zeros(N, dtype=np.int16)
        self._agent_cols = np.zeros(N, dtype=np.int16)

        # Dancer positions and properties
        self._dancer_rows = np.zeros((N, D), dtype=np.int16)
        self._dancer_cols = np.zeros((N, D), dtype=np.int16)
        self._dancer_codes = np.zeros((N, D), dtype=np.uint8)
        self._dancer_values = np.zeros((N, D), dtype=np.float32)

        # Dance sequences: pre-computed deltas
        self._dance_deltas = np.zeros((N, D, 16, 2), dtype=np.int16)

        # Phase management
        self._task_phase = np.zeros(N, dtype=np.uint8)
        self._current_dancing_idx = np.full(N, -1, dtype=np.int16)
        self._dance_step_idx = np.zeros(N, dtype=np.int16)
        self._time_until_next = np.ones(N, dtype=np.int16)
        self._dance_order_idx = np.zeros(N, dtype=np.int16)
        self._frames = np.zeros(N, dtype=np.int32)
        self._game_over = np.zeros(N, dtype=bool)

        # Instruction (LANG_DICT index)
        self._instructions = np.zeros(N, dtype=np.int64)
        self._choice_instructions = np.zeros(N, dtype=np.int64)

        # Observation LUT (non-symbolic modes)
        if not self._symbolic:
            C = self._channels
            self._compact_lut = np.zeros(
                (N, _NUM_COMPACT_ENTRIES, UPSAMPLE_SIZE, UPSAMPLE_SIZE, C),
                dtype=np.uint8)
            self._char_map = np.zeros((N, 256), dtype=np.uint8)

            # Pre-fill floor/wall/agent templates (shared across all envs)
            # Compact index: 0=floor (zeros), 1=wall, 2=agent
            if self._easy_mode:
                wall_tmpl = _CHAR_TO_TEMPLATE_BASE[ballet_core.WALL_CHAR][:, :, :1].astype(np.uint8)
                agent_tmpl = _CHAR_TO_TEMPLATE_BASE[ballet_core.AGENT_CHAR][:, :, :1].astype(np.uint8)
            else:
                wall_tmpl = _CHAR_TO_TEMPLATE_BASE[ballet_core.WALL_CHAR].astype(np.uint8)
                agent_tmpl = _CHAR_TO_TEMPLATE_BASE[ballet_core.AGENT_CHAR].astype(np.uint8)
            # Index 0 = floor (already zeros)
            self._compact_lut[:, 1] = wall_tmpl   # broadcast
            self._compact_lut[:, 2] = agent_tmpl   # broadcast
            # char_map for floor/wall/agent (same for all envs)
            self._char_map[:, _FLOOR_CODE] = 0
            self._char_map[:, _WALL_CODE] = 1
            self._char_map[:, _AGENT_CODE] = 2

        self._initialized = True

    def reset(self, seed=None):
        """Reset all environments. Returns (obs, info)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if not self._initialized:
            self._allocate_state()
        self._reset_envs(np.arange(self.num_envs))
        return self._get_obs(), self._get_info()

    def _reset_envs(self, idx):
        """Reset environments at given indices."""
        M = len(idx)
        D = self._num_dancers

        # Reset boards to base
        self._boards[idx] = _BASE_BOARD

        # Place agent at start
        ar, ac = ballet_core.AGENT_START
        self._agent_rows[idx] = ar
        self._agent_cols[idx] = ac
        self._boards[idx, ar, ac] = _AGENT_CODE

        # Sample dancer properties per env
        all_motions = list(ballet_core.DANCE_SEQUENCES.keys())
        all_positions = list(ballet_core.DANCER_POSITIONS)
        all_colors = list(COLORS.keys())
        all_shapes = list(DANCER_SHAPES)

        for i_local, i_env in enumerate(idx):
            target_idx = int(self.rng.integers(D))
            motions = all_motions.copy()
            positions = all_positions.copy()
            colors = all_colors.copy()
            shapes = all_shapes.copy()
            self.rng.shuffle(positions)
            self.rng.shuffle(motions)
            self.rng.shuffle(colors)
            self.rng.shuffle(shapes)

            choice_instruction = "watch"
            for d in range(D):
                char = ballet_core.POSSIBLE_DANCER_CHARS[d]
                code = ord(char)
                r, c = positions[d]
                motion = motions[d]
                value = 1.0 if d == target_idx else 0.0

                self._dancer_rows[i_env, d] = r
                self._dancer_cols[i_env, d] = c
                self._dancer_codes[i_env, d] = code
                self._dancer_values[i_env, d] = value
                self._dance_deltas[i_env, d] = _DANCE_DELTAS[motion]
                self._boards[i_env, r, c] = code

                if value > 0.0:
                    choice_instruction = motion

                # Update compact LUT for this dancer (non-symbolic)
                if not self._symbolic:
                    compact_idx = 3 + d  # 0=floor, 1=wall, 2=agent, 3+=dancers
                    if self._easy_mode:
                        color_shape = "red triangle"
                    else:
                        color_shape = colors[d] + " " + shapes[d]
                    template = _generate_template(color_shape,
                                                  easy_mode=self._easy_mode)
                    self._compact_lut[i_env, compact_idx] = template.astype(np.uint8)
                    self._char_map[i_env, code] = compact_idx

            self._choice_instructions[i_env] = _MOTION_TO_LANG[choice_instruction]

        # Reset phase state
        self._task_phase[idx] = 0
        self._current_dancing_idx[idx] = -1
        self._dance_step_idx[idx] = 0
        self._time_until_next[idx] = 1
        self._dance_order_idx[idx] = 0
        self._frames[idx] = 0
        self._game_over[idx] = False
        self._instructions[idx] = LANG_DICT["watch"]

        # Execute first game step (PyColab its_showtime compatibility).
        # This matches BalletGame.play(0) which runs both Phase 1 (scheduling)
        # and Phase 2 (dancer movement) on the first frame.
        self._frames[idx] += 1
        self._step_dance_scheduling(idx)
        # After scheduling, move dancer if one was started
        dance_active = idx[(self._task_phase[idx] == 0) &
                           (self._current_dancing_idx[idx] >= 0)]
        if len(dance_active) > 0:
            self._step_dancer_movement(dance_active)

    def _get_obs(self):
        """Build batched observation tuple."""
        if self._symbolic:
            obs = _BOARD_CHAR_MAP[self._boards].reshape(self.num_envs, -1)
            return (obs, self._instructions.copy())
        else:
            N = self.num_envs
            # compact_indices[n, r, c] = compact LUT index for that cell
            n_idx = np.arange(N)[:, None, None]  # (N, 1, 1)
            compact_indices = self._char_map[n_idx, self._boards]  # (N, 11, 11)
            # tiles[n, r, c] = template pixel block for that cell: (9, 9, C)
            # Use env-indexed LUT: _compact_lut[n, k] = (9, 9, C)
            # Flatten spatial dims to index efficiently
            flat_idx = compact_indices.reshape(N, -1)  # (N, 121)
            n_flat = np.arange(N)[:, None]  # (N, 1)
            tiles_flat = self._compact_lut[n_flat, flat_idx]  # (N, 121, 9, 9, C)
            tiles = tiles_flat.reshape(N, BOARD_ROWS, BOARD_COLS,
                                       UPSAMPLE_SIZE, UPSAMPLE_SIZE,
                                       self._channels)  # (N, 11, 11, 9, 9, C)
            images = tiles.transpose(0, 1, 3, 2, 4, 5).reshape(
                N, BOARD_ROWS * UPSAMPLE_SIZE, BOARD_COLS * UPSAMPLE_SIZE,
                self._channels)
            return (np.ascontiguousarray(images), self._instructions.copy())

    def _get_info(self):
        """Build batched info dict."""
        return {
            "instruction_string": self._instructions.copy(),
            "num_dancers": np.full(self.num_envs, self._num_dancers, dtype=np.int32),
            "dance_delay": np.full(self.num_envs, self._dance_delay, dtype=np.int32),
        }

    def step(self, actions):
        """Step all N environments. actions: (N,) int array.

        Returns: (obs, reward, terminated, truncated, info)
        """
        actions = np.asarray(actions, dtype=np.int64)
        N = self.num_envs
        reward = np.zeros(N, dtype=np.float32)

        self._frames += 1

        # --- Phase 1: Agent update ---
        # Snapshot masks before any phase changes in this step
        dance_mask = (self._task_phase == 0)
        choice_mask = (self._task_phase == 1)

        # Dance phase: handle countdown and dancer scheduling
        dance_idx = np.where(dance_mask)[0]
        if len(dance_idx) > 0:
            self._step_dance_scheduling(dance_idx)

        # Choice phase: move agent
        choice_idx = np.where(choice_mask)[0]
        if len(choice_idx) > 0:
            self._step_agent_movement(choice_idx, actions)

        # --- Phase 2: Dancer/collision update ---
        # Dance phase: move current dancer (use original dance_mask snapshot)
        dance_active = np.where(
            dance_mask & (self._current_dancing_idx >= 0))[0]
        if len(dance_active) > 0:
            self._step_dancer_movement(dance_active)

        # Choice phase: check agent-dancer collision (use updated phase)
        choice_idx2 = np.where(self._task_phase == 1)[0]
        if len(choice_idx2) > 0:
            self._step_collision_check(choice_idx2, reward)

        # Termination
        terminated = self._game_over.copy()
        truncated = (self._frames >= self._max_steps) & ~terminated

        # Build obs and info before auto-reset
        obs = self._get_obs()
        info = self._get_info()

        # Auto-reset
        done = terminated | truncated
        info["final_observation"] = [None] * N
        info["final_info"] = [None] * N
        if done.any():
            done_idx = np.where(done)[0]
            for i in done_idx:
                info["final_observation"][i] = (obs[0][i].copy(), obs[1][i].copy())
                info["final_info"][i] = {
                    "instruction_string": self._instructions[i].copy(),
                    "num_dancers": self._num_dancers,
                    "dance_delay": self._dance_delay,
                }
            self._reset_envs(done_idx)
            # Overwrite obs for reset envs
            new_obs = self._get_obs()
            obs[0][done_idx] = new_obs[0][done_idx]
            obs[1][done_idx] = new_obs[1][done_idx]

        return obs, reward, terminated, truncated, info

    def _step_dance_scheduling(self, idx):
        """Handle dance phase scheduling: countdown, start next dancer, phase transition."""
        # Decrement countdown
        countdown_active = idx[self._time_until_next[idx] > 0]
        self._time_until_next[countdown_active] -= 1

        # Where countdown just reached 0 (only from those that were decremented)
        just_zero = countdown_active[self._time_until_next[countdown_active] == 0]
        if len(just_zero) == 0:
            return

        # Snapshot dance_order_idx BEFORE any modifications to correctly split:
        # - has_more: envs that still have a dancer to start
        # - no_more: envs that have exhausted all dancers -> choice phase
        order_snapshot = self._dance_order_idx[just_zero]
        has_more = just_zero[order_snapshot < self._num_dancers]
        no_more = just_zero[order_snapshot >= self._num_dancers]

        # Start next dancer (if more dancers remain)
        if len(has_more) > 0:
            self._current_dancing_idx[has_more] = self._dance_order_idx[has_more]
            self._dance_order_idx[has_more] += 1
            self._dance_step_idx[has_more] = 0

        # Transition to choice phase (if no more dancers)
        if len(no_more) > 0:
            self._task_phase[no_more] = 1
            self._instructions[no_more] = self._choice_instructions[no_more]

    def _step_dancer_movement(self, idx):
        """Move the currently dancing dancer for envs at idx."""
        cidx = self._current_dancing_idx[idx]
        sidx = self._dance_step_idx[idx]

        # Only move if within sequence
        active = idx[sidx < 16]
        if len(active) == 0:
            return
        cidx_a = self._current_dancing_idx[active]
        sidx_a = self._dance_step_idx[active]

        # Get deltas
        dr = self._dance_deltas[active, cidx_a, sidx_a, 0]
        dc = self._dance_deltas[active, cidx_a, sidx_a, 1]

        # Old position
        old_r = self._dancer_rows[active, cidx_a]
        old_c = self._dancer_cols[active, cidx_a]

        # New position with wall collision
        new_r = old_r + dr
        new_c = old_c + dc
        wall_hit = (self._boards[active, new_r, new_c] == _WALL_CODE)
        final_r = np.where(wall_hit, old_r, new_r)
        final_c = np.where(wall_hit, old_c, new_c)

        # Update board: clear old, stamp new
        codes = self._dancer_codes[active, cidx_a]
        self._boards[active, old_r, old_c] = _FLOOR_CODE
        self._dancer_rows[active, cidx_a] = final_r
        self._dancer_cols[active, cidx_a] = final_c
        self._boards[active, final_r, final_c] = codes

        # Advance step counter
        self._dance_step_idx[active] += 1

        # Handle sequence completion
        seq_done = active[self._dance_step_idx[active] >= 16]
        if len(seq_done) > 0:
            self._current_dancing_idx[seq_done] = -1
            self._time_until_next[seq_done] = self._dance_delay

    def _step_agent_movement(self, idx, actions):
        """Move agents for envs in choice phase."""
        deltas = _DIRECTION_DELTAS[actions[idx]]  # (M, 2)
        new_r = self._agent_rows[idx] + deltas[:, 0]
        new_c = self._agent_cols[idx] + deltas[:, 1]
        blocked = (self._boards[idx, new_r, new_c] == _WALL_CODE)

        # Clear old position
        self._boards[idx, self._agent_rows[idx], self._agent_cols[idx]] = _FLOOR_CODE

        # Update position
        self._agent_rows[idx] = np.where(blocked, self._agent_rows[idx], new_r)
        self._agent_cols[idx] = np.where(blocked, self._agent_cols[idx], new_c)

        # Stamp new position
        self._boards[idx, self._agent_rows[idx], self._agent_cols[idx]] = _AGENT_CODE

    def _step_collision_check(self, idx, reward):
        """Check agent-dancer collisions for envs in choice phase."""
        # Compare agent position with all dancer positions
        match_r = (self._agent_rows[idx, None] == self._dancer_rows[idx])  # (M, D)
        match_c = (self._agent_cols[idx, None] == self._dancer_cols[idx])  # (M, D)
        reached = match_r & match_c  # (M, D)
        reward[idx] = (reached * self._dancer_values[idx]).sum(axis=1)
        self._game_over[idx] = reached.any(axis=1)
