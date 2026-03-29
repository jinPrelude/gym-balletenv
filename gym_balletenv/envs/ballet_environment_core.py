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

"""Pure NumPy game engine for the ballet environment.

Replaces the PyColab-based engine with direct numpy array manipulation
for higher throughput. The game logic is identical: dancers perform
choreography sequences on an 11x11 grid, then the agent navigates to
the target dancer.
"""
import enum
import numpy as np

ROOM_SIZE = (11, 11)  # one square around edge will be wall.
DANCER_POSITIONS = [(2, 2), (2, 5), (2, 8),
                    (5, 2), (5, 8),  # space in center for agent
                    (8, 2), (8, 5), (8, 8)]
AGENT_START = (5, 5)
AGENT_CHAR = "A"
WALL_CHAR = "#"
FLOOR_CHAR = " "
RESERVED_CHARS = [AGENT_CHAR, WALL_CHAR, FLOOR_CHAR]
POSSIBLE_DANCER_CHARS = [
    chr(i) for i in range(65, 91) if chr(i) not in RESERVED_CHARS
]

DANCE_SEQUENCE_LENGTHS = 16


class DIRECTIONS(enum.IntEnum):
  N = 0
  NE = 1
  E = 2
  SE = 3
  S = 4
  SW = 5
  W = 6
  NW = 7


DANCE_SEQUENCES = {
    "circle_cw": [
        DIRECTIONS.N, DIRECTIONS.E, DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.W,
        DIRECTIONS.W, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.E, DIRECTIONS.E,
        DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.N,
        DIRECTIONS.E
    ],
    "circle_ccw": [
        DIRECTIONS.N, DIRECTIONS.W, DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.E,
        DIRECTIONS.E, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.W, DIRECTIONS.W,
        DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.N,
        DIRECTIONS.W
    ],
    "up_and_down": [
        DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.N,
        DIRECTIONS.S, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.S,
        DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.S,
        DIRECTIONS.N
    ],
    "left_and_right": [
        DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E,
        DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.W,
        DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.W,
        DIRECTIONS.E
    ],
    "diagonal_uldr": [
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SE, DIRECTIONS.NW
    ],
    "diagonal_urdl": [
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW, DIRECTIONS.NE
    ],
    "plus_cw": [
        DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.S,
        DIRECTIONS.N, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.N, DIRECTIONS.S,
        DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.W,
        DIRECTIONS.E
    ],
    "plus_ccw": [
        DIRECTIONS.N, DIRECTIONS.S, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.S,
        DIRECTIONS.N, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.N, DIRECTIONS.S,
        DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.S, DIRECTIONS.N, DIRECTIONS.E,
        DIRECTIONS.W
    ],
    "times_cw": [
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.NW, DIRECTIONS.SE,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SE, DIRECTIONS.NW,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.NW, DIRECTIONS.SE
    ],
    "times_ccw": [
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.SE, DIRECTIONS.NW, DIRECTIONS.NE, DIRECTIONS.SW,
        DIRECTIONS.NW, DIRECTIONS.SE, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.SE, DIRECTIONS.NW, DIRECTIONS.NE, DIRECTIONS.SW
    ],
    "zee": [
        DIRECTIONS.NE, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.E, DIRECTIONS.E,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.SW,
        DIRECTIONS.E, DIRECTIONS.E, DIRECTIONS.W, DIRECTIONS.W, DIRECTIONS.NE,
        DIRECTIONS.SW, DIRECTIONS.NE
    ],
    "chevron_down": [
        DIRECTIONS.NW, DIRECTIONS.S, DIRECTIONS.SE, DIRECTIONS.NE, DIRECTIONS.N,
        DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.NE,
        DIRECTIONS.S, DIRECTIONS.SW, DIRECTIONS.NW, DIRECTIONS.N, DIRECTIONS.SE,
        DIRECTIONS.NW, DIRECTIONS.SE
    ],
    "chevron_up": [
        DIRECTIONS.SE, DIRECTIONS.N, DIRECTIONS.NW, DIRECTIONS.SW, DIRECTIONS.S,
        DIRECTIONS.NE, DIRECTIONS.SW, DIRECTIONS.NE, DIRECTIONS.SW,
        DIRECTIONS.N, DIRECTIONS.NE, DIRECTIONS.SE, DIRECTIONS.S, DIRECTIONS.NW,
        DIRECTIONS.SE, DIRECTIONS.NW
    ],
}

# Direction deltas: indexed by DIRECTIONS enum value -> (row_delta, col_delta)
_DIRECTION_DELTAS = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    dtype=np.int8)

# Pre-convert dance sequences to delta arrays for fast indexing
_DANCE_DELTAS = {}
for _name, _seq in DANCE_SEQUENCES.items():
    _DANCE_DELTAS[_name] = _DIRECTION_DELTAS[np.array(_seq, dtype=np.int8)]

# ASCII codes for fast board operations
_FLOOR_CODE = ord(FLOOR_CHAR)
_WALL_CODE = ord(WALL_CHAR)
_AGENT_CODE = ord(AGENT_CHAR)

# Pre-computed base board (walls + floor)
_BASE_BOARD = np.full(ROOM_SIZE, _FLOOR_CODE, dtype=np.uint8)
_BASE_BOARD[0, :] = _WALL_CODE
_BASE_BOARD[-1, :] = _WALL_CODE
_BASE_BOARD[:, 0] = _WALL_CODE
_BASE_BOARD[:, -1] = _WALL_CODE


class BalletGame:
    """Pure numpy game engine for the ballet environment.

    Manages an 11x11 board, entity positions, dance phases, and collision.
    Designed for maximum throughput: no object creation per step, no string
    comparisons in the hot path, array index counters instead of list pops.
    """

    def __init__(self, dancers_and_properties, dance_delay=16):
        """Initialize the game.

        Args:
            dancers_and_properties: list of (character, (row, col), motion,
                shape, color, value) tuples.
            dance_delay: steps to wait between dances.
        """
        self.num_dancers = len(dancers_and_properties)
        self.dance_delay = dance_delay
        self.game_over = False
        self.frame = 0

        # Board state
        self.board = _BASE_BOARD.copy()

        # Agent
        self.agent_row, self.agent_col = AGENT_START
        self.board[self.agent_row, self.agent_col] = _AGENT_CODE

        # Dancers -- store properties in parallel arrays for speed
        self.dancer_chars = []      # character strings (for external reference)
        self.dancer_codes = []      # ASCII codes (for board ops)
        self.dancer_rows = []
        self.dancer_cols = []
        self.dancer_motions = []    # motion name strings
        self.dancer_deltas = []     # pre-computed (16, 2) delta arrays
        self.dancer_values = []     # reward values

        # Metadata for external use (templates, instructions)
        self.char_to_color_shape = []
        choice_instruction_string = "watch"

        # Dance scheduling
        self.dance_order = []       # list of dancer indices to perform

        for i, (char, pos, motion, shape, color, value) in enumerate(
                dancers_and_properties):
            self.dancer_chars.append(char)
            self.dancer_codes.append(ord(char))
            self.dancer_rows.append(pos[0])
            self.dancer_cols.append(pos[1])
            self.dancer_motions.append(motion)
            self.dancer_deltas.append(_DANCE_DELTAS[motion])
            self.dancer_values.append(value)
            self.char_to_color_shape.append((char, color + " " + shape))
            self.dance_order.append(i)
            if value > 0.:
                choice_instruction_string = motion

            # Place dancer on board
            self.board[pos[0], pos[1]] = ord(char)

        # Phase state machine
        # task_phase: 0 = dance, 1 = choice
        self.task_phase = 0
        self.instruction_string = "watch"
        self.choice_instruction_string = choice_instruction_string
        self.time_until_next_dance = 1  # first dancer starts after 1 step
        self.current_dancing_idx = -1   # no dancer currently dancing
        self.dance_step_idx = 0         # step within current dance
        self.dance_order_idx = 0        # next dancer in dance_order to start

        # Reward accumulator for current step
        self._reward = 0.0

    def play(self, action):
        """Execute one game step. Returns (reward,).

        Update order matches PyColab's schedule:
        1. Agent (PlayerSprite) update -- handles phase transitions + agent movement
        2. Dancer updates -- handle dance moves + collision with agent

        Args:
            action: int, 0-7 corresponding to DIRECTIONS enum.

        Returns:
            reward: float, 0.0 or dancer's value if agent reached it.
        """
        self._reward = 0.0
        self.frame += 1

        # --- Phase 1: Agent update (matches PlayerSprite.update) ---
        if self.task_phase == 0:  # dance phase
            # Agent cannot move during dance. Handle dance scheduling.
            if self.time_until_next_dance > 0:
                self.time_until_next_dance -= 1
                if self.time_until_next_dance == 0:
                    if self.dance_order_idx < len(self.dance_order):
                        # Start next dancer
                        self.current_dancing_idx = self.dance_order[
                            self.dance_order_idx]
                        self.dance_order_idx += 1
                        self.dance_step_idx = 0
                    else:
                        # All dances done -- enter choice phase
                        self.task_phase = 1
                        self.instruction_string = (
                            self.choice_instruction_string)
        elif self.task_phase == 1:  # choice phase
            # Agent can move
            self._move_agent(action)

        # --- Phase 2: Dancer updates (matches DancerSprite.update) ---
        if self.task_phase == 0 and self.current_dancing_idx >= 0:
            di = self.current_dancing_idx
            if self.dance_step_idx < DANCE_SEQUENCE_LENGTHS:
                # Move the dancing dancer
                self._move_dancer(di, self.dance_step_idx)
                self.dance_step_idx += 1
                if self.dance_step_idx >= DANCE_SEQUENCE_LENGTHS:
                    # Dance finished
                    self.current_dancing_idx = -1
                    self.time_until_next_dance = self.dance_delay
        elif self.task_phase == 1:
            # In choice phase, check if agent reached any dancer
            for di in range(self.num_dancers):
                if (self.agent_row == self.dancer_rows[di] and
                        self.agent_col == self.dancer_cols[di]):
                    self._reward = self.dancer_values[di]
                    self.game_over = True
                    break

        return self._reward

    def _move_agent(self, action):
        """Move agent in the given direction if destination is not a wall."""
        dr, dc = _DIRECTION_DELTAS[action]
        new_row = self.agent_row + dr
        new_col = self.agent_col + dc
        if self.board[new_row, new_col] != _WALL_CODE:
            # Clear old position, set new
            self.board[self.agent_row, self.agent_col] = _FLOOR_CODE
            self.agent_row = new_row
            self.agent_col = new_col
            self.board[self.agent_row, self.agent_col] = _AGENT_CODE

    def _move_dancer(self, dancer_idx, step_idx):
        """Move a dancer according to its dance sequence."""
        deltas = self.dancer_deltas[dancer_idx]
        dr = int(deltas[step_idx, 0])
        dc = int(deltas[step_idx, 1])
        old_row = self.dancer_rows[dancer_idx]
        old_col = self.dancer_cols[dancer_idx]
        new_row = old_row + dr
        new_col = old_col + dc
        code = self.dancer_codes[dancer_idx]
        if self.board[new_row, new_col] != _WALL_CODE:
            self.board[old_row, old_col] = _FLOOR_CODE
            self.dancer_rows[dancer_idx] = new_row
            self.dancer_cols[dancer_idx] = new_col
            self.board[new_row, new_col] = code


def make_game(dancers_and_properties, dance_delay=16):
    """Create a BalletGame (drop-in for the old pycolab make_game).

    Args:
        dancers_and_properties: list of (character, (row, col), motion,
            shape, color, value) tuples.
        dance_delay: steps between dances.

    Returns:
        BalletGame instance.
    """
    return BalletGame(dancers_and_properties, dance_delay)
