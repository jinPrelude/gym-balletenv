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

"""Gymnasium wrapper for the ballet environment.

Wraps the pure-numpy BalletGame engine with Gymnasium API, observation
rendering, and template management. Rendering uses vectorized numpy
operations for maximum throughput.
"""
from absl import logging

import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete

import numpy as np
import cv2

from gym_balletenv.envs import ballet_environment_core as ballet_core

UPSAMPLE_SIZE = 9  # pixels per game square
SCROLL_CROP_SIZE = 11  # in game squares
MAX_DANCERS = 8  # maximum number of dancers (len of DANCER_POSITIONS)
BOARD_ROWS, BOARD_COLS = ballet_core.ROOM_SIZE  # 11 x 11
SYMBOLIC_OBS_SIZE = BOARD_ROWS * BOARD_COLS  # 121: flattened board

# Categorical encoding for board characters -> normalized [0, 1].
# floor=0, wall=1, agent=2, dancer_0=3, ..., dancer_7=10
_NUM_CATEGORIES = 2 + 1 + MAX_DANCERS  # 11 (floor, wall, agent, 8 dancers)
_BOARD_CHAR_MAP = np.zeros(256, dtype=np.float32)  # ASCII lookup table
_BOARD_CHAR_MAP[ord(ballet_core.FLOOR_CHAR)] = 0.0 / (_NUM_CATEGORIES - 1)
_BOARD_CHAR_MAP[ord(ballet_core.WALL_CHAR)] = 1.0 / (_NUM_CATEGORIES - 1)
_BOARD_CHAR_MAP[ord(ballet_core.AGENT_CHAR)] = 2.0 / (_NUM_CATEGORIES - 1)
for _i, _ch in enumerate(ballet_core.POSSIBLE_DANCER_CHARS[:MAX_DANCERS]):
  _BOARD_CHAR_MAP[ord(_ch)] = (3.0 + _i) / (_NUM_CATEGORIES - 1)

DANCER_SHAPES = [
    "triangle", "empty_square", "plus", "inverse_plus", "ex", "inverse_ex",
    "circle", "empty_circle", "tee", "upside_down_tee",
    "h", "u", "upside_down_u", "vertical_stripes", "horizontal_stripes"
]

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([128, 0, 128]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "brown": np.array([128, 64, 0]),
    "pink": np.array([255, 64, 255]),
    "cyan": np.array([0, 255, 255]),
    "dark_green": np.array([0, 100, 0]),
    "dark_red": np.array([100, 0, 0]),
    "dark_blue": np.array([0, 0, 100]),
    "olive": np.array([100, 100, 0]),
    "teal": np.array([0, 100, 100]),
    "lavender": np.array([215, 200, 255]),
    "peach": np.array([255, 210, 170]),
    "rose": np.array([255, 205, 230]),
    "light_green": np.array([200, 255, 200]),
    "light_yellow": np.array([255, 255, 200]),
}


def _generate_template(object_name, easy_mode=False):
  """Generates a template object image, given a name with color and shape."""
  object_color, object_type = object_name.split()
  template = np.zeros((UPSAMPLE_SIZE, UPSAMPLE_SIZE))
  half = UPSAMPLE_SIZE // 2
  if object_type == "triangle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (j <= half and i >= 2 * (half - j)) or (j > half and i >= 2 *
                                                   (j - half)):
          template[i, j] = 1.
  elif object_type == "square":
    template[:, :] = 1.
  elif object_type == "empty_square":
    template[:2, :] = 1.
    template[-2:, :] = 1.
    template[:, :2] = 1.
    template[:, -2:] = 1.
  elif object_type == "plus":
    template[:, half - 1:half + 2] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "inverse_plus":
    template[:, :] = 1.
    template[:, half - 1:half + 2] = 0.
    template[half - 1:half + 2, :] = 0.
  elif object_type == "ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1:
          template[i, j] = 1.
  elif object_type == "inverse_ex":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if not (abs(i - j) <= 1 or abs(UPSAMPLE_SIZE - 1 - j - i) <= 1):
          template[i, j] = 1.
  elif object_type == "circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if (i - half)**2 + (j - half)**2 <= half**2:
          template[i, j] = 1.
  elif object_type == "empty_circle":
    for i in range(UPSAMPLE_SIZE):
      for j in range(UPSAMPLE_SIZE):
        if abs((i - half)**2 + (j - half)**2 - half**2) < 6:
          template[i, j] = 1.
  elif object_type == "tee":
    template[:, half - 1:half + 2] = 1.
    template[:3, :] = 1.
  elif object_type == "upside_down_tee":
    template[:, half - 1:half + 2] = 1.
    template[-3:, :] = 1.
  elif object_type == "h":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[half - 1:half + 2, :] = 1.
  elif object_type == "u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[-3:, :] = 1.
  elif object_type == "upside_down_u":
    template[:, :3] = 1.
    template[:, -3:] = 1.
    template[:3, :] = 1.
  elif object_type == "vertical_stripes":
    for j in range(half + UPSAMPLE_SIZE % 2):
      template[:, 2*j] = 1.
  elif object_type == "horizontal_stripes":
    for i in range(half + UPSAMPLE_SIZE % 2):
      template[2*i, :] = 1.
  else:
    raise ValueError("Unknown object: {}".format(object_type))

  if easy_mode:
    return (template * 255).astype(np.uint8).reshape(
        UPSAMPLE_SIZE, UPSAMPLE_SIZE, 1)

  if object_color not in COLORS:
    raise ValueError("Unknown color: {}".format(object_color))

  template = np.tensordot(template, COLORS[object_color], axes=0)

  return template


# Agent and wall templates (RGB, used as base -- sliced for easy mode)
_CHAR_TO_TEMPLATE_BASE = {
    ballet_core.AGENT_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([255, 255, 255]),
            axes=0),
    ballet_core.WALL_CHAR:
        np.tensordot(
            np.ones([UPSAMPLE_SIZE, UPSAMPLE_SIZE]),
            np.array([40, 40, 40]),
            axes=0),
}

LANG_DICT = {
  "watch": 0,
  "circle_cw": 1,
  "circle_ccw": 2,
  "up_and_down": 3,
  "left_and_right": 4,
  "diagonal_uldr": 5,
  "diagonal_urdl": 6,
  "plus_cw": 7,
  "plus_ccw": 8,
  "times_cw": 9,
  "times_ccw": 10,
  "zee": 11,
  "chevron_down": 12,
  "chevron_up": 13
}


class BalletEnvironment(gym.Env):
  """A Gymnasium environment for ballet tasks."""
  metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

  def __init__(self, level_name, max_steps=None, render_mode="rgb_array",
               num_dancers_range=None, dance_delay_range=None, symbolic=False):
    super(BalletEnvironment, self).__init__()

    name_infos = level_name.split("_")
    if len(name_infos) == 3:
      num_dancers, dance_delay, level = name_infos
      assert level == "easy"
      easy_mode = True
    elif len(name_infos) == 2:
      num_dancers, dance_delay = name_infos
      easy_mode = False

    num_dancers = int(num_dancers)
    dance_delay = int(dance_delay[5:])

    assert num_dancers in range(1, 9)
    self._num_dancers = num_dancers
    self._dance_delay = dance_delay
    if max_steps is None:
      worst_delay = max(dance_delay_range) if dance_delay_range else dance_delay
      max_steps = 320 if worst_delay <= 16 else 1024
    self._max_steps = max_steps
    self._easy_mode = easy_mode
    self._symbolic = symbolic

    # Store sampling ranges (None = fixed from level_name)
    self._num_dancers_range = num_dancers_range
    self._dance_delay_range = dance_delay_range

    if self._symbolic:
      obs_box = Box(low=0.0, high=1.0, shape=(SYMBOLIC_OBS_SIZE,),
                    dtype=np.float32)
    else:
      channels = 1 if self._easy_mode else 3
      img_size = (SCROLL_CROP_SIZE * UPSAMPLE_SIZE,
                  SCROLL_CROP_SIZE * UPSAMPLE_SIZE, channels)
      obs_box = Box(low=0, high=255, shape=img_size, dtype=np.uint8)
    self.observation_space = Tuple((obs_box, Discrete(14)))
    self.action_space = Discrete(8)

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    self.curr_img_obs = None
    self.curr_lang_obs = None

    # Internal state
    self._current_game = None
    self._done = False
    self._game_over = None
    self._char_to_template = None

    # Template lookup table for vectorized rendering (non-symbolic only)
    # Shape: (256, 9, 9, C) indexed by ASCII code
    if not self._symbolic:
      self._channels = 1 if self._easy_mode else 3
      self._template_lut = np.zeros(
          (256, UPSAMPLE_SIZE, UPSAMPLE_SIZE, self._channels),
          dtype=np.uint8)
      # Pre-fill agent and wall entries (constant across episodes)
      if self._easy_mode:
        self._template_lut[ord(ballet_core.AGENT_CHAR)] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.AGENT_CHAR][:, :, :1]
            .astype(np.uint8))
        self._template_lut[ord(ballet_core.WALL_CHAR)] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.WALL_CHAR][:, :, :1]
            .astype(np.uint8))
      else:
        self._template_lut[ord(ballet_core.AGENT_CHAR)] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.AGENT_CHAR].astype(np.uint8))
        self._template_lut[ord(ballet_core.WALL_CHAR)] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.WALL_CHAR].astype(np.uint8))

  def _game_factory(self):
    """Samples dancers and positions, returns a BalletGame."""
    target_dancer_index = self.np_random.integers(self._num_dancers)
    motions = list(ballet_core.DANCE_SEQUENCES.keys())
    if self._easy_mode:
      positions = ballet_core.DANCER_POSITIONS.copy()
      colors = ["red"] * self._num_dancers
      shapes = ["triangle"] * self._num_dancers
    else:
      positions = ballet_core.DANCER_POSITIONS.copy()
      colors = list(COLORS.keys())
      shapes = DANCER_SHAPES.copy()
    self.np_random.shuffle(positions)
    self.np_random.shuffle(motions)
    self.np_random.shuffle(colors)
    self.np_random.shuffle(shapes)
    dancers_and_properties = []
    for dancer_i in range(self._num_dancers):
      if dancer_i == target_dancer_index:
        value = 1.
      else:
        value = 0.
      dancers_and_properties.append(
          (ballet_core.POSSIBLE_DANCER_CHARS[dancer_i],
           positions[dancer_i],
           motions[dancer_i],
           shapes[dancer_i],
           colors[dancer_i],
           value))

    logging.info("Making level with dancers_and_properties: %s",
                 dancers_and_properties)

    return ballet_core.make_game(
        dancers_and_properties=dancers_and_properties,
        dance_delay=self._dance_delay)

  def _get_symbolic_obs(self):
    """Flattens the raw board into a categorical float vector."""
    obs = _BOARD_CHAR_MAP[self._current_game.board].flatten()
    instruct_str = self._current_game.instruction_string
    self.curr_lang_obs = instruct_str
    return obs, instruct_str

  def _get_obs(self):
    """Renders board to image using vectorized template lookup."""
    board = self._current_game.board
    # Vectorized blit: lookup all templates at once, reshape to image
    tiles = self._template_lut[board]  # (11, 11, 9, 9, C)
    image = tiles.transpose(0, 2, 1, 3, 4).reshape(
        BOARD_ROWS * UPSAMPLE_SIZE, BOARD_COLS * UPSAMPLE_SIZE,
        self._channels)
    # Make contiguous copy (transpose makes it non-contiguous)
    image = np.ascontiguousarray(image)
    self.curr_lang_obs = self._current_game.instruction_string
    self.curr_img_obs = image
    return (self.curr_img_obs, self.curr_lang_obs)

  def reset(self, seed=None, options=None):
    """Start a new episode."""
    super().reset(seed=seed)
    # Per-episode sampling
    if self._num_dancers_range is not None:
        self._num_dancers = int(self.np_random.choice(self._num_dancers_range))
    if self._dance_delay_range is not None:
        self._dance_delay = int(self.np_random.choice(self._dance_delay_range))
    # Build game
    self._current_game = self._game_factory()
    # Set up templates for current episode's dancers
    if not self._symbolic:
      self._char_to_template = {}
      for char, color_shape in self._current_game.char_to_color_shape:
        template = _generate_template(color_shape, easy_mode=self._easy_mode)
        self._char_to_template[char] = template
        self._template_lut[ord(char)] = template.astype(np.uint8)
      # Also store agent/wall in _char_to_template for test compatibility
      if self._easy_mode:
        self._char_to_template[ballet_core.AGENT_CHAR] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.AGENT_CHAR][:, :, :1]
            .astype(np.uint8).copy())
        self._char_to_template[ballet_core.WALL_CHAR] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.WALL_CHAR][:, :, :1]
            .astype(np.uint8).copy())
      else:
        self._char_to_template[ballet_core.AGENT_CHAR] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.AGENT_CHAR])
        self._char_to_template[ballet_core.WALL_CHAR] = (
            _CHAR_TO_TEMPLATE_BASE[ballet_core.WALL_CHAR])
    self._done = False
    # Run one game step to match PyColab's its_showtime() behavior:
    # its_showtime() runs the first update cycle, so the initial observation
    # already has the first dance move applied.
    self._current_game.play(0)
    # Build observation
    if self._symbolic:
      vec_obs, instruct_str = self._get_symbolic_obs()
      observation = (vec_obs, LANG_DICT[instruct_str])
    else:
      img_obs, instruct_str = self._get_obs()
      observation = (img_obs, LANG_DICT[instruct_str])
    info = {
        "instruction_string": instruct_str,
        "num_dancers": self._num_dancers,
        "dance_delay": self._dance_delay,
    }
    return observation, info

  def step(self, action):
    """Apply action, step the world forward, and return observations."""
    reward = self._current_game.play(action)
    self._game_over = self._current_game.game_over

    if self._symbolic:
      vec_obs, instruct_str = self._get_symbolic_obs()
      observation = (vec_obs, LANG_DICT[instruct_str])
    else:
      img_obs, instruct_str = self._get_obs()
      observation = (img_obs, LANG_DICT[instruct_str])

    terminated = self._current_game.game_over
    truncated = self._current_game.frame >= self._max_steps
    self._done = terminated or truncated

    info = {
        "instruction_string": instruct_str,
        "num_dancers": self._num_dancers,
        "dance_delay": self._dance_delay,
    }
    return observation, reward, terminated, truncated, info

  def render(self):
    if self.render_mode == "rgb_array":
      canvas = np.zeros((99, 200, 3), dtype=np.uint8)
      if self._symbolic:
        label = "[symbolic]"
        if self.curr_lang_obs:
          label = f"[symbolic] {self.curr_lang_obs}"
        cv2.putText(canvas, label, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return canvas
      img = self.curr_img_obs
      if self._easy_mode:
          img = np.repeat(img, 3, axis=2)
      canvas[:, :99, :] = img
      if self.curr_lang_obs:
          cv2.putText(canvas, self.curr_lang_obs, (100, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                      cv2.LINE_AA)
      return canvas

  @property
  def max_episode_length(self):
    return self._max_steps
