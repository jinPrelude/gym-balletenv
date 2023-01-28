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

"""A pycolab environment for going to the ballet.

A pycolab-based environment for testing memory for sequences of events. The
environment contains some number of "dancer" characters in (implicit) 3 x 3
squares within a larger 9 x 9 room. The agent starts in the center of the room.
At the beginning of an episode, the dancers each do a dance solo of a fixed
length, separated by empty time of a fixed length. The agent's actions do
nothing during the dances. After the last dance ends, the agent must go up to a
dancer, identified using language describing the dance. The agent is rewarded +1
for approaching the correct dancer, 0 otherwise.

The room is upsampled at a size of 9 pixels per square to render a view for the
agent, which is cropped in egocentric perspective, i.e. the agent is always in
the center of its view (see https://arxiv.org/abs/1910.00571).
"""
from absl import app
from absl import flags
from absl import logging

import gym
from gym.spaces import Tuple, Box, MultiBinary, Discrete

import numpy as np
from pycolab import cropping

from gym_balletenv.envs import ballet_environment_core as ballet_core

FLAGS = flags.FLAGS

UPSAMPLE_SIZE = 9  # pixels per game square
SCROLL_CROP_SIZE = 11  # in game squares

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


def _generate_template(object_name):
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

  if object_color not in COLORS:
    raise ValueError("Unknown color: {}".format(object_color))

  template = np.tensordot(template, COLORS[object_color], axes=0)

  return template


# Agent and wall templates
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


def get_scrolling_cropper(rows=9, cols=9, crop_pad_char=" "):
  return cropping.ScrollingCropper(rows=rows, cols=cols,
                                   to_track=[ballet_core.AGENT_CHAR],
                                   pad_char=crop_pad_char,
                                   scroll_margins=(None, None))

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
  """A Python environment API for pycolab ballet tasks."""
  metadata = {"render_modes": ["rgb_array"]}

  def __init__(self, num_dancers, dance_delay, max_steps, rng=None):
    """Construct a BalletEnvironment that wraps pycolab games for agent use.

    This class inherits from gym and has all the expected methods and specs.

    Args:
      num_dancers: The number of dancers to use, between 1 and 8 (inclusive).
      dance_delay: How long to delay between the dances.
      max_steps: The maximum number of steps to allow in an episode, after which
          it will terminate.
      rng: An optional numpy Random Generator, to set a fixed seed use e.g.
          `rng=np.random.default_rng(seed=...)`
    """
    super(BalletEnvironment, self).__init__()

    assert num_dancers in range(1, 9)
    self._num_dancers = num_dancers
    self._dance_delay = dance_delay
    self._max_steps = max_steps

    img_size = (SCROLL_CROP_SIZE * UPSAMPLE_SIZE, SCROLL_CROP_SIZE * UPSAMPLE_SIZE, 3)
    self.observation_space = Tuple(
      (Box(low=0, high=255, shape=img_size, dtype=np.uint8),
      Discrete(14))
    )
    self.action_space = Discrete(8)
    self.reward_range = (0, 1)

    # internal state
    if rng is None:
      rng = np.random.default_rng()
    self._rng = rng
    self._current_game = None       # Current pycolab game instance.
    self._done = False              # Current game done.
    self._game_over = None          # Whether the game has ended.
    self._char_to_template = None   # Mapping of chars to sprite images.

    # rendering tools
    self._cropper = get_scrolling_cropper(SCROLL_CROP_SIZE, SCROLL_CROP_SIZE,
                                          " ")

  def _game_factory(self):
    """Samples dancers and positions, returns a pycolab core game engine."""
    target_dancer_index = self._rng.integers(self._num_dancers)
    motions = list(ballet_core.DANCE_SEQUENCES.keys())
    positions = ballet_core.DANCER_POSITIONS.copy()
    colors = list(COLORS.keys())
    shapes = DANCER_SHAPES.copy()
    self._rng.shuffle(positions)
    self._rng.shuffle(motions)
    self._rng.shuffle(colors)
    self._rng.shuffle(shapes)
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

  def _render_observation(self, observation):
    """Renders from raw pycolab image observation to agent-usable ones."""
    observation = self._cropper.crop(observation)
    obs_rows, obs_cols = observation.board.shape
    image = np.zeros([obs_rows * UPSAMPLE_SIZE, obs_cols * UPSAMPLE_SIZE, 3],
                     dtype=np.uint8)
    for i in range(obs_rows):
      for j in range(obs_cols):
        this_char = chr(observation.board[i, j])
        if this_char != ballet_core.FLOOR_CHAR:
          image[
              i * UPSAMPLE_SIZE:(i + 1) * UPSAMPLE_SIZE, j *
              UPSAMPLE_SIZE:(j + 1) * UPSAMPLE_SIZE] = self._char_to_template[
                  this_char]
    language = self._current_game.the_plot["instruction_string"]
    full_observation = (image, language)
    return full_observation

  def reset(self):
    """Start a new episode."""
    # Build a new game and retrieve its first set of state/reward/discount.
    self._current_game = self._game_factory()
    # set up rendering, cropping, and state for current game
    self._char_to_template = {
        k: _generate_template(v) for k, v in self._current_game.the_plot[
            "char_to_color_shape"]}
    self._char_to_template.update(_CHAR_TO_TEMPLATE_BASE)
    self._cropper.set_engine(self._current_game)
    self._done = False
    # let's go!
    observation, _, _ = self._current_game.its_showtime()
    img_obs, instruct_str = self._render_observation(observation)
    observation = (img_obs, LANG_DICT[instruct_str])
    return observation

  def step(self, action):
    """Apply action, step the world forward, and return observations."""

    # Execute the action in pycolab.
    observation, reward, discount = self._current_game.play(action)

    self._game_over = self._is_game_over()
    reward = reward if reward is not None else 0.
    img_obs, instruct_str = self._render_observation(observation)
    observation = (img_obs, LANG_DICT[instruct_str])

    # Check the current status of the game.
    if self._game_over:
      self._done = True
    else:
      self._done = False

    # create info dict which contains real language string
    info = {"instruction_string": instruct_str}
    return observation, reward, self._done, info


  def _is_game_over(self):
    """Returns whether it is game over, either from the engine or timeout."""
    return (self._current_game.game_over or
            (self._current_game.the_plot.frame >= self._max_steps))


def simple_builder(level_name):
  """Simplifies building from fixed defs.

  Args:
    level_name: '{num_dancers}_delay{delay_length}', where each variable is an
        integer. The levels used in the paper were:
        ['2_delay16', '4_delay16', '8_delay16',
         '2_delay48', '4_delay48', '8_delay48']

  Returns:
    A BalletEnvironment with the requested settings.
  """
  num_dancers, dance_delay = level_name.split("_")
  num_dancers = int(num_dancers)
  dance_delay = int(dance_delay[5:])
  max_steps = 320 if dance_delay == 16 else 1024
  level_args = dict(
      num_dancers=num_dancers,
      dance_delay=dance_delay,
      max_steps=max_steps)
  return BalletEnvironment(**level_args)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  env = simple_builder("4_delay16")
  for _ in range(3):
    obs = env.reset().observation
    for _ in range(300):
      obs = env.step(0).observation
    print(obs)

if __name__ == "__main__":
  app.run(main)