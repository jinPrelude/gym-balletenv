# gym-balletenv
Gymnasium-compatible implementation of the original dm_env style [Ballet environment](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory) which is introduced in [Towards mental time travel: a hierarchical memory for reinforcement learning agents](https://arxiv.org/abs/2105.14039).

<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/215299001-faeb4aa1-8665-4772-a2a8-6139615a5a25.gif" width="250" />
</p>

# Features
- Gymnasium-compatible implementation of the original dm_env style [Ballet environment](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory)
- Support [gymnasium](https://github.com/Farama-Foundation/Gymnasium)>=1.0.0
- Implement wrappers for gym-balletenv:
    - [x] [GrayScaleObservation](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/gray_scale_observation.py)
    - [x] [TransposeObservation](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/transpose_observation.py)
    - [x] [OnehotLanguage](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/onehot_language.py)

# Installation
```bash
git clone https://github.com/jinPrelude/gym-balletenv.git
cd gym-balletenv
pip install -e .
```

# Usage

## Using `gymnasium.make`
```python
import gym_balletenv
import gymnasium as gym

# Default: 2 dancers, delay 16, max_steps 500
env = gym.make("gym_balletenv/balletenv-v0")

# Custom parameters
env = gym.make("gym_balletenv/balletenv-v0", level_name="4_delay16", max_steps=320)

obs, info = env.reset(seed=0)
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Using the class directly
```python
from gym_balletenv.envs import BalletEnvironment
from gym_balletenv.wrappers import GrayScaleObservation

env = BalletEnvironment(level_name="2_delay16", max_steps=320)
env = GrayScaleObservation(env)

obs, info = env.reset(seed=0)
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

Check out the repo [gym-balletenv-example](https://github.com/jinPrelude/gym-balletenv-example) for more usage of gym-balletenv and its wrappers.

## Easy mode
**Easy mode** is a custom feature that removes various color options and limits dancer shapes by the number of dancers. All dancer colors are set to "red", and only N shapes are sampled for an N-dancer environment. This feature is made for rapid model evaluation.

Simply add `_easy` at the end of the `level_name`:
```python
from gym_balletenv.envs import BalletEnvironment

env = BalletEnvironment(level_name="2_delay2_easy", max_steps=320)
```
