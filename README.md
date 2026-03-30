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

# Default: 2 dancers, delay 16 (max_steps auto-set to 320)
env = gym.make("gym_balletenv/balletenv-v0")

# 8 dancers, delay 48 (max_steps auto-set to 1024)
env = gym.make("gym_balletenv/balletenv-v0", level_name="8_delay48")

# Override max_steps manually
env = gym.make("gym_balletenv/balletenv-v0", level_name="4_delay16", max_steps=500)

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

env = BalletEnvironment(level_name="2_delay16")  # max_steps=320 auto
env = GrayScaleObservation(env)

obs, info = env.reset(seed=0)
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

Check out the repo [gym-balletenv-example](https://github.com/jinPrelude/gym-balletenv-example) for more usage of gym-balletenv and its wrappers.

## Easy mode
**Easy mode** is a custom feature designed for rapid model evaluation. It simplifies the observation in several ways:

- **Single-channel rendering**: Observations are `(99, 99, 1)` instead of RGB `(99, 99, 3)`, rendered directly in the pipeline without wrapper overhead.
- **Unified dancer shape**: All dancers use the same shape ("triangle"), since shape identity is not semantically meaningful.

Pass `easy_mode=True` to enable:
```python
from gym_balletenv.envs import BalletEnvironment

env = BalletEnvironment(level_name="2_delay2", easy_mode=True)
# obs[0].shape == (99, 99, 1)
```

## Per-episode Sampling (v1)
The `BalletEnvironment-v1` registration allows per-episode uniform sampling of `num_dancers` and `dance_delay`. Each `reset()` call samples new values from the provided ranges.

```python
import gymnasium as gym
import gym_balletenv

# Sample num_dancers from {2, 4, 6} and delay from {16, 48} each episode
env = gym.make("gym_balletenv/BalletEnvironment-v1", level_name="2_delay16",
               num_dancers_range=[2, 4, 6], dance_delay_range=[16, 48])

# max_steps is auto-computed from worst-case (max delay)
obs, info = env.reset(seed=0)
```

When ranges are `None` (default), behavior is identical to `balletenv-v0`.

## Symbolic Observation Mode

> **Note**: Not part of the original paper. A custom extension for accelerating memory and length extrapolation experiments.

Symbolic mode replaces pixel observations with the **flattened 11x11 game board** (121 float32 values), eliminating 9x upsampling and color rendering while preserving the identical spatial information, task structure, action space (`Discrete(8)`), language instruction (`Discrete(14)`), episode flow, and reward. The temporal model (transformer/LSTM) infers dancer movement patterns from the sequence of board states, exactly as it would from pixel frames.

```python
# Quick start
env = gym.make("gym_balletenv/balletenv-symbolic-v0")

# v1 with per-episode sampling
env = gym.make("gym_balletenv/BalletEnvironment-v1",
               level_name="2_delay2",
               num_dancers_range=[2, 3, 4],
               dance_delay_range=[2, 4, 6],
               symbolic=True)
```

### Observation Layout

`Tuple(Box(121,), Discrete(14))` -- the 11x11 board flattened, with categorical encoding normalized to `[0, 1]`:

| Value | Meaning |
|-------|---------|
| `0.0` | Floor |
| `0.1` | Wall |
| `0.2` | Agent |
| `0.3`--`1.0` | Dancer 0--7 |

No hand-crafted features (phase, move deltas, etc.) -- the board is a direct representation of what the pixel observation shows, just without rendering. Can be reshaped to `(11, 11)` for spatial models.

**Wrapper compatibility**: `OnehotLanguage` works as-is. `TransposeObservation` is not compatible (nor needed).

## Batched Environment

> **Note**: For high-throughput RL training. Processes N environments in single numpy operations.

`BatchedBalletEnv` eliminates `SyncVectorEnv`'s per-env Python loop overhead by managing all N game states as contiguous numpy arrays. A single `step()` call advances all environments simultaneously.

```python
from gym_balletenv.envs import BatchedBalletEnv
import numpy as np

env = BatchedBalletEnv(
    num_envs=512,
    level_name="4_delay16",
    symbolic=True,        # or False for RGB/Easy B&W
    seed=42,
)

obs, info = env.reset()
# obs: tuple(ndarray (512, 121), ndarray (512,))

for _ in range(1000):
    actions = np.random.randint(0, 8, size=512)
    obs, reward, terminated, truncated, info = env.step(actions)
    # reward: (512,), terminated: (512,), truncated: (512,)

env.close()
```

**Key properties:**
- All environments in a batch share the same `num_dancers` and `dance_delay`
- Terminated environments are automatically reset within the same `step()` call
- Terminal observations are stored in `info["final_observation"]` (list of length N, `None` for non-done envs)
- Supports all observation modes: RGB `(N, 99, 99, 3)`, Easy B&W `(N, 99, 99, 1)`, Symbolic `(N, 121)`
- `observation_space` and `action_space` describe single-env spaces (not batched)

### Throughput comparison (N=512, `4_delay16`)

| Mode | `SyncVectorEnv` | `BatchedBalletEnv` | Speedup |
|------|----------------:|-------------------:|--------:|
| Symbolic | 220K steps/s | **1.80M steps/s** | **8.2x** |
| Easy B&W | 55K steps/s | **70K steps/s** | 1.3x |
| RGB | 48K steps/s | 39K steps/s | ~1x |

Symbolic mode benefits most because the game-logic vectorization dominates. RGB/B&W modes are memory-bandwidth limited by the 11x11 → 99x99 tile rendering.
