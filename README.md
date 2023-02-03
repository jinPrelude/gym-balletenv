# gym-balletenv
Gym-style implementation of the original dm_env style [Ballet environment](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory) which is introduced in [Towards mental time travel: a hierarchical memory for reinforcement learning agents](https://arxiv.org/abs/2105.14039).

<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/215299001-faeb4aa1-8665-4772-a2a8-6139615a5a25.gif" width="250" />
</p>

# Featuers
- Gym-style implementation of the original dm_env style [Ballet environment](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory)
- Support [gymnasium](https://github.com/Farama-Foundation/Gymnasium)>=0.27.0 in [v.0.0.2](https://github.com/jinPrelude/gym-balletenv/releases/tag/v0.0.2) (older gym style is supported in [v.0.0.1](https://github.com/jinPrelude/gym-balletenv/releases/tag/v0.0.1))
- Implement wrappers for gym-balletenv:
    - [x] [gray_scale_observation](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/gray_scale_observation.py)
    - [x] [record_video](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/record_video.py)
    - [x] [transpose_observation](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/transpose_observation.py)
    - [x] [OhehotLanguage](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/onehot_language.py)
- And the following gymnasium wrappers are also tested for gym-balletenv:
    - [x] [gymnasium.wrappers.RecordEpisodeStatistics](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/record_episode_statistics.py)
    - [x] [gymnasium.vector.SyncVectorEnv](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py)
    - [x] [gymnasium.vector.AsyncVectorEnv](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/async_vector_env.py)

# Installation
```bash
git clone https://github.com/jinPrelude/gym-balletenv.git
cd gym-balletenv
pip install -e .
```
# Usage
This is very example code for gym-balletenv :
```python
from gym_balletenv.envs import BalletEnvironment
from gym_balletenv.wrappers import GrayScaleObservation


env = BalletEnvironment(env_id="2_delay16", max_steps=320)

obs, info = env.reset(seed=0)
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```


Check out the repo [gym-balletenv-example](https://github.com/jinPrelude/gym-balletenv-example) more usage of gym-balletenv and its wrappers.

## Prerequisite for RecordVideo wrapper
[RecordVideo Wrapper](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/record_video.py) uses [pygifsicle](https://github.com/LucaCappelletti94/pygifsicle) for optimizing the recorded gif file, so you need to install gifsicle to use it:
```bash
sudo apt-get install gifsicle
```
## Easy mode
**Easy mode** is a custom feature that removes various color options and limited dancer shape by the number of dancers. So all the colors of the dancers are "red", and only 4 shape of the dancers will be sampling for 4 dancers environment. This feature is made for rapid model evaluation purpose.

Simply add "_easy" at the end of the env_id :
```python
from gym_balletenv.envs import BalletEnvironment

env = BalletEnvironment(env_id="2_delay2_easy", max_step=320)

```