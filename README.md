# gym-balletenv
Gym-style implementation of the original dm_env style [Ballet environment](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory) which is introduced in [Towards mental time travel: a hierarchical memory for reinforcement learning agents](https://arxiv.org/abs/2105.14039).

![images](https://user-images.githubusercontent.com/16518993/215299001-faeb4aa1-8665-4772-a2a8-6139615a5a25.gif)

# What's changed
- Gym-style implementation of the original dm_env style [Ballet environment](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory)
- Implement wrappers for gym-balletenv:
    - [x] [gray_scale_observation](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/gray_scale_observation.py)
    - [x] [record_video](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/record_video.py)
    - [x] [transpose_observation](https://github.com/jinPrelude/gym-balletenv/blob/master/gym_balletenv/wrappers/transpose_observation.py)
- And the following gym wrappers are also tested for gym-balletenv:
    - [x] [gym.wrappers.RecordEpisodeStatistics](https://github.com/openai/gym/blob/master/gym/wrappers/record_episode_statistics.py)
    - [x] [gym.vector.SyncVectorEnv](https://github.com/openai/gym/blob/master/gym/vector/sync_vector_env.py)

# Installation
```bash
git clone https://github.com/jinPrelude/gym-balletenv.git
cd gym-balletenv
pip install -e .
```
# Usage
### Check out the repo [gym-balletenv-example](https://github.com/jinPrelude/gym-balletenv-example) for the usage of gym-balletenv and its wrappers.

# Prerequisite for RecordVideo wrapper
if you want to use the wrapper RecordVideo, you need to install gifsicle first:
```bash
sudo apt-get install gifsicle
```