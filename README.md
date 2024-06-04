# Deep-RL-Toolkit

<p align="center">
<img src="docs/images/logo.png" alt="logo" width="1000"/>
</p>

## Overview

Deep RL Toolkit is a flexible and high-efficient reinforcement learning framework. RLToolkit is developed for practitioners with the following advantages:

- **Reproducible**. We provide algorithms that stably reproduce the result of many influential reinforcement learning algorithms.

- **Extensible**. Build new algorithms quickly by inheriting the abstract class in the framework.

- **Reusable**.  Algorithms provided in the repository could be directly adapted to a new task by defining a forward network and training mechanism will be built automatically.

- **Elastic**: allows to elastically and automatically allocate computing resources on the cloud.

- **Lightweight**: the core codes \<1,000 lines (check [Demo](./examples/tutorials/lesson3/DQN/train.py)).

- **Stable**: much more stable than [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) by utilizing various ensemble methods.

## Table of Content

- [Deep-RL-Toolkit](#deep-rl-toolkit)
  - [Overview](#overview)
  - [Table of Content](#table-of-content)
  - [Supported Algorithms](#supported-algorithms)
  - [Supported Envs](#supported-envs)
  - [Examples](#examples)
    - [Quick Start](#quick-start)
  - [References](#references)
    - [Reference Papers](#reference-papers)
    - [References code](#references-code)

## Supported Algorithms

RLToolkit implements the following model-free deep reinforcement learning (DRL) algorithms:

![../_images/rl_algorithms_9_15.svg](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

## Supported Envs

- **OpenAI Gym**
- **Atari**
- **MuJoCo**
- **PyBullet**

For the details of DRL algorithms, please check out the educational webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

## Examples

<p align="center">
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/docs/images/trained.gif" alt="logo" width="810"/>
</p>
<p align="center">
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/examples/tutorials/assets/img/breakout.gif" width = "200" height ="200"/>
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/examples/tutorials/assets/img/spaceinvaders.gif" width = "200" height ="200"/>
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/examples/tutorials/assets/img/seaquest.gif" width = "200" height ="200"/>
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/docs/images/Breakout.gif" width = "200" height ="200" alt="Breakout"/>
<br>

<p align="center">
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/docs/images/performance.gif" width = "265" height ="200" alt="NeurlIPS2018"/>
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/docs/images/Half-Cheetah.gif" width = "265" height ="200" alt="Half-Cheetah"/>
<img src="https://github.com/jianzhnie/RLToolkit/blob/main/examples/tutorials/assets/img/snowballfight.gif" width = "265" height ="200"/>
<br>

If you want to learn more about deep reinforcemnet learning, please read the [deep-rl-class](https://jianzhnie.github.io/llmtech/) and run the [examples](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples).

- [Classic Control](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples/discrete)
- [Atari Benchmark](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples/atari)
- [Box2d Benchmark](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples/box2d)
- [Mujuco Benchmark](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples/mujoco)
- [Petting Zoo](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples/pettingzoo)

### Quick Start

```bash
git clone https://github.com/jianzhnie/deep-rl-toolkit.git
cd examples/cleanrl/

python cleanrl_runner.py  --logger tensorboard --env CartPole-v0 --algo dqn
```

## References

### Reference Papers

01. Deep Q-Network (DQN) <sub><sup> ([V. Mnih et al. 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) </sup></sub>
02. Double DQN (DDQN) <sub><sup> ([H. Van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461)) </sup></sub>
03. Advantage Actor Critic (A2C)
04. Vanilla Policy Gradient (VPG)
05. Natural Policy Gradient (NPG) <sub><sup> ([S. Kakade et al. 2002](http://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)) </sup></sub>
06. Trust Region Policy Optimization (TRPO) <sub><sup> ([J. Schulman et al. 2015](https://arxiv.org/abs/1502.05477)) </sup></sub>
07. Proximal Policy Optimization (PPO) <sub><sup> ([J. Schulman et al. 2017](https://arxiv.org/abs/1707.06347)) </sup></sub>
08. Deep Deterministic Policy Gradient (DDPG) <sub><sup> ([T. Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971)) </sup></sub>
09. Twin Delayed DDPG (TD3) <sub><sup> ([S. Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) </sup></sub>
10. Soft Actor-Critic (SAC) <sub><sup> ([T. Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)) </sup></sub>
11. SAC with automatic entropy adjustment (SAC-AEA) <sub><sup> ([T. Haarnoja et al. 2018](https://arxiv.org/abs/1812.05905)) </sup></sub>

### References code

- rllib

  - https://github.com/ray-project/ray
  - https://docs.ray.io/en/latest/rllib/index.html

- coach

  - https://github.com/IntelLabs/coach
  - https://intellabs.github.io/coach

- Pearl

  - https://github.com/facebookresearch/Pearl
  - https://pearlagent.github.io/

- tianshou

  - https://github.com/thu-ml/tianshou
  - https://tianshou.org/en/stable/

- stable-baselines3

  - https://github.com/DLR-RM/stable-baselines3
  - https://stable-baselines3.readthedocs.io/en/master/

- PARL

  - https://github.com/PaddlePaddle/PARL
  - https://parl.readthedocs.io/zh-cn/latest/

- openrl

  - https://github.com/OpenRL-Lab/openrl/
  - https://openrl-docs.readthedocs.io/zh/latest/

- cleanrl

  - https://github.com/vwxyzjn/cleanrl
  - https://docs.cleanrl.dev/
