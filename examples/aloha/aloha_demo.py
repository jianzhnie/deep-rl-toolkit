# example.py
import gymnasium as gym
from gym_aloha.env import AlohaEnv

env = AlohaEnv(task='alohainsertion-v0')
env = gym.make('gym_aloha/AlohaInsertion-v0',
               obs_type='pixels',
               render_mode='rgb_array')
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation.keys())
    print(observation['top'].shape)
    print(action.shape)
    print(info)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
