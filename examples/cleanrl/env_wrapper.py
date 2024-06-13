import gymnasium as gym


def make_env(env_id, seed):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def main() -> None:
    env = make_env('CartPole-v1', seed=0)()
    observation, info = env.reset()
    for _ in range(1000):
        action = (env.action_space.sample()
                  )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward, terminated, truncated, info)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == '__main__':
    main()
