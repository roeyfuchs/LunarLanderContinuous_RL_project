from simple_dqn import SimpleDQN
import gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(0)
    np.random.seed(0)

    agent = SimpleDQN(env.observation_space.shape[0])
    rewards = agent.solve_env(env)

    plt.xlabel("Number Episode")
    plt.ylabel("Score Per Episode")
    plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
    plt.show()