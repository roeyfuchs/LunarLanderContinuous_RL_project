import gym
import matplotlib.pyplot as plt
from dqn_basic import DQN_BASIC
from dqn_epsilon_greedy import DQN_EPSILON_GREEDY
import numpy as np


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(0)
    np.random.seed(0)

    agent = DQN_EPSILON_GREEDY(env.observation_space.shape[0])
    scores, lr, batch_size = agent.solve_game(env)

    plt.plot([i + 1 for i in range(0, len(scores), 2)], scores[::2])
    plt.xlabel("Number Episode")
    plt.ylabel("Score Per Episode")
    plt.show()