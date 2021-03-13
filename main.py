from simple_dqn import SimpleDQN
from queue import PriorityQueue
from dqn_prioritized_experience import DQNPrioritizedExperience
import gym
import os
import time
import numpy as np
import matplotlib.pyplot as plt


BASE_SAVING_PATH = "graphs"
os.makedirs(
    BASE_SAVING_PATH, exist_ok=True
)  # make sure that we have a directory to save
FILE_SAVEING_TYPE = "png"


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(0)
    np.random.seed(0)

    agent = DQNPrioritizedExperience(env.observation_space.shape[0])
    rewards = agent.solve_env(env)

    filename = time.strftime("%H-%M-%S", time.localtime())
    file_path = os.path.join(BASE_SAVING_PATH, filename)
    plt.xlabel("Number Episode")
    plt.ylabel("Score Per Episode")
    plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
    plt.savefig(".".join([file_path, FILE_SAVEING_TYPE]))
    plt.show()
