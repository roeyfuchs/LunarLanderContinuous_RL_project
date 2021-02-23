import numpy as np
import gym

#Building the environment
env = gym.make('FrozenLake-v0')

# Defining the different parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initializing the Q-matrix
Q = np.zeros((env.observation_space.n, env.action_space.n))
print(Q.shape)
