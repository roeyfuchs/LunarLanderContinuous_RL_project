import gym
import numpy as np

MEAN = 0
STD = 0.05


class uncertainty_env(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        noise_x, noise_y = np.random.normal(MEAN, STD, 2)
        obs[0] += noise_x
        obs[1] += noise_y
        return obs
