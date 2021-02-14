import bisect  # for using to discrete values

import numpy as np

obs_discrete_range = np.arange(-1.0, 1.0, 0.2)
n = len(obs_discrete_range)


class table_method:
    def __init__(self):
        array_shape = [n] * (n - 2)
        array_shape.extend([2, 2])  # for 0 1 values
        self.table = np.zeros(array_shape)
        pass

    def discrete_obs(self, obs):
        new_obs = []
        for i in obs:
            new_obs.extend([bisect.bisect_left(obs_discrete_range, i)])
        print(obs[-2:])
        new_obs[-2:] = obs[-2:]  # 0 or 1
        return new_obs
