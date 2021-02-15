import bisect  # for using to discrete values

import numpy as np

# discrete_range = np.arange(-1.0, 1.0, 0.2)
discrete_range = [-1, -0.8, -0.6, 0, 0.6, 0.8, 1]
table_indexs = {discrete_range[x]: x for x in range(len(discrete_range))}
n = len(discrete_range)

discrete_actions = [(x, y) for x in discrete_range for y in discrete_range]
action_index = {discrete_actions[x]: x for x in range(len(discrete_range))}

alpha = 0.1
gamma = 0.1


class table_method:
    def __init__(self):
        array_shape = [n] * (n - 2)
        array_shape.extend([2, 2])  # for 0 1 values
        array_shape.extend([len(discrete_actions)])  # |S| * |A|
        self.Q = np.zeros(array_shape)
        pass

    def get_action(self, state):
        def get_best_action_index(self, state):
            """ TODO: change to greedy epsilon """
            return np.argmax(self.Q[state])

        state = self.discrete_obs(state)
        action_index = get_best_action_index(self, state)
        return discrete_actions[action_index]

    def discrete_obs(self, obs):
        new_obs = []
        for i in obs:
            new_obs.extend([bisect.bisect_left(discrete_range, i) - 1])
        new_obs[-2:] = obs[-2:].astype(int)  # 0 or 1
        return new_obs

    def train(self, env):
        observation = env.reset()
        for _ in range(1000):
            print(_)
            action = self.get_action(observation)
            observation_tag, reward, done, info = env.step(action)

            currentQ = self.Q[self.discrete_obs(observation), action_index[action]]
            self.Q[
                self.discrete_obs(observation), action_index[action]
            ] = currentQ + alpha * (
                reward
                + gamma
                * (
                    self.Q[self.discrete_obs(observation_tag), action_index[action]]
                    - self.Q[self.discrete_obs(observation), action_index[action]]
                )
            )

            if done:
                observation = env.reset()
