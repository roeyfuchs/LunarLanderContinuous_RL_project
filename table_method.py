import bisect  # for using to discrete values
import random

import numpy as np

from plot_creator import Plot

# discrete_range = np.arange(-1.0, 1.0, 0.2)
discrete_range = [-1, -0.8, -0.6, 0, 0.6, 0.8, 1]
table_indexs = {discrete_range[x]: x for x in range(len(discrete_range))}
n = len(discrete_range)
numer_of_features = 8

discrete_actions = [(x, y) for x in discrete_range for y in discrete_range]
action_index = {discrete_actions[x]: x for x in range(len(discrete_actions))}

alpha = 0.8
gamma = 0.1
epsilon = 0.05  # for epsilon-greedy


class table_method:
    def __init__(self):
        array_shape = [n] * (numer_of_features - 2)
        array_shape.extend([2, 2])  # for 0 1 values
        array_shape.extend([len(discrete_actions)])  # |S| * |A|
        self.Q = np.zeros(array_shape)

    def get_action(self, state):
        def get_best_action_index(self, state):
            """ epsilon-greedy """
            p = random.random()
            state = tuple(state)
            if p < epsilon:
                return random.randrange(len(self.Q[state]))
            else:
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
        def update(state, state2, reward, action, action2):
            """ by example https://www.geeksforgeeks.org/sarsa-reinforcement-learning/ """

            def get_indexes(state, action):
                """ get valid indexes of state and action """
                a = state
                a.append(action)
                return tuple(a)

            state = self.discrete_obs(state)
            state2 = self.discrete_obs(state2)
            action = action_index[action]
            action2 = action_index[action2]
            q1_indx = get_indexes(state, action)
            q2_indx = get_indexes(state2, action2)
            predict = self.Q[q1_indx]
            target = reward + gamma * self.Q[q2_indx]
            self.Q[q1_indx] = self.Q[q2_indx] + alpha * (target - predict)

        total_episodes = 500
        plot = Plot(f"SARSA, alpha = {alpha}, gamma = {gamma}", "episode #", "rewards")
        for episode in range(total_episodes):
            print(f"episode #{episode}")
            episode_reward = 0
            state1 = env.reset()
            action1 = self.get_action(state1)

            done = False
            while not done:
                env.render()
                state2, reward, done, info = env.step(action1)
                episode_reward += reward
                action2 = self.get_action(state2)

                update(state1, state2, reward, action1, action2)
                state1 = state2
                action1 = action2
            plot.add_point(episode, episode_reward)
            print(episode_reward)
        input()
