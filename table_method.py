import random

import numpy as np

from plot_creator import Plot

# discrete values for states
x_discrete = np.arange(-0.2, 0.2, 0.1)
y_discrete = np.concatenate((np.arange(0, 1.1, 0.3), [-0.01]))
x_velocity_discrete = np.arange(-0.5, 0.5, 0.25)
y_velocity_discrete = np.arange(-0.8, 0.2, 0.25)
angle_discrete = np.arange(-0.3, 0.3, 0.15)
angular_velocity_discrete = np.arange(-0.3, 0.3, 0.15)
left_leg = np.array([0, 1])
right_leg = np.array([0, 1])
values_table = np.array(
    [
        x_discrete,
        y_discrete,
        x_velocity_discrete,
        y_velocity_discrete,
        angle_discrete,
        angular_velocity_discrete,
        left_leg,
        right_leg,
    ],
    dtype=object,
)

# discrete values for action
# main_engine_values = [0, 0.2, 0.8, 1]
main_engine_values = [0, 1]
# sec_engine_values = [0, -1, -0.6, 0.6, 1]
sec_engine_values = [0, -1, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
action_index = {discrete_actions[x]: x for x in range(len(discrete_actions))}

default_alpha = 0.3
default_gamma = 0.99
epsilon = 0.1  # for epsilon-greedy


class table_method:
    def __init__(self, alpha=default_alpha, gamma=default_gamma):
        array_shape = [
            len(x_discrete),
            len(y_discrete),
            len(x_velocity_discrete),
            len(y_velocity_discrete),
            len(angle_discrete),
            len(angular_velocity_discrete),
            len(left_leg),
            len(right_leg),
        ]
        array_shape.extend([len(discrete_actions)])  # |S| * |A|
        self.Q = np.zeros(array_shape)
        self.alpha = alpha
        self.gamma = gamma

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
        for i in np.abs(obs - values_table):
            new_obs.append(np.argmin(i))
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
            target = self.Q[q2_indx]
            self.Q[q1_indx] = self.Q[q1_indx] + self.alpha * (
                reward + self.gamma * target - predict
            )

        total_episodes = 500
        reward_array = []
        plot = Plot(
            f"SARSA, alpha = {self.alpha}, gamma = {self.gamma}", "episode #", "rewards"
        )
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
            reward_array.append(episode_reward)
        print(f"Avg. reawrd: {np.mean(reward_array)}")
        input()
