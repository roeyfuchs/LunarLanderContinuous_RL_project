import os
import random

import numpy as np

from plot_creator import Plot

# discrete values for states
x_discrete = np.linspace(-0.2, 0.2, 9)
y_discrete = np.concatenate((np.linspace(0, 1.1, 9), [-0.01]))
x_velocity_discrete = np.linspace(-0.3, 0.3, 6)
y_velocity_discrete = np.linspace(-0.5, 0.1, 10)
angle_discrete = np.linspace(-0.3, 0.3, 8)
angular_velocity_discrete = np.linspace(-0.2, 0.2, 6)
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
main_engine_values = [0, 0.5, 1]
sec_engine_values = [0, -1, -0.75, 0.75, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
action_index = {discrete_actions[x]: x for x in range(len(discrete_actions))}

default_alpha = 0.3
default_gamma = 0.99
epsilon = [0.3, 0.2, 0.1, 0.05, 0.01, 0]  # for epsilon-greedy
epsilon_bins = [0, 150, 300, 500, 1000, 1250, 3000]

BASE_PATH_Q_SAVING = os.path.join("SARSA", "Q")
BASE_PATH_REWARD = os.path.join("SARSA", "reward")
os.makedirs(BASE_PATH_Q_SAVING, exist_ok=True)
os.makedirs(BASE_PATH_REWARD, exist_ok=True)  # make sure we have directoris to save


class table_method:
    def __init__(
        self,
        alpha=default_alpha,
        gamma=default_gamma,
        save=False,
        load=None,
        render=True,
        verbose=True,
    ):
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
        print(array_shape)
        array_shape.extend([len(discrete_actions)])  # |S| * |A|
        if load:
            self.Q = np.load(load)
        else:
            self.Q = np.zeros(array_shape)
        self.alpha = alpha
        self.gamma = gamma
        self.save = save
        self.render = render
        self.verbose = verbose

    def get_action(self, state, it):
        def get_best_action_index(self, state, it):
            """ epsilon-greedy """

            def get_prob_by_iteration(it):
                for i in range(len(epsilon_bins) - 1):
                    if it >= epsilon_bins[i] and it < epsilon_bins[i + 1]:
                        return epsilon[i]
                return epsilon[-1]

            p = random.random()
            state = tuple(state)
            if p < get_prob_by_iteration(it):
                return random.randrange(len(self.Q[state]))
            else:
                return np.argmax(self.Q[state])

        state = self.discrete_obs(state)
        action_index = get_best_action_index(self, state, it)
        return discrete_actions[action_index]

    def discrete_obs(self, obs):
        new_obs = []
        for i in np.abs(obs - values_table):
            new_obs.append(np.argmin(i))
        return new_obs

    def play(self, env):
        def update(state, state2, reward, action, action2, done):
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
            if done:
                target = 0
            self.Q[q1_indx] = self.Q[q1_indx] + self.alpha * (reward + self.gamma * target - predict)

        total_episodes = 100000
        reward_array = []
        plot = Plot(
            f"SARSA, alpha = {self.alpha}, gamma = {self.gamma}",
            "episode #",
            "rewards",
            verbose=self.verbose,
            win=100,
        )
        for episode in range(total_episodes):
            if self.verbose:
                print(f"episode #{episode}")
            episode_reward = 0
            state1 = env.reset()
            action1 = self.get_action(state1, episode)

            done = False
            while not done:
                if self.render:
                    env.render()
                state2, reward, done, info = env.step(action1)
                episode_reward += reward
                action2 = self.get_action(state2, episode)

                update(state1, state2, reward, action1, action2, done)
                state1 = state2
                action1 = action2
            plot.add_point(episode, episode_reward)
            if self.verbose:
                print(episode_reward)
            reward_array.append(episode_reward)
        if self.verbose:
            print(f"Avg. reawrd: {np.mean(reward_array)}")
        if self.save:
            file_name = "-".join([str(self.alpha), str(self.gamma)])
            plot.save(file_name)
            np.save(os.path.join(BASE_PATH_Q_SAVING, file_name), self.Q)
            np.save(os.path.join(BASE_PATH_REWARD, file_name), reward_array)
