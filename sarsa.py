import random

import numpy as np

from utils import discrete_actions, main_engine_values, sec_engine_values

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
action_index = {discrete_actions[x]: x for x in range(len(discrete_actions))}

default_alpha = 0.3
default_gamma = 0.99
epsilon = [0.3, 0.2, 0.1, 0.05, 0.01, 0]  # for epsilon-greedy
epsilon_bins = [0, 150, 300, 500, 1000, 1250, 3000]


class SARSA:
    def __init__(
        self,
        verbose,
        alpha=default_alpha,
        gamma=default_gamma,
        render=True,
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
        array_shape.extend([len(discrete_actions)])  # |S| * |A|
        self.Q = np.zeros(array_shape)
        self.alpha = alpha
        self.gamma = gamma
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
            self.Q[q1_indx] = self.Q[q1_indx] + self.alpha * (
                reward + self.gamma * target - predict
            )

        total_episodes = 100000
        reward_array = []
        for episode in range(total_episodes):
            self.versobe(f"episode #{episode}")
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
            self.verbose(
                "episode: {}/{}, score: {}".format(
                    episode, total_episodes, episode_reward
                )
            )
            reward_array.append(episode_reward)
            is_solved = np.mean(reward_array[-100:])
            if is_solved > 200:
                self.verbose("Task Completed!")
                break
        return reward_array
