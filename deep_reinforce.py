import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from plot_creator import Plot

# discrete values for action
main_engine_values = [
    0,
    1,
]
main_engine_values.sort()
sec_engine_values = [0, 1, -1]
sec_engine_values.sort()
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
action_index = {discrete_actions[x]: x for x in range(len(discrete_actions))}


BASE_PATH_MODEL_SAVING = os.path.join("reinforce", "model")
BASE_PATH_REWARD = os.path.join("reinforce", "reward")
os.makedirs(BASE_PATH_MODEL_SAVING, exist_ok=True)
os.makedirs(BASE_PATH_REWARD, exist_ok=True)  # make sure we have directoris to save

# from https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
class reinforce:
    def __init__(
        self,
        env,
        render=True,
        save=False,
        alpha=0.35,
        gamma=0.99,
        lr=0.001,
        verbose=True,
        load=None,
    ):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_shape = len(discrete_actions)
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = lr

        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_reawrds = []
        if load:
            self.model = keras.models.load_model(load)
        else:
            self.model = self._create_model()
        self.verbose = verbose
        self.render = render
        self.save = save

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def remember(self, state, action, action_prob, reward):
        encoded_action = self.hot_encode_action(action)
        encoded_action = action
        self.gradients.append(encoded_action - action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def _create_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(128, input_shape=self.state_shape, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.action_shape, activation="softmax"))
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt)

        return model

    def get_action(self, state):
        state = state.reshape([1, state.shape[0]])

        action_probability_distribution = self.model.predict(state).flatten()

        action_probability_distribution /= np.sum(action_probability_distribution)
        action = np.random.choice(
            self.action_shape, 1, p=action_probability_distribution
        )[0]

        return action, action_probability_distribution

    def get_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_total_return = 0

        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)

        norm_discounted_rewards = (discounted_rewards - mean_rewards) / (
            std_rewards + 1e-7
        )

        return norm_discounted_rewards

    def update_policy(self):
        states = np.vstack(self.states)
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)

        discounted_rewards = self.get_discounted_rewards(rewards)

        gradients *= discounted_rewards
        gradients = self.alpha * np.vstack([gradients]) + self.probs

        history = self.model.train_on_batch(states, gradients)

        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

        return history

    def train(self, episodes, rollout_n=1):
        env = self.env
        total_rewards = np.zeros(episodes)

        plot = Plot(
            f"Reinforce, $\\alpha = {self.alpha}$, $\\gamma = {self.gamma}$, learning rate = {self.learning_rate}",
            "episode #",
            "rewards",
            verbose=self.verbose,
            win=100,
        )
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, prob = self.get_action(state)
                next_state, reward, done, info = env.step(discrete_actions[action])

                self.remember(state, action, prob, reward)

                state = next_state

                episode_reward += reward

                if self.render:
                    env.render()

                if done:
                    if episode % rollout_n == 0:
                        history = self.update_policy()
            total_rewards[episode] = episode_reward
            plot.add_point(episode, episode_reward)
            if self.verbose:
                print(f"#episode {episode}, reward {episode_reward}")
        self.total_reawrds = total_rewards
        if self.save:
            file_name = "-".join(
                [str(self.alpha), str(self.gamma), str(self.learning_rate)]
            )
            plot.save(file_name)
            self.model.save(os.path.join(BASE_PATH_MODEL_SAVING, file_name))
            np.save(os.path.join(BASE_PATH_REWARD, file_name), self.total_reawrds)
