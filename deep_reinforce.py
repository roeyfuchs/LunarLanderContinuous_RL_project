import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# discrete values for action
main_engine_values = [0, 0.1, 0.9, 1]
sec_engine_values = [0, -1, -0.9, 0.9, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
action_index = {discrete_actions[x]: x for x in range(len(discrete_actions))}


# from https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
class reinforce:
    def __init__(self, env):
        self.env = env
        self.state_shape = env.observation_space.shape
        # self.action_shape = env.action_shape.shape
        self.action_shape = len(discrete_actions)
        self.gamma = 0.99
        self.alpha = 0.0001
        self.learning_rate = 0.01

        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_reawrds = []
        self.model = self._create_model()

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def remember(self, state, action, action_prob, reward):
        encoded_action = self.hot_encode_action(action)
        self.gradients.append(encoded_action - action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def _create_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(24, input_shape=self.state_shape, activation="relu"))
        model.add(layers.Dense(12, activation="relu"))

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

    def train(self, episodes, rollout_n=1, render_n=50):
        env = self.env
        total_rewards = np.zeros(episodes)

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

                if episode % render_n == 0:
                    env.render()
                if done:
                    if episode % rollout_n == 0:
                        history = self.update_policy()
            total_rewards[episode] = episode_reward
            print(f"#episode {episode}, reward {episode_reward}")
        self.total_reawrds = total_rewards
