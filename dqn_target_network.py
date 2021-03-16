from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
import numpy as np
import utils
import random


class DQNTargetNetwork:
    """ Implementation of deep q learning algorithm with replay memory, epsilon greedy and target network"""

    def __init__(self, verbose, render):
        self.action_space = len(utils.discrete_actions)
        self.state_space = utils.state_space
        self.gamma = .99
        self.batch_size = 64
        self.lr = 0.0005
        self.replay_memory = deque(maxlen=1000000)
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996
        self.network = self.build_network()
        self.target_network = self.build_network()
        self.verbose = verbose
        self.render = render

    def build_network(self):
        network = Sequential()
        network.add(Dense(150, input_dim=self.state_space, activation=relu))
        network.add(Dense(120, activation=relu))
        network.add(Dense(self.action_space, activation=linear))
        network.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return network

    def save_state(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.network.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return
        minibatch = random.sample(self.replay_memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.target_network.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.network.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.network.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def solve_env(self, env):
        rewards = []
        for episode in range(utils.episodes):
            state = env.reset()
            state = np.reshape(state, (1, 8))
            score = 0
            max_steps = 3000
            update_target_model = utils.updateTargetNetwork
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(utils.discrete_actions[action])
                if self.render:
                    env.render()
                score += reward
                update_target_model -= 1
                next_state = np.reshape(next_state, (1, 8))
                self.save_state(state, action, reward, next_state, done)
                state = next_state
                if update_target_model == 0:
                    update_target_model = utils.updateTargetNetwork
                    weights = self.network.get_weights()
                    self.target_network.set_weights(weights)
                self.replay()
                if done:
                    self.verbose("episode: {}/{}, score: {}".format(episode, utils.episodes, score))
                    break
            rewards.append(score)

            # Average score of last 100 episode
            is_solved = np.mean(rewards[-100:])
            if is_solved > 200:
                self.verbose("Task Completed!")
                break
            self.verbose("Average over last 100 episode: {0:.2f}".format(is_solved))
        return rewards
