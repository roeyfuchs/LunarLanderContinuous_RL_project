from keras import Sequential
from queue import PriorityQueue
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
from itertools import count
import numpy as np
import utils
import random


class DQNPrioritizedExperience:
    """ Implementation of deep q learning algorithm with prioritized replay memory and epsilon greedy"""

    def __init__(self, verbose, render):
        self.action_space = len(utils.discrete_actions)
        self.state_space = utils.state_space
        self.gamma = .99
        self.batch_size = 64
        self.lr = 0.0005
        self.replay_memory = PriorityQueue()
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996
        self.epsilon_loss = -0.0001
        self.model = self.build_model()
        self.tie_breaker = count()
        self.verbose = verbose
        self.render = render

    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def save_state(self, loss, reward, tie_breaker, state, action, next_state, done):
        self.replay_memory.put((loss, reward, tie_breaker, state, action, next_state, done))
        next(self.tie_breaker)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if self.replay_memory.qsize() < self.batch_size:
            return

        minibatch = []
        for i in range(self.batch_size):
            minibatch.append(self.replay_memory.get())

        states = np.array([i[3] for i in minibatch])
        actions = np.array([i[4] for i in minibatch])
        rewards = np.array([i[1] for i in minibatch])
        next_states = np.array([i[5] for i in minibatch])
        dones = np.array([i[6] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def solve_env(self, env):
        rewards = []
        for episode in range(utils.episodes):
            state = env.reset()
            state = np.reshape(state, (1, 8))
            score = 0
            max_steps = 3000
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(utils.discrete_actions[action])
                score += reward
                next_state = np.reshape(next_state, (1, 8))

                target = reward + self.gamma * (np.amax(self.model.predict_on_batch(next_state), axis=1)) * (1 - done)
                prediction = np.amax(self.model.predict_on_batch(state), axis=1)
                loss = -1*((target - prediction)**2 + self.epsilon_loss)

                self.save_state(loss, reward, self.tie_breaker, state, action, next_state, done)
                state = next_state
                self.replay()
                if done:
                    print("episode: {}/{}, score: {}".format(episode, utils.episodes, score))
                    break
            rewards.append(score)

            # Average score of last 100 episode
            is_solved = np.mean(rewards[-100:])
            if is_solved > 200:
                print('\n Task Completed! \n')
                break
            print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        return rewards
