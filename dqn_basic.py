import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import utils

import numpy as np

# env = gym.make("LunarLanderContinuous-v2")
# env.seed(0)
# np.random.seed(0)


main_engine_values = [0, 0.5, 1]
sec_engine_values = [-1, -0.75, 0, 0.75, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]


class DQN_BASIC:
    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space):
        self.action_space = len(utils.discrete_actions)
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.0005
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()
        self.select_action = self.default_select_action

    def set_select_action(self, select_action):
        self.select_action = select_action


    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def store_current_state(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def default_select_action(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    # def learn(self,):
    #     current_state = self.memory[-1]
    #     states = np.array(current_state[0])
    #     actions = np.array(current_state[1])
    #     rewards = np.array(current_state[2])
    #     next_states = np.array(current_state[3])
    #     dones = np.array(current_state[4])
    #
    #
    #     print(states)
    #     target = rewards + self.gamma * (np.amax(self.model.predict(next_states), axis=0)) * (1 - dones)
    #     targets_full = self.model.predict(states)
    #     targets_full[[ind], [actions]] = targets
    #     print(targets_full)
    #
    #     self.model.fit(states, targets_full, epochs=1, verbose=0)
    #     print("yuv")




    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def solve_game(self, env):
        scores = []

        for episode in range(utils.episodes):
            state = env.reset()
            state = np.reshape(state, (1, 8))
            score = 0
            max_steps = 3000
            for i in range(max_steps):
                action = self.select_action(state)
                # env.render()
                next_state, reward, done, _ = env.step(discrete_actions[action])
                score += reward
                next_state = np.reshape(next_state, (1, 8))
                self.store_current_state(state, action, reward, next_state, done)
                state = next_state
                #self.learn()
                self.replay()
                if done:
                    print("episode: {}/{}, score: {}".format(episode, utils.episodes, score))
                    break
            scores.append(score)

            # Average score of last 100 episode
            is_solved = np.mean(scores[-100:])
            if is_solved > 200:
                print('\n Task Completed! \n')
                break
            print("Average score over last 100 episode: {0:.2f} \n".format(is_solved))
        return scores, self.lr, self.batch_size

