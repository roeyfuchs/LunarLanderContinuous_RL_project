from dqn_basic import DQN_BASIC
import random
import numpy as np


class DQN_EPSILON_GREEDY(DQN_BASIC):
    def __init__(self, state_space):
        super().__init__(state_space)
        super().set_select_action(self.greedy_select_action)

    def greedy_select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
