import os

os.environ["LANG"] = "en_US"
import gym

import table_method

env = gym.make("LunarLanderContinuous-v2")
alphas = [0.35, 0.7, 0.9, 1]
gammas = [0, 0.35, 0.7, 0.9, 0.99]

for alpha in alphas:
    for gamma in gammas:
        print(alpha, gamma)
        agent = table_method.table_method(
            alpha=alpha, gamma=gamma, render=False, save=True, verbose=False
        )
        agent.play(env)

env.close()
