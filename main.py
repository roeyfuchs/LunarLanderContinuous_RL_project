import os

from deep_reinforce import reinforce

os.environ["LANG"] = "en_US"
import gym
from table_method import table_method

env = gym.make("LunarLanderContinuous-v2")
env.seed(42)
agent = table_method(render=False, save=True)
agent.play(env)
# lr = [0.1, 0.01, 0.001, 0.0001]
# al = [0.35, 0.7, 0.9, 1]
# gm = [0.35, 0.7, 0.99]
# for learn in lr:
#    for a in al:
#        for g in gm:
#            agent = reinforce(
#                env, render=False, save=True, alpha=a, gamma=g, lr=learn, verbose=False
#            )
#            print(learn, a,g)
#            agent.train(600, 1)
env.close()
