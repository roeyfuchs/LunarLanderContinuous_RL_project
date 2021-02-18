import os

os.environ["LANG"] = "en_US"
import gym

import table_method

env = gym.make("LunarLanderContinuous-v2")


agent = table_method.table_method()
agent.train(env)
observation = env.reset()
for _ in range(1000):
    env.render()
    # action = env.action_space.sample()
    action = agent.get_action(observation)
    observation, reward, done, info = env.step(action)
    print("action:", action)
    print("obs:", agent.discrete_obs(observation))
    if done:
        print("done")
        observation = env.reset()
env.close()
