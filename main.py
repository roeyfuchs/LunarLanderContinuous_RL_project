import argparse
import os
import sys
import time
from queue import PriorityQueue

import gym
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide debug msgs
from ddqn import DDQN
from dqn_prioritized_experience import DQNPrioritizedExperience
from dqn_target_network import DQNTargetNetwork
from sarsa import SARSA
from simple_dqn import SimpleDQN
from uncertainty_env import uncertainty_env
import utils

BASE_SAVING_PATH = "graphs"
os.makedirs(
    BASE_SAVING_PATH, exist_ok=True
)  # make sure that we have a directory to save
FILE_SAVEING_TYPE = "png"


alogs_dict = {
    "sarsa": SARSA,
    "simple_dqn": SimpleDQN,
    "ddqn": DDQN,
    "dqn_p": DQNPrioritizedExperience,
    "dqn_target": DQNTargetNetwork,
}

def update_parser_options(parser):
    parser.add_argument(
        "-v",
        action="store_true",
        help="verbose. if set will print out information and graph",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        help="Render. if set will render the lunar lander environment",
    )
    parser.add_argument(
        "-n",
        action="store_true",
        default=False,
        help="Noise. if set will add noise to the environment",
    )
    parser.add_argument(
        "agent", choices=[i for i in alogs_dict], help="which agent to use"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    update_parser_options(parser)
    args = parser.parse_args()

    if args.n:
        env = uncertainty_env(gym.make("LunarLanderContinuous-v2"))
    else:
        env = gym.make("LunarLanderContinuous-v2")

    utils.set_verbose(args.v)
    print(alogs_dict[args.agent])
    print(args)

    env.seed(0)
    np.random.seed(0)
    agent = alogs_dict[args.agent](utils.printout, args.r)
    rewards = agent.solve_env(env)

    filename = time.strftime("%H-%M-%S", time.localtime())
    file_path = os.path.join(BASE_SAVING_PATH, filename)
    plt.xlabel("Number Episode")
    plt.ylabel("Score Per Episode")
    plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
    plt.savefig(".".join([file_path, FILE_SAVEING_TYPE]))
    plt.show()
