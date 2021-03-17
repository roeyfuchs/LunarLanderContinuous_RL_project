# Solving The Lunar Lander Continuous
This repo submitted as final project for Reinforcement Learning course (896873), Bar-ilan University, 2021.

We have implemented the following algorithms/methods:
1. SARSA.
2. Simple DQN.
3. DQN with \epsilon-greedy.
4. DQN with PER (prioritized experience replay).
5. DQN with target network.
6. Double DQN.

In addtion, its possible to run the environment with noises. The enviroment will return to the lander the regular observation, but will add Gaussian noises (mean = 0 and STD = 0.05) to the location (x and y)

## Getting Started
### Prerequisites
Please use python 3.6.
We have use basic python libraries excluding gym environment:

-NumPy
-TensorFlow
-Keras
-Matplotlib

You can run `pip -r requirements.txt` to install them.


### How To Run
Simply run the `main.py` file with the name of the algorithm that you prefer:
`sarsa`, `simple_dqn`, `dqn_eg` (DQN with epsilon-greedy), `dqn_target` (DQN with target network`, `dqn_p` (DQN with PER), `ddqn` (Double DQN)

### Flags
Run the program with  `-n` to add noises to the enviroment, `-v` (for verbose) to add information messages during the run and graph in the end, `-r` to render the enviroment during the run.


For example, to run DDQN, verbose and render, use the following command:
`python main.py -vr ddqn.

In any case, you can run `python main.py -h` for the help message.

## Acknowledgments
We use code from [https://github.com/shivaverma/OpenAIGym/tree/master/lunar-lander/discrete](this repo) and [https://www.geeksforgeeks.org/sarsa-reinforcement-learning/](this tutorial) as a skelaton for our code.

