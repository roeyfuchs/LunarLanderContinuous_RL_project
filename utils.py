import sys
import os

main_engine_values = [0, 0.5, 1]
sec_engine_values = [-1, -0.75, 0, 0.75, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
episodes = 1000
updateTargetNetwork = 50
state_space = 8

output = sys.stdout


def set_verbose(v):
    global output
    if v:
        output = sys.stdout
    else:
        output = open(os.devnull, "w")


def printout(str):
    global output
    print(str, file=output)






