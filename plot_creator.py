import os
import time

import matplotlib.pyplot as plt

BASE_SAVEING_PATH = "graphs"
os.makedirs(
    BASE_SAVEING_PATH, exist_ok=True
)  # make sure that we have a directory to save
FILE_SAVEING_TYPE = "png"


class Plot:
    def __init__(self, title, xlabel, ylabel, verbose=True):
        self.fig = plt.figure(5)
        self.line = plt.plot([], [])[0]
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.x = []
        self.y = []
        self.ax = self.fig.get_children()[1]

        if verbose:
            plt.show(block=False)

    def add_point(self, xP, yP):
        self.x.append(xP)
        self.y.append(yP)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        self.fig.canvas.flush_events()

    def save(self, filename=None):
        if not filename:
            filename = time.strftime("%H-%M-%S", time.localtime())
        file_path = os.path.join(BASE_SAVEING_PATH, filename)
        plt.savefig(".".join([file_path, FILE_SAVEING_TYPE]))
