import os
import time

import matplotlib.pyplot as plt
import numpy as np

BASE_SAVEING_PATH = "graphs"
os.makedirs(
    BASE_SAVEING_PATH, exist_ok=True
)  # make sure that we have a directory to save
FILE_SAVEING_TYPE = "png"


class Plot:
    def __init__(self, title, xlabel, ylabel, verbose=True, win=1):
        self.fig = plt.figure()
        self.line = plt.plot([], [])[0]
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.x = []
        self.y = []
        self.xmean = []
        self.ymean = []
        self.ax = self.fig.get_children()[1]
        self.win = win

        if verbose:
            plt.show(block=False)

    def add_point(self, xP, yP):
        self.x.append(xP)
        self.y.append(yP)
        if len(self.xmean) > 0 or len(self.x) == self.win:
            if len(self.xmean) > 0:
                i = self.xmean[-1] + 1
            else:
                i = self.win
            self.xmean.append(i)
            self.ymean.append(np.mean(self.y[i - self.win : i]))

        self.line.set_xdata(self.xmean)
        self.line.set_ydata(self.ymean)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        self.fig.canvas.flush_events()

    def save(self, filename=None):
        if not filename:
            filename = time.strftime("%H-%M-%S", time.localtime())
        file_path = os.path.join(BASE_SAVEING_PATH, filename)
        self.fig.savefig(".".join([file_path, FILE_SAVEING_TYPE]))
