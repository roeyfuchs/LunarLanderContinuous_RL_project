import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, title, xlabel, ylabel):
        self.fig = plt.figure(5)
        self.line = plt.plot([], [])[0]
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.x = []
        self.y = []
        self.ax = self.fig.get_children()[1]

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
