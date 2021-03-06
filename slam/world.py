import math
import numpy as np
import pygame
from common import matrix_utils

minDist = 2
nClasses = 3

colors = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255)
]

bshape = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [-1, 0, 1],
    [0, -1, 1],
    [1, 0, 1]
]) * np.array([0.25, 0.25, 1])

# defines the points of interest that can be detected by the sensor
class World():
    def __init__(self, nPoints):
        self.points = []
        while(len(self.points) < nPoints):
            p = (np.random.rand(2) + np.array([-0.5, -0.5])) * 10
            for b in self.points:
                if np.linalg.norm(b.posn - p) < minDist:
                    break
            else:
                self.points.append(Beacon(p[0], p[1]))

    def getDrawnObjs(self):
        return self.points

class Beacon():
    def __init__(self, x, y):
        trans = matrix_utils.translation2d(x, y)
        self.posn = np.array([x, y])
        self.points = matrix_utils.tfPoints(bshape, trans)
        self.colorClass = np.random.random_integers(0, nClasses-1)
        self.color = colors[self.colorClass]

    def getPoints(self):
        return self.points