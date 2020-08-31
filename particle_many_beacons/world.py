import math
import numpy as np
import pygame
from common import matrix_utils

minDist = 2

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
            for pt in self.points:
                if np.linalg.norm(pt - p) < minDist:
                    break
            else:
                self.points.append(p)

    def getDrawnObjs(self):
        objs = []
        for p in self.points:
            objs.append(Beacon(p[0], p[1]))
        return objs

class Beacon():
    def __init__(self, x, y):
        trans = matrix_utils.translation2d(x, y)
        self.points = matrix_utils.tfPoints(bshape, trans)
        self.color = (128, 255, 0)

    def getPoints(self):
        return self.points