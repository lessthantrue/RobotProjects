import numpy as np
from common import matrix_utils

shape = np.array([
    [1, 0, 1],
    [0, -1, 1],
    [-1, 0, 1],
    [0, 1, 1],
    [1, 0, 1]
]) * np.array([0.2, 0.2, 1])

class World():
    def __init__(self, beaconX, beaconY):
        self.bX = beaconX
        self.bY = beaconY
        self.color = (255, 0, 0)

    def getPoints(self):
        tf = matrix_utils.translation2d(self.bX, self.bY)
        return matrix_utils.tfPoints(shape, tf)