import numpy as np
import pygame
import motion
from common import matrix_utils

robot_shape = np.array([ 
    [1, 0, 1], 
    [-1, -0.5, 1], 
    [-1, 0.5, 1],
    [1, 0, 1]
]) * np.array([0.25, 0.25, 1])

class Robot():
    def __init__(self, startX, startY, startT):
        self.x = startX
        self.y = startY
        self.t = startT
        self.color = (0, 0, 255)

    def act(self, v, w, dt):
        tNext = self.t + w * dt
        self.x += v * (np.cos(self.t) + np.cos(tNext)) * dt / 2
        self.y += v * (np.sin(self.t) + np.sin(tNext)) * dt / 2
        self.t = tNext

    def actNoisy(self, v, w, dt):
        diff = motion.getInst().getNoisyDiff(v, w, dt, self.t)
        # print(diff)
        self.x += diff[0]
        self.y += diff[1]
        self.t += diff[2]

    def getStateVector(self):
        return np.array([self.x, self.y, self.t])
    
    def getFrame(self):
        return matrix_utils.translation2d(self.x, self.y) @ matrix_utils.rotation2d(self.t)

    def getPoints(self):
        return matrix_utils.tfPoints(robot_shape, self.getFrame())