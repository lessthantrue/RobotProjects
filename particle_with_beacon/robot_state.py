import numpy as np
import math
from common import matrix_utils

shape = np.array([
    [0, 0, 1],
    [0.5, 0, 1],
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0.5, 0, 1]
])

class Robot():
    def __init__(self, startX, startY, startT):
        self.x = startX
        self.y = startY
        self.t = startT
        self.processCov = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.color = (0, 0, 0)
        # self.processCov = np.zeros((3, 3))
    
    def getNoisyAct(self, v, w):
        dx = v * math.cos(self.t) 
        dy = v * math.sin(self.t)
        dt = w
        actVector = np.array([dx, dy, dt])
        return np.random.multivariate_normal(actVector, self.processCov * np.linalg.norm(actVector))

    def act(self, v, w, dt):
        self.x += v * math.cos(self.t) * dt
        self.y += v * math.sin(self.t) * dt
        self.t += w * dt

    def noisyAct(self, v, w, dt):
        noisyAction = self.getNoisyAct(v, w)
        self.x += noisyAction[0] * dt
        self.y += noisyAction[1] * dt
        self.t += noisyAction[2] * dt

    def getStateVector(self):
        return np.array(self.x, self.y, self.t)

    def getFrame(self):
        return matrix_utils.translation2d(self.x, self.y) @ matrix_utils.rotation2d(self.t)

    def getPoints(self):
        return matrix_utils.tfPoints(shape, self.getFrame())