import numpy as np
import math
from common import matrix_utils

CameraCov = np.array([
    [1, 0],
    [0, 1]
])
GyroCov = np.array([[np.pi / 180]])

cameraShape = np.array([
    [0, 0, 1],
    [1, 0, 1]
])

class CameraServo():
    # rate is number of readings per second
    def __init__(self, rate, parentFrame):
        self.rate = int(1000 / rate)
        self.time = rate
        self.lastReading = np.zeros(2)
        self.hasNewReading = False
        self.parentFrame = parentFrame
        self.color = (0, 0, 255)

    def step(self, rs, ws, dt):
        self.time -= dt
        if self.time < 0:
            self.time = self.rate
            dx = rs.x - ws.bX
            dy = rs.y - ws.bY
            dt = math.pi - rs.t + math.atan2(dy, dx)
            self.lastReading = np.array([math.sqrt(dx * dx + dy * dy), dt])
            self.hasNewReading = True

    def getAbsoluteReading(self):
        return self.lastReading

    def getNoisyReading(self):
        return np.random.multivariate_normal(self.lastReading, CameraCov * 0.5)

    def getPoints(self):
        tf = matrix_utils.rotation2d(self.lastReading[1]) @ matrix_utils.scale2d(self.lastReading[0], 0)
        return matrix_utils.tfPoints(cameraShape, self.parentFrame() @ tf)

class Gyroscope():
    def __init__(self, rate):
        self.rate = (1000 / rate)
        self.time = rate
        self.lastReading = np.zeros(1)
        self.hasNewReading = False

    def step(self, rs, ws, dt):
        self.time -= dt
        if self.time < 0:
            self.time = self.rate
            self.lastReading = np.array([rs.t])
            self.hasNewReading = True

    def getAbsoluteReading(self):
        return self.lastReading

    def getNoisyReading(self):
        return np.random.multivariate_normal(self.lastReading, GyroCov)