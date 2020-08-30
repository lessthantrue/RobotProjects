import numpy as np
import math
from common import matrix_utils

sensorCov = np.array([
    [0.001, 0, 0],
    [0, 0.001, 0],
    [0, 0, 0.001]
])
# sensorCov = np.zeros((2, 2))

class CameraServo():
    def __init__(self):
        self.lastReading = [0, 0]

    def getAbsoluteReading(self, rs, ws):
        dx = rs.x - ws.bX
        dy = rs.y - ws.bY
        dt = math.pi - rs.t + math.atan2(dy, dx)
        return (math.sqrt(dx * dx + dy * dy), dt, rs.t)

    def getNoisyReading(self, rs, ws):
        d, tr, t = getAbsoluteReading(rs, ws)
        sensorVector = np.array([d, tr, t])
        return np.random.multivariate_normal(sensorVector, sensorCov)
