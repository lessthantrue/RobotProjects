import numpy as np
import math
from common import matrix_utils

shape = np.array([
    [0.5, 0, 1],
    [0, 0.5, 1],
    [-0.5, 0, 1],
    [0, -0.5, 1],
    [0.5, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [-1, 0, 1],
    [0, -1, 1],
    [1, 0, 1]
])

class EKF:
    def __init__(self, initialM):
        self.mu = np.array(initialM, dtype='float64')
        self.cov = np.zeros((3, 3))
        self.color = (0, 255, 0)

    def predict(self, v, w, dt, processCov):
        dx = v * math.cos(self.mu[2]) * dt
        dy = v * math.sin(self.mu[2]) * dt
        dth = w * dt
        self.mu += np.array([dx, dy, dth])
        self.cov += processCov * dt * np.linalg.norm([v, w])

    def update(self, ws, Z, measureCov):
        dx = self.mu[0] - ws.bX
        dy = self.mu[1] - ws.bY
        dTotal = dx * dx + dy * dy

        # dh/dx
        C = np.array([
            [dx / math.sqrt(dTotal), dy / math.sqrt(dTotal), 0],
            [-dy / dTotal, dx/dTotal, -1],
            [0, 0, 1]
        ])

        # h(xhat)
        H = np.array([
            math.sqrt(dTotal),
            math.pi - self.mu[2] + math.atan2(dy, dx),
            self.mu[2]
        ])

        K = self.cov @ C.T @ np.linalg.inv(C @ self.cov @ C.T + measureCov)

        err = Z - H

        # fix stupid radian wrapping stuff
        if abs(err[1]) > np.pi:
            err[1] = -(np.sign(err[1]) * np.pi * 2 - err[1])

        self.mu += K @ (err)
        self.cov = (np.eye(3) - (K @ C)) @ self.cov

    def getPoints(self):
        cov_truncated = np.copy(self.cov)
        cov_truncated[2] = np.array([0, 0, 1])
        trans = matrix_utils.translation2d(self.mu[0], self.mu[1])
        rot = matrix_utils.rotation2d(self.mu[2])
        return matrix_utils.tfPoints(shape, trans @ cov_truncated @ rot)