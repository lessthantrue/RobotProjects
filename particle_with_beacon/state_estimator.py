import numpy as np
import math

class EKF:
    def __init__(self, initialM):
        self.mu = np.array(initialM, dtype='float64')
        self.cov = np.zeros((3, 3))

    def predict(self, v, w, dt, processCov):
        dx = v * math.cos(self.mu[2]) * dt
        dy = v * math.sin(self.mu[2]) * dt
        dth = w * dt
        self.mu += np.array([dx, dy, dth])
        self.cov += processCov * dt

    def update(self, ws, Z, measureCov):
        dx = self.mu[0] - ws.bX
        dy = self.mu[1] - ws.bY
        dTotal = dx * dx + dy * dy

        Q = measureCov
        C = np.array([
            [dx / math.sqrt(dTotal), dy / math.sqrt(dTotal), 0],
            [-dy / dTotal, dx/dTotal, -1],
            [0, 0, 1]
        ])
        H = np.array([
            math.sqrt(dTotal),
            math.pi - self.mu[2] + math.atan2(dy, dx),
            self.mu[2]
        ])

        K = self.cov @ C.T @ np.linalg.inv(C @ (self.cov @ C.T) + Q)
        self.mu += K @ (Z - H)
        self.cov = (np.eye(3) - (K @ C)) @ self.cov

