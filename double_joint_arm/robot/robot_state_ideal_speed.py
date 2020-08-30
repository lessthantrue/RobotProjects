import numpy as np
import kinematics
from matrix_utils import *

def close(v1, v2, tol=0.0001):
    return abs(np.linalg.norm(v1-v2)) < tol

class Robot():
    def __init__(self, t1_0, t2_0, x0=1, y0=1):
        self.len1 = 1
        self.len2 = 2
        
        # frames 
        self.Rw_0 = translation2d(x0, y0)
        self.R0_1_base = translation2d(self.len1, 0)
        self.R1_2_base = translation2d(self.len2, 0)
        self.R0_1 = rotation2d(t1_0) @ self.R0_1_base
        self.R1_2 = rotation2d(t2_0) @ self.R1_2_base

        self.x0, self.y0 = x0, y0
        self.t1, self.t2 = t1_0, t2_0
        self.w1, self.w2 = 0, 0

    # c1, c2 = w1, w2
    def act(self, c1, c2, dt):
        self.t1 += (self.w1 + c1) / 2 * dt
        self.t2 += (self.w2 + c2) / 2 * dt

        self.w1 = c1
        self.w2 = c2

        self.R0_1 = rotation2d(self.t1) @ self.R0_1_base
        self.R1_2 = rotation2d(self.t2) @ self.R1_2_base

    def getStateVector(self):
        return np.array([self.t1, self.t2, self.w1, self.w2])

    def getVelVector(self):
        return np.array([self.w1, self.w2, 0, 0])

    def getJoints(self):
        q0 = self.Rw_0 @ np.array([0, 0, 1])
        q1 = self.Rw_0 @ self.R0_1 @ np.array([0, 0, 1])
        q2 = self.Rw_0 @ self.R0_1 @ self.R1_2 @ np.array([0, 0, 1])

        return (q0, q1, q2)