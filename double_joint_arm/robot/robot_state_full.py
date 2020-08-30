import numpy as np
from .matrix_utils import translation2d, rotation2d
from .motor_dynamics import Motor
import control
import control.matlab

def close(v1, v2, tol=0.0001):
    return abs(np.linalg.norm(v1-v2)) < tol

class Robot():
    def __init__(self, t1_0, t2_0, x0=1, y0=1, vbus=12):
        # parameters (meters)
        self.len1 = 0.3
        self.len2 = 0.6

        self.pen = True

        # frames 
        self.Rw_0 = translation2d(x0, y0)
        self.R0_1_base = translation2d(self.len1, 0)
        self.R1_2_base = translation2d(self.len2, 0)
        self.R0_1 = rotation2d(t1_0) @ self.R0_1_base
        self.R1_2 = rotation2d(t2_0) @ self.R1_2_base

        # Andymark Neverest motor specs
        tStall = 350 * 0.0070615518333333 # oz*in -> n*m
        iStall = 11.25
        iFree = 0.4
        sFree = 160 * (2*np.pi/60) # rot / min -> rad / sec
        # friction = tStall / 10 # lb-ft (unused)
        self.vbus = vbus
        self.m1 = Motor(tStall, sFree, iStall, iFree)
        self.m2 = Motor(tStall, sFree, iStall, iFree)

        # initial conditions
        self.x0, self.y0 = x0, y0
        self.m1.t, self.m2.t = t1_0, t2_0

    def act(self, c1, c2, dt):
        # clamp control signals to -1, 1
        c1 = max(-1, min(1, c1))
        c2 = max(-1, min(1, c2))

        self.m1.act(c1 * self.vbus, dt)
        self.m2.act(c2 * self.vbus, dt)

        # update frames
        self.R0_1 = rotation2d(self.m1.t) @ self.R0_1_base
        self.R1_2 = rotation2d(self.m2.t) @ self.R1_2_base

    def getStateVector(self):
        return np.array([self.m1.t, self.m2.t, self.m1.w, self.m2.w])

    def getMeasurement(self):
        m1_m, m2_m = self.m1.getMeasurement(), self.m2.getMeasurement()
        return np.array([m1_m[0], m2_m[0], m1_m[1], m2_m[1]])

    def getVelVector(self):
        return np.array([self.m1.w, self.m2.w, self.m1.a, self.m2.a])

    # gets joints in world frame
    def getJoints(self):
        q0 = self.Rw_0 @ np.array([0, 0, 1])
        q1 = self.Rw_0 @ self.R0_1 @ np.array([0, 0, 1])
        q2 = self.Rw_0 @ self.R0_1 @ self.R1_2 @ np.array([0, 0, 1])

        return (q0, q1, q2)