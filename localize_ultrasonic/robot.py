import numpy as np
import pygame
import sensor, motion, estimator, regulator
from common.matrix_utils import *

reg_kp = np.array([1, 1, 0.8])
reg_kf = np.array([0.0, 0.0, 0.0])

class Robot():
    def __init__(self, startV):
        self.state = np.array(startV, dtype="float64")
        self.color = (255, 0, 255)
        self.us1 = sensor.Ultrasonic(rotation2d(np.pi / 3) @ translation2d(0.3, 0), self)
        self.us2 = sensor.Ultrasonic(rotation2d(np.pi) @ translation2d(0.3, 0), self)
        self.us3 = sensor.Ultrasonic(rotation2d(np.pi * 5 / 3) @ translation2d(0.3, 0), self)
        self.ekf = estimator.EKF(self.state, self.us1.frame, self.us2.frame, self.us3.frame)

        ctv = 0.5 / np.cos(np.pi / 6)
        ctm = ctv *  np.sin(np.pi / 6)
        self.shape = np.array([
            [ctv, 0, 1],
            [-ctm, 0.5, 1],
            [-ctm, -0.5, 1],
            [ctv, 0, 1],
            [0, 0, 1]
        ])

    def init_feedback(self):
        self.ctrl = regulator.Regulator(reg_kp, reg_kf)
        self.ctrl.setGoal(np.zeros(3))

    def setWaypoint(self, w):
        self.ctrl.setGoal(w)

    def t(self):
        return self.state[2]

    def x(self):
        return self.state[0]

    def y(self):
        return self.state[1]

    def act(self, act, dt):
        self.state = motion.move_robot(self.state, act, dt)

    def autoAct(self, ff, dt):
        fb = self.ctrl.get(self.ekf.est)
        fb = rotation2d(-self.ekf.est[2]) @ fb
        act = fb + rotation2d(-self.ekf.est[2]) @ ff
        np.clip(act, -1, 1)
        self.act(act, dt)
        return act

    def getDrawnObjects(self):
        return [self.us1, self.us2, self.us3, self.ekf] + self.us1.rays + self.us2.rays + self.us3.rays

    def getPoints(self):
        return tfPoints(self.shape, self.getFrame())

    def getEstPts(self):
        tf = self.ekf.getFrame()
        points = []
        for i in range(len(self.shape)):
            points.append(np.array(tf @ self.shape[i])[0:2])

        return points

    def getSegments(self):
        pts = self.getPoints()
        segs = []
        for i in range(len(pts) - 2):
            segs.append([pts[i], pts[i+1]])
        return segs

    def getEstSegments(self):
        pts = self.getEstPts()
        segs = []
        for i in range(len(pts) - 2):
            segs.append([pts[i], pts[i+1]])
        return segs

    def getFrame(self):
        return motion.getFrame(self.state)

    def stepEkf(self, act, segs, rsegs, esegs, dt):
        self.ekf.act(act, dt / 1000.0)
        if np.linalg.norm(act) > 0.01 and np.random.rand() < 0.1:
            ekfFrame = self.ekf.getFrame()
            robFrame = self.getFrame()
            sas, mas, dSens, dEst = [], [], [], []
            for us in [self.us1, self.us2, self.us3]:
                (_, as1, ar1, de1) = us.getModelOutput(segs + esegs, ekfFrame)                
                (_, _, _, ds1) = us.getModelOutput(segs + rsegs, robFrame)
                sas.append(ar1)
                mas.append(as1)
                dEst.append(de1)
                dSens.append(ds1)

            self.ekf.sense(np.array(dSens), np.array(dEst), np.array(sas), np.array(mas))

if __name__ == "__main__":
    r = Robot([0, 0, 0])
    print(r.getPoints())
    r.act(1, 1, 1, 1)
    print(r.getPoints())