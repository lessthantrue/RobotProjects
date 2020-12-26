import numpy as np
import pygame
from common import matrix_utils

cameraPtShape = np.array([
    [-1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1]
]) * np.array([0.15, 0.15, 1])

class Camera():
    def __init__(self, _range, fov, eframe, rframe, npts):
        self.range = _range # avoiding python keywords
        self.minrange = _range / 5
        self.fov = fov
        self.cov = np.array([
            [0.2, 0],
            [0, 10]
        ]) * 0.003
        self.fail_rate = 0.25
        self.eframe = eframe
        self.rframe = rframe

        fovrad = np.deg2rad(self.fov)
        self.shape = np.array([
            [np.cos(fovrad / 2) * self.minrange, np.sin(fovrad / 2) * self.minrange, 1],
            [self.minrange, 0, 1],
            [np.cos(fovrad / 2) * self.minrange, -np.sin(fovrad / 2) * self.minrange, 1],
            [np.cos(fovrad / 2) * self.range, -np.sin(fovrad / 2) * self.range, 1],
            [self.range, 0, 1],
            [np.cos(fovrad / 2) * self.range, np.sin(fovrad / 2) * self.range, 1],
            [np.cos(fovrad / 2) * self.minrange, np.sin(fovrad / 2) * self.minrange, 1]

        ])

        self.color = (255, 0, 255)

        self.objs = []
        for _ in range(npts):
            self.objs.append(CameraPoint())

    # gets all points in range of the sensor
    # in the form of distance and angle from
    # robot heading
    def getSeenPoints(self, world, posn, head):
        fovrad = np.deg2rad(self.fov)
        toRet = []

        for o in self.objs:
            o.setOffscreen()

        objIdx = 0

        for b in world.points:
            p = b.posn
            disp = p - posn
            th = np.arctan2(disp[1], disp[0])
            # get angle delta to point
            thDelt = th - head
            thDelt %= np.pi * 2        
            if abs(thDelt) > np.pi:
                thDelt -= np.pi * 2

            # test angle and distance in range and not random fail
            dist = np.linalg.norm(disp)
            if abs(thDelt) < fovrad / 2 and dist < self.range and dist > self.minrange and np.random.rand() > self.fail_rate:
                toRet.append(np.random.multivariate_normal(np.array([thDelt, dist]), self.cov))
                self.objs[objIdx].setLocation(dist, thDelt, self.eframe())
                self.objs[objIdx].setColor(b.color)
                objIdx += 1

        return toRet

    # gets all points in the world relative to a position and heading
    def getPointsPerfect(self, world, posn, head):
        toRet = []
        for p in world.points:
            disp = p - posn
            th = np.arctan2(disp[1], disp[0])
            # get angle delta to point
            thDelt = th - head
            thDelt %= np.pi * 2        
            if abs(thDelt) > np.pi:
                thDelt -= np.pi * 2

            # test angle and distance in range and not random fail
            dist = np.linalg.norm(disp)
            toRet.append(np.array([thDelt, dist]))

        return toRet

    def getPoints(self):
        return matrix_utils.tfPoints(self.shape, self.rframe())

    def getDrawnObjs(self):
        return self.objs

class CameraPoint():
    def __init__(self):
        self.setOffscreen()
        self.color = (0, 0, 255)

    def setColor(self, c):
        self.color = c

    def setLocation(self, d, th, cFrame):
        trans = matrix_utils.translation2d(d, 0)
        rot = matrix_utils.rotation2d(th)
        self.points = matrix_utils.tfPoints(cameraPtShape, cFrame @ rot @ trans)

    def setOffscreen(self):
        trans = matrix_utils.translation2d(30, 30)
        self.points = matrix_utils.tfPoints(cameraPtShape, trans)

    def getPoints(self):
        return self.points