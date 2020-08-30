import numpy as np
import pygame

class Camera():
    def __init__(self, _range, fov):
        self.range = _range # avoiding python keywords
        self.minrange = _range / 5
        self.fov = fov
        self.cov = np.array([
            [0.2, 0.1],
            [0.1, 10]
        ]) * 0.003
        self.fail_rate = 0.25

    # gets all points in range of the sensor
    # in the form of distance and angle from
    # robot heading
    def getPoints(self, world, posn, head):
        fovrad = np.deg2rad(self.fov)
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
            if abs(thDelt) < fovrad / 2 and dist < self.range and dist > self.minrange and np.random.rand() > self.fail_rate:
                toRet.append(np.random.multivariate_normal(np.array([thDelt, dist]), self.cov))

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

    def draw(self, surf, color, transform, posn, head):
        # posn = transform(posn[0], posn[1])
        fovrad = np.deg2rad(self.fov)
        shape = [
            np.array([np.cos(head - fovrad / 2) * self.minrange, np.sin(head - fovrad / 2) * self.minrange]),
            np.array([np.cos(head) * self.minrange, np.sin(head) * self.minrange]),
            np.array([np.cos(head + fovrad / 2) * self.minrange, np.sin(head + fovrad / 2) * self.minrange]),
            np.array([np.cos(head + fovrad / 2) * self.range, np.sin(head + fovrad / 2) * self.range]),
            np.array([np.cos(head) * self.range, np.sin(head) * self.range]),
            np.array([np.cos(head - fovrad / 2) * self.range, np.sin(head - fovrad / 2) * self.range])
        ]

        for i in range(len(shape)):
            shape[i] = transform(shape[i][0] + posn[0], shape[i][1] + posn[1])

        pygame.draw.lines(
            surf,
            color,
            True,
            shape
        )
