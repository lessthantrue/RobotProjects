import numpy as np
import kinematics
from paths import circle
from profile import profile

class Controller():
    def __init__(self, len1, len2):
        self.l1 = len1
        self.l2 = len2
        self.t = 0
        self.tau = 0.015
        self.tmax = 0
    
    def initMotorSpeedControllers(self, m1, m2):
        pass

    def initCirclePath(self, xc, yc, r, robot):
        self.path = circle.Circle(xc, yc, r)
        self.genMorePath(robot)

    def genMorePath(self, robot):
        pfn =   lambda t: self.path.getPathParams(t)[0:2]
        dpfn =  lambda t: self.path.getPathParams(t)[2:4]
        d2pfn = lambda t: self.path.getPathParams(t)[4:6]
        jfn = lambda x: kinematics.forward_jacobian(x[0], x[1], robot.len1, robot.len2)
        hfn = lambda x: kinematics.forward_hessian(x[0], x[1], robot.len1, robot.len2)
        t1, t2, _, _ = robot.getStateVector()
        self.profile = profile.Profile(pfn, dpfn, d2pfn, self.tau, t0=self.t)
        self.profile.transform(np.array([t1, t2]), jfn, hfn)
        self.tmax += self.tau

    def getActuation(self, robot, dt):
        t1, t2, _, _ = robot.getStateVector()
        _, v, _, _ = self.profile.get(self.t)

        # J = kinematics.forward_jacobian(t1, t2, robot.len1, robot.len2)
        # print(J @ v)

        self.t += dt
        if self.t > self.tmax:
            print("Making more path")
            self.genMorePath(robot)

        return v