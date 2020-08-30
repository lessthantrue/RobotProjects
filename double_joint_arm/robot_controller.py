import numpy as np
from controllers import state_space
from controllers import velocity_regulator
from paths import circle
import kinematics
from profile import profile

class Controller():
    def __init__(self, len1, len2):
        self.len1 = len1
        self.len2 = len2
        self.tau = 0.2 # amount of time to generate in a profile
        self.t = 0 # time
        self.tmax = self.tau
        self.path = None

    def initMotorSpeedControllers(self, m1, m2):
        # self.initStateSpaceControl(m1, m2)
        self.initPidControl(m1, m2)

    def initStateSpaceControl(self, m1, m2):
        evals = [-0.1, -0.2, -0.3]
        self.vm1 = state_space.StateSpace(m1.getA(1), m1.getB(), evals)
        self.vm2 = state_space.StateSpace(m2.getA(1), m2.getB(), evals)
        self.m1 = m1
        self.m2 = m2 # it's bad but I'm doing it

    def initPidControl(self, m1, m2):
        kfv1 = (1 / m1.analyticalSpeedAt(12))
        kfv2 = (1 / m2.analyticalSpeedAt(12))
        # kfv1, kfv2 = 0, 0
        kfa = 0
        kp = 3
        kd = 0.1
        kv = 1
        self.vm1 = velocity_regulator.VelRegulator(kp, kfv1, kfa, kd, kv)
        self.vm2 = velocity_regulator.VelRegulator(kp, kfv2, kfa, kd, kv)

    def setPath(self, path, robot):
        self.path = path
        self.t = 0
        self.tmax = 0
        self.genMorePath(robot)

    def genMorePath(self, robot):
        pfn =   lambda t: self.path.getPathParams(t)[0:2] - (robot.Rw_0 @ np.array([0, 0, 1]))[0:2]
        dpfn =  lambda t: self.path.getPathParams(t)[2:4]
        d2pfn = lambda t: self.path.getPathParams(t)[4:6]
        qfn =   lambda x: kinematics.forward_state(x[0], x[1], robot.len1, robot.len2)
        jfn =   lambda x: kinematics.forward_jacobian(x[0], x[1], robot.len1, robot.len2)
        hfn =   lambda x: kinematics.forward_hessian(x[0], x[1], robot.len1, robot.len2)
        t1, t2, _, _ = robot.getStateVector()
        self.profile = profile.Profile(pfn, dpfn, d2pfn, self.tau, t0=self.t)
        self.profile.transform(np.array([t1, t2]), qfn, jfn, hfn)
        self.tmax += self.tau

    def getActuation(self, robot, dt):
        # _, _, q2 = robot.getJoints()
        q, v, a, _ = self.profile.get(self.t)

        # state space stuff below
        # inertia assumed to be 1
        # i1 = (1/self.m1.K) * (a[0] + self.m1.B * v[0])
        # i2 = (1/self.m2.K) * (a[1] + self.m2.B * v[1])

        # g1 = np.array([q[0], v[0], i1])
        # g2 = np.array([q[1], v[1], i2])

        # u1 = self.vm1.get(robot.m1.getStateVector(), g1)[0] / 12
        # u2 = self.vm2.get(robot.m2.getStateVector(), g2)[0] / 12

        # PID stuff below
        t1, t2, w1, w2 = robot.getMeasurement()

        # print(q, v, a)

        u1 = self.vm1.get(t1, q[0], w1, v[0], a[0])
        u2 = self.vm2.get(t2, q[1], w2, v[1], a[1])

        self.t += dt
        if self.t > self.tmax:
            self.genMorePath(robot)

        return [u1, u2]

    def done(self):
        return self.path is None or self.t > self.path.time * 1.1 # some extra time to settle