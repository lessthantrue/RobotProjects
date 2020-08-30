import numpy as np

stdev_ctrl = 0.03
cov_sens = [
    [ 0.01**2, 0, 0 ],
    [ 0, 0.1**2, 0 ],
    [ 0, 0, 0 ]
]

class Motor():
    def __init__(self, tStall, sFree, iStall, iFree, friction=0):
        self.K = tStall / iStall # motor constant
        self.R = self.K * iFree / sFree # motor resistance

        self.B = 0.1 # viscous friction in N*s/m
        self.L = 0.001 # inductance assumed to be 1 mH

        self.t = 0.0 # position
        self.w = 0.0 # speed
        self.i = 0.0 # current
        self.a = 0.0 # acceleration
        
    def getA(self, inertia):
        return np.array([
            [ 0, 1, 0 ],
            [ 0, -self.B / inertia, self.K / inertia ],
            [ 0, self.R / self.L, -self.K / self.L ]
        ])

    def getB(self):
        return np.array([
            [ 0 ],
            [ 0 ],
            [ 1 / self.L ]
        ])

    def getStateVector(self):
        return np.array([self.t, self.w, self.i])
        
    def getMeasurement(self):
        # rounding position better simulates an encoder
        return np.array([round(self.t, 4), self.w, self.i]) + np.random.multivariate_normal(np.zeros(3), cov_sens)

    def act(self, voltage, dt, inertia=1):
        voltage += np.random.normal(0, stdev_ctrl)
        X = self.getStateVector()
        dX = self.getA(inertia) @ X + self.getB() @ np.array([voltage])
        X2 = X + dX * dt 
        dX2 = self.getA(inertia) @ X2 + self.getB() @ np.array([voltage])
        X += (dX + dX2) * dt / 2
        self.t = X[0]
        self.w = X[1]
        self.i = X[2]
        self.a = (dX[1] + dX2[1]) / 2

    def analyticalSpeedAt(self, voltage):
        return voltage / (self.B - self.R)