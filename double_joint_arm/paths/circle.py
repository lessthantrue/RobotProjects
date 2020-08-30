import numpy as np

class Circle():
    def __init__(self, xc, yc, r, w0, a=0, wf=None):
        self.xc = xc
        self.yc = yc
        self.r = r
        self.w0 = w0 # radians/sec
        self.a = a
        self.wf = w0 if wf is None else wf

    def getPathParams(self, t):
        if self.w0 + self.a * t <= self.wf:
            th = (self.w0 + self.a * t / 2) * t
            w = self.w0 + self.a * t
            dw = self.a
        else:
            phi = (self.wf * self.wf - self.w0 * self.w0) / (2 * self.a)
            th = self.wf * t - phi
            w = self.wf
            dw = 0

        x = self.r * np.cos(th) + self.xc
        y = self.r * np.sin(th) + self.yc
        dx = self.r * (w * -np.sin(th))
        dy = self.r * (w * np.cos(th))
        d2x = self.r * (dw * -np.sin(th) + w * w * -np.cos(th))
        d2y = self.r * (dw * np.cos(th) + w * w * -np.sin(th))

        return np.array([x, y, dx, dy, d2x, d2y])

