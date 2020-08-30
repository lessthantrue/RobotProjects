import numpy as np

class MotionModel():
    def __init__(self):
        defVal = 1
        self.a1 = defVal
        self.a2 = defVal
        self.a3 = defVal
        self.a4 = defVal
        self.a5 = defVal
        self.a6 = defVal        

    def simpleTriDist(self, val):
        return np.random.triangular(-val, np.zeros(len(val)), val)

    def setParams(self, a1, a2, a3, a4, a5, a6):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6

    def getNoisyDiff(self, v, w, dt, th):
        ctrl = np.array([abs(v), abs(w)])
        if np.linalg.norm(ctrl) == 0:
            return np.zeros(3)

        params1 = np.array([self.a1, self.a2])
        params2 = np.array([self.a3, self.a4])
        params3 = np.array([self.a5, self.a6])

        vh, wh, gh = self.simpleTriDist(np.array([params1 @ ctrl, params2 @ ctrl, params3 @ ctrl]))

        vh += v
        wh += w

        # vh = v + self.simpleTriDist(self.a1 * abs(v) + self.a2 * abs(w))
        # wh = w + self.simpleTriDist(self.a3 * abs(v) + self.a4 * abs(w))
        # gh = self.simpleTriDist(self.a5 * abs(v) + self.a6 * abs(w))

        if wh == 0:
            return np.array([0, 0, 0])
        else:
            vw = vh / wh

        x = -vw * np.sin(th) + vw * np.sin(th + wh * dt)
        y = vw * np.cos(th) - vw * np.cos(th + wh * dt)
        t = wh * dt + gh * dt

        return np.array([x, y, t])

inst = MotionModel()
def getInst():
    return inst
