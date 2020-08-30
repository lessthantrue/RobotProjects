import numpy as np

class Arc():
    def __init__(self, x0, y0, x1, y1, xc, yc, vmax, amax, cw=False):
        self.s0 = np.array([x0, y0])
        self.s1 = np.array([x1, y1])
        self.sc = np.array([xc, yc])

        self.th_0 = np.arctan2((self.s0-self.sc)[1], (self.s0-self.sc)[0])
        self.th_f = np.arctan2((self.s1-self.sc)[1], (self.s1-self.sc)[0])
        th_diff = self.th_f - self.th_0

        self.vmax = vmax
        self.amax = amax

        self.r = np.linalg.norm(self.s0 - self.sc)
        # assert abs(self.r - np.linalg.norm(self.s1 - self.sc)) < 0.001, "bad start/end/center points: {} != {}".format(self.r, np.linalg.norm(self.s1 - self.sc))
        
        wmax = vmax / self.r
        accelRadialMax = wmax * wmax * self.r

        # fix some clockwise/counterclockwise stuff
        if (cw and th_diff > 0) or ((not cw) and th_diff < 0):
            th_diff = (np.pi * 2 - th_diff)
            if cw:
                th_diff = -abs(th_diff)
            else:
                th_diff = abs(th_diff)

        if accelRadialMax >= amax * 3 / 4:
            # worst case scenario: centripetal acceleration at the current
            # angular speed is too fast. Decrease max angular speed.

            # need to assume a value for accelAngularMax in this case
            accelAngularMax = amax / 2

            wmax = np.power(amax**2 - accelAngularMax**2, 1/4) / np.sqrt(self.r)

            # recalculate acceleration values
            accelRadialMax = wmax * wmax * self.r
            accelMax = np.sqrt(accelAngularMax**2 + accelRadialMax**2)
            assert accelMax - amax < 0.00005, "Acceleration too high: {} > {}".format(accelMax, amax)

        # get max angular and radial acceleration along the path
        accelAngularMax = np.sqrt(amax**2 - accelRadialMax**2)
        alph = accelAngularMax / self.r

        # overapproximation of max acceleration along path
        accelMax = np.sqrt(accelAngularMax**2 + accelRadialMax**2)

        if accelMax > amax:
            # arc starting acceleration is too great
            alph = np.sqrt(amax**2 - accelRadialMax**2) / self.r

            # recalculate accelerations
            accelAngularMax = alph * self.r
            accelMax = np.sqrt(accelAngularMax**2 + accelRadialMax**2)

        assert accelMax - amax < 0.0005, "Acceleration too high: {} > {}".format(accelMax, amax)
        assert abs(wmax) * self.r - vmax < 0.0005, "Velocity too high: {} > {}".format(wmax * self.r, vmax)

        # set motion in the right direction
        if cw:
            alph = -abs(alph)
            wmax = -abs(wmax)
        else:
            alph = abs(alph)
            wmax = abs(wmax)

        assert(np.sign(alph) == np.sign(th_diff))
            
        self.wmax = wmax
        self.alph = alph

        # now start the trapezoidal profile
        self.astop = wmax / alph
        thAccel = self.astop * self.astop * alph / 2

        # if stopping doesn't leave enough distance to decellerate:
        if abs(thAccel) > abs(th_diff / 2):
            # stop accelerating at half distance
            self.astop = np.sqrt(abs(th_diff / alph))
            thAccel = self.astop * self.astop * alph / 2
            # adjust max speed
            assert self.alph * self.astop < self.wmax
            self.wmax = self.alph * self.astop        
            # in this case, decelleration needs to start immediately
            self.dstart = self.astop
            thcw = 0
        else:
            # find angle traversed and time spent at constant angular velocity
            thcw = th_diff - 2 * thAccel
            tcw = thcw / wmax
            # add starting time
            self.dstart = self.astop + tcw
        assert(abs(thcw + thAccel * 2 - th_diff) < 0.0001)

        self.time = self.dstart + self.astop

    def getPathParams(self, t):
        # get angular acceleration, velocity, position
        thAccum = 0
        if t < self.astop:
            thAccum = self.alph / 2 * t * t
            w = self.alph * t
            alph = self.alph
        elif t < self.dstart:
            tadj = t - self.astop
            thAccum = self.alph / 2 * self.astop * self.astop
            thAccum += self.wmax * (tadj)
            w = self.wmax
            alph = 0
        elif t < self.dstart + self.astop:
            tadj = t - self.dstart
            thAccum += self.alph / 2 * self.astop * self.astop
            thAccum += self.wmax * (self.dstart - self.astop)
            thAccum += self.wmax * tadj - self.alph / 2 * tadj * tadj
            w = self.wmax - tadj * self.alph
            alph = -self.alph
        else:
            thAccum = self.th_f - self.th_0
            w = 0
            alph = 0

        # convert to x/y velocity, acceleration, position
        th = self.th_0 + thAccum
        x = np.cos(th) * self.r + self.sc[0]
        y = np.sin(th) * self.r + self.sc[1]
        vx = -np.sin(th) * self.r * w
        vy = np.cos(th) * self.r * w
        ax = self.r * (-np.cos(th) * w * w - np.sin(th) * alph)
        ay = self.r * (-np.sin(th) * w * w + np.cos(th) * alph)

        return np.array([x, y, vx, vy, ax, ay])
