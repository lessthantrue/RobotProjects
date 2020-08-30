import numpy as np

# moves a point along a straight line
class Line():
    def __init__(self, x0, y0, x1, y1, vmax, amax):
        self.s0 = np.array([x0, y0])
        self.s1 = np.array([x1, y1])
        self.amax = amax
        self.vmax = vmax
        
        # need 2 time thresholds:
        # time to stop accelerating and continue at constant velocity,
        # and time to start decellerating
        # these can happen at the same time for a triangle profile
        d = np.linalg.norm(self.s0 - self.s1)

        # attempt 1: stop accelerating when accelerating more would pass max velocity
        self.astop = vmax / amax
        daccel = amax * self.astop * self.astop / 2

        # failure condition: stopping then doesn't leave enough distance to decellerate
        if daccel > d/2:
            # attempt 2: stop accelerating at half the distance
            self.astop = np.sqrt(d / amax)
            # in this case, decelleration needs to start immediately
            self.dstart = self.astop
            # adjust max velocity to the speed that the end was able to get to
            self.vmax = self.amax * self.astop
        else:
            # start by finding time traveled at constant velocity
            dcvel = d - 2 * daccel
            tcvel = dcvel / vmax
            # add starting time
            self.dstart = self.astop + tcvel

        self.time = self.dstart + self.astop
            
    def getPathParams(self, t):
        d = np.linalg.norm(self.s0 - self.s1)
        v = 0
        a = 0

        # get distance traveled along the path
        # and velocity/acceleration
        dAccum = 0
        if t < self.astop:
            dAccum = self.amax / 2 * t * t
            v = self.amax  * t
            a = self.amax
        elif t < self.dstart:
            dAccum = self.amax / 2 * self.astop * self.astop
            dAccum += self.vmax * (t - self.astop)
            v = self.vmax
            a = 0
        elif t < self.dstart + self.astop:
            tadj = t - self.dstart
            dAccum += self.amax / 2 * self.astop * self.astop
            dAccum += self.vmax * (self.dstart - self.astop)
            dAccum += self.vmax * tadj - self.amax / 2 * tadj * tadj        
            v = self.vmax - tadj * self.amax
            a = -self.amax
        else:
            dAccum = d
            v = 0
            a = 0

        u = (self.s1 - self.s0) / d # direction unit vector
        vx, vy = u * v
        ax, ay = u * a
        x, y = u * dAccum + self.s0

        return np.array([x, y, vx, vy, ax, ay])

