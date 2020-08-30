import math
import numpy as np

#https://www.chiefdelphi.com/t/paper-trapezoidal-motion-profile-generator/142297
class SinPlanner():
    def __init__(self, amax, vmax):
        self.amax = amax
        self.vmax = vmax
    
    def setProfile(self, start, end):
        diff = (end - start)
        self.time = math.sqrt(2 * math.pi * abs(diff) / self.amax)
        amaxTemp = self.amax * np.sign(diff)

        # if velocity constraint has been violated
        if self.time / math.pi * self.amax > self.vmax:
            self.time = math.pi * self.vmax
            amaxTemp = 2 * math.pi * diff / (self.time ^ 2)

        # total distance hasn't changed
        assert(abs(amaxTemp*(self.time*self.time)/(2*math.pi)-diff) < 0.0001)
        
        # velocity constraint is respected
        assert(self.vmax >= self.time / math.pi * amaxTemp)

        # construct acceleration function of time
        self.timeToAccel = lambda t: amaxTemp * math.sin(2 * math.pi * t / self.time)
