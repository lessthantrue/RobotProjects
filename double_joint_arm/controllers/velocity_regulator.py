import math

class VelRegulator():
    def __init__(self, kp, fv, fa, kd, kv):
        self.kp = kp
        self.fa = fa
        self.fv = fv
        self.kd = kd
        self.kv = kv
    
    def get(self, xcur, xgoal, vcur, vgoal, a):
        return self.kp * (xgoal - xcur) - self.kd * vcur + self.fv * vgoal + self.kv * (vgoal - vcur) + self.fa * a