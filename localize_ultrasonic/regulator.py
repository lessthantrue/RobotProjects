import math
import numpy as np

class Regulator():
    def __init__(self, kp, kf=0):
        self.kp = kp
        self.kf = kf

    def setGoal(self, goal):
        self.goal = goal

    def get(self, cur):
        return self.kp * (self.goal - cur) + self.kf * self.goal
        