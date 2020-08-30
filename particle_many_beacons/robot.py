import numpy as np
import pygame
import motion

class Robot():
    def __init__(self, startX, startY, startT):
        self.x = startX
        self.y = startY
        self.t = startT

    def act(self, v, w, dt):
        tNext = self.t + w * dt
        self.x += v * (np.cos(self.t) + np.cos(tNext)) * dt / 2
        self.y += v * (np.sin(self.t) + np.sin(tNext)) * dt / 2
        self.t = tNext

    def actNoisy(self, v, w, dt):
        diff = motion.getInst().getNoisyDiff(v, w, dt, self.t)
        # print(diff)
        self.x += diff[0]
        self.y += diff[1]
        self.t += diff[2]

    def getStateVector(self):
        return np.array([self.x, self.y, self.t])
    
    def draw(self, surf, color, transform):
        center = transform(self.x, self.y)
        rot = np.array([
            [np.cos(-self.t), -np.sin(-self.t)],
            [np.sin(-self.t), np.cos(-self.t)]
        ])
        
        rot = rot @ (np.identity(2) * 10)

        shape = [ 
            np.array([1, 0]), 
            np.array([-1, -0.5]), 
            np.array([-1, 0.5])
        ]

        for i in range(len(shape)):
            shape[i] = rot @ shape[i]
            shape[i] += center

        pygame.draw.lines(
            surf,
            color,
            True,
            shape
        )