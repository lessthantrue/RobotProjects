import math
import numpy as np
import pygame

minDist = 2

# defines the points of interest that can be detected by the sensor
class World():
    def __init__(self, nPoints):
        self.points = []
        while(len(self.points) < nPoints):
            p = (np.random.rand(2) + np.array([-0.5, -0.5])) * 10
            for pt in self.points:
                if np.linalg.norm(pt - p) < minDist:
                    break
            else:
                self.points.append(p)

    def draw(self, surf, color, transform):
        for p in self.points:
            pygame.draw.circle(
                surf, 
                color,
                transform(p[0], p[1]),
                10
            )