import numpy as np
import pygame

class Gyro():
    def __init__(self):
        # define noise covariance and such here
        self.heading = 0
        self.stdev = 0.1

    # returns a noisy sensor reading given robot heading
    def get(self, r_heading):
        self.heading = np.random.normal(loc=r_heading, scale=self.stdev)
        return np.random.normal(loc=r_heading, scale=self.stdev)

    def draw(self, surf, color, transform, r_posn):
        grposn = transform(r_posn[0], r_posn[1])
        end = np.array([np.cos(-self.heading) * 10, np.sin(-self.heading) * 10]) + grposn

        pygame.draw.line(
            surf,
            color,
            grposn,
            end
        )
