import numpy as np
import pygame

keyList = [
    pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP4,
    pygame.K_KP6, pygame.K_KP7, pygame.K_KP8, pygame.K_KP9,
    pygame.K_LEFT, pygame.K_RIGHT
]

# dirs in normal - tangential
# positive normal is to the right of positive tangential
dirs = [
    [-0.707, 0.707, 0],
    [-1, 0, 0],
    [-0.707, -0.707, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0.707, 0.707, 0],
    [1, 0, 0],
    [0.707, -0.707, 0],
    [0, 0, 1],
    [0, 0, -1]
]

actions = {}
for i in range(len(dirs)):
    actions[keyList[i]] = dirs[i]

class KeyInput():
    def __init__(self):
        self.keysDown = []

    def onKeyDown(self, k):
        self.keysDown += [k]

    def onKeyUp(self, k):
        for i in range(len(self.keysDown)):
            if self.keysDown[i] == k:
                del self.keysDown[i]
                break

    def getAction(self):
        dirsum = np.array([0, 0, 0], dtype="float64")
        for k in self.keysDown:
            if k in actions:
                dirsum += np.array(actions[k], dtype="float64") / len(self.keysDown)
        return dirsum * 0.5