import numpy as np
from common import matrix_utils as mu
from motion import step

ctv = 0.5 / np.cos(np.pi / 6)
ctm = ctv *  np.sin(np.pi / 6)
shape = np.array([
    [ctv, 0, 1],
    [-ctm, 0.5, 1],
    [-ctm, -0.5, 1],
    [ctv, 0, 1],
    [0, 0, 1]
])

class Robot():
    def __init__(self, startV):
        self.state = np.array(startV, dtype="float64")
        self.color = (255, 0, 255)
    
    def getFrame(self):
        return mu.translation2d(self.state[0], self.state[1]) @ mu.rotation2d(self.state[2])

    def getPoints(self):
        return mu.tfPoints(shape, self.getFrame())

    def act(self, s, r, dt):
        self.state = np.array(step(self.state, s * dt, r))