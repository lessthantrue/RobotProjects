import numpy as np
from common import matrix_utils
import motion

class CirclePattern():
    def __init__(self, n, r):
        self.points_pol = []
        anglestep = np.pi * 2 / n
        for i in range(n):
            self.points_pol.append([i * anglestep, r, i * anglestep])

class Controller():
    def __init__(self, x0, n_robs):
        self.n_robs = n_robs
        self.state = x0

        self.color = (255, 255, 0)

        radius = 3
        self.pattern = CirclePattern(n_robs, radius)
        self.frames = []
        initial = matrix_utils.translation2dh(radius, 0, dth=np.pi)
        for i in range(n_robs):
            self.frames.append(matrix_utils.rotation2dh(np.pi * 2 / n_robs * i) @ initial)
        
    def t(self):
        return self.state[2]

    def x(self):
        return self.state[0]

    def y(self):
        return self.state[1]

    def getFrame(self):
        return motion.getFrame(self.state)

    def act(self, act, dt):
        self.state = motion.move_robot(self.state, act, dt)

    def getFF(self, act):
        vn, vt, w = act
        actv = np.array([vn, vt, w])
        actv = matrix_utils.rotation2d(self.t()) @ actv

        av = []
        for i in range(self.n_robs):
            d = self.pattern.points_pol[i][1]
            th_0 = self.pattern.points_pol[i][0] + self.t()
            vang = np.cross(d * np.array([np.cos(th_0), np.sin(th_0), 0]), np.array([0, 0, w]))
            vt = -vang + actv

            av.append(vt)

        return av

    # gets array of waypoint vectors for robots 
    def getWaypoints(self):
        wpts = []
        for i in range(self.n_robs):
            local = self.frames[i] @ np.array([0, 0, 0, 1])
            th = local[2]
            posn = np.array([local[0], local[1], 1])
            total = self.getFrame() @ posn
            total[2] = th + self.t()
            wpts.append(total)
        return wpts
            
    def getPoints(self):
        shape = []
        for th, d, _ in self.pattern.points_pol:
            shape.append([np.cos(th) * d, np.sin(th) * d, 1])

        shape.append(shape[0])
        return matrix_utils.tfPoints(np.array(shape), self.getFrame())