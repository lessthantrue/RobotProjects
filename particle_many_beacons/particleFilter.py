import numpy as np
import pygame
import motion
import robot
from common import matrix_utils

class ParticleFilter():
    def __init__(self, nParticles, s0):
        self.particles = []
        self.objs = []
        for _ in range(0, nParticles):
            self.particles.append(np.copy(s0))
            self.objs.append(particleSprite(s0[0], s0[1], s0[2]))
        self.color = (0, 255, 255)

    def updateSprites(self):
        for i in range(len(self.particles)):
            st = self.particles[i]
            self.objs[i].setState(st[0], st[1], st[2])

    def act(self, actV, dt):
        for i in range(len(self.particles)):
            self.particles[i] += motion.getInst().getNoisyDiff(actV[0], actV[1], dt, self.particles[i][2])
        self.updateSprites()

    def sense(self, sensV, sensCov, world, camera, gyro, gyroStdev):
        weights = [0] * len(self.particles)
        wSum = 0

        for i in range(len(self.particles)):
            weights[i] = self.getProbSens(sensV, sensCov, self.particles[i], world, camera, gyro, gyroStdev)
            wSum += weights[i]

        # low variance resampling
        M = wSum / len(self.particles)
        start = np.random.rand() * M
        
        newParticles = []
        j = 0
        c = weights[0]
        for i in range(len(self.particles)):
            u = start + i * M
            while c < u:
                c += weights[j]
                j += 1
                j %= len(self.particles)
            newParticles.append(np.copy(self.particles[j]))

        assert len(newParticles) == len(self.particles)

        self.particles = newParticles
        self.updateSprites()

    # P(z|x)
    def getProbSens(self, sensV, sensCov, state, world, camera, gyro, gyroStdev):
        # This is going to require some explaining:
        # To associate each sensed point with each world point is an assignment
        # problem, minimizing the sum of distances from each of the associated
        # points. Because each sensed point can only be associated with one
        # actual point and vice versa, this is now a bipartite weighted assignment
        # problem, which fortunately has many solutions. However:
        #   * I'm way too lazy for that
        #   * the best solution only offers O(V^3) complexity
        #   * By nature of the problem, a greedy heuristic
        #     will yield an optimal solution most of the time anyways.
        # 
        # The greedy solution proposed:
        #   - For each sensed point Ps,
        #   -   Find the closest world point to Ps, Pw
        #   -   Pair Ps and Pw
        #   
        # This works under the assumption that any improper pairing will result 
        # in a distance much greater than any proper pairing, or that no two sensed
        # points will both be closest to the same world point
        
        pts = camera.getPointsPerfect(world, state[0:2], state[2])
        assoc = []

        for i in range(len(sensV)):
            minPt = None
            minDist = 1000000000000000000
            for j in range(len(pts)):
                d = np.linalg.norm(sensV[i] - pts[j])
                if d < minDist:
                    minDist = d
                    minPt = pts[j]

            assoc.append([sensV[i], np.copy(minPt)])

        # now that we have associated points, start on the math
        pTotal = 0
        for (s, w) in assoc:
            diff = s - w
            pTotal -= (diff.T @ np.linalg.inv(sensCov) @ diff) / 2

        pTotal -= (state[2] - gyro)**2 / (gyroStdev**2 * 2)

        return np.exp(pTotal)
        
    def draw(self, surf, color, transform):
        shape = [
            np.array([1, 0]),
            np.array([-1, -0.5]),
            np.array([-1, 0.5])
        ]

        for p in self.particles:
            center = transform(p[0], p[1])
            rot = np.array([
                [np.cos(-p[2]), -np.sin(-p[2])],
                [np.sin(-p[2]), np.cos(-p[2])]
            ]) @ (np.identity(2) * 5)

            tf = []
            for v in shape:
                tf.append((rot @ v) + center)

            pygame.draw.lines(
                surf, 
                color,
                True,
                tf
            )
        
    def avg(self):
        return sum(self.particles) / len(self.particles)

    def drawAvg(self, surf, color, tf):
        shape =[
            np.array([1, 0]),
            np.array([-1, -0.5]),
            np.array([-1, 0.5])
        ]

        avg = self.avg()

        center = tf(avg[0], avg[1])
        rot = np.array([
            [np.cos(-avg[2]), -np.sin(-avg[2])],
            [np.sin(-avg[2]), np.cos(-avg[2])]
        ]) @ (np.identity(2) * 10)

        for i in range(len(shape)):
            shape[i] = (rot @ shape[i]) + center

        pygame.draw.lines(
            surf,
            color,
            True,
            shape
        )

    def getFrame(self):
        est = self.avg()
        return matrix_utils.translation2d(est[0], est[1]) @ matrix_utils.rotation2d(est[2])

    def getPoints(self):
        return matrix_utils.tfPoints(robot.robot_shape, self.getFrame())

    def getDrawnObjs(self):
        return self.objs

class particleSprite():
    def __init__(self, x, y, t):
        self.setState(x, y, t)
        self.color = (255, 0, 0)

    def setState(self, x, y, t):
        trans = matrix_utils.translation2d(x, y)
        rot = matrix_utils.rotation2d(t)
        scl = matrix_utils.scale2d(0.7, 0.7)
        self.points = matrix_utils.tfPoints(robot.robot_shape, trans @ rot @ scl)
    
    def getPoints(self):
        return self.points