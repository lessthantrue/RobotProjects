import numpy as np
from common import matrix_utils

# FastSLAM 2.0 implementation

# s :: x, y, h (robot state) [SE(2)]
# u :: v, w
# returns sH :: x, y, h (expected robot state) [SE(2)]
def h(s, u, dt):
    v, w = u
    sH = np.copy(s)
    sH[0] += v * np.cos(sH[2]) * dt
    sH[1] += v * np.sin(sH[2]) * dt
    sH[2] += w * dt
    return sH

# mu :: x, y (landmark location)
# s :: x, y, h (robot state) [SE(2)]
# returns zH :: d, dh (expected distance and relative heading to landmark)
def g(mu, s):
    p = np.array([s[0], s[1]])
    disp = p - mu
    dh = np.arctan2(disp[1], disp[0]) - s[2]
    dh %= np.pi * 2
    if abs(dh) > np.pi:
        dh -= np.pi * 2

    return np.array([np.linalg.norm(disp), dh])

# z :: d, dh (sensed landmark location)
# s :: x, y, h (robot state) [SE(2)]
# returns mu :: x, y (expected landmark location)
def g_inv(z, s):
    d, dh = z
    x, y, h = s
    th = h + dh
    x = x + np.cos(th) * d
    y = y + np.sin(th) * d
    return np.array([x, y])

R = np.eye(2) * 0.1
P = np.eye(2) * 0.1

p0 = 0.1

rplus = 0.3
rminus = 0.1

featureShape = np.array([
    [0.7, 0.7, 1],
    [-0.7, 0.7, 1],
    [-0.7, -0.7, 1],
    [0.7, -0.7, 1]
])

class Feature():
    def __init__(self):
        self.est = np.zeros(2) # X and Y mean (mu)
        self.cov = np.eye(len(self.est)) # covariance (sigma)
        self.exist = rplus # probability it actually exists (tau)

    def copy(self):
        f = Feature()
        f.est = np.copy(self.est)
        f.cov = np.copy(self.cov)
        f.exist = self.exist
        return f

    # derivative of g with respect to a feature
    # returns G :: R(2x2)
    # TODO: implement
    def G_th(self, s):
        x, y = self.est
        sx, sy, _ = s
        dx = x - sx
        dy = y - sy
        distsqr = dx * dx + dy * dy
        dist = np.sqrt(distsqr)

        # G = [
        #   [ d(dist)/dx, d(dist)/dy ]
        #   [ d(head)/dx, d(head)/dy ]
        # ]
        G = [
            [dx / dist, dy / dist],
            [-dy / distsqr, dx / distsqr]
        ]

        return np.array(G)

    # derivative of g with respect to the robot state
    # returns G :: R(2x3)
    # TODO: implement
    def G_s(self, s):
        x, y = self.est
        dx = x - s.x
        dy = y - s.y
        distsqr = dx * dx + dy * dy
        dist = np.sqrt(distsqr)

        # G = [
        #   [ d(dist)/dx, d(dist)/dy, d(dist)/dh ]
        #   [ d(head)/dx, d(head)/dy, d(head)/dh ]
        # ]

        G = [
            [dx / dist, dy / dist, 0],
            [-dy / distsqr, dx / distsqr, -1]
        ]

        return np.array(G)

    # s :: x, y, h (robot state) [SE(2)]
    # z :: d, dh (distance and relative heading to one sensed landmark)
    # p :: likelihood that the landmark sensed corresponds to this feature
    def p_sensed(self, s, z):
        Gth = self.G_th(s)
        Gs = self.G_s(s)
        Q = R + Gth @ self.cov @ Gth.T
        zH = g(self.est, s)

        s_pos = np.array([s[0], s[1]])

        sigma = np.linalg.inv(Gs.T @ np.linalg.inv(Q) @ Gs + np.linalg.inv)
        # mu = sigma @ Gs.T @ np.linalg.inv(Q) @ (z - zH) + s_pos

        # note: The paper says to sample a new robot state S from
        #       the landmark probability distribution, but that
        #       makes absolutely no sense, so I'm going to use
        #       the state passed in to this function.
        #       This is equivalent to using zH from above.
        # s ~ N(mu, sigma)

        # cholesky ~= sqrt
        z_diff = z - zH
        p = np.linalg.inv(np.linalg.cholesky(2 * np.pi * Q)) * np.exp((z_diff.T @ np.linalg.inv(Q) @ z_diff) / -2)
        return p

    def getPoints(self):
        trans = matrix_utils.translation2d(self.est[0], self.est[1])
        covTf = np.block([
            [ self.cov, np.zeros(1, 3) ],
            [ np.zeros(3, 1), [1] ]
        ])
        return matrix_utils.tfPoints(featureShape, trans @ covTf)

class Particle():
    def __init__(self):
        self.pose = np.zeros(3) # SE(2)
        self.features = [] # list of features (above)

    def copy(self):
        p = Particle()
        p.pose = np.copy(self.pose)
        for f in self.features:
            p.pose.append(f.copy())
        return p

    def copyTo(self, other):
        other.pose = self.pose

    # u :: v, w
    def act(self, u, dt):
        # propagate pose estimate forwards with time
        self.pose = h(self.pose, u, dt)
        # pose covariance is not tracked, so that's all folks

    # zs :: [d, dh] (distance and relative heading to sensed landmarks)
    # returns w :: double (weight of this particle)
    # modifies the current particle to incorporate sensor data
    def sense(self, zs):
        Ns = []
        for z in zs:
            P = [x.p_sensed(self.pose, z) for x in self.features]
            P.append(p0)
            Ns.append(np.argmax(P)) # index of most likely feature

        w = 0
        # handle observed features
        for i in range(len(Ns)):
            n = Ns[i]
            z = zs[i]
            if n < len(self.features): # Known feature case
                f = self.features[n]
                f.exist += rplus
                # lots of the following are recomputations, can be optimized
                Gth = f.G_th(self.pose)
                Q = R + Gth @ f.cov @ Gth.T
                zH = g(f.est, self.pose)

                K = f.cov @ f.G_th(self.pose) @ np.linalg.inv(Q)
                f.est += K @ (z - zH)
                f.cov = (np.eye(len(f.est)) - K @ Gth) @ f.cov

                Gs = f.G_s(self.pose)
                L = Gs @ P @ Gs.T + Gth @ f.cov @ Gth.T + R
                zDiff = z - zH
                w += np.linalg.inv(np.linalg.cholesky(2 * np.pi * L)) * np.exp((zDiff.T @ np.linalg.inv(L) @ zDiff) / -2)
            elif n == len(self.features): # New feature case
                f = Feature()
                f.est = g_inv(z, self.pose)
                G_th = f.G_th(self.pose)
                f.cov = G_th @ np.linalg.inv(R) @ G_th.T        
                w += p0    
        # later: handle unobserved features within sensor range
        return w

class SLAM():
    def __init__(self, nParticles=20):
        self.particles = [] # list of particles (above)
        for _ in range(nParticles):
            self.particles.append(Particle())
        self.p_visualized = Particle()

    # u :: v, w
    def act(self, u, dt=0.001):
        for p in self.particles:
            p.act(u, dt)

    def sense(self, zs):
        ws = []
        for p in self.particles:
            ws.append(p.sense(zs))
        ws = np.array(ws, dtype="float64")
        ws *= (1.0/sum(ws)) # normalize weights

        # resample particles
        newP = []
        for _ in range(0, len(self.particles)):
            newP.append(np.random.choice(self.particles, replace=True, p=ws).copy())

        # set the visualized particle to the max probability particle
        i_vis = np.argmax(ws)
        self.particles[i_vis].copyTo(self.p_visualized)

        # set new particles
        self.particles = newP

    def getDrawnObjects(self):
        return [self.p_visualized]
