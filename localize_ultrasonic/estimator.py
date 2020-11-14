if __name__ == "__main__":
    import sys
    sys.path.append("C:\\Users\\Nick\\Documents\\Programming\\RobotProjects")

import numpy as np
from common import matrix_utils
import motion

shape = np.array([
    [0.5, 0, 1],
    [0, 0.5, 1],
    [-0.5, 0, 1],
    [0, -0.5, 1],
    [0.5, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [-1, 0, 1],
    [0, -1, 1],
    [1, 0, 1]
])

ctv = 0.5 / np.cos(np.pi / 6)
ctm = ctv *  np.sin(np.pi / 6)
robShape = np.array([
    [ctv, 0, 1],
    [-ctm, 0.5, 1],
    [-ctm, -0.5, 1],
    [ctv, 0, 1],
    [0, 0, 1]
])

def jacobian_sense(sa, ma):
    if type(sa) == type(None):
        return np.array([0, 0, 0])

    md, sd = ma[1] - ma[0], sa[1] - sa[0]
    psiM, psiS = np.arctan2(md[1], md[0]), np.arctan2(sd[1], sd[0])
    sinms = np.sin(psiM - psiS)

    # This should never happen. I don't know why it does, but it's an 
    # error case anyways.
    if sinms == 0:
        return np.array([0, 0, 0])
        
    dldx = -np.sin(psiM) / sinms
    dldy = np.cos(psiM) / sinms
    dldth = np.cos(psiM - psiS) / (sinms * sinms) * np.sign(sinms)
    return np.array([dldx, dldy, dldth])
    
def jacobian_act(x, dt):
    return matrix_utils.rotation2d(x[2]) * dt

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    if denom == 0:
        return None
    return (num / denom.astype(float))*db + b1

class EKF():
    def __init__(self, x0, stdevs):
        self.est = np.copy(x0)
        self.P = np.identity(len(x0)) * 1
        self.Q = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.3]
        ]) * 0.3
        self.R = np.diag(stdevs) ** 2
        self.color = (255, 0, 0)
        self.drawnObjs = []

    # A = ID; B = ID * dt
    def act(self, actV, dt):
        self.est = motion.move_robot(self.est, actV, dt)
        self.P += self.Q * np.linalg.norm(actV) * dt

    def getFrame(self):
        return motion.getFrame(self.est)

    def sense(self, sensV, sensVhat, sa, ma):
        H = []
        for i in range(len(sa)):
            H.append(jacobian_sense(sa[i], ma[i]))

        H = np.array(H)

        L = self.jumpFilter(sensV, sensVhat)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S) @ L
        self.est += K @ (sensV - sensVhat)
        self.P = (np.identity(len(self.est)) - K @ H) @ self.P

    def jumpFilter(self, sensV, sensVhat):
        eps = 0.1 * np.linalg.det(self.P)
        L = np.zeros(len(sensV))
        for i in range(len(sensV)):
            if abs(sensV[i] - sensVhat[i]) < eps:
                L[i] = 1
        return np.diag(L)

    def getPoints(self):
        cov_truncated = np.copy(self.P)
        cov_truncated[2] = np.array([0, 0, 1])
        trans = matrix_utils.translation2d(self.est[0], self.est[1])
        rot = matrix_utils.rotation2d(self.est[2])
        return (trans @ cov_truncated @ rot @ shape.T).T

import scipy
import scipy.linalg


class sigmaPointSprite():
    def __init__(self, s0):
        self.setPosition(s0)
        self.color = (128, 0, 0)

    def setPosition(self, pose):
        self.frame = matrix_utils.translation2d(pose[0], pose[1]) @ matrix_utils.rotation2d(pose[2])

    def getPoints(self):
        return matrix_utils.tfPoints(robShape, self.frame)

k = 0
alph = 0.6
L = 3 # dimensions
lam = alph * alph * L - L
beta = 2
w0 = 0.4
class UKF():
    def __init__(self, x0):
        self.est = np.copy(x0)
        self.P = np.identity(len(x0), dtype='float64')

        self.Q = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.3]
        ])

        self.R = np.identity(3) * 3
        self.wCalc = self.weights()

        self.drawnObjs = []
        for _ in self.wCalc[0]:
            self.drawnObjs.append(sigmaPointSprite([0, 0, 0]))

        self.color = (255, 0, 0)

    def getFrame(self):
        return motion.getFrame(self.est)

    def resetSigmaSprites(self):
        X = self.sigmas()
        for i in range(len(X)):
            self.drawnObjs[i].setPosition(X[i])

    def sigmas(self):
        sqrtp = np.linalg.cholesky(self.P) * np.sqrt(L / (1 - w0))
        sigmapts = []
        for i in range(-L, L+1):
            sigmapts.append(np.copy(self.est))
            if i != 0:
                col = sqrtp[:, abs(i) - 1]
                sigmapts[-1] += col * np.sign(i)
        return sigmapts

    def weights(self):
        Wm, Wc = [], []
        for i in range(-L, L+1):
            if i == 0:
                Wm.append(w0)
                Wc.append(w0)
            else:
                w = (1 - w0) / (2 * L)
                Wm.append(w)
                Wc.append(w)
        return Wm, Wc

    def getYs(self, sas, mas, X):
        estFrame = self.getFrame()
        ys = []        
        for i in range(len(X)):
            yi = [0, 0, 0]
            for j in range(len(sas)):
                sa = sas[j]
                ma = mas[j]
                xi = X[i]

                if type(sa) == type(None):
                    yi[j] = 15
                    continue

                # transform active sensor segment to this sigma point frame
                xFrame = matrix_utils.translation2d(xi[0], xi[1]) @ matrix_utils.rotation2d(xi[2])
                tf = xFrame @ np.linalg.inv(estFrame)
                sat1 = tf @ matrix_utils.toAffine(sa[0])
                sat2 = tf @ matrix_utils.toAffine(sa[1])
                ma1 = matrix_utils.toAffine(ma[0])
                ma2 = matrix_utils.toAffine(ma[1])

                # get intersection point with transformed segment
                intersect = seg_intersect(sat1, sat2, ma1, ma2)

                if type(intersect) == type(None):
                    yi[j] = 15

                yi[j] = min(np.linalg.norm(intersect[0:2] - X[i][0:2]), 15)
            ys.append(yi)
        return np.array(ys)

    def act(self, actV, dt):
        X = np.array(list(map(lambda x: motion.move_robot(x, actV, dt), self.sigmas())))
        Wm, Wc = self.wCalc
        self.est = np.average(X, weights=Wm, axis=0)

        P = np.zeros((3, 3))
        for i in range(len(X)):
            Xe = X[i] - self.est
            P += Wc[i] * np.outer(Xe, Xe)

        self.P = P + self.Q * np.linalg.norm(actV) * dt
        self.resetSigmaSprites()

    def sense(self, sensV, sensVhat, sa, ma):
        X = self.sigmas()

        # get expected sensor readings from sigma points
        ys = self.getYs(sa, ma, X)
        print(ys)

        Wm, Wc = self.wCalc

        ys_avg = np.average(ys, weights=Wm, axis=0)

        # compute cross covariance
        C = np.zeros((3, 3))
        S = np.copy(self.R)
        for i in range(len(X)):
            ys_res = ys[i] - ys_avg
            C += Wc[i] * np.outer(X[i] - self.est, ys_res)
            S += Wc[i] * np.outer(ys_res, ys_res)

        # rest of kalman filter
        K = C @ np.linalg.inv(S)

        self.est += K @ (sensV - ys_avg)
        self.P -= K @ S @ K.T
        self.resetSigmaSprites()

    def getPoints(self):
        cov_truncated = np.copy(self.P)
        cov_truncated[2] = np.array([0, 0, 1])
        trans = matrix_utils.translation2d(self.est[0], self.est[1])
        rot = matrix_utils.rotation2d(self.est[2])
        return (trans @ cov_truncated @ rot @ shape.T).T

if __name__ == "__main__":
    ukf = UKF(np.array([0, 0, 0], dtype='float64'), 0, 0, 0)
    ukf.act(np.array([1, 1, 1]), 1)
    print(ukf.est)