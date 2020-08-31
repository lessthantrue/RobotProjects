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
    def __init__(self, x0, uf1, uf2, uf3):
        self.est = np.copy(x0)
        self.uframes = [uf1, uf2, uf3]
        self.P = np.identity(len(x0)) * 1
        self.Q = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.3]
        ]) * 0.3
        self.R = np.identity(3)
        self.color = (255, 0, 0)

    # A = ID; B = ID * dt
    def act(self, actV, dt):
        self.est = motion.move_robot(self.est, actV, dt)
        self.P += self.Q * np.linalg.norm(actV) * dt

    def getFrame(self):
        return motion.getFrame(self.est)

    def sense(self, sensV, sensVhat, sa, ma):
        H = np.array([
            jacobian_sense(sa[0], ma[0]),
            jacobian_sense(sa[1], ma[1]),
            jacobian_sense(sa[2], ma[2])
        ])
        L = self.jumpFilter(sensV, sensVhat)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R) @ L
        self.est += K @ (sensV - sensVhat)
        self.P = (np.identity(len(self.est)) - K @ H) @ self.P

    def jumpFilter(self, sensV, sensVhat):
        eps = 0.3
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

# k = 0
# alph = 0.6
# L = 3 # dimensions
# lam = alph * alph * L - L
# beta = 2
# class UKF():
#     def __init__(self, x0, uf1, uf2, uf3):
#         self.x = np.copy(x0)
#         self.uframes = [uf1, uf2, uf3]
#         self.P = np.identity(len(x0), dtype='float64')

#         self.Q = np.array([
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 0.3]
#         ])

#         self.R = np.identity(3) * 3

#     def getFrame(self):
#         return motion.getFrame(self.x)

#     def sigmas(self):
#         sqrtp = scipy.linalg.sqrtm(self.P * (lam + L))
#         sigmapts = []
#         for i in range(-L, L+1):
#             sigmapts.append(np.copy(self.x))
#             if i != 0:
#                 col = sqrtp[:, abs(i) - 1]
#                 sigmapts[-1] += col * np.sign(i)
#         return sigmapts

#     def weights(self):
#         Wm, Wc = [], []
#         for i in range(-L, L+1):
#             if i == 0:
#                 Wm.append(lam / (L + lam))
#                 Wc.append(lam / (L + lam) + (1 - alph * alph + beta))
#             else:
#                 w = 1 / (2 * (L + lam))
#                 Wm.append(w)
#                 Wc.append(w)
#         return Wm, Wc

#     def getYs(self, sa, ma, X):
#         for i in range(len(X)):
#             # TODO: get measurements from each sigma state in this function
#             # do so by creating a transform from the estimate to the sigma
#             # point and putting sa through it, then finding the intersection
#             # with ma
#             tf = 0

#     def act(self, actV, dt):
#         actV = matrix_utils.rotation2d(self.x[2]) @ actV
#         X = np.array(list(map(lambda x: motion.move_robot(x, actV, dt), self.sigmas())))
#         Wm, Wc = self.weights()
#         self.x = np.average(X, weights=Wm, axis=0)
#         self.P = np.cov(X.T, aweights=Wc, ddof=0) + self.Q * np.linalg.norm(actV) * dt

#     def sense(self, sensV, sensVhat, sa, ma):
#         pass


# if __name__ == "__main__":
#     ukf = UKF(np.array([0, 0, 0], dtype='float64'), 0, 0, 0)
#     ukf.act(np.array([1, 1, 1]), 1)
#     print(ukf.x)