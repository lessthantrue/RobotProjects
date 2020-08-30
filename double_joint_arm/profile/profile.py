import numpy as np
from collections import namedtuple

# position, velocity, acceleration, time
Point = namedtuple("Point", "q v a t")

def close(v1, v2, tol=0.0001):
    return abs(np.linalg.norm(v1-v2)) < tol

# generates and stores a motion profile in configuration space
# given a path in world space and its derivatives
# numerical integration stays within 0.0001 tolerance for about 0.15 seconds
class Profile():
    # path, dpath, d2path are functions of time
    def __init__(self, path, dpath, d2path, t, dt=0.001, t0=0):
        self.dt = dt
        self.t0 = t0
        n = int(t / dt)

        self.path = []

        for i in range(0, n):
            tn = t0 + i * dt
            q = path(tn)
            v = dpath(tn)
            a = d2path(tn)
            self.path.append(Point(q, v, a, tn))

    def get(self, t):
        i = (t - self.t0) / self.dt
        if i <= 0:
            return self.path[0]
        if i >= len(self.path):
            return self.path[-1]
                    
        return self.path[int(i)]

    # fn    :: configuration position -> world position
    # jfn   :: configuration position -> (configuration velocity -> world velocity)
    # hfn   :: configuration position, velocity -> (configuration acceleration -> world acceleration)
    def transform(self, nq0, fn, jfn, hfn):
        npath = []

        # if not close(fn(nq0), self.path[0][0]):
        #     print(fn(nq0))
        #     print(self.path[0][0])
        #     print(nq0)
        #     assert(False)

        for p in self.path:
            q, v, a, t = p

            if len(npath) == 0:
                cq = nq0
            else:
                cq = npath[-1].q

            # step time forwards by integrating the inverse dynamics
            J = jfn(cq)
            cv = np.linalg.inv(J) @ v

            H = hfn(cq) # Hessian tensor
            ca = np.linalg.inv(J) @ (a - np.tensordot(np.outer(cv, cv), H))

            if len(npath) != 0:
                cq = cq + cv * self.dt

            # some newton's method to keep away the accumulating error
            cq -= np.linalg.inv(J) @ (fn(cq) - q)
            
            npath.append(Point(cq, cv, ca, t))

        self.path = npath
