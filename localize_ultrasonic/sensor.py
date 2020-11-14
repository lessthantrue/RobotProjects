import numpy as np
from common import matrix_utils

minRange = 0.3
maxRange = 7
sendfov = 15
recvfov = 30

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

# returns true if p is within rectangle pa, pb
def pt_in(p, pa, pb):
    return (p[0] >= min(pa[0], pb[0]) and 
        p[0] <= max(pa[0], pb[0]) and
        p[1] >= min(pa[1], pb[1]) and
        p[1] <= max(pa[1], pb[1]))

shape = np.array([
    [-0.5, 0, 1],
    [0.5, 0.577, 1],
    [0.5, 0.2, 1],
    [0.2, 0, 1],
    [0.5, -0.2, 1],
    [0.5, -0.577, 1],
    [-0.5, 0, 1]
], dtype="float64") * [0.2, 0.2, 1]

class Ultrasonic():
    def __init__(self, localFrame, parent):
        self.frame = localFrame
        self.parentFrame = parent.getFrame
        self.color = (0, 0, 255)

        # ray parameters in form of heading, distance
        anglestep = np.pi / 18
        rayParams = np.array([
            [anglestep * 3, 7],
            [anglestep * 2, 8.0],
            [anglestep * 1, 11],
            [anglestep * 0, 15],
            [anglestep * -1, 11],
            [anglestep * -2, 8],
            [anglestep * -3, 7]
        ])

        self.rays = []
        for rp in rayParams:
            self.rays.append(UltrasonicRay(rp[0], rp[1], self.getFrame))
        self.drawnObjs = self.rays

        self.stdev = 0.1

    def getFrame(self):
        return self.parentFrame() @ self.frame

    def getPoints(self):
        return matrix_utils.tfPoints(shape, self.getFrame())

    # returns :
    # * intersection point
    # * origin and direction vector of active ray
    # * active map line segment
    # * distance from sensor origin to intersection point
    def getModelOutput(self, segs, rFrame):
        sFrame = rFrame @ self.frame
        pts = list(map(lambda x: x.getIntersection(segs, sFrame) + [x], self.rays))

        minPt = None
        minSeg = None
        minRay = None
        minD = 15
        for p, s, d, r in pts:
            r.color = (0, 255, 255)
            if (type(p) != type(None)) and d < minD:
                minD = d
                minPt = p
                minSeg = s
                minRay = r

        if minRay != None:
            minRay.color = (0, 255, 0)
            minRay = minRay.getIntPoints(sFrame)
            minRay[0] = matrix_utils.toAffine(minRay[0])
            minRay[1] = matrix_utils.toAffine(minRay[1])

        return (minPt, minSeg, minRay, minD)

    def addSensorNoise(self, minD):
        if np.random.rand() > 0.8 or minD < 0.5:
            return 15
        else:
            return np.random.triangular(minD * 4 / 5, minD, minD * 6 / 5)

class UltrasonicRay():
    def __init__(self, heading, distance, pFrame):
        self.color = (0, 255, 255)
        self.shape = np.array([
            [0, 0, 1],
            [1, 0, 1]
        ], dtype="float64")

        self.parentFrame = pFrame

        rot = matrix_utils.rotation2d(heading)
        scl = matrix_utils.scale2d(distance, 1)

        for i in range(len(self.shape)):
            self.shape[i] = rot @ scl @ self.shape[i]

    def getIntPoints(self, sensFrame):
        pts = []

        for i in range(len(self.shape)):
            pts.append((sensFrame @ self.shape[i])[0:2])

        return pts

    def getPoints(self):
        return matrix_utils.tfPoints(self.shape, self.parentFrame())

    # returns:
    # * intersection point
    # * map line segment intersected with
    # * distance to from origin to intersection
    def getIntersection(self, wldSegs, sensFrame):
        minD = 100000000
        minP = None
        minSeg = None
        raypts = self.getIntPoints(sensFrame)
        for i in range(len(wldSegs)):
            wpa, wpb = wldSegs[i]
            if np.array_equal(wpa, wpb):
                continue

            p = seg_intersect(raypts[0][0:2], raypts[1][0:2], wpa[0:2], wpb[0:2])
            if type(p) is type(None):
                continue

            d = np.linalg.norm(raypts[0] - p)
            if (pt_in(p, wpa, wpb) and 
                pt_in(p, raypts[0], raypts[1]) and 
                d < minD and d > minRange):
                minD = d
                minP = p
                minSeg = wpa, wpb

        return [minP, minSeg, minD]
