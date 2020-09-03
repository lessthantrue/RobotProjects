import numpy as np
from common import matrix_utils

class World():
    def __init__(self, w, h, space=2, buf=1):
        self.walls = []
        self.dim = (w, h)

        hlen = w - (buf * 2)
        hspan = (w - space) / 2 - buf
        vlen = h - (buf * 2)
        vspan = (h - space) / 2 - buf

        self.walls.append(self.make_wall(vlen, buf, vspan, buf, buf))
        self.walls.append(self.make_wall(hlen, buf, hspan, buf, buf))
        self.walls.append(self.make_wall(vlen, buf, vspan, buf, buf))
        self.walls.append(self.make_wall(hlen, buf, hspan, buf, buf))

        self.color = (255, 255, 255)

    def make_wall(self, count, minVal, maxVal, startVal, endVal):
        toRet = []
        prev = startVal
        toRet.append(prev)
        for _ in range(count-1):
            val = np.random.triangular(minVal, prev, maxVal+0.99)
            toRet.append(float(int(val)))
            prev = val
        toRet.append(endVal)
        return toRet

    def getPoints(self):
        scl2 = matrix_utils.scale2d(100.0/60.0, 100.0/60.0)

        pts = []
        for w in range(len(self.walls)):
            pttmp = []
            rot = matrix_utils.rotation2d(np.pi / 2 * (w + 1))
            tdif = (self.dim[0] - self.dim[1]) / 2
            t1 = matrix_utils.translation2d(-len(self.walls[w]) / 2, tdif * (1 - w % 2) + 3)
            scl1 = matrix_utils.scale2d(-1, 1)

            tf1 = rot @ scl1 @ t1

            for i in range(len(self.walls[w])):
                pttmp.append([i, self.walls[w][i], 1])
                pttmp.append([i+1, self.walls[w][i], 1])

            for i in range(len(pttmp)):
                pttmp[i] = tf1 @ np.array(pttmp[i], dtype="float64")

            pts += pttmp

        for i in range(len(pts)):
            pts[i] = scl2 @ np.array(pts[i])
            pts[i] = np.array(pts[i], dtype="float64")

        return np.array(pts)

    def getSegments(self):
        pts = self.getPoints()
        segs = []
        for i in range(len(pts)-1):
            if not np.array_equal(pts[i], pts[i+1]):
                segs.append([pts[i], pts[i+1]])
        return segs

if __name__ == "__main__":
    w = World(12, 9)
    print(w.getPoints())