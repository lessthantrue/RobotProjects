import numpy as np

def rotation2d(theta):
    st = np.sin(theta)
    ct = np.cos(theta)

    return np.array([
        [ct, -st, 0],
        [st, ct, 0],
        [0, 0, 1]
    ], dtype="float64")

def translation2d(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ], dtype="float64")

def scale2d(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype="float64")

# 2dh: 2d + heading
def rotation2dh(theta):
    st = np.sin(theta)
    ct = np.cos(theta)

    return np.array([
        [ct, -st, 0, 0],
        [st, ct, 0, 0],
        [0, 0, 1, theta],
        [0, 0, 0, 1]
    ])

def translation2dh(dx, dy, dth=0):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dth],
        [0, 0, 0, 1]
    ])

def tfPoints(points, frame):
    return (frame @ points.T).T