import numpy as np

def rotation2d(theta):
    st = np.sin(theta)
    ct = np.cos(theta)

    return np.array([
        [ct, -st, 0],
        [st, ct, 0],
        [0, 0, 1]
    ])

def translation2d(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])