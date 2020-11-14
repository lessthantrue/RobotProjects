import numpy as np

s0 = (0, 0, 0)

def step(state, stepSize, radius):
    x, y, th = state
    if radius != 0:
        angle = stepSize / radius
        x1 = radius * (np.sin(angle + th) - np.sin(th)) + x
        y1 = radius * (-np.cos(angle + th) + np.cos(th)) + y
        th1 = th + angle
    else:
        x1 = np.cos(th) * stepSize + x
        y1 = np.sin(th) * stepSize + y
        th1 = th

    return (x1, y1, th1)