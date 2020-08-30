import numpy as np
from common import matrix_utils as mu

# steps x with act along dt
# x : [x, y, heading] in global frame
# act : [v_normal, v_tangential, w] in robot frame
# dt : double
def move_robot(x, act, dt):
    x = np.copy(x)
    th_f = x[2] + act[2] * dt
    th_avg  = (th_f + x[2]) / 2
    
    # transform local normal-tangential velocity to global frame
    x += mu.rotation2d(th_avg) @ act * dt

    # make sure angle is set properly
    x[2] = th_f
    return x

# returns a matrix transform from robot local frame to global frame
# x : [x, y, heading] in global frame
def getFrame(x):
    return mu.translation2d(x[0], x[1]) @ mu.rotation2d(x[2])