from profile.profile import Profile
from paths.circle import Circle
import kinematics
import numpy as np

###################################################
# Tests a circular path in the profiler.
# Generates the profile, transforms it into 
# configuration space, then tests each of the
# configuration space points against the 
# original profile.
###################################################

def close(v1, v2, tol=0.0001):
    return abs(np.linalg.norm(v1-v2)) < tol

q0 = np.array([0, 3.14 / 3])
l1 = 1
l2 = 2
x0 = kinematics.forward_state(q0[0], q0[1], l1, l2)
r = 0.75

c = Circle(x0[0] - r, x0[1], r, 1)

path =   lambda t: c.getPathParams(t)[0:2]
dpath =  lambda t: c.getPathParams(t)[2:4]
d2path = lambda t: c.getPathParams(t)[4:6]

p_original = Profile(path, dpath, d2path, 3)
p_transformed = Profile(path, dpath, d2path, 3)

qfn = lambda x: kinematics.forward_state(x[0], x[1], l1, l2)
jfn = lambda x: kinematics.forward_jacobian(x[0], x[1], l1, l2)
hfn = lambda x: kinematics.forward_hessian(x[0], x[1], l1, l2)

p_transformed.transform(q0, qfn, jfn, hfn)
i = 0

with open("log.csv", "w+") as f:
    f.write(", ".join(["x_tf", "y_tf", "x_or", "y_or"]) + '\n')
    for p1, p2 in zip(p_original.path, p_transformed.path):
        q_tf = kinematics.forward_state(p2.q[0], p2.q[1], l1, l2)
        q_or = np.array([p1.q[0], p1.q[1]])
        f.write(", ".join(map(str, np.concatenate([q_tf, q_or]))) + '\n')
        if not close(q_tf, q_or):
            print("Within tolerance for " + str(i) + " time steps")
            # print(p1)
            # print(p2)
            # print(str(q_tf) + " != " + str(q_or))
            break
        else:
            i += 1
    else:
        print("Within tolerance for entire time")