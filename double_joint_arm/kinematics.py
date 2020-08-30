import numpy as np
from math import sin, cos

def forward_state(t1, t2, l1, l2):
    x = l1 * cos(t1) + l2 * cos(t1 + t2)
    y = l1 * sin(t1) + l2 * sin(t1 + t2)
    return np.array([x, y])

def forward_jacobian_full_state(t1, t2, w1, w2, l1, l2):
    # [ dx/dt1,  dx/dt2,  dx/dw1,  dx/dw2  ]
    # [ dy/dt1,  dy/dt2,  dy/dw1,  dy/dw2  ]
    # [ dvx/dt1, dvx/dt2, dvx/dw1, dvx/dw2 ]
    # [ dvy/dt1, dvy/dt2, dvy/dw1, dvy/dw2 ]
    j = forward_jacobian(t1, t2, l1, l2)
    return np.array([
        [ j[0][0], j[0][1], 0, 0 ],
        [ j[1][0], j[1][1], 0, 0 ],
        [ -w1*l1*cos(t1)-w1*l2*cos(t2+t1), -w2*l2*cos(t2+t1), j[0][0], j[0][1]],
        [ -w1*l1*sin(t1)-w1*l2*sin(t2+t1), -w2*l2*cos(t2+t1), j[1][0], j[1][1]]
    ])

def forward_jacobian(theta1, theta2, len1, len2):
    # [ dx/dt1, dx/dt2 ]
    # [ dy/dt1, dy/dt2 ]
    return np.array([
        [-len1*sin(theta1)-len2*sin(theta2+theta1), -len2*sin(theta2+theta1)],
        [len1*cos(theta1)+len2*cos(theta2+theta1), len2*cos(theta2+theta1)]
    ])

def forward(w1, w2, t1, t2, l1, l2):
    u = np.array([w1, w2])
    J = forward_jacobian(t1, t2, l1, l2)
    return J @ u

def inverse(vx, vy, t1, t2, l1, l2):
    u = np.array([vx, vy])
    J = forward_jacobian(t1, t2, l1, l2)
    return np.linalg.inv(J) @ u

# yeah, we're really about to get into triads
def forward_hessian(theta1, theta2, len1, len2):
    hess = np.array([
        [   # H(f1) where f1 : [ t1, t2 ] -> x
            [ -len1*cos(theta1)-len2*cos(theta2+theta1), -len2*cos(theta2+theta1) ],
            [ -len2*cos(theta2+theta1), -len2*cos(theta2+theta1)]
        ],
        [   # H(f2) where f2 : [ t1, t2 ] -> y
            [ -len1*sin(theta1)-len2*sin(theta2+theta1), -len2*sin(theta2+theta1)],
            [ -len2*sin(theta2+theta1), -len2*sin(theta2+theta1) ]
        ]
    ])
    return hess

# def forward_twice(theta1, theta2, w1, w2, a1, a2, len1, len2):
#     H = forward_hessian(theta1, theta2, len1, len2)
#     J = forward_jacobian(theta1, theta2, len1, len2)
#     du = np.array([a1, a2])
#     u = np.array([w1, w2])
#     return (J @ du) + np.tensordot(np.outer(u, u), H)

# def inverse_twice(t1, t2, w1, w2, ax, ay, l1, l2):
#     H = forward_hessian(t1, t2, l1, l2)
#     J = forward_jacobian(t1, t2, l1, l2)
#     dx = np.array([ax, ay])
#     u = np.array([w1, w2])
#     return np.linalg.inv(J) @ ( dx - np.tensordot(np.outer(u, u), H) )