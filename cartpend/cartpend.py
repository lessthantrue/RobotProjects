import math
import numpy as np
import control
import control.matlab

m = 1
M = 5
L = 2
g = -9.81
d = 1

# x, v, t, w
initialState = np.array([-3, 0, math.pi+0.3, 0])

def derivative(state, u):
    def y(idx):
        return state[idx - 1]
    
    Sy = math.sin(y(3))
    Cy = math.cos(y(3))
    D = m * L * L * (m + M * (1 - Cy * Cy))

    dState = np.zeros(4)
    dState[0] = y(2)
    dState[1] = (1/D)*(-(m**2)*(L**2)*g*Cy*Sy + m*(L**2)*(m*L*(y(4)**2)*Sy - d*y(2))) + m*L*L*(1/D)*u
    dState[2] = y(4)
    dState[3] = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*(y(4)**2)*Sy - d*y(2))) - m*L*Cy*(1/D)*u + .01*np.random.rand()
    return dState

    

def linearize(state):
    if state[2] < math.pi / 2 and state[2] > -math.pi / 2:
        # pendulum facing down
        print("linearized with pendulum down")
        s = -1
    else:
        s = 1
    
    A = np.array([
        [0, 1, 0, 0],
        [0, -d/M, -m*g/M, 0],
        [0, 0, 0, 1],
        [0, -s*d/(M*L), -s*(m+M)*g/(M*L), 0]
    ])
    
    B = np.array([
        [0], 
        [1/M], 
        [0], 
        [s*1/(M*L)]
    ])
    return (A, B)

A, B = linearize([0, 0, math.pi, 0])

# place poles manually
initialPole = -3
poleDelta = -0.1
evals_desired = [0] * len(initialState)
evals_desired[0] = initialPole
for i in range(1, len(initialState)):
    evals_desired[i] = evals_desired[i-1] + poleDelta
K = control.matlab.place(A, B, evals_desired)

# Q = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 10, 0],
#     [0, 0, 0, 100]
# ])

# R = np.array([0.001])

# K = control.matlab.lqr(A, B, Q, R)

if __name__ == "__main__":
    # check controllability
    # A, B = linearize([0, 0, 0, 0])
    # ct = control.matlab.ctrb(A, B)
    # print(ct)
    # print(np.linalg.matrix_rank(ct))

    # print controller information
    print(K)
    print(np.linalg.eig(A - B * K))
