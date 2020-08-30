import numpy as np
import control
import control.matlab

L = 1 #electric inductance
J = 0.01 #moment of inertia
b = 0.1 #motor damping constant
R = 1 #electric resistance
K = 0.01 #motor torque constant

# state is e_w, i
# input is volts
# output is w

# matrices for the actual motor system

A = np.array([
    [-b/J, K/J],
    [-K/L, -R/L]
])

B = np.array([
    [0],
    [1/L]
])

C =  np.array([
    [1, 0]
])

def derivative(state, u):
    dState = A * state
    dState += B * u
    return dState

def getctrl(state, K, Kf, setpoint):
    innerCtrl = -np.matmul(K, state)
    inerCtrl += Kf

# matrices for the whole thing (motor + feedforward + control)
# start with feedback for inner controller

eigs = [-3, -3.1]
K = control.matlab.place(A, B, eigs)
Kf = np.array([1])

obs = control.matlab.obsv(A, C)
print(obs) # observable check: system is observable (no surprises there)

An = A - np.matmul(B, K*C)
Bn = B * (K+Kf)

print(np.linalg.eig(An)[0]) # surprisingly, An is stable on its own
# print(An)
# print(Bn)
