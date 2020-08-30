import math
import numpy as np
import control
import control.matlab

def euler(state, derivative, dt):
    for i in range(0, len(state)):
        state[i] += derivative[i] * dt
    return state

def getFeedbackControl(state, K):
    state = np.array(state)
    ctrl = -np.matmul(K, state)
    return ctrl
    