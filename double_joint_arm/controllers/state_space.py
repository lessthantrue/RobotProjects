import numpy as np
import control
import control.matlab

class StateSpace():
    def __init__(self, A, B, evals):
        self.K = control.matlab.place(A, B, evals)

    def get(self, state, ref):
        return -self.K @ np.array(state - ref)