import numpy as np
import pygame

class Controller():
    def __init__(self, v_def, w_def):
        self.v = 0
        self.w = 0
        self.manual = {
            pygame.K_KP1: np.array([v_def * -0.75, w_def * -0.75]),
            pygame.K_KP2: np.array([v_def * -1, 0]),
            pygame.K_KP3: np.array([v_def * -0.75, w_def * 0.75]),
            pygame.K_KP4: np.array([0, w_def]),
            pygame.K_KP6: np.array([0, -w_def]),
            pygame.K_KP7: np.array([v_def * 0.75, w_def * 0.75]),
            pygame.K_KP8: np.array([v_def, 0]),
            pygame.K_KP9: np.array([v_def * 0.75, w_def * -0.75])
        }

        self.ctrlCov = np.array([
            [0.1, 0.01],
            [0.01, 0.1]
        ])

    def act_manual(self, key):
        if (key in self.manual.keys()):
            self.v, self.w = self.manual[key]

    def stop(self):
        self.v, self.w = 0, 0

    def getCtrl(self):
        return np.array([self.v, self.w])
