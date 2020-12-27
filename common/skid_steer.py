import numpy as np

try:
    import matrix_utils as mu
except ModuleNotFoundError:
    from common import matrix_utils as mu

import pygame

# TODO:0
# Other common files that should be made:
# * keyboard controls (controller.py from particle_many_beacons)
# * camera
# * distance/heading sensor? (or just a camera that rotates?)
# * gyroscope
# * encoders?
# * EKF?
# Split simulation and rendering
# Redo rendering interface (addObjects -> getPoints)

# basic robot that can drive forwards/backwards and turn in a 2d plane
# state represented as a vector of x, y, and heading
# action space is linear and angular velocity
class SkidSteer():
    def __init__(self):
        self.state = np.zeros(3)
        self.noise = np.zeros((3, 3))

    # state :: [ x, y, h ]
    def setState(self, state):
        self.state = state

    def setStateManual(self, x, y, h):
        self.state = np.array([x, y, h])

    # get or set one of the robot's state variables
    def stateVar(self, idx, newVal=None):
        if not (newVal is None):
            self.state[idx] = newVal
        return self.state[idx]

    # get or set the x coordinate of the robot
    def x(self, newVal=None):
        return self.stateVar(0, newVal)

    def y(self, newVal=None):
        return self.stateVar(1, newVal)

    def h(self, newVal=None):
        return self.stateVar(2, newVal)

    # return the transformation from the robot frame to the world frame
    def getFrame(self):
        return mu.translation2d(self.x(), self.y()) @ mu.rotation2d(self.h())
    
    # u :: [ v, w ]
    def act(self, u, dt):
        v, w = u
        self.state[2] += w * dt / 2
        self.state[0] += v * np.cos(self.state[2]) * dt
        self.state[1] += v * np.sin(self.state[2]) * dt
        self.state[2] += w * dt / 2

    def actManual(self, v, w, dt):
        self.act([v, w], dt)

# jacobian of a skid steer robot state vector with respect to the control vector
# state :: [ x, y, h ]
# returns d(state)/du = [ dx/dv dx/dw ]
#                       [ dy/dv dy/dw ]
#                       [ dh/dv dh/dw ]
def skidSteerJacobian(state):
    h = state[2]
    return np.array([
        [ -np.sin(h), 0 ],
        [ np.cos(h), 0 ],
        [ 0, 1 ]
    ])

v_def = 3
w_def = 1

def numpadDrive(event):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_KP1:
            return (v_def * -0.75, w_def * -0.75)
        elif event.key == pygame.K_KP2:
            return (v_def * -1, 0)
        elif event.key == pygame.K_KP3:
            return (v_def * -0.75, w_def * 0.75)
        elif event.key == pygame.K_KP4:
            return (0, w_def)
        elif event.key == pygame.K_KP6:
            return (0, -w_def)
        elif event.key == pygame.K_KP7:
            return (v_def * 0.75, w_def * 0.75)
        elif event.key == pygame.K_KP8:
            return (v_def, 0)
        elif event.key == pygame.K_KP9:
            return (v_def * 0.75, w_def * -0.75)
    elif event.type == pygame.KEYUP:
        return (0, 0)

defaultShape = np.array([ 
    [1, 0, 1], 
    [-1, -0.5, 1], 
    [-1, 0.5, 1],
    [1, 0, 1]
]) * np.array([0.25, 0.25, 1])

class DefaultSkidSteer(SkidSteer):
    def __init__(self, color):
        super().__init__()
        self.setState([0, 0, 0])
        self.cmd = [0, 0]
        self.color = color
    
    def onKey(self, event):
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            self.cmd = numpadDrive(event)

    def onLoop(self, dt):
        for _ in range(dt):
            self.act(self.cmd, 0.001)

    def getPoints(self):
        return mu.tfPoints(defaultShape, self.getFrame())

if __name__ == "__main__":
    import simulation
    ss = DefaultSkidSteer((255, 100, 100))
    width = 640
    height = 480
    dynamics_to_graphics_ratio = 40
    cmd = (0, 0)

    sim = simulation.Simulation(width, height)
    sim.initGame()
    sim.addObjects([ss])
    sim.run(ss.onLoop, ss.onKey)
