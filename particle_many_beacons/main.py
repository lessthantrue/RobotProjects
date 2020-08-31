import sys
sys.path.append("C:\\Users\\Nick\\Documents\\Programming\\RobotProjects")
from common import simulation

import numpy as np
import pygame
import robot
import controller
import world
import camera
import gyro
import particleFilter

# graphics and interface stuff
width = 640
height = 480
dynamics_to_graphics_ratio = 40 # pixels/unit
running = True
cmd = (0, 0)
clock = pygame.time.Clock()
js = None

v_def = 3.0
w_def = 1.0
ctrl = controller.Controller(v_def, w_def)
rob = robot.Robot(0.0, 0.0, 0)
wld = world.World(10)
filt = particleFilter.ParticleFilter(100, rob.getStateVector())
cam = camera.Camera(7, 60, filt.getFrame, rob.getFrame, len(wld.points))
gyr = gyro.Gyro()

def setup_joysticks():
    global js
    js = pygame.joystick.Joystick(0)
    js.init()

def on_js_loop():
    global cmd
    fwd = js.get_axis(1) * v_def * -2
    turn = js.get_axis(2) * w_def * 6
    cmd = (fwd, turn)

def on_key_event(event):
    if event.type == pygame.KEYDOWN:
        ctrl.act_manual(event.key)
    elif event.type == pygame.KEYUP:
        ctrl.stop()

def on_loop(dt):
    for _ in range(0, dt):
        act = ctrl.getCtrl()

        if np.linalg.norm(act) > 0:
            rob.actNoisy(act[0], act[1], 0.001)
        gyr.get(rob.t)

    filt.act(act, dt * 0.001)

    if np.random.rand() < 0.3:
        sensCov = cam.cov
        pts = cam.getSeenPoints(wld, np.array([rob.x, rob.y]), rob.t)
        if len(pts) > 0 and np.linalg.norm(act) > 0:
            filt.sense(pts, sensCov, wld, cam, gyr.heading, gyr.stdev)

sim = simulation.Simulation(width, height)
sim.initGame()
sim.addObjects(wld.getDrawnObjs() + cam.getDrawnObjs() + filt.getDrawnObjs() + [rob, cam, filt])
sim.run(on_loop, on_key_event)