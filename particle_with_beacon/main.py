import sys
sys.path.append("C:\\Users\\Nick\\Documents\\Programming\\RobotProjects")
from common import simulation

import numpy as np
import math
import pygame
import world_state
import robot_state
import sensor
import state_estimator
import scipy
import scipy.linalg

ws = world_state.World(3, 3)
rs = robot_state.Robot(0, 0, 0)
se = state_estimator.EKF([0, 0, 0])

# graphics and interface stuff
width = 640
height = 480
dynamics_to_graphics_ratio = 40 # pixels/unit
running = True
cmd = (0, 0)
clock = pygame.time.Clock()
cam = sensor.CameraServo(10, rs.getFrame)
gyr = sensor.Gyroscope(250)
js = None

v_def = 3
w_def = 1

pygame.init()
display_surf = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

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
    global running
    global cmd
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_KP1:
            cmd = (v_def * -0.75, w_def * 0.75)
        elif event.key == pygame.K_KP2:
            cmd = (v_def * -1, 0)
        elif event.key == pygame.K_KP3:
            cmd = (v_def * -0.75, w_def * -0.75)
        elif event.key == pygame.K_KP4:
            cmd = (0, w_def)
        elif event.key == pygame.K_KP6:
            cmd = (0, -w_def)
        elif event.key == pygame.K_KP7:
            cmd = (v_def * 0.75, w_def * 0.75)
        elif event.key == pygame.K_KP8:
            cmd = (v_def, 0)
        elif event.key == pygame.K_KP9:
            cmd = (v_def * 0.75, w_def * -0.75)
    elif event.type == pygame.KEYUP:
        cmd = (0, 0)

def on_loop(dt):
    for _ in range(0, dt):
        cam.step(rs, ws, 1)
        gyr.step(rs, ws, 1)
        rs.noisyAct(cmd[0], cmd[1], 0.001)
    se.predict(cmd[0], cmd[1], dt * 0.001, rs.processCov)

    if cam.hasNewReading and gyr.hasNewReading:
        cam.hasNewReading = False
        gyr.hasNewReading = False
        Z1 = cam.getNoisyReading()
        Z2 = gyr.getNoisyReading()
        R = scipy.linalg.block_diag(sensor.CameraCov, sensor.GyroCov)
        se.update(ws, np.concatenate([Z1, Z2]), R)

sim = simulation.Simulation(width, height)
sim.initGame()
sim.addObjects([rs, ws, se, cam])
sim.run(on_loop, on_key_event)