import sys
sys.path.append("C:\\Users\\Nick\\Documents\\Programming\\RobotProjects")
from common import simulation

import robot_state
import pygame
import world
import camera
import numpy as np
import slam

# graphics and interface stuff
width = 640
height = 480
dynamics_to_graphics_ratio = 40 # pixels/unit
cmd = (0, 0)

rs = robot_state.Robot(0, 0, 0)
ws = world.World(10)
cam = camera.Camera(7, 60, rs.getFrame, rs.getFrame, len(ws.points))
s = slam.SLAM()

v_def = 3
w_def = 1

def on_key_event(event):
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
        rs.noisyAct(cmd[0], cmd[1], 0.001)

    s.act(cmd, dt=0.001*dt)
    pts = cam.getSeenPoints(ws, np.array([rs.x, rs.y]), rs.t)
    s.sense(pts)

sim = simulation.Simulation(width, height)
sim.initGame()
sim.addObjects([rs, cam])
sim.addObjects(ws.getDrawnObjs())
sim.addObjects(cam.getDrawnObjs())
sim.addObjects(s.getDrawnObjects())
sim.run(on_loop, on_key_event)