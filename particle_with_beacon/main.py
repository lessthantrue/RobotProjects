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

ws = world_state.World(5, 5)
rs = robot_state.Robot(0, 0, 0)
se = state_estimator.EKF([0, 0, 0])

# graphics and interface stuff
width = 640
height = 480
dynamics_to_graphics_ratio = 40 # pixels/unit
running = True
cmd = (0, 0)
clock = pygame.time.Clock()
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
    if event.type == pygame.QUIT:
        running = False
    elif event.type == pygame.KEYDOWN:
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
    if cmd == (0, 0):
        return
    for _ in range(0, dt):
        rs.noisyAct(cmd[0], cmd[1], 0.001)
        se.predict(cmd[0], cmd[1], 0.001, rs.processCov)
        Z = sensor.getNoisyReading(rs, ws)
        se.update(ws, Z, sensor.sensorCov)

def on_render():
    display_surf.fill((255, 255, 255))

    z_d, z_tr, z_tg = sensor.getAbsoluteReading(rs, ws)

    bx_graphics, by_graphics = dynamics_to_graphics(ws.bX, ws.bY)
    rs_xf, rs_yf = dynamics_to_graphics(rs.x + 0.3 * math.cos(rs.t), rs.y + 0.3 * math.sin(rs.t))
    rx_graphics, ry_graphics = dynamics_to_graphics(rs.x, rs.y)
    se_xf, se_yf = dynamics_to_graphics(se.mu[0] + 0.3 * math.cos(se.mu[2]), se.mu[1] + 0.3 * math.sin(se.mu[2]))
    se_gx, se_gy = dynamics_to_graphics(se.mu[0], se.mu[1])
    sensor_xf, sensor_yf = dynamics_to_graphics(ws.bX - z_d * math.cos(z_tr + rs.t), ws.bY - z_d * math.sin(z_tr + rs.t))
    gyro_xf, gyro_yf = dynamics_to_graphics(rs.x + 0.3 * math.cos(z_tg), rs.y + 0.3 * math.sin(z_tg))

    # get points to draw the covariance with
    points = [ np.array([0.1, 0]), np.array([0, 0.1]), np.array([-0.1, 0]), np.array([0, -0.1]) ]
    cov_truncated = np.array([
        [se.cov[0][0], se.cov[0][1]],
        [se.cov[1][0], se.cov[1][1]]
    ])
    for i in range(len(points)):
        points[i] = cov_truncated @ points[i]
        points[i] += se.mu[:2]
        points[i] = dynamics_to_graphics(points[i][0], points[i][1])

    # draw beacon:
    pygame.draw.circle(
        display_surf,
        (0, 0, 0),
        (bx_graphics, by_graphics),
        int(10)
    )

    # draw roobit:
    pygame.draw.line(
        display_surf,
        (0, 0, 0),
        (rx_graphics, ry_graphics),
        (rs_xf, rs_yf),
        int(2)
    )

    # draw sensor reading:
    # pygame.draw.line(
    #     display_surf,
    #     (255, 0, 0),
    #     (bx_graphics, by_graphics),
    #     (sensor_xf, sensor_yf),
    #     int(1)
    # )
    # pygame.draw.line(
    #     display_surf,
    #     (255, 0, 0),
    #     (rx_graphics, ry_graphics),
    #     (gyro_xf, gyro_yf),
    #     int(1)
    # )
    # print(se_gx, se_gy)
    # draw estimate:
    pygame.draw.line(
        display_surf,
        (255, 0, 255),
        (se_gx, se_gy),
        (se_xf, se_yf),
        int(2)
    )

    pygame.draw.lines(
        display_surf,
        (0, 255, 0),
        True,
        points
    )

sim = simulation.Simulation(width, height)
sim.initGame()
sim.addObjects([rs])
sim.run(on_loop, on_key_event)