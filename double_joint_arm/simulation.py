import numpy as np
import math
import pygame
from robot.robot_state_full import Robot
import kinematics
from controllers.regulator import Regulator
from controllers.velocity_regulator import VelRegulator
from paths.circle import *
from paths.line import *
from paths.arc import *
from robot_controller import Controller
from planner import Planner
from interpreter import Interpreter
import sys
import os

# graphics and interface stuff
width = 640
height = 480

dyn_width = 1.5
dyn_height = 1.5

dynamics_to_graphics_ratio = min(width / dyn_width, height / dyn_height) # pixels/unit
running = True
cmd = (0, 0)

display_surf = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
pathDrawn = pygame.surface.Surface((width, height))
pathDrawn.fill((255, 255, 255))

def dynamics_to_graphics(x, y):
    return int(x * dynamics_to_graphics_ratio), int(height - y * dynamics_to_graphics_ratio)

def graphics_to_dynamics(x, y):
    return x / dynamics_to_graphics_ratio, (height - y) / dynamics_to_graphics_ratio

# world/entities/whatever
center = graphics_to_dynamics(width/2, height/2)
robit_offset = np.array([-0.5, -0.5])
robit = Robot(0, math.pi / 4, center[0]+robit_offset[0], center[1]+robit_offset[1])

xi, yi = kinematics.forward_state(robit.m1.t, robit.m2.t, robit.len1, robit.len2)

t = 0
u = (0, 0)

clock = pygame.time.Clock()

# controller stuff
fname = "test_2.gcode"
ctrl = Controller(robit.len1, robit.len2)
ctrl.initMotorSpeedControllers(robit.m1, robit.m2)

plan = Planner(xi, yi, center[0], center[1])
intr = Interpreter()
intr.setPlanner(plan)
intr.loadFile(os.getcwd() + "\\double_joint_arm\\programs\\" + fname)

# start up

pygame.init()

def on_key_event(event):
    global running
    global target
    global t
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
        target = pygame.mouse.get_pos()
        target = graphics_to_dynamics(target[0], target[1])

def on_loop():
    dt = clock.tick(60)
    global u
    global t

    # save old position for line drawing
    (_, _, q2_0) = robit.getJoints()

    # check for path complete
    if ctrl.done():
        # update robot location in planner
        plan.setPosition(q2_0[0], q2_0[1])
        # get the next GCode line
        while not intr.nextLine():
            pass # keep processing until a move needs to be made
        if not intr.EOF:
            # set up the controller
            ctrl.setPath(plan.path, robit)

    robit.pen = intr.mac.pos.vector[2] < 0

    for _ in range(dt):
        t += 0.001
        
        u = ctrl.getActuation(robit, 0.001)
        robit.act(u[0], u[1], 0.001)

    (_, _, q2_1) = robit.getJoints()
    if robit.pen:
        pygame.draw.line(
            pathDrawn,
            (255, 0, 0),
            dynamics_to_graphics(q2_0[0], q2_0[1]),
            dynamics_to_graphics(q2_1[0], q2_1[1]),
            2
        )

def on_render():
    display_surf.fill((255, 255, 255))
    display_surf.blit(pathDrawn, (0, 0))

    # joints
    (q0, q1, q2) = robit.getJoints()
    gx0, gy0 = dynamics_to_graphics(q0[0], q0[1])
    gx1, gy1 = dynamics_to_graphics(q1[0], q1[1])
    gx2, gy2 = dynamics_to_graphics(q2[0], q2[1])

    # velocities
    (t1, t2, w1, w2) = robit.getStateVector()
    (_, _, a1, a2) = robit.getVelVector()
    [vx, vy, ax, ay] = kinematics.forward_jacobian_full_state(t1, t2, w1, w2, robit.len1, robit.len2) @ robit.getVelVector()
    gvx, gvy = dynamics_to_graphics(q2[0] + vx*0.5, q2[1] + vy*0.5)
    gax, gay = dynamics_to_graphics(q2[0] + ax*0.5, q2[1] + ay*0.5)

    px, py = plan.path.getPathParams(ctrl.t)[0:2]
    gp = dynamics_to_graphics(px, py)

    # draw arm 1
    pygame.draw.line(
        display_surf, 
        (255, 0, 0),
        (gx0, gy0),
        (gx1, gy1),
        int(3)
    )

    # draw arm 2
    pygame.draw.line(
        display_surf,
        (0, 0, 255),
        (gx1, gy1),
        (gx2, gy2),
        int(3)
    )

    # draw joint
    pygame.draw.circle(
        display_surf, 
        (255, 0, 255),
        (gx1, gy1),
        int(5)
    )

    # draw end effector velocity
    pygame.draw.line(
        display_surf, 
        (0, 0, 0),
        (gx2, gy2),
        (gvx, gvy),
        int(3)
    )

    # draw end effector acceleration
    pygame.draw.line(
        display_surf, 
        (80, 80, 80),
        (gx2, gy2),
        (gax, gay),
        int(3)
    )

    # current path location
    pygame.draw.circle(
        display_surf,
        (0, 128, 255),
        gp, 
        int(5)
    )

    pygame.display.flip()

while running:
    for event in pygame.event.get():
        on_key_event(event)

    on_loop()
    on_render()


pygame.quit()