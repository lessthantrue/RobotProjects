import sys
sys.path.append("C:\\Users\\Nick\\Documents\\Programming\\RobotProjects")
from common import simulation
import numpy as np
import pygame
import robot, world, key_input, controller

# graphics and interface stuff
width = 1280
height = 960

# simulation state setup
nrobs = 3
robs = []
for i in range(nrobs):
    robs.append(robot.Robot(np.array([i, i, i])))

kin = key_input.KeyInput()
wld = world.World(12, 9)
ctrl = controller.Controller([0.0, 0.0, 0.0], nrobs)

drawnObjs = [wld, ctrl]

for r in robs:
    drawnObjs += [r] + r.getDrawnObjects()
    r.init_feedback()

def makeSegments(points):
    segs = []
    for i in range(len(pts) - 1):
        if not np.array_equal(pts[i], pts[i+1]):
            segs.append([pts[i], pts[i+1]])
    return segs

def on_event(event):
    if event.type == pygame.KEYDOWN:
        kin.onKeyDown(event.key)
    elif event.type == pygame.KEYUP:
        kin.onKeyUp(event.key)

wsegs = wld.getSegments()
def on_loop(dt):
    act = kin.getAction()
    ctrl.act(act, dt / 1000.0)
    ways = ctrl.getWaypoints()

    esegs = []
    rsegs = []
    for i in range(len(robs)):
        rsegs += robs[i].getSegments()
        esegs += robs[i].getEstSegments()
        robs[i].setWaypoint(ways[i])

    acts = ctrl.getFF(act)
    for i in range(len(robs)):
        r = robs[i]
        for _ in range(0, dt):
            acted = r.autoAct(acts[i], 0.001)

        r.stepEkf(acted, wsegs, rsegs, esegs, dt)

sim = simulation.Simulation(width, height)
sim.initGame()
sim.addObjects(drawnObjs)
sim.run(on_loop, on_event)