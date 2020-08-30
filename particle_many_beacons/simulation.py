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

def dynamics_to_graphics(x, y):
    offset_y = height / 2
    offset_x = width / 2
    return np.array([int(x * dynamics_to_graphics_ratio + offset_x), int(offset_y - y * dynamics_to_graphics_ratio)])

pygame.init()
display_surf = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

# simulation state setup

v_def = 3.0
w_def = 1.0
ctrl = controller.Controller(v_def, w_def)
rob = robot.Robot(0.0, 0.0, 0)
wld = world.World(10)
cam = camera.Camera(7, 60)
gyr = gyro.Gyro()
filt = particleFilter.ParticleFilter(100, rob.getStateVector())

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
    if event.type == pygame.QUIT:
        running = False
    elif event.type == pygame.KEYDOWN:
        ctrl.act_manual(event.key)
    elif event.type == pygame.KEYUP:
        ctrl.stop()

def on_loop():
    dt = clock.tick(60)

    for _ in range(0, dt):
        act = ctrl.getCtrl()

        if np.linalg.norm(act) > 0:
            rob.actNoisy(act[0], act[1], 0.001)
        gyr.get(rob.t)

    filt.act(act, dt / 1000.0)

    if np.random.rand() < 0.3:
        sensCov = cam.cov
        pts = cam.getPoints(wld, np.array([rob.x, rob.y]), rob.t)
        if len(pts) > 0 and np.linalg.norm(act) > 0:
            filt.sense(pts, sensCov, wld, cam, gyr.heading, gyr.stdev)

def on_render():
    display_surf.fill((255, 255, 255))
    
    rob_posn = filt.avg()
    rob_t = rob_posn[2]
    rob_posn = rob_posn[0:2]

    # rob.draw(display_surf, (0, 0, 0), dynamics_to_graphics)
    wld.draw(display_surf, (0, 0, 255), dynamics_to_graphics)
    cam.draw(display_surf, (0, 255, 0), dynamics_to_graphics, rob_posn, rob_t)
    gyr.draw(display_surf, (255, 0, 0), dynamics_to_graphics, rob_posn)
    # filt.draw(display_surf, (0, 0, 255), dynamics_to_graphics)
    filt.drawAvg(display_surf, (0, 0, 255), dynamics_to_graphics)

    # draw camera readings
    pts = cam.getPoints(wld, rob_posn, rob_t)
    for p in pts:
        # heading, distance
        (t, d) = p

        th = t + rob_t
        posn = np.array([d * np.cos(th), d * np.sin(th)]) + rob_posn

        pygame.draw.circle(
            display_surf,
            (255, 125, 0),
            dynamics_to_graphics(posn[0], posn[1]),
            5
        )

    pygame.display.flip()

if pygame.joystick.get_count() > 0:
    setup_joysticks()

while running:
    for event in pygame.event.get():
        on_key_event(event)
    if js != None:
        on_js_loop()

    on_loop()
    on_render()


pygame.quit()