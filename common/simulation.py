import numpy as np
import pygame
import common.matrix_utils as mu

class Simulation():
    def __init__(self, width, height, graphics_ratio=40):
        # graphics and interface stuff
        self.width = width
        self.height = height
        self.graphics_ratio = graphics_ratio # pixels/unit
        self.clock = pygame.time.Clock()
        self.js = None

        self.offset_y = height / 2
        self.offset_x = width / 2

        self.drawnObjs = []

        self.dynamics_to_graphics = (
            mu.translation2d(self.offset_x, self.offset_y) @ 
            mu.scale2d(graphics_ratio, -graphics_ratio)
        )

        self.graphics_to_dynamics = np.linalg.inv(self.dynamics_to_graphics)

    def setup_joysticks(self):
        self.js = pygame.joystick.Joystick(0)
        self.js.init()

    def initGame(self):
        pygame.init()
        self.display_surf = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def drawObjs(self, fill=(255, 255, 255)):
        if fill != None:
            self.display_surf.fill(fill)
        
        for obj in self.drawnObjs:
            pts = obj.getPoints()

            # transform dynamics to graphics and take only first 2 columns
            pts = (self.dynamics_to_graphics @ pts.T).T[:, :2]
            
            pygame.draw.lines(self.display_surf, np.array(obj.color) * 0.8, False, pts, 2)

        pygame.display.flip()

    def addObjects(self, objs):
        self.drawnObjs += objs

    def stepClock(self, fps):
        return self.clock.tick(fps)

    def run(self, loopfn, eventfn, fps=60, renderfn=None):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                eventfn(event)
            
            dt = self.clock.tick(fps)
            loopfn(dt)
            self.drawObjs()
            if renderfn != None:
                renderfn()

        pygame.quit()

