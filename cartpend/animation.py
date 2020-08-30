import pygame
import math
import dynamics
import cartpend
import numpy as np

BG = (255, 255, 255) #white
STUFF = (0, 0, 0) #black

width = 640
height = 480
dynamics_to_graphics_ratio = 40 # pixels/meter

def dynamics_to_graphics(x, y):
    offset_y = height / 2
    offset_x = width / 2
    return x * dynamics_to_graphics_ratio + offset_x, y * dynamics_to_graphics_ratio + offset_y

class App:
    def __init__(self, initialState, control=None):
        self.controller = control
        if control == None:
            control = lambda x: 0
        
        self.state = initialState
        self.clock = pygame.time.Clock()
        self.size = self.width, self.height = width, height

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        return True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        
    def on_loop(self):
        dt = self.clock.tick(60)
        # advance dynamics
        for _ in range(0, dt):
            dState = cartpend.derivative(self.state, self.controller(self.state))
            self.state = dynamics.euler(self.state, dState, 0.001)

    def on_render(self):
        self._display_surf.fill(BG)
        cartPosnX, cartPosnY = self.state[0], 0
        ballPosnRelativeX = math.sin(self.state[2]) * cartpend.L
        ballPosnRelativeY = math.cos(self.state[2]) * cartpend.L
        ballPosnX, ballPosnY = ballPosnRelativeX + cartPosnX, ballPosnRelativeY + cartPosnY
        cartPosnXg, cartPosnYg = dynamics_to_graphics(cartPosnX, cartPosnY)
        ballPosnXg, ballPosnYg = dynamics_to_graphics(ballPosnX, ballPosnY)

        # draw track
        pygame.draw.line(
            self._display_surf,
            STUFF,
            (0, cartPosnYg),
            (self.width, cartPosnYg),
            1
        )

        # draw pole
        pygame.draw.line(
            self._display_surf,
            STUFF,
            (cartPosnXg, cartPosnYg),
            (ballPosnXg, ballPosnYg),
            3
        
        pygame.draw.circle(
            self._display_surf,
            STUFF,
            (ballPosnXg, ballPosnYg),
            int(5)
        )
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

if __name__ == "__main__":
    # unactuated
    # app = App(dynamics.initialState)
    # app.on_execute()

    # simple linear feedback
    ref = [0, 0, math.pi, 0]
    def ctrlFn(state):
        stateOffset = np.array(state) - np.array(ref)
        return dynamics.getFeedbackControl(stateOffset, cartpend.K)
    app = App(cartpend.initialState, control=ctrlFn)
    app.on_execute()