import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
import math
from enum import Enum

from ACL_bench.gym_flowers_v2.envs.utils.custom_rendering import CustomViewer, EmptyViewer
from gym.envs.classic_control import rendering

#region Utils
class ContactEnum(Enum):
    GROUND_CONTACT = 1
    BASKET_CONTACT = 2

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.ball in [contact.fixtureA.body, contact.fixtureB.body]: #First check that this is a collision of the ball
            if self.env.basket in [contact.fixtureA.body, contact.fixtureB.body]: #If collision with basket
                if self.env.basket.fixtures[2] in [contact.fixtureA, contact.fixtureB]: #If basket's bottom
                    self.env.end_contact = ContactEnum.BASKET_CONTACT
            elif any(x in [contact.fixtureA.body, contact.fixtureB.body] for x in self.env.terrain): #If collision with ground
                self.env.end_contact = ContactEnum.GROUND_CONTACT
#endregion

#region Constants

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
VIEWPORT_W = 600
VIEWPORT_H = 400

INITIAL_RANDOM = 5
FRICTION = 2.5

GAME_WIDTH = math.ceil(VIEWPORT_W/SCALE)
GAME_HEIGHT = math.ceil(VIEWPORT_H/SCALE)
TERRAIN_STEP   = 50/SCALE   # in steps
TERRAIN_HEIGHT = 15/SCALE
#endregion

class ParametricContinuousDrawingPath(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, max_steps = 20):
        super(ParametricContinuousDrawingPath, self).__init__()

        # Seed env and init Box2D
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None

        self.max_steps = max_steps

        # TODO : Bézier curve
        #self.action_space = spaces.Box(np.array([0, 0, 0, 0, 0]),
        #                               np.array([GAME_WIDTH, GAME_HEIGHT, GAME_WIDTH, GAME_HEIGHT, 1]),
        #                               dtype=np.float32) # control point (x, y), end point (x, y), draw ? (bool)
        self.action_space = spaces.Box(np.array([0, 0, 0]),
                                       np.array([GAME_WIDTH, GAME_HEIGHT, 1]),
                                       dtype=np.float32) # end point (x, y), draw ? (bool)
        self.observation_space = spaces.Box(0, 255, shape=(GAME_WIDTH, GAME_HEIGHT, 3), dtype=np.float32) # RGB image
        self.environment_settings_space = spaces.Box(np.array([0, 0, 0]),
                                                     np.array([GAME_WIDTH, GAME_WIDTH, 1]),
                                                     dtype=np.float32)  # basket_x, basket_width, balls_gravity

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.drawlist:
            self.world.DestroyBody(t)
        self.terrain = []
        self.drawlist = []
        self.basket = []

    # Gather parameters for procedural track generation, make sure to call this before each new episode
    def set_environment(self, basket_x, basket_width=10, balls_gravity=None):
        self.basket_x = basket_x
        self.basket_width = basket_width
        self.balls_gravity = balls_gravity if balls_gravity is not None else 0.5
        self.create_terrain_fixtures()

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.end_contact = None

        if self.viewer is None:
            self.viewer = CustomViewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(0, GAME_WIDTH, 0, GAME_HEIGHT)

        self.drawlist = []
        self.generate_game()
        self.drawlist.extend(self.terrain)
        self.current_action_position = self.ball.position
        self.nb_steps = 0

        # TODO
        #return self.step(np.array([0]*self.walker_body.nb_motors))[0]
        return self.current_action_position

    def step(self, action):
        if action[2] > 0.5: # Draw ?
            new_body = self.get_action_drawing_body(action[0], action[1])
            self.drawlist.append(new_body)
        self.current_action_position = (action[0], action[1])
        self.nb_steps += 1

        self.world.Step(1.0/FPS, 6*30, 2*30)

        # TODO : Get image
        self.draw_game()
        state = self.viewer.get_rgb_array()

        reward = 0
        reward -= np.linalg.norm(self.basket.position - self.ball.position)

        done = False
        if self.end_contact is not None:
            done = True
            if self.end_contact == ContactEnum.GROUND_CONTACT:
                reward -= 100
            elif self.end_contact == ContactEnum.BASKET_CONTACT:
                reward += 100
        elif self.nb_steps >= self.max_steps:
            done = True
            reward -= 100

        return np.array(state), reward, done, {}

    def draw_game(self):
        # Draw viewport
        self.viewer.draw_polygon([
            (0, 0),
            (GAME_WIDTH, 0),
            (GAME_WIDTH, GAME_HEIGHT),
            (0, GAME_HEIGHT),
        ], color=(0.9, 0.9, 1.0))

        # Draw terrain
        for poly, color in self.terrain_poly:
            self.viewer.draw_polygon(poly, color=color)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

    def render(self, mode='human', debug=False):
        #### DEBUG ####
        if debug : self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.draw_game()
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    #region Actions utils
    # ------------------------------------------ Actions Utils------------------------------------------
    def get_action_drawing_body(self, x, y):
        drawing_vector = (self.current_action_position[0] - x,
                          self.current_action_position[1] - y)

        drawing_vector_length = np.linalg.norm(drawing_vector)
        fixture = fixtureDef(
            shape=polygonShape(box=(2 / SCALE, drawing_vector_length / 2)),
            density=5.0,
            friction=0.1,
            categoryBits=0x1,
            maskBits=0xFFFF,
            restitution=0.0)

        # Get the angle
        '''
        Put the straight drawing (segment) with its bottom at current_action_position,
        calculate the end point and get the resulting vector (0, drawing_vector_length) :
        
        tmp_x = self.current_action_position[0]
        tmp_y = self.current_action_position[1] + drawing_vector_length
        tmp_vector = (self.current_action_position[0] - tmp_x,
                      self.current_action_position[1] - tmp_y) # = (0, drawing_vector_length)
        
        Then calculate the cosine between the vector to be drawn and the straight one with
        cosine = np.dot(vec1, vec2) / np.dot(||vec1||, ||vec2||).
        As both the vectors have the same norm, this is the same as ||vec1||^2 :
        cos_angle = np.dot(tmp_vector, drawing_vector) / np.square(drawing_vector_length)
        
        Finally as the first vector equals (0, drawing_vector_length), we have
        np.dot(vec1, vec2) = 0*vec2[0] + (-drawing_vector_length)*vec2[1]). So the final simplified
        equation is :
        '''
        cos_angle = (-drawing_vector_length*drawing_vector[1]) / np.square(drawing_vector_length)
        angle = math.acos(cos_angle)

        # Find middle point of the segment
        drawing_x = (self.current_action_position[0] + x) / 2
        drawing_y = (self.current_action_position[1] + y) / 2

        drawing = self.world.CreateStaticBody(
            position=(drawing_x, drawing_y),
            angle = angle,
            fixtures=fixture
        )
        drawing.color1 = (0.5, 0.4, 0.9)
        drawing.color2 = (0.3, 0.3, 0.5)

        return drawing
    #endregion

    #region Fixtures Initialization
    # ------------------------------------------ FIXTURES INITIALIZATION ------------------------------------------
    def create_terrain_fixtures(self):
        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=
                            [(0, 0),
                             (1, 1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )

        basket_polygons = [
            [(-self.basket_width / 2 - 1, +5), (-self.basket_width / 2 - 1, +40),
             (-self.basket_width / 2 + 1, +40), (-self.basket_width / 2 + 1, +5)],
            [(+self.basket_width / 2 - 1, +5), (+self.basket_width / 2 - 1, +40),
             (+self.basket_width / 2 + 1, +40), (+self.basket_width / 2 + 1, +5)],
            [(-self.basket_width / 2 - 1, +1), (-self.basket_width / 2 - 1, +5),
             (+self.basket_width / 2 + 1, +5), (+self.basket_width / 2 + 1, +1)],
        ]

        self.fd_basket = [
            fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in polygon]),
                density=5.0,
                friction=0.1,
                categoryBits=0x2,
                maskBits=0xFFFF,
                restitution=0.0)
            for polygon in basket_polygons
        ]

        self.fd_ball = fixtureDef(
            shape=circleShape(radius=0.2),
            density=1.0,
            restitution=self.balls_gravity,
            categoryBits=0x20,
            maskBits=0x000F
        )
    #endregion

    # region Game generation
    # ------------------------------------------ GAME GENERATION ------------------------------------------
    def generate_game(self):
        self._generate_terrain()
        self._generate_ball()
        self._generate_basket()

    def _generate_terrain(self):
        y = TERRAIN_HEIGHT
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        x = 0
        max_x = GAME_WIDTH * TERRAIN_STEP

        # Generation of terrain
        while x < max_x:
            self.terrain_x.append(x)
            self.terrain_y.append(y)
            x += TERRAIN_STEP

        # Draw terrain
        self.terrain_poly = []
        assert len(self.terrain_x) == len(self.terrain_y)
        for i in range(len(self.terrain_x)-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
            ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge,
                userData='grass')
            color = (0.3, 1.0 if (i % 2) == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_ball(self):
        init_x = GAME_WIDTH - 2
        init_y = GAME_HEIGHT - 2
        self.ball = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=self.fd_ball
        )
        self.ball.color1 = (0.5, 0.4, 0.9)
        self.ball.color2 = (0.3, 0.3, 0.5)
        self.drawlist.append(self.ball)

    def _generate_basket(self):
        init_y = TERRAIN_HEIGHT
        self.basket = self.world.CreateStaticBody(
            position=(self.basket_x, init_y),
            fixtures=self.fd_basket
        )
        self.basket.color1 = (0.5, 0.4, 0.9)
        self.basket.color2 = (0.3, 0.3, 0.5)
        self.drawlist.append(self.basket)
    #endregion