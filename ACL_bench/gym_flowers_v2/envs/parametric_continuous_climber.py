import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
import math
from copy import copy, deepcopy

from ACL_bench.gym_flowers_v2.envs.bodies.BodiesEnum import BodiesEnum
from ACL_bench.gym_flowers_v2.envs.utils.custom_user_data import CustomUserDataObjectTypes, CustomUserData, CustomBodyUserData

#region Utils
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        for body in [contact.fixtureA.body, contact.fixtureB.body]:
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                body.userData.has_contact = True
                if body.userData.is_contact_critical:
                    self.env.head_contact = True

    def EndContact(self, contact):
        for body in [contact.fixtureA.body, contact.fixtureB.body]:
            if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                body.userData.has_contact = False

class LidarCallback(Box2D.b2.rayCastCallback):
    def __init__(self, agent_mask_filter):
        Box2D.b2.rayCastCallback.__init__(self)
        self.agent_mask_filter = agent_mask_filter
        self.fixture = None
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & self.agent_mask_filter) == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        return fraction
#endregion

#region Constants

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
VIEWPORT_W = 600
VIEWPORT_H = 600#400

INITIAL_RANDOM = 5
FRICTION = 2.5

NB_LIDAR = 15
# LIDAR_GROUP = 6
LIDAR_GROUP_THICKNESS = 0.15 # angle in radian
# LIDAR_GROUP_ANGLE_DIFFERENCE = LIDAR_GROUP_THICKNESS / (LIDAR_GROUP-1)
LIDAR_RANGE   = 160/SCALE

WALL_HEIGHT = 500 / SCALE
WALL_WIDTH = 300 / SCALE

GRIPS_CLOSENESS_LIMIT = 0.01
NO_GRIP_EXTENSION = 0.5
TERRAIN_STEP   = 50/SCALE   # in steps
TERRAIN_HEIGHT = 15/SCALE
#endregion

class ParametricContinuousClimber(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, body_type, lidar_group = 6, **body_args):
        super(ParametricContinuousClimber, self).__init__()

        # Seed env and init Box2D
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.wall = None
        self.lidar_group = lidar_group

        self.prev_shaping = None

        self.agent_body = BodiesEnum[body_type].value(SCALE, **body_args)
        init_x = VIEWPORT_W / SCALE / 2
        init_y = TERRAIN_HEIGHT + 2 * self.agent_body.LEG_H
        self.no_grips_zone = [VIEWPORT_W / SCALE / 2 - self.agent_body.AGENT_WIDTH / 2 - NO_GRIP_EXTENSION, # min_x
                              VIEWPORT_W / SCALE / 2 + self.agent_body.AGENT_WIDTH / 2 + NO_GRIP_EXTENSION, # max_x
                              TERRAIN_HEIGHT - NO_GRIP_EXTENSION, # min_y
                              self.agent_body.AGENT_HEIGHT + NO_GRIP_EXTENSION] # max_y

        self.action_space = spaces.Box(np.array([-1] * self.agent_body.nb_motors),
                                       np.array([1] * self.agent_body.nb_motors), dtype=np.float32)
        high = np.array([np.inf] * (self.agent_body.state_size +  # motors infos
                                    4 +  # head infos
                                    NB_LIDAR))  # lidars infos
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        #TODO
        self.environment_settings_space = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        for t in self.wall:
            self.world.DestroyBody(t)
        self.wall = []

        for motor in self.agent_body.motors:
            self.world.DestroyJoint(motor.box2d_object)
        self.agent_body.motors = []

        for body_part in self.agent_body.body_parts:
            self.world.DestroyBody(body_part.box2d_object)
        self.agent_body.body_parts = []

    # Gather parameters for procedural track generation, make sure to call this before each new episode
    def set_environment(self, splitting_height, circle_radius,
        space_between_grips, y_variations_std,
        group_sections, group_x_variations_std):
        self.splitting_height = splitting_height
        self.circle_radius = circle_radius.item() if isinstance(circle_radius, np.float32) else circle_radius
        self.space_between_grips = space_between_grips
        self.y_variations_std = y_variations_std
        self.group_sections = group_sections
        self.group_x_variations_std = group_x_variations_std

        self.create_terrain_fixtures()

    def reset(self):
        self._destroy()

        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.end_contact = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self.drawlist = []
        self.generate_game()

        self.drawlist = self.terrain + self.wall + self.agent_body.body_parts

        self.lidar = [LidarCallback(self.agent_body.reference_head_object.fixtures[0].filterData.maskBits)
                      for _ in range(NB_LIDAR * self.lidar_group)]

        return self.step(np.array([0] * self.agent_body.nb_motors))[0]

    def step(self, action):
        self.agent_body.activate_motors(action)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        head = self.agent_body.reference_head_object
        pos = head.position
        vel = head.linearVelocity

        lidar_group_size = (self.lidar_group - 1) if self.lidar_group > 1 else self.lidar_group
        lidar_group_angle_difference = LIDAR_GROUP_THICKNESS / lidar_group_size
        for i in range(0, NB_LIDAR):
            for j in range(self.lidar_group):
                index = i * self.lidar_group + j
                self.lidar[index].fraction = 1.0
                self.lidar[index].p1 = pos
                self.lidar[index].p2 = (
                    pos[0] + math.sin(2 * np.pi * i / NB_LIDAR + j*lidar_group_angle_difference) * LIDAR_RANGE,
                    pos[1] - math.cos(2 * np.pi * i / NB_LIDAR + j*lidar_group_angle_difference) * LIDAR_RANGE)
                self.world.RayCast(self.lidar[index], self.lidar[index].p1, self.lidar[index].p2)
        state = [
            head.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * head.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS]

        # add motors state
        state.extend(self.agent_body.get_motors_state())

        state += [
            min(map(lambda l:l.fraction, self.lidar[i:i+self.lidar_group]))
            for i in range(0, NB_LIDAR*self.lidar_group, self.lidar_group)
        ]

        # state += [
        #     l.fraction for l in self.lidar
        # ]

        self.scroll = max(0, pos[1] - VIEWPORT_H / SCALE / 5) - TERRAIN_HEIGHT + 1/SCALE # 1 = grass

        # TODO : Check if still ok ??!! Don't forget success thresholds
        shaping = 130 * pos[1] / SCALE  # moving up is a way to receive reward (normalized to get 300 on completion)
        #shaping -= 5.0 * abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= self.agent_body.TORQUE_PENALTY * 80 * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        # TODO : end contact ? (head ground contact ?)
        if pos[1] >= WALL_HEIGHT + TERRAIN_HEIGHT:
            done = True
            reward += 100

        # if self.nb_steps >= self.max_steps:
        #     reward = -100
        #     done = True

        return np.array(state), reward, done, {}

    def draw_game(self, draw_lidars):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(0, VIEWPORT_W / SCALE, self.scroll, VIEWPORT_H / SCALE + self.scroll)
        # Draw viewport
        self.viewer.draw_polygon([
            (0, self.scroll),
            (0, self.scroll + VIEWPORT_H / SCALE),
            (VIEWPORT_W / SCALE, self.scroll + VIEWPORT_H / SCALE),
            (VIEWPORT_W / SCALE, self.scroll),
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

        # Draw lidars
        if draw_lidars:
            for i in range(len(self.lidar)):
                l = self.lidar[i]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

    def render(self, mode='human', draw_lidars=True):
        self.draw_game(draw_lidars)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        self._destroy()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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

        self.fd_wall_section = fixtureDef(
            shape=polygonShape(box=(WALL_WIDTH / 2, self.splitting_height / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x100,
            maskBits=0x0 # Do not collide with body
        )

        self.fd_grip = fixtureDef(
            shape=circleShape(radius=self.circle_radius),
            density=1.0,
            restitution=0,
            categoryBits=0x1,
            maskBits=0xFFFF
        )
    #endregion

    # region Game generation
    # ------------------------------------------ GAME GENERATION ------------------------------------------
    def generate_game(self):
        self._generate_terrain()
        self._generate_wall()
        self._generate_agent()

    def _generate_terrain(self):
        y = TERRAIN_HEIGHT
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        x = 0
        max_x = VIEWPORT_W/SCALE * TERRAIN_STEP

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

    def sampler_function(self, mean, std, lower_bound, upper_bound):
        sampled_value = np.random.normal(mean, std)
        if sampled_value < lower_bound:
            return lower_bound
        elif sampled_value > upper_bound:
            return upper_bound
        else:
            return sampled_value

    def draw_grips(self, init_x, init_y, section_height):
        space = self.sampler_function(self.space_between_grips,
                                      0.1,
                                      2*self.circle_radius + GRIPS_CLOSENESS_LIMIT,
                                      WALL_WIDTH)
        current_x = init_x + space
        while current_x < init_x + WALL_WIDTH - self.circle_radius - GRIPS_CLOSENESS_LIMIT:
            y = init_y + section_height / 2 * self.sampler_function(0,
                                                                    self.y_variations_std,
                                                                    -1,
                                                                    1) # 0 = init_y, -1 = bottom of section, 1 = top of section

            y = min(y, init_y + section_height / 2 - self.circle_radius - GRIPS_CLOSENESS_LIMIT)  # not above the section's limit
            y = max(y, init_y - section_height / 2 + self.circle_radius + GRIPS_CLOSENESS_LIMIT)  # not below the section's limit

            if not (self.no_grips_zone[0] - self.circle_radius <= current_x <= self.no_grips_zone[1] + self.circle_radius and \
                    self.no_grips_zone[2] - self.circle_radius <= y <= self.no_grips_zone[3] + self.circle_radius):
                current_grip = self.world.CreateStaticBody(
                    position=(current_x, y),
                    fixtures=self.fd_grip,
                    userData=CustomUserData("grip", CustomUserDataObjectTypes.TERRAIN)
                )
                current_grip.color1 = (1, 1, 1)
                current_grip.color2 = (0.3, 0.3, 0.5)
                self.wall.append(current_grip)

                space = self.sampler_function(self.space_between_grips,
                                              0.1,
                                              2*self.circle_radius + GRIPS_CLOSENESS_LIMIT,
                                              WALL_WIDTH)
            current_x += space

    def _generate_wall(self):
        self.wall = []
        remaining_height = WALL_HEIGHT
        current_x = VIEWPORT_W/SCALE/2
        current_y = TERRAIN_HEIGHT #+ self.agent_body.AGENT_HEIGHT + 2
        group_size = math.ceil(self.sampler_function(self.group_sections,
                                                     0.5,
                                                     1,
                                                     math.floor(WALL_HEIGHT/self.splitting_height)))
        while remaining_height > self.splitting_height:
            current_section = self.world.CreateStaticBody(
                position=(current_x, current_y + self.splitting_height/2),
                fixtures=self.fd_wall_section)
            current_section.color1 = (0.5, 0.4, 0.9)
            current_section.color2 = (0.5, 0.4, 0.9)
            self.wall.append(current_section)

            self.draw_grips(current_x - WALL_WIDTH/2, #left edge of section
                            current_y + self.splitting_height / 2,
                            self.splitting_height)

            current_y += self.splitting_height
            remaining_height -= self.splitting_height
            group_size -= 1
            if group_size == 0:
                group_size = math.ceil(self.sampler_function(self.group_sections,
                                                     0.5,
                                                     1,
                                                     math.floor(WALL_HEIGHT/self.splitting_height)))
                # TODO : Add curve parameter ?
                current_x = VIEWPORT_W/SCALE/2 + self.sampler_function(0,
                                                                       self.group_x_variations_std,
                                                                        -VIEWPORT_W/SCALE/2 + WALL_WIDTH/2,
                                                                        VIEWPORT_W / SCALE / 2 - WALL_WIDTH / 2)
        smaller_section_fd = fixtureDef(
            shape=polygonShape(box=(WALL_WIDTH / 2, remaining_height / 2)),
            density=1.0,
            restitution=0.0,
            categoryBits=0x100,
            maskBits=0x0 # Do not collide with body
        )
        smaller_section = self.world.CreateStaticBody(
            position=(current_x, current_y + remaining_height / 2),
            fixtures=smaller_section_fd
        )

        smaller_section.color1 = (0.5, 0.4, 0.9)
        smaller_section.color2 = (0.5, 0.4, 0.9)
        self.wall.append(smaller_section)

        if remaining_height > self.circle_radius * 2 + 2 * GRIPS_CLOSENESS_LIMIT:
            self.draw_grips(current_x - WALL_WIDTH / 2,  # left edge of section
                            current_y + remaining_height / 2,
                            remaining_height)

    def _generate_agent(self):
        init_x = VIEWPORT_W / SCALE / 2
        init_y = TERRAIN_HEIGHT + self.agent_body.AGENT_CENTER_HEIGHT

        self.agent_body.draw(
            self.world,
            init_x,
            init_y,
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        )

        # self.drawlist.extend([body_part.box2d_object for body_part in self.agent_body.body_parts])

    #endregion