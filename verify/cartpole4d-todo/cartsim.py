import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import superp
import ann

class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1000
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 1.0  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        #self.force_mag = 30.0
        self.force_mag = 1.0
        self.tau = 0.001  # seconds between state updates 
                        # sample period
        self.min_action = -1.0e16
        self.max_action = 1.0e16 # bounded control input: [-1,1]*force_mag

        # Angle at which to fail the episode
        # safe region 
        # unsafe region -12 degree to 12 degree; -2.4m to 2.4m
        self.theta_threshold_radians = 12 * math.pi / 180
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        x, x_dot, theta, theta_dot = self.state
        return torch.tensor([x, theta, x_dot, theta_dot]) ## reorder state

    def stepPhysics(self, normed_force): # the continuous dynamics
        x, x_dot, theta, theta_dot = self.state # old state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        ## the second-order derivative of x and theta
        thetaacc = (self.gravity * sintheta - normed_force * costheta) / self.length
        xacc = normed_force

        x = x + self.tau * x_dot # yes
        x_dot = x_dot + self.tau * xacc #yes
        theta = theta + self.tau * theta_dot #yes
        theta_dot = theta_dot + self.tau * thetaacc #yes

        self.state = (x, x_dot, theta, theta_dot)

        return (x, x_dot, theta, theta_dot) #new state

    def step(self, action):#update state with control input: action
        action = np.expand_dims(action, 0)
        assert self.action_space.contains(action),             "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * action
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state # (x, dx, theta, dtheta)
        done = x < -self.x_threshold             or x > self.x_threshold             or theta < -self.theta_threshold_radians             or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state), reward, done, {}

    def reset(self): #initialization of the cart pole
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) #initial state
        #self.state = (0, 0, 0, 0)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 5
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 3.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()


############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


# loading the controller NN model
ctrl_nn = ann.gen_nn(superp.N_H_C, superp.D_H_C, superp.DIM_S, superp.DIM_C, superp.CTRL_ACT, superp.CTRL_OUT_BOUND) # generate the nn model for the controller
ctrl_nn.load_state_dict(torch.load('pre_trained_ctrl.pt'), strict=True)

env = ContinuousCartPoleEnv()

with torch.no_grad():
    while True:
        env.reset() #initialization
        env.render()
        for i in range(10000):
            normed_force = ctrl_nn(env.get_state()).item()
            env.stepPhysics(normed_force)
            env.render()

