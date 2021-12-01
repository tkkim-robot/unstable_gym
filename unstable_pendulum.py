import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class UnstablePendulumEnv(gym.Env):

    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, dt=0.05, wind_type="sine", max_wind=1.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = dt
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.viewer = None

        self.t = 0.
        self.w = 0.
        self.prev_w = 0.
        self.wind_type = wind_type
        self.max_w = max_wind

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_wind(self, t):
        if self.wind_type == "random":
            dw = self.np_random.uniform(low=-0.5, high=0.5)
            w = max(min(self.prev_w + dw, self.max_w), -self.max_w)
            self.prev_w = w
        elif self.wind_type == "sine":
            w = self.max_w*np.sin(t)
        else:
            raise NotImplementedError("Not Implemented Wind Type !!")
        return w

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        self.t += dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        w = self.update_wind(self.t)

        self.last_u = u  # for rendering
        self.last_w = w  # for rendering

        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        final_torque = u + w*l**2*np.cos(th)/2
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) +
                            3.0 / (m * l ** 2) * final_torque) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            fname = path.join(path.dirname(__file__), "assets/wind.png")
            self.wind_img = rendering.Image(fname, 1.0, 1.0) #width, height

            self.imgtrans = rendering.Transform()
            self.wind_imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans) # just appending to self.attrs in Geom Class
            self.wind_img.add_attr(self.wind_imgtrans) # just appending to self.attrs in Geom Class

        self.viewer.add_onetime(self.img)
        self.viewer.add_onetime(self.wind_img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
            self.wind_imgtrans.scale = (-self.last_w , np.abs(self.last_w) )
            wind_x = 1.5 if self.last_w > 0 else -1.5
            self.wind_imgtrans.set_translation(wind_x, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    env = UnstablePendulumEnv(wind_type="random", max_wind=1.0)
    #env = gym.wrappers.Monitor(env, '/tmp/unstable_gym/', force=True)
    import time

    obs = env.reset()
    for step in range(500):
        action = env.action_space.sample()
        nobs, reward, done, info = env.step(action)
        env.render()
            
