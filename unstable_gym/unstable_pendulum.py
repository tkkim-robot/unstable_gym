__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class UnstablePendulumEnv(gym.Env):
    """
    This environment implements an inverted pendulum with added side wind.
    It combines the standard pendulum dynamics and torque visualization from gymnasium's PendulumEnv
    with an additional side wind effect applied at the pole's center.
    
    The torque applied by the agent is visualized using the asset "assets/clockwise.png".
    The side wind is visualized using "assets/wind.png": if the wind is negative, the image appears on the left;
    if positive, the image is rotated 180° and appears on the right. The size of the wind image scales with wind strength.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, g=10.0, wind_type: str = "sine", max_wind: float = 1.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode
        self.screen_dim = 500  # screen dimension in pixels
        self.screen = None
        self.clock = None
        self.isopen = True

        # Define action and observation spaces.
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Wind-related parameters.
        self.wind_type = wind_type
        self.max_w = max_wind
        self.t = 0.0
        self.prev_w = 0.0
        self.last_u = None  # last applied torque (for rendering)
        self.last_w = None  # last computed wind value (for rendering)

        self.np_random = None
        self.seed()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def update_wind(self, t):
        if self.wind_type == "random":
            dw = self.np_random.uniform(low=-0.5, high=0.5)
            w = np.clip(self.prev_w + dw, -self.max_w, self.max_w)
            self.prev_w = w
        elif self.wind_type == "sine":
            w = self.max_w * np.sin(t)
        else:
            raise NotImplementedError("Wind type not implemented!")
        return w

    def step(self, u):
        th, thdot = self.state  # th := theta
        g, m, l, dt = self.g, self.m, self.l, self.dt

        # Clip and record the applied torque.
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        # Update time and compute wind.
        self.t += dt
        w = self.update_wind(self.t)
        self.last_w = w

        # Compute cost (negative reward).
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        # Apply wind force as an additional torque at the pole's center.
        final_torque = u + w * l**2 * np.cos(th) / 2
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * final_torque) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            x = options.get("x_init", DEFAULT_X)
            y = options.get("y_init", DEFAULT_Y)
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # symmetric limits
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        self.last_w = None
        self.t = 0.0
        self.prev_w = 0.0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create a new surface for drawing.
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        # Draw the pendulum rod as a rotated rectangle.
        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l_val, r_val, t_val, b_val = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l_val, b_val), (l_val, t_val), (r_val, t_val), (r_val, b_val)]
        transformed_coords = []
        for c in coords:
            vec = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            vec = (vec[0] + offset, vec[1] + offset)
            transformed_coords.append(vec)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        # Draw the axle (center).
        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        # Draw the pendulum endpoint.
        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        # --- Torque Visualization ---
        # Load torque visualization asset.
        fname_torque = path.join(path.dirname(__file__), "assets", "clockwise.png")
        try:
            torque_img = pygame.image.load(fname_torque)
        except Exception as e:
            raise DependencyNotInstalled(
                "Could not load torque asset. Make sure assets/clockwise.png exists."
            ) from e
        if self.last_u is not None:
            scale_img_dim = max(int(scale * np.abs(self.last_u) / 2), 1)
            scaled_torque_img = pygame.transform.smoothscale(torque_img, (scale_img_dim, scale_img_dim))
            # Flip the image horizontally if last_u is positive.
            is_flip = bool(self.last_u > 0)
            scaled_torque_img = pygame.transform.flip(scaled_torque_img, is_flip, True)
            torque_pos = (
                offset - scaled_torque_img.get_rect().centerx,
                offset - scaled_torque_img.get_rect().centery,
            )
            self.surf.blit(scaled_torque_img, torque_pos)
        # --- End Torque Visualization ---

        # --- Wind Visualization ---
        # Load the wind image asset.
        fname_wind = path.join(path.dirname(__file__), "assets", "wind.png")
        try:
            wind_img = pygame.image.load(fname_wind)
        except Exception as e:
            raise DependencyNotInstalled(
                "Could not load wind asset. Make sure assets/wind.png exists."
            ) from e
        if self.last_w is not None:
            wind_scale_factor = scale * np.abs(self.last_w) / 2
            new_dim = max(int(wind_scale_factor), 1)
            scaled_wind_img = pygame.transform.smoothscale(wind_img, (new_dim, new_dim))
            # Rotate the wind image if wind is from the right.
            if self.last_w > 0:
                scaled_wind_img = pygame.transform.rotate(scaled_wind_img, 180)
            # Position: wind appears left if wind is negative, right if positive.
            wind_x = -1.5 if self.last_w < 0 else 1.5
            wind_pos = (offset + int(wind_x * scale) - scaled_wind_img.get_width() // 2,
                        offset - scaled_wind_img.get_height() // 2)
            self.surf.blit(scaled_wind_img, wind_pos)
        # --- End Wind Visualization ---

        # Flip the drawing surface vertically to match expected orientation.
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:  # mode == "rgb_array"
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# Default main function to test the environment.
if __name__ == "__main__":
    import time

    env = UnstablePendulumEnv(render_mode="human", wind_type="random", max_wind=1.0)
    obs, info = env.reset(seed=42)
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(env.dt)
    env.close()