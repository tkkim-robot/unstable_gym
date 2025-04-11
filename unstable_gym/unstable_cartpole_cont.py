"""
Classic cart-pole system with continuous control, extended with side wind.
Based on the gymnasium CartPoleEnv but modified to:
  1. Accept continuous control actions.
  2. Add a side wind force to the pole dynamics.
  3. Visualize the wind using an asset (assets/wind.png).
  
The wind is computed using a sine or random update; it is applied as an additional
torque at the pole’s center. The wind icon’s size is scaled according to its strength,
and its placement is adjusted based on the wind’s sign.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from os import path


class UnstableCartPoleContEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, 
        sutton_barto_reward: bool = False, 
        render_mode: Optional[str] = None,
        wind_type: str = "sine",
        max_wind: float = 1.0
    ):
        self._sutton_barto_reward = sutton_barto_reward

        # System dynamics parameters.
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        # Continuous force magnitude limits.
        self.min_action = -10.0
        self.max_action = 10.0
        self.dt = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Wind parameters.
        self.wind_type = wind_type
        self.max_w = max_wind
        self.t = 0.0
        self.prev_w = 0.0
        self.last_w = 0.0  # For visualization

        # Angle threshold for termination.
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Observation: [cart position, cart velocity, pole angle, pole angular velocity]
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        # Use a continuous action space.
        self.action_space = spaces.Box(
            low=np.array([self.min_action], dtype=np.float32),
            high=np.array([self.max_action], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: Optional[np.ndarray] = None
        self.steps_beyond_terminated = None

        self.np_random = None
        self.seed()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def update_wind(self, t: float) -> float:
        if self.wind_type == "random":
            dw = self.np_random.uniform(low=-0.5, high=0.5)
            w = np.clip(self.prev_w + dw, -self.max_w, self.max_w)
            self.prev_w = w
        elif self.wind_type == "sine":
            w = self.max_w * math.sin(t)
        else:
            raise NotImplementedError("Wind type not implemented!")
        return w

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Ensure state is available.
        assert self.state is not None, "Call reset() before step()."
        # Convert continuous action.
        force = float(np.array(action).item())
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Update time and wind.
        self.t += self.dt
        w = self.update_wind(self.t)
        self.last_w = w
        # Compute wind-induced torque.
        wind_torque = 2.0 * w * self.length * costheta / self.masspole

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp + wind_torque) / (
            self.length * (4.0 / 3.0 - self.masspole * (costheta**2) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * xacc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * thetaacc
        else:  # semi-implicit Euler
            x_dot = x_dot + self.dt * xacc
            x = x + self.dt * x_dot
            theta_dot = theta_dot + self.dt * thetaacc
            theta = theta + self.dt * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "Call reset() before taking additional steps."
                )
            self.steps_beyond_terminated += 1
            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Use custom reset bounds if provided.
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(np.float32)
        self.steps_beyond_terminated = None
        self.t = 0.0
        self.prev_w = 0.0
        self.last_w = 0.0
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self) -> Union[None, np.ndarray]:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "Render mode not specified. Pass render_mode at initialization, e.g. gym.make(..., render_mode='rgb_array')."
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # for rgb_array mode
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x, x_dot, theta, theta_dot = self.state
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Draw cart.
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0
        carty = 100  # vertical position of the cart
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # Draw pole.
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            vec = pygame.math.Vector2(coord).rotate_rad(-theta)
            vec = (vec[0] + cartx, vec[1] + carty + axleoffset)
            pole_coords.append(vec)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # Draw axle.
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # Draw track.
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # --- Wind Visualization ---
        # Load the wind asset from "assets/wind.png".
        fname_wind = path.join(path.dirname(__file__), "assets", "wind.png")
        try:
            wind_img = pygame.image.load(fname_wind)
        except Exception as e:
            raise DependencyNotInstalled(
                "Could not load wind asset. Ensure assets/wind.png exists."
            ) from e
        if self.last_w is not None:
            # Scale the image based on wind strength.
            wind_scale_factor = scale * abs(self.last_w) / 2
            new_dim = max(int(wind_scale_factor), 1)
            scaled_wind_img = pygame.transform.smoothscale(wind_img, (new_dim, new_dim))
            # If wind is positive, rotate image 180°.
            if self.last_w < 0:
                scaled_wind_img = pygame.transform.rotate(scaled_wind_img, 180)
            # Position: if wind < 0, place near the right; if wind > 0, near the left.
            wind_x = 550 if self.last_w < 0 else 50
            wind_y = 200  # fixed vertical placement
            self.surf.blit(
                scaled_wind_img,
                (wind_x - scaled_wind_img.get_width() // 2, wind_y - scaled_wind_img.get_height() // 2),
            )
        # --- End Wind Visualization ---

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
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

    env = UnstableCartPoleContEnv(render_mode="human", wind_type="sine", max_wind=1.0)
    obs, info = env.reset(seed=1)
    for _ in range(2000):
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(env.dt)
    env.close()