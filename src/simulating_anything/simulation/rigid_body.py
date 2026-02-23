"""2D projectile with quadratic drag simulation."""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ProjectileSimulation(SimulationEnvironment):
    """2D projectile motion with quadratic air drag.

    State vector: [x, y, vx, vy]
    Uses symplectic Euler integration for energy accuracy.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.gravity = p.get("gravity", 9.81)
        self.drag = p.get("drag_coefficient", 0.1)
        self.mass = p.get("mass", 1.0)
        self.v0 = p.get("initial_speed", 30.0)
        self.angle = np.radians(p.get("launch_angle", 45.0))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Launch from origin with configured speed and angle."""
        vx = self.v0 * np.cos(self.angle)
        vy = self.v0 * np.sin(self.angle)
        self._state = np.array([0.0, 0.0, vx, vy], dtype=np.float64)
        self._step_count = 0
        self._landed = False
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep with drag and gravity."""
        if self._landed:
            return self._state

        x, y, vx, vy = self._state
        dt = self.config.dt

        # Quadratic drag: F_drag = -c * |v| * v
        speed = np.sqrt(vx**2 + vy**2)
        drag_factor = self.drag / self.mass

        ax = -drag_factor * speed * vx
        ay = -self.gravity - drag_factor * speed * vy

        # Symplectic Euler: update velocity first, then position
        vx_new = vx + dt * ax
        vy_new = vy + dt * ay
        x_new = x + dt * vx_new
        y_new = y + dt * vy_new

        # Ground collision
        if y_new < 0.0 and self._step_count > 0:
            # Linear interpolation to find landing time
            t_land = -y / (y_new - y) * dt if (y_new - y) != 0 else dt
            x_new = x + t_land * vx_new
            y_new = 0.0
            vx_new = 0.0
            vy_new = 0.0
            self._landed = True

        self._state = np.array([x_new, y_new, vx_new, vy_new], dtype=np.float64)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, vx, vy]."""
        return self._state
