"""Magnetic pendulum simulation -- fractal basin boundaries.

A pendulum bob (magnet) swings over N fixed magnets arranged symmetrically.
With friction, the pendulum settles over one magnet, but which one depends
sensitively on initial conditions, creating fractal basin boundaries.

Equations of motion (2D, small-angle approximation):
  d^2x/dt^2 = -gamma*dx/dt - omega0^2*x + sum_i [alpha*(x_i - x) / r_i^3]
  d^2y/dt^2 = -gamma*dy/dt - omega0^2*y + sum_i [alpha*(y_i - y) / r_i^3]

where r_i = sqrt((x - x_i)^2 + (y - y_i)^2 + d^2), d = height above magnets.

Target rediscoveries:
- Fractal basin boundaries (boundary fraction as complexity measure)
- 3-fold rotational symmetry of basin structure
- All trajectories settle to one of N attractors (no sustained chaos)
- Sensitivity: nearby ICs can reach different magnets
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MagneticPendulumSimulation(SimulationEnvironment):
    """Magnetic pendulum over N equally-spaced magnets.

    State vector: [x, y, vx, vy]

    Parameters:
        gamma: damping coefficient (default 0.1)
        omega0_sq: restoring force omega_0^2 (default 0.5)
        alpha: magnet strength (default 1.0)
        R: magnet distance from center (default 1.0)
        d: height of pendulum above magnet plane (default 0.3)
        n_magnets: number of magnets (default 3)
        x_0, y_0: initial position (default 0.5, 0.5)
        vx_0, vy_0: initial velocity (default 0.0, 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.gamma = p.get("gamma", 0.1)
        self.omega0_sq = p.get("omega0_sq", 0.5)
        self.alpha = p.get("alpha", 1.0)
        self.R = p.get("R", 1.0)
        self.d = p.get("d", 0.3)
        self.n_magnets = int(p.get("n_magnets", 3))
        self.x_0 = p.get("x_0", 0.5)
        self.y_0 = p.get("y_0", 0.5)
        self.vx_0 = p.get("vx_0", 0.0)
        self.vy_0 = p.get("vy_0", 0.0)

        # Compute magnet positions: equally spaced on a circle of radius R
        self.magnet_positions = self._compute_magnet_positions()

    def _compute_magnet_positions(self) -> np.ndarray:
        """Compute positions of N magnets equally spaced on a circle."""
        angles = np.array([
            2 * np.pi * i / self.n_magnets for i in range(self.n_magnets)
        ])
        positions = np.column_stack([
            self.R * np.cos(angles),
            self.R * np.sin(angles),
        ])
        return positions

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize pendulum state [x, y, vx, vy]."""
        self._state = np.array(
            [self.x_0, self.y_0, self.vx_0, self.vy_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, vx, vy]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for the magnetic pendulum EOM."""
        x, y, vx, vy = state

        # Magnetic attraction: sum over all magnets
        ax_mag = 0.0
        ay_mag = 0.0
        for mx, my in self.magnet_positions:
            dx = mx - x
            dy = my - y
            r = np.sqrt(dx**2 + dy**2 + self.d**2)
            r3 = r**3
            ax_mag += self.alpha * dx / r3
            ay_mag += self.alpha * dy / r3

        # Restoring force + damping + magnetic attraction
        ax = -self.gamma * vx - self.omega0_sq * x + ax_mag
        ay = -self.gamma * vy - self.omega0_sq * y + ay_mag

        return np.array([vx, vy, ax, ay])

    def find_attractor(self, tol: float = 0.01, t_max: float = 200.0) -> int:
        """Run until settled, return which magnet (0, 1, 2, ...) the bob reaches.

        Args:
            tol: velocity and acceleration threshold for "settled".
            t_max: maximum simulation time before giving up.

        Returns:
            Index of the nearest magnet at rest, or -1 if not settled.
        """
        dt = self.config.dt
        max_steps = int(t_max / dt)

        for _ in range(max_steps):
            self.step()
            x, y, vx, vy = self._state
            speed = np.sqrt(vx**2 + vy**2)
            if speed < tol:
                # Find nearest magnet
                return self._nearest_magnet(x, y)

        # Did not settle -- return nearest magnet anyway
        return self._nearest_magnet(self._state[0], self._state[1])

    def _nearest_magnet(self, x: float, y: float) -> int:
        """Return index of nearest magnet to position (x, y)."""
        dists = np.sqrt(
            (self.magnet_positions[:, 0] - x) ** 2
            + (self.magnet_positions[:, 1] - y) ** 2
        )
        return int(np.argmin(dists))

    def basin_map(
        self,
        grid_size: int = 50,
        t_max: float = 200.0,
        x_range: tuple[float, float] = (-1.5, 1.5),
        y_range: tuple[float, float] = (-1.5, 1.5),
    ) -> np.ndarray:
        """Compute basin-of-attraction map on a grid of initial conditions.

        For each (x0, y0) on the grid (with vx=vy=0), determine which magnet
        the pendulum settles to.

        Returns:
            Integer array of shape (grid_size, grid_size) with magnet indices.
        """
        xs = np.linspace(x_range[0], x_range[1], grid_size)
        ys = np.linspace(y_range[0], y_range[1], grid_size)
        basin = np.zeros((grid_size, grid_size), dtype=np.int32)

        for i, y0 in enumerate(ys):
            for j, x0 in enumerate(xs):
                config = SimulationConfig(
                    domain=self.config.domain,
                    dt=self.config.dt,
                    n_steps=self.config.n_steps,
                    parameters={
                        **{k: v for k, v in self.config.parameters.items()},
                        "x_0": x0,
                        "y_0": y0,
                        "vx_0": 0.0,
                        "vy_0": 0.0,
                    },
                )
                sim = MagneticPendulumSimulation(config)
                sim.reset()
                basin[i, j] = sim.find_attractor(t_max=t_max)

        return basin

    def compute_basin_entropy(
        self,
        grid_size: int = 50,
        t_max: float = 200.0,
        x_range: tuple[float, float] = (-1.5, 1.5),
        y_range: tuple[float, float] = (-1.5, 1.5),
    ) -> float:
        """Measure boundary complexity as fraction of border cells with mixed neighbors.

        A cell is a "boundary" cell if any of its 4-connected neighbors maps to
        a different attractor. The basin entropy is the fraction of all cells
        that are boundary cells.
        """
        basin = self.basin_map(grid_size, t_max, x_range, y_range)
        boundary_count = 0
        total = 0

        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                total += 1
                val = basin[i, j]
                neighbors = [
                    basin[i - 1, j], basin[i + 1, j],
                    basin[i, j - 1], basin[i, j + 1],
                ]
                if any(n != val for n in neighbors):
                    boundary_count += 1

        if total == 0:
            return 0.0
        return boundary_count / total

    def trajectory_to_attractor(
        self, x0: float, y0: float, t_max: float = 200.0, tol: float = 0.01,
    ) -> np.ndarray:
        """Return full trajectory from (x0, y0) until the pendulum settles.

        Args:
            x0, y0: initial position (vx=vy=0).
            t_max: maximum simulation time.
            tol: velocity threshold for "settled".

        Returns:
            Array of shape (N, 4) with [x, y, vx, vy] at each step.
        """
        config = SimulationConfig(
            domain=self.config.domain,
            dt=self.config.dt,
            n_steps=self.config.n_steps,
            parameters={
                **{k: v for k, v in self.config.parameters.items()},
                "x_0": x0,
                "y_0": y0,
                "vx_0": 0.0,
                "vy_0": 0.0,
            },
        )
        sim = MagneticPendulumSimulation(config)
        sim.reset()

        dt = self.config.dt
        max_steps = int(t_max / dt)
        trajectory = [sim._state.copy()]

        for _ in range(max_steps):
            sim.step()
            trajectory.append(sim._state.copy())
            speed = np.sqrt(sim._state[2] ** 2 + sim._state[3] ** 2)
            if speed < tol:
                break

        return np.array(trajectory)

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous energy (not conserved due to damping).

        E = 0.5*(vx^2 + vy^2) + 0.5*omega0^2*(x^2 + y^2)
            - sum_i [alpha / r_i]
        """
        if state is None:
            state = self._state
        x, y, vx, vy = state

        ke = 0.5 * (vx**2 + vy**2)
        pe_restoring = 0.5 * self.omega0_sq * (x**2 + y**2)

        pe_magnetic = 0.0
        for mx, my in self.magnet_positions:
            r = np.sqrt((x - mx)**2 + (y - my)**2 + self.d**2)
            pe_magnetic -= self.alpha / r

        return ke + pe_restoring + pe_magnetic
