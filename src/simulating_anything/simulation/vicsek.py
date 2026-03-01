"""Vicsek model of self-propelled particles (flocking / active matter).

Target rediscoveries:
- Order-disorder phase transition: low noise -> ordered (flocking), high noise -> disordered
- Order parameter: phi = |1/N * sum_i exp(i*theta_i)| in [0, 1]
- Critical noise eta_c depends on density rho = N/L^2
- Connection to XY model and Kuramoto synchronization
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class VicsekSimulation(SimulationEnvironment):
    """Vicsek model: minimal model of collective motion / flocking.

    N self-propelled particles move at constant speed v0.  At each step,
    each particle aligns its heading to the average heading of neighbors
    within radius R, plus random noise.

    Update rules (synchronous):
        theta_i(t+dt) = <theta_j>_{|r_j - r_i| < R} + eta * U(-pi, pi)
        x_i(t+dt) = x_i(t) + v0 * cos(theta_i(t+dt)) * dt
        y_i(t+dt) = y_i(t) + v0 * sin(theta_i(t+dt)) * dt

    with periodic boundary conditions on [0, L) x [0, L).

    State vector: [x_1, y_1, theta_1, x_2, y_2, theta_2, ...] (3*N array).

    Parameters:
        N: number of particles (default 100)
        L: box side length (default 10.0)
        v0: particle speed (default 0.5)
        R: interaction radius (default 1.0)
        eta: noise strength in [0, 1] (default 0.3)
        dt: timestep (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 100))
        self.L = p.get("L", 10.0)
        self.v0 = p.get("v0", 0.5)
        self.R = p.get("R", 1.0)
        self.eta = p.get("eta", 0.3)

        # Internal arrays: positions (N, 2) and headings (N,)
        self._pos: np.ndarray = np.zeros((self.N, 2))
        self._theta: np.ndarray = np.zeros(self.N)
        self._rng: np.random.Generator = np.random.default_rng(42)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize random positions in [0, L)^2 and headings in [-pi, pi)."""
        self._rng = np.random.default_rng(seed)
        self._pos = self._rng.uniform(0, self.L, size=(self.N, 2))
        self._theta = self._rng.uniform(-np.pi, np.pi, size=self.N)
        self._step_count = 0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using synchronous Vicsek update rules."""
        new_theta = self._compute_new_headings()
        self._theta = new_theta

        # Update positions
        dx = self.v0 * np.cos(self._theta) * self.config.dt
        dy = self.v0 * np.sin(self._theta) * self.config.dt
        self._pos[:, 0] += dx
        self._pos[:, 1] += dy

        # Periodic boundary conditions
        self._pos[:, 0] = self._pos[:, 0] % self.L
        self._pos[:, 1] = self._pos[:, 1] % self.L

        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return state as [x1, y1, theta1, x2, y2, theta2, ...] (3*N array)."""
        state = np.empty(3 * self.N, dtype=np.float64)
        state[0::3] = self._pos[:, 0]
        state[1::3] = self._pos[:, 1]
        state[2::3] = self._theta
        return state

    def _compute_new_headings(self) -> np.ndarray:
        """Compute new heading for each particle from neighbor averaging + noise.

        Uses periodic boundary conditions for neighbor detection.
        """
        new_theta = np.empty(self.N, dtype=np.float64)

        for i in range(self.N):
            # Find neighbors within radius R (periodic BC)
            neighbor_mask = self._find_neighbors(i)

            # Average heading using circular mean (atan2 of mean sin/cos)
            neighbor_thetas = self._theta[neighbor_mask]
            mean_sin = np.mean(np.sin(neighbor_thetas))
            mean_cos = np.mean(np.cos(neighbor_thetas))
            avg_heading = np.arctan2(mean_sin, mean_cos)

            # Add noise
            noise = self.eta * self._rng.uniform(-np.pi, np.pi)
            new_theta[i] = avg_heading + noise

        # Wrap headings to [-pi, pi)
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        return new_theta

    def _find_neighbors(self, i: int) -> np.ndarray:
        """Find all particles within radius R of particle i (periodic BC).

        Returns a boolean mask of shape (N,). Particle i is always included
        in its own neighborhood.
        """
        # Displacement with minimum image convention
        delta = self._pos - self._pos[i]
        delta -= self.L * np.round(delta / self.L)
        dist_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2
        return dist_sq < self.R ** 2

    def order_parameter(self) -> float:
        """Compute the Vicsek order parameter phi = |1/N * sum exp(i*theta)|.

        Returns a scalar in [0, 1]:
            phi ~ 0: disordered (random headings)
            phi ~ 1: ordered (aligned flock)
        """
        z = np.mean(np.exp(1j * self._theta))
        return float(np.abs(z))

    def order_parameter_sweep(
        self,
        eta_values: np.ndarray | list[float],
        n_steps: int = 200,
        n_avg: int = 50,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Sweep noise eta, measure steady-state order parameter.

        Args:
            eta_values: Noise values to sweep.
            n_steps: Total steps per noise value.
            n_avg: Number of final steps over which to average phi.
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys 'eta', 'phi_mean', 'phi_std'.
        """
        eta_values = np.asarray(eta_values)
        phi_mean = np.empty(len(eta_values))
        phi_std = np.empty(len(eta_values))

        original_eta = self.eta
        for j, eta in enumerate(eta_values):
            self.eta = eta
            self.reset(seed=seed)

            # Run transient
            n_transient = n_steps - n_avg
            for _ in range(n_transient):
                self.step()

            # Measure
            phi_vals = []
            for _ in range(n_avg):
                self.step()
                phi_vals.append(self.order_parameter())

            phi_mean[j] = np.mean(phi_vals)
            phi_std[j] = np.std(phi_vals)

        self.eta = original_eta
        return {
            "eta": eta_values,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
        }

    def compute_density(self) -> float:
        """Compute particle density rho = N / L^2."""
        return self.N / (self.L ** 2)

    def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) position arrays each of shape (N,)."""
        return self._pos[:, 0].copy(), self._pos[:, 1].copy()

    def get_headings(self) -> np.ndarray:
        """Return heading array of shape (N,)."""
        return self._theta.copy()
