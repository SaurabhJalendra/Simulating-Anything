"""Kuramoto model of coupled oscillators.

Target rediscoveries:
- Order parameter: r(K) transition from 0 to 1 as K increases
- Critical coupling: K_c = 2/(pi*g(0)) for symmetric unimodal distributions
  - For uniform distribution on [-1, 1]: g(0) = 0.5, K_c = 4/pi ~ 1.273
  - For standard normal: g(0) = 1/sqrt(2*pi), K_c = sqrt(2*pi) ~ 2.507
- Self-consistency equation: r = 1 - K_c/K for K > K_c (mean-field)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class KuramotoSimulation(SimulationEnvironment):
    """Kuramoto model: d(theta_i)/dt = omega_i + (K/N)*sum_j sin(theta_j - theta_i).

    State vector: [theta_1, theta_2, ..., theta_N] (phases of N oscillators).

    Parameters:
        N: number of oscillators
        K: coupling strength
        omega_std: std of natural frequency distribution (default 1.0)
        omega_mean: mean of natural frequency distribution (default 0.0)
        distribution: 'uniform' or 'normal' (default 'uniform')
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 50))
        self.K = p.get("K", 1.0)
        self.omega_std = p.get("omega_std", 1.0)
        self.omega_mean = p.get("omega_mean", 0.0)
        self.distribution = "uniform"  # String params can't go in parameters dict

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize phases uniformly on [0, 2*pi] and draw natural frequencies."""
        rng = np.random.default_rng(seed)
        self._state = rng.uniform(0, 2 * np.pi, self.N).astype(np.float64)
        self._step_count = 0

        # Draw natural frequencies from chosen distribution
        if self.distribution == "normal":
            self.omega = (
                self.omega_mean + self.omega_std * rng.standard_normal(self.N)
            ).astype(np.float64)
        else:
            # Uniform on [omega_mean - omega_std, omega_mean + omega_std]
            half_width = self.omega_std
            self.omega = rng.uniform(
                self.omega_mean - half_width,
                self.omega_mean + half_width,
                self.N,
            ).astype(np.float64)

        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current phases."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, theta: np.ndarray) -> np.ndarray:
        """Compute d(theta)/dt for all oscillators."""
        # Coupling term: (K/N) * sum_j sin(theta_j - theta_i)
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]  # diff[i,j] = theta_j - theta_i
        coupling = (self.K / self.N) * np.sum(np.sin(diff), axis=1)
        return self.omega + coupling

    @property
    def order_parameter(self) -> tuple[float, float]:
        """Compute the Kuramoto order parameter r*exp(i*psi).

        Returns (r, psi) where r in [0, 1] measures synchronization
        and psi is the mean phase.
        """
        z = np.mean(np.exp(1j * self._state))
        return float(np.abs(z)), float(np.angle(z))

    @property
    def order_parameter_r(self) -> float:
        """Synchronization order parameter r in [0, 1]."""
        return self.order_parameter[0]

    @property
    def critical_coupling(self) -> float:
        """Theoretical critical coupling K_c = 2/(pi*g(0)).

        For uniform on [-w, w]: g(0) = 1/(2w), so K_c = 4w/pi
        For normal(0, sigma): g(0) = 1/(sigma*sqrt(2*pi)), so K_c = sigma*2*sqrt(2*pi)/pi
        """
        if self.distribution == "normal":
            return 2 * self.omega_std * np.sqrt(2 * np.pi) / np.pi
        else:
            # Uniform on [-omega_std, omega_std]
            return 4 * self.omega_std / np.pi

    def measure_steady_state_r(
        self,
        n_transient_steps: int = 5000,
        n_measure_steps: int = 2000,
        seed: int | None = None,
    ) -> float:
        """Measure the steady-state order parameter after transient."""
        self.reset(seed=seed)

        for _ in range(n_transient_steps):
            self.step()

        r_values = []
        for _ in range(n_measure_steps):
            self.step()
            r_values.append(self.order_parameter_r)

        return float(np.mean(r_values))
