"""1D Gray-Scott reaction-diffusion simulation.

Target rediscoveries:
- Self-replicating pulses in 1D
- Pulse splitting bifurcation as a function of (f, k)
- Pulse speed measurement
- Existence of stable localized pulses

Equations:
    du/dt = D_u * d^2u/dx^2 - u*v^2 + f*(1 - u)
    dv/dt = D_v * d^2v/dx^2 + u*v^2 - (f + k)*v

Uses Karl Sims convention: D_u=0.16, D_v=0.08 with unscaled Laplacian.
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 1D Laplacian with periodic boundary conditions."""
    return (
        np.roll(field, 1) + np.roll(field, -1) - 2.0 * field
    ) / (dx * dx)


class GrayScott1DSimulation(SimulationEnvironment):
    """1D Gray-Scott reaction-diffusion model.

    State is a (2, N) array of u and v concentrations on a 1D periodic grid.
    Self-replicating pulses, splitting, and traveling pulses emerge from
    the interplay of diffusion, autocatalytic growth (u*v^2), and removal.

    Parameters:
        D_u: diffusion coefficient for u (default 0.16)
        D_v: diffusion coefficient for v (default 0.08)
        f: feed rate (default 0.04)
        k: kill rate (default 0.06)
        N: number of grid points (default 256)
        L: domain length (default 2.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.D_u = p.get("D_u", 0.16)
        self.D_v = p.get("D_v", 0.08)
        self.f = p.get("f", 0.04)
        self.k = p.get("k", 0.06)
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 2.5)
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # CFL stability check: dt < dx^2 / (4 * max(D_u, D_v))
        D_max = max(self.D_u, self.D_v)
        self.cfl_limit = self.dx**2 / (4.0 * D_max) if D_max > 0 else np.inf

        self._u = np.ones(self.N, dtype=np.float64)
        self._v = np.zeros(self.N, dtype=np.float64)
        self._state = np.stack([self._u, self._v], axis=0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize u=1, v=0 everywhere with a localized perturbation.

        A small region in the center seeds v with a Gaussian bump to
        trigger pulse formation.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        self._u = np.ones(self.N, dtype=np.float64)
        self._v = np.zeros(self.N, dtype=np.float64)

        # Seed a localized perturbation in the center
        center = self.N // 2
        width = max(self.N // 20, 3)
        sigma = width / 3.0
        x_idx = np.arange(self.N)
        gaussian = np.exp(-((x_idx - center) ** 2) / (2 * sigma**2))
        self._u -= 0.5 * gaussian
        self._v += 0.25 * gaussian

        # Small noise for symmetry breaking
        self._u += 0.01 * rng.standard_normal(self.N)
        self._v += 0.01 * np.abs(rng.standard_normal(self.N))

        # Clamp to physically valid range
        self._u = np.clip(self._u, 0.0, 1.0)
        self._v = np.clip(self._v, 0.0, 1.0)

        self._step_count = 0
        self._state = np.stack([self._u, self._v], axis=0)
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using explicit Euler."""
        dt = self.config.dt
        dx = self.dx

        lap_u = _laplacian_1d(self._u, dx)
        lap_v = _laplacian_1d(self._v, dx)

        uvv = self._u * self._v * self._v
        du = self.D_u * lap_u - uvv + self.f * (1.0 - self._u)
        dv = self.D_v * lap_v + uvv - (self.f + self.k) * self._v

        self._u = self._u + dt * du
        self._v = self._v + dt * dv

        self._step_count += 1
        self._state = np.stack([self._u, self._v], axis=0)
        return self._state

    def observe(self) -> np.ndarray:
        """Return current (u, v) field as shape (2, N)."""
        return self._state

    def count_pulses(self, threshold: float | None = None) -> int:
        """Count the number of pulses (peaks) in the v field.

        A pulse is detected as a local maximum in v that exceeds a threshold.

        Args:
            threshold: Minimum v value to count as a pulse. If None, uses
                half the maximum v value, with a floor of 0.01.

        Returns:
            Number of detected pulses.
        """
        v = self._v
        if threshold is None:
            threshold = max(0.5 * np.max(v), 0.01)

        # Find local maxima: v[i] > v[i-1] and v[i] > v[i+1]
        # Use periodic boundary conditions via roll
        left = np.roll(v, 1)
        right = np.roll(v, -1)
        is_peak = (v > left) & (v > right) & (v > threshold)
        return int(np.sum(is_peak))

    def measure_pulse_speed(
        self, n_steps: int = 100
    ) -> float:
        """Measure pulse propagation speed by tracking the peak of v.

        Runs n_steps forward and measures peak displacement.
        Only meaningful when a single pulse exists.

        Args:
            n_steps: Number of steps to track.

        Returns:
            Estimated pulse speed in spatial units per time unit.
        """
        v_before = self._v.copy()
        peak_before = np.argmax(v_before)

        for _ in range(n_steps):
            self.step()

        v_after = self._v.copy()
        peak_after = np.argmax(v_after)

        # Handle periodic wrap-around
        displacement = (peak_after - peak_before) * self.dx
        if abs(displacement) > self.L / 2:
            if displacement > 0:
                displacement -= self.L
            else:
                displacement += self.L

        total_time = n_steps * self.config.dt
        speed = abs(displacement) / total_time if total_time > 0 else 0.0
        return speed

    @property
    def total_v(self) -> float:
        """Total v concentration (integral of v)."""
        return float(np.sum(self._v) * self.dx)

    @property
    def total_u(self) -> float:
        """Total u concentration (integral of u)."""
        return float(np.sum(self._u) * self.dx)

    @property
    def max_v(self) -> float:
        """Maximum v concentration."""
        return float(np.max(self._v))
