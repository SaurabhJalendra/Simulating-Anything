"""Langford system simulation -- Hopf-Hopf (double Hopf) bifurcation.

The Langford system exhibits aperiodic flow on a torus:
    dx/dt = (z - b)*x - d*y
    dy/dt = d*x + (z - b)*y
    dz/dt = c + a*z - z^3/3 - (x^2 + y^2)*(1 + e*z) + f*z*x^3

Parameters:
    a: linear growth rate (default 0.95)
    b: Hopf parameter (default 0.7)
    c: constant forcing (default 0.6)
    d: rotation rate (default 3.5)
    e: nonlinear coupling (default 0.25)
    f: higher-order coupling (default 0.1)

Key physics:
- 3D system exhibiting Hopf-Hopf (double Hopf) bifurcation
- Can produce torus (quasiperiodic) dynamics
- Rich bifurcation structure: limit cycles, tori, chaos
- System has rotational symmetry in (x,y) plane
- Parameter c controls the vertical offset / folding

Target rediscoveries:
- SINDy recovery of Langford ODEs
- Quasiperiodic vs chaotic behavior detection (power spectrum)
- Torus characterization (two incommensurate frequencies)
- Lyapunov exponent estimation
- Parameter sweep for Hopf-Hopf bifurcation
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LangfordSimulation(SimulationEnvironment):
    """Langford system: Hopf-Hopf bifurcation and torus dynamics.

    State vector: [x, y, z]

    ODEs:
        dx/dt = (z - b)*x - d*y
        dy/dt = d*x + (z - b)*y
        dz/dt = c + a*z - z^3/3 - (x^2 + y^2)*(1 + e*z) + f*z*x^3

    The system exhibits rotational symmetry in the (x, y) plane. The
    parameter b controls the Hopf bifurcation, and the interplay between
    parameters produces limit cycles, tori, and chaos.

    Parameters:
        a: linear growth rate (default 0.95)
        b: Hopf parameter (default 0.7)
        c: constant forcing (default 0.6)
        d: rotation rate (default 3.5)
        e: nonlinear coupling (default 0.25)
        f: higher-order coupling (default 0.1)
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.95)
        self.b = p.get("b", 0.7)
        self.c = p.get("c", 0.6)
        self.d = p.get("d", 3.5)
        self.e = p.get("e", 0.25)
        self.f = p.get("f", 0.1)
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.0)
        self.z_0 = p.get("z_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Langford state."""
        self._state = np.array(
            [self.x_0, self.y_0, self.z_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, z]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Langford system equations."""
        x, y, z = state
        r2 = x * x + y * y
        dx = (z - self.b) * x - self.d * y
        dy = self.d * x + (z - self.b) * y
        dz = (
            self.c
            + self.a * z
            - z**3 / 3.0
            - r2 * (1.0 + self.e * z)
            + self.f * z * x**3
        )
        return np.array([dx, dy, dz])

    def estimate_lyapunov(
        self, n_steps: int = 30000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby
        trajectories, renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def compute_radius_history(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> np.ndarray:
        """Compute the cylindrical radius r = sqrt(x^2 + y^2) over time.

        Useful for detecting torus dynamics and quasiperiodic behavior.

        Args:
            n_steps: Number of steps to record after transient.
            n_transient: Steps to skip for transient.

        Returns:
            Array of radius values, shape (n_steps,).
        """
        self.reset()
        for _ in range(n_transient):
            self.step()

        radii = np.empty(n_steps)
        for i in range(n_steps):
            state = self.step()
            radii[i] = np.sqrt(state[0] ** 2 + state[1] ** 2)

        return radii

    def compute_frequency_spectrum(
        self, n_steps: int = 8192, n_transient: int = 5000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the power spectrum of the x-component.

        Returns (frequencies, power) for detecting quasiperiodic behavior
        (two incommensurate peaks) vs chaos (broadband spectrum).

        Args:
            n_steps: Number of data points (preferably power of 2 for FFT).
            n_transient: Steps to skip for transient.

        Returns:
            Tuple of (frequencies, power_spectrum) arrays.
        """
        self.reset()
        for _ in range(n_transient):
            self.step()

        x_vals = np.empty(n_steps)
        for i in range(n_steps):
            state = self.step()
            x_vals[i] = state[0]

        # Remove mean
        x_vals -= np.mean(x_vals)

        # FFT
        dt = self.config.dt
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        power = np.abs(np.fft.rfft(x_vals)) ** 2 / n_steps

        return freqs, power

    def compute_trajectory_statistics(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Compute time-averaged statistics of the trajectory.

        Args:
            n_steps: Number of steps to measure after transient.
            n_transient: Steps to skip for transient.

        Returns:
            Dict with mean, std, min, max for each component and radius.
        """
        self.reset()

        for _ in range(n_transient):
            self.step()

        xs, ys, zs = [], [], []
        for _ in range(n_steps):
            state = self.step()
            xs.append(state[0])
            ys.append(state[1])
            zs.append(state[2])

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        rs = np.sqrt(xs**2 + ys**2)

        return {
            "x_mean": float(np.mean(xs)),
            "y_mean": float(np.mean(ys)),
            "z_mean": float(np.mean(zs)),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "z_std": float(np.std(zs)),
            "r_mean": float(np.mean(rs)),
            "r_std": float(np.std(rs)),
            "r_max": float(np.max(rs)),
            "z_min": float(np.min(zs)),
            "z_max": float(np.max(zs)),
        }
