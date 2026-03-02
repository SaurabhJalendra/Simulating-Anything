"""Lorenz-Haken (Maxwell-Bloch) laser system simulation.

The Lorenz-Haken system describes a single-mode laser in terms of
the slowly-varying envelope of the electric field, the macroscopic
polarization, and the population inversion. The equations are:

    dx/dt = sigma * (y - x)
    dy/dt = (r - z) * x - y
    dz/dt = x * y - b * z

where:
    x = electric field amplitude
    y = macroscopic polarization
    z = population inversion
    sigma = kappa / gamma_perp (cavity decay / polarization decay)
    r = pump parameter normalized to threshold (r=1 is lasing threshold)
    b = gamma_perp / gamma_parallel (polarization / inversion decay)

These equations are structurally identical to the Lorenz convection
system but arise from the Maxwell-Bloch equations for a resonant
two-level medium in a ring cavity (Haken 1975).

Classic laser parameters: sigma=3.0, r=25.0, b=1.0
(differs from Lorenz standard sigma=10, rho=28, beta=8/3)

Target rediscoveries:
- SINDy recovery of Maxwell-Bloch ODEs
- Lasing threshold at r = 1
- Chaos transition via pump-parameter sweep
- Lyapunov exponent estimation
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LorenzHakenSimulation(SimulationEnvironment):
    """Lorenz-Haken (Maxwell-Bloch) single-mode laser model.

    State vector: [x, y, z]
        x: electric field amplitude
        y: macroscopic polarization
        z: population inversion

    ODEs:
        dx/dt = sigma * (y - x)
        dy/dt = (r - z) * x - y
        dz/dt = x * y - b * z

    Parameters:
        sigma: ratio of cavity decay rate to polarization decay rate
               (kappa / gamma_perp). Classic laser value: 3.0
        r: pump parameter normalized to threshold. r=1 is the first
           lasing threshold. Classic chaotic value: 25.0
        b: ratio of polarization decay rate to population inversion
           decay rate (gamma_perp / gamma_parallel). Classic: 1.0
        x_0, y_0, z_0: initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.sigma = p.get("sigma", 3.0)
        self.r = p.get("r", 25.0)
        self.b = p.get("b", 1.0)
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Lorenz-Haken state."""
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
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Maxwell-Bloch laser equations.

        dx/dt = sigma*(y - x)
        dy/dt = (r - z)*x - y
        dz/dt = x*y - b*z
        """
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = (self.r - z) * x - y
        dz = x * y - self.b * z
        return np.array([dx, dy, dz])

    @property
    def lasing_threshold(self) -> float:
        """Return the lasing threshold pump parameter.

        The first lasing instability occurs at r = 1, independent of
        sigma and b. Below r=1 the only stable state is the trivial
        (non-lasing) fixed point at the origin.
        """
        return 1.0

    @property
    def fixed_points(self) -> list[np.ndarray]:
        """Compute the fixed points of the Lorenz-Haken system.

        For r <= 1: only the origin (non-lasing state).
        For r > 1: origin plus two symmetric lasing states:
            C+/- = (+/- sqrt(b*(r-1)), +/- sqrt(b*(r-1)), r-1)
        """
        points = [np.array([0.0, 0.0, 0.0])]
        if self.r > 1:
            c = np.sqrt(self.b * (self.r - 1))
            points.append(np.array([c, c, self.r - 1]))
            points.append(np.array([-c, -c, self.r - 1]))
        return points

    @property
    def second_threshold(self) -> float:
        """Compute the second (Hopf) instability threshold.

        For the lasing fixed points, the Hopf bifurcation occurs at:
            r_H = sigma * (sigma + b + 3) / (sigma - b - 1)

        This requires sigma > b + 1; otherwise no second instability.
        Returns np.inf if sigma <= b + 1.
        """
        if self.sigma <= self.b + 1:
            return np.inf
        return self.sigma * (self.sigma + self.b + 3) / (self.sigma - self.b - 1)

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby trajectories,
        renormalize when they diverge too far.
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

    def measure_intensity(self, n_transient: int = 5000, n_measure: int = 10000) -> dict:
        """Measure laser intensity statistics after transient.

        The laser intensity is proportional to x^2 (electric field squared).

        Returns dict with mean, std, max intensity and the time series.
        """
        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect intensity = x^2
        intensities = np.empty(n_measure)
        for i in range(n_measure):
            self.step()
            intensities[i] = self._state[0] ** 2

        return {
            "mean_intensity": float(np.mean(intensities)),
            "std_intensity": float(np.std(intensities)),
            "max_intensity": float(np.max(intensities)),
            "intensities": intensities,
        }
