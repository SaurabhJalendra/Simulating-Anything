"""Kepler two-body orbital mechanics simulation.

Target rediscoveries:
- Kepler's third law: T^2 = (4*pi^2 / GM) * a^3, i.e. T = 2*pi * a^(3/2) / sqrt(GM)
- Energy conservation: E = 0.5*(v_r^2 + v_theta^2) - GM/r = const
- Angular momentum conservation: L = r * v_theta = const
- Semi-major axis: a = -GM / (2*E) for bound orbits
- Eccentricity: e = sqrt(1 + 2*E*L^2 / (GM)^2)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class KeplerOrbit(SimulationEnvironment):
    """Two-body Kepler orbit in polar coordinates (reduced mass system).

    State vector: [r, theta, v_r, v_theta]

    Equations of motion:
        dr/dt = v_r
        dtheta/dt = v_theta / r
        dv_r/dt = v_theta^2 / r - GM / r^2
        dv_theta/dt = -v_r * v_theta / r

    Parameters:
        GM: gravitational parameter (default 1.0, natural units)
        initial_r: initial radial distance (default 1.0)
        eccentricity: orbital eccentricity 0 <= e < 1 (default 0.5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.GM = p.get("GM", 1.0)
        self.initial_r = p.get("initial_r", 1.0)
        self.ecc = p.get("eccentricity", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize orbit at perihelion.

        At perihelion: r = a*(1-e), v_r = 0,
        v_theta = sqrt(GM * (1+e) / (a*(1-e)))
        where a is the semi-major axis derived from initial_r and eccentricity.
        For initial_r interpreted as semi-major axis a:
            r_peri = a * (1 - e)
        """
        a = self.initial_r
        e = self.ecc
        r_peri = a * (1.0 - e)
        # v_theta at perihelion from vis-viva and angular momentum
        v_theta_peri = np.sqrt(self.GM * (1.0 + e) / (a * (1.0 - e)))

        self._state = np.array(
            [r_peri, 0.0, 0.0, v_theta_peri], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [r, theta, v_r, v_theta]."""
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
        """Equations of motion in polar coordinates.

        dr/dt = v_r
        dtheta/dt = v_theta / r
        dv_r/dt = v_theta^2 / r - GM / r^2
        dv_theta/dt = -v_r * v_theta / r
        """
        r, theta, v_r, v_theta = state
        dr = v_r
        dtheta = v_theta / r
        dv_r = v_theta**2 / r - self.GM / r**2
        dv_theta = -v_r * v_theta / r
        return np.array([dr, dtheta, dv_r, dv_theta])

    @property
    def energy(self) -> float:
        """Specific orbital energy: E = 0.5*(v_r^2 + v_theta^2) - GM/r."""
        r, _, v_r, v_theta = self._state
        return 0.5 * (v_r**2 + v_theta**2) - self.GM / r

    @property
    def angular_momentum(self) -> float:
        """Specific angular momentum: L = r * v_theta (conserved)."""
        r, _, _, v_theta = self._state
        return r * v_theta

    @property
    def semi_major_axis(self) -> float:
        """Semi-major axis: a = -GM / (2*E) for bound orbits (E < 0)."""
        E = self.energy
        if E >= 0:
            return float("inf")  # Unbound orbit
        return -self.GM / (2.0 * E)

    @property
    def eccentricity_from_state(self) -> float:
        """Eccentricity computed from current energy and angular momentum.

        e = sqrt(1 + 2*E*L^2 / (GM)^2)
        """
        E = self.energy
        L = self.angular_momentum
        discriminant = 1.0 + 2.0 * E * L**2 / self.GM**2
        if discriminant < 0:
            return 0.0  # Numerical noise for near-circular
        return np.sqrt(discriminant)

    @property
    def period(self) -> float:
        """Orbital period from Kepler's third law: T = 2*pi*a^(3/2)/sqrt(GM)."""
        a = self.semi_major_axis
        if a == float("inf"):
            return float("inf")
        return 2.0 * np.pi * a**1.5 / np.sqrt(self.GM)

    @property
    def perihelion(self) -> float:
        """Perihelion distance: r_min = a * (1 - e)."""
        return self.semi_major_axis * (1.0 - self.eccentricity_from_state)

    @property
    def aphelion(self) -> float:
        """Aphelion distance: r_max = a * (1 + e)."""
        return self.semi_major_axis * (1.0 + self.eccentricity_from_state)
