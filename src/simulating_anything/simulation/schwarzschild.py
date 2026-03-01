"""Schwarzschild geodesic simulation -- massive particle orbits around a black hole.

Simulates timelike geodesics in the Schwarzschild spacetime using the
effective potential formulation in natural units (G=c=1).

Target rediscoveries:
- ISCO radius: r_isco = 6M
- Effective potential: V_eff(r) = -M/r + L^2/(2r^2) - ML^2/r^3
- Energy conservation along geodesics
- Orbital precession (non-Keplerian, GR effect)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SchwarzschildGeodesic(SimulationEnvironment):
    """Schwarzschild geodesic integrator for massive particles.

    Integrates the geodesic equations for a massive test particle in the
    Schwarzschild metric using the effective potential formulation.  The
    parameterization is in proper time tau.

    State vector: [r, phi, p_r, dphi_dtau] where
        r = radial coordinate (Schwarzschild)
        phi = azimuthal angle
        p_r = dr/dtau (radial proper-time velocity)
        dphi_dtau = L/r^2 (angular proper-time velocity)

    Parameters:
        M: black hole mass (default 1.0)
        L: specific angular momentum (default 4.0)
        r_0: initial radial coordinate (default 10.0)
        pr_0: initial radial velocity dr/dtau (default 0.0)
    """

    # Minimum allowed radius as a fraction above the event horizon.
    # Below this, the coordinate singularity makes integration meaningless.
    _R_CAPTURE_FACTOR = 1.005

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.M = p.get("M", 1.0)
        self.L = p.get("L", 4.0)
        self.r_0 = p.get("r_0", 10.0)
        self.pr_0 = p.get("pr_0", 0.0)
        self._captured = False

    @property
    def schwarzschild_radius(self) -> float:
        """Event horizon radius r_s = 2M."""
        return 2.0 * self.M

    def effective_potential(self, r: float | np.ndarray) -> float | np.ndarray:
        """Compute the effective potential V_eff(r).

        V_eff(r) = -M/r + L^2/(2*r^2) - M*L^2/r^3

        This combines the Newtonian gravity, centrifugal barrier, and the
        general-relativistic correction term (-ML^2/r^3).
        """
        M, L = self.M, self.L
        return -M / r + L**2 / (2.0 * r**2) - M * L**2 / r**3

    def effective_potential_derivative(self, r: float) -> float:
        """Compute dV_eff/dr.

        dV_eff/dr = M/r^2 - L^2/r^3 + 3*M*L^2/r^4
        """
        M, L = self.M, self.L
        return M / r**2 - L**2 / r**3 + 3.0 * M * L**2 / r**4

    @property
    def energy(self) -> float:
        """Specific orbital energy E = (1/2)*p_r^2 + V_eff(r).

        This is a constant of motion along the geodesic.
        """
        r = self._state[0]
        pr = self._state[2]
        return 0.5 * pr**2 + self.effective_potential(r)

    @property
    def angular_momentum(self) -> float:
        """Specific angular momentum L (constant of motion)."""
        return self.L

    def is_captured(self) -> bool:
        """Return True if the particle has fallen within the capture threshold.

        The integration is frozen when r < 2M * _R_CAPTURE_FACTOR to avoid
        the coordinate singularity at r = 2M.  Once frozen, the particle is
        considered captured.
        """
        return self._captured

    def find_isco(self) -> float:
        """Return the ISCO radius.

        For the Schwarzschild metric, the innermost stable circular orbit
        is at r_isco = 6M, independent of L.
        """
        return 6.0 * self.M

    def find_circular_orbit_radius(self, L: float | None = None) -> float | None:
        """Find the radius of a stable circular orbit for given L.

        Circular orbits satisfy dV_eff/dr = 0, which gives:
            M*r^2 - L^2*r + 3*M*L^2 = 0

        Returns the larger root (stable orbit) or None if L is too small
        for a stable orbit to exist.
        """
        if L is None:
            L = self.L
        M = self.M
        # Quadratic in r: M*r^2 - L^2*r + 3*M*L^2 = 0
        a_coeff = M
        b_coeff = -L**2
        c_coeff = 3.0 * M * L**2
        discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
        if discriminant < 0:
            return None
        sqrt_disc = np.sqrt(discriminant)
        # Larger root is the stable orbit
        r_stable = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)
        if r_stable <= self.schwarzschild_radius:
            return None
        return float(r_stable)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize geodesic state [r, phi, p_r, dphi/dtau]."""
        r = self.r_0
        phi = 0.0
        pr = self.pr_0
        dphi = self.L / r**2
        self._state = np.array([r, phi, pr, dphi], dtype=np.float64)
        self._step_count = 0
        self._captured = False
        return self._state

    def step(self) -> np.ndarray:
        """Advance one proper-time step using RK4.

        If the particle approaches too close to the event horizon
        (r < 2.01*M), the step is rejected and the state is frozen to
        avoid numerical singularity.
        """
        if self._captured:
            return self._state
        self._rk4_step()
        self._step_count += 1
        # Check capture
        if self._state[0] < self.schwarzschild_radius * self._R_CAPTURE_FACTOR:
            self._captured = True
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [r, phi, p_r, dphi/dtau]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration of geodesic equations."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        y2 = y + 0.5 * dt * k1
        if y2[0] < self.schwarzschild_radius * self._R_CAPTURE_FACTOR:
            self._state = y2
            self._captured = True
            return

        k2 = self._derivatives(y2)
        y3 = y + 0.5 * dt * k2
        if y3[0] < self.schwarzschild_radius * self._R_CAPTURE_FACTOR:
            self._state = y3
            self._captured = True
            return

        k3 = self._derivatives(y3)
        y4 = y + dt * k3
        if y4[0] < self.schwarzschild_radius * self._R_CAPTURE_FACTOR:
            self._state = y4
            self._captured = True
            return

        k4 = self._derivatives(y4)
        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute the geodesic equations of motion.

        dr/dtau = p_r
        dphi/dtau = L / r^2
        dp_r/dtau = -M/r^2 + L^2/r^3 - 3*M*L^2/r^4

        The fourth component (dphi/dtau) is derived, not independently
        integrated, since L is conserved.
        """
        r = state[0]
        pr = state[2]
        M, L = self.M, self.L

        dr = pr
        dphi = L / r**2
        # Radial acceleration from effective potential gradient
        dpr = -M / r**2 + L**2 / r**3 - 3.0 * M * L**2 / r**4
        # dphi/dtau changes as r changes: d(L/r^2)/dtau = -2L*dr/(r^3*dtau)
        ddphi = -2.0 * L * pr / r**3

        return np.array([dr, dphi, dpr, ddphi])

    def compute_trajectory_xy(
        self, states: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert (r, phi) trajectory to Cartesian (x, y) for visualization.

        Args:
            states: Array of shape (n, 4). If None, uses the last collected
                trajectory from run().

        Returns:
            Tuple of (x, y) arrays.
        """
        if states is None:
            states = np.array(self._trajectory_states)
        r = states[:, 0]
        phi = states[:, 1]
        return r * np.cos(phi), r * np.sin(phi)
