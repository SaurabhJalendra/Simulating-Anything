"""Lorenz-Stommel coupled atmosphere-ocean simulation.

A novel coupled dynamical system combining the Lorenz atmospheric chaos model
with the Stommel thermohaline circulation model. This coupling is designed to
explore emergent behavior when chaotic atmospheric forcing drives ocean
thermohaline dynamics.

Lorenz atmosphere (chaotic 3D ODE):
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho_eff - z) - y
    dz/dt = x*y - beta*z

Stommel ocean (thermohaline box model):
    dT/dt = eta1_eff - T - |T - S| * T
    dS/dt = eta2 - delta * S - |T - S| * S

Coupling mechanism:
    eta1_eff = eta1 + coupling_strength * x
        (Lorenz atmospheric temperature anomaly x modulates ocean heat flux)
    rho_eff = rho + ocean_feedback * |T - S|
        (Ocean overturning circulation strength modulates atmospheric convection)

This is a novel model -- no closed-form analytical results exist for the
coupled system. The goal is to discover emergent phenomena: synchronization,
regime transitions, and coupling-dependent Lyapunov exponents.

Target discoveries:
- Coupling-dependent Lyapunov exponent (does coupling stabilize or enhance chaos?)
- Atmosphere-ocean synchronization metrics
- SINDy ODE recovery of the 5D coupled system
- PySR expressions for coupling-dependent quantities
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LorenzStommelSimulation(SimulationEnvironment):
    """Coupled Lorenz-Stommel atmosphere-ocean model.

    State vector: [x, y, z, T, S]
        x, y, z: Lorenz atmospheric variables
        T: ocean temperature difference (equatorial - polar)
        S: ocean salinity difference (equatorial - polar)

    The Lorenz subsystem (atmosphere) chaotically forces the Stommel subsystem
    (ocean thermohaline circulation) through the heat flux parameter. The ocean
    feeds back into the atmosphere by modulating the effective Rayleigh number.

    Parameters:
        sigma: Lorenz Prandtl number (default 10.0)
        rho: Lorenz Rayleigh number (default 28.0)
        beta: Lorenz geometric factor (default 8/3)
        eta1: Stommel thermal forcing (default 3.0)
        eta2: Stommel haline forcing (default 1.0)
        delta: Stommel salinity relaxation rate (default 0.3)
        coupling_strength: atmosphere -> ocean coupling (default 0.1)
        ocean_feedback: ocean -> atmosphere feedback (default 0.01)
        x_0, y_0, z_0: Lorenz initial conditions
        T_0, S_0: Stommel initial conditions
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        # Lorenz parameters
        self.sigma = p.get("sigma", 10.0)
        self.rho = p.get("rho", 28.0)
        self.beta = p.get("beta", 8.0 / 3.0)
        # Stommel parameters
        self.eta1 = p.get("eta1", 3.0)
        self.eta2 = p.get("eta2", 1.0)
        self.delta = p.get("delta", 0.3)
        # Coupling parameters
        self.coupling_strength = p.get("coupling_strength", 0.1)
        self.ocean_feedback = p.get("ocean_feedback", 0.01)
        # Initial conditions
        self.x_0 = p.get("x_0", 1.0)
        self.y_0 = p.get("y_0", 1.0)
        self.z_0 = p.get("z_0", 1.0)
        self.T_0 = p.get("T_0", 2.0)
        self.S_0 = p.get("S_0", 1.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the coupled atmosphere-ocean state."""
        self._state = np.array(
            [self.x_0, self.y_0, self.z_0, self.T_0, self.S_0],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, z, T, S]."""
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
        """Compute the coupled Lorenz-Stommel derivatives.

        The coupling is bidirectional:
        - Lorenz x modulates Stommel eta1 (atmosphere heats ocean)
        - Stommel |T-S| modulates Lorenz rho (ocean circulation affects
          atmospheric convection)
        """
        x, y, z, T, S = state

        # Ocean overturning flow rate
        q = abs(T - S)

        # Effective parameters with coupling
        eta1_eff = self.eta1 + self.coupling_strength * x
        rho_eff = self.rho + self.ocean_feedback * q

        # Lorenz atmosphere equations (with ocean feedback on rho)
        dx = self.sigma * (y - x)
        dy = x * (rho_eff - z) - y
        dz = x * y - self.beta * z

        # Stommel ocean equations (with atmospheric forcing on eta1)
        dT = eta1_eff - T - q * T
        dS = self.eta2 - self.delta * S - q * S

        return np.array([dx, dy, dz, dT, dS])

    def _lorenz_derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute standalone Lorenz derivatives (no coupling).

        Used for Lyapunov estimation on the atmospheric subsystem.
        """
        x, y, z = state[:3]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def atmosphere_state(self) -> np.ndarray:
        """Return the Lorenz atmospheric variables [x, y, z]."""
        return self._state[:3].copy()

    def ocean_state(self) -> np.ndarray:
        """Return the Stommel ocean variables [T, S]."""
        return self._state[3:].copy()

    def flow_rate(self) -> float:
        """Compute the thermohaline overturning rate q = |T - S|."""
        T, S = self._state[3], self._state[4]
        return float(abs(T - S))

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent of the coupled system.

        Uses the Wolf et al. (1985) method: track two nearby trajectories
        in the full 5D state space and renormalize periodically.

        Args:
            n_steps: Number of integration steps for estimation.
            dt: Timestep (uses config.dt if None).

        Returns:
            Estimated largest Lyapunov exponent.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1.copy()
        state2[0] += eps  # Perturb x

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            state1 = self._rk4_advance(state1, dt)
            state2 = self._rk4_advance(state2, dt)

            # Bail out if trajectory diverges
            if not (np.all(np.isfinite(state1))
                    and np.all(np.isfinite(state2))):
                break

            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def _rk4_advance(
        self, state: np.ndarray, dt: float
    ) -> np.ndarray:
        """Advance a state vector one RK4 step (pure function)."""
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * dt * k1)
        k3 = self._derivatives(state + 0.5 * dt * k2)
        k4 = self._derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def atmosphere_ocean_correlation(
        self, n_steps: int = 10000, n_transient: int = 2000
    ) -> dict[str, float]:
        """Measure cross-correlation between atmosphere and ocean variables.

        Skips transient, then computes Pearson correlations between
        Lorenz x and Stommel T, S, and q = |T - S|.

        Args:
            n_steps: Number of measurement steps.
            n_transient: Steps to skip for transient decay.

        Returns:
            Dict with correlation coefficients.
        """
        self.reset()

        # Skip transient
        for _ in range(n_transient):
            self.step()

        x_vals = np.empty(n_steps)
        T_vals = np.empty(n_steps)
        S_vals = np.empty(n_steps)
        q_vals = np.empty(n_steps)

        for i in range(n_steps):
            state = self.step()
            x_vals[i] = state[0]
            T_vals[i] = state[3]
            S_vals[i] = state[4]
            q_vals[i] = abs(state[3] - state[4])

        def _pearson(a: np.ndarray, b: np.ndarray) -> float:
            """Pearson correlation coefficient."""
            a_c = a - np.mean(a)
            b_c = b - np.mean(b)
            denom = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
            if denom < 1e-15:
                return 0.0
            return float(np.sum(a_c * b_c) / denom)

        return {
            "corr_x_T": _pearson(x_vals, T_vals),
            "corr_x_S": _pearson(x_vals, S_vals),
            "corr_x_q": _pearson(x_vals, q_vals),
            "mean_q": float(np.mean(q_vals)),
            "std_q": float(np.std(q_vals)),
            "mean_x": float(np.mean(x_vals)),
            "std_x": float(np.std(x_vals)),
        }
