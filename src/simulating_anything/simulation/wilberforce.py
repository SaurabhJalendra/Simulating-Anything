"""Wilberforce pendulum simulation (coupled spring-torsion).

The Wilberforce pendulum is a mass on a spring that also rotates, with coupling
between the vertical (translational) and torsional oscillation modes. Energy
transfers back and forth between these modes at the beat frequency.

Equations of motion:
    m * z'' = -k * z - (eps/2) * theta
    I * theta'' = -kappa * theta - (eps/2) * z

Target rediscoveries:
- Normal mode frequencies: omega_z = sqrt(k/m), omega_theta = sqrt(kappa/I)
- Beat frequency (when omega_z ~ omega_theta): |omega_z - omega_theta|
- Energy transfer period: T_transfer = 2*pi / |omega_z - omega_theta|
- SINDy recovery of coupled ODEs
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class Wilberforce(SimulationEnvironment):
    """Wilberforce pendulum: coupled translational-torsional oscillator.

    Equations of motion:
        m * z'' = -k * z - (eps/2) * theta
        I * theta'' = -kappa * theta - (eps/2) * z

    State vector: [z, z_dot, theta, theta_dot]

    Parameters:
        m: mass (kg), default 0.5
        k: spring constant (N/m), default 5.0
        I: moment of inertia (kg*m^2), default 1e-4
        kappa: torsion constant (N*m/rad), default 1e-3
        eps: coupling constant (N/rad or N*m/m), default 1e-3
        z_0: initial vertical displacement (m), default 0.1
        z_dot_0: initial vertical velocity (m/s), default 0.0
        theta_0: initial angular displacement (rad), default 0.0
        theta_dot_0: initial angular velocity (rad/s), default 0.0
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.m = p.get("m", 0.5)
        self.k = p.get("k", 5.0)
        self.I = p.get("I", 1e-4)
        self.kappa = p.get("kappa", 1e-3)
        self.eps = p.get("eps", 1e-3)
        self.z_0 = p.get("z_0", 0.1)
        self.z_dot_0 = p.get("z_dot_0", 0.0)
        self.theta_0 = p.get("theta_0", 0.0)
        self.theta_dot_0 = p.get("theta_dot_0", 0.0)

    @property
    def omega_z(self) -> float:
        """Translational (vertical) normal mode frequency: sqrt(k/m)."""
        return np.sqrt(self.k / self.m)

    @property
    def omega_theta(self) -> float:
        """Torsional normal mode frequency: sqrt(kappa/I)."""
        return np.sqrt(self.kappa / self.I)

    @property
    def beat_frequency(self) -> float:
        """Beat frequency: |omega_z - omega_theta|."""
        return abs(self.omega_z - self.omega_theta)

    @property
    def energy_transfer_period(self) -> float:
        """Energy transfer period: 2*pi / beat_frequency."""
        bf = self.beat_frequency
        if bf < 1e-15:
            return float("inf")
        return 2.0 * np.pi / bf

    @property
    def total_energy(self) -> float:
        """Total mechanical energy: translational KE + PE + rotational KE + PE + coupling.

        E = 0.5*m*z_dot^2 + 0.5*k*z^2
          + 0.5*I*theta_dot^2 + 0.5*kappa*theta^2
          + 0.5*eps*z*theta
        """
        z, z_dot, theta, theta_dot = self._state
        return (
            0.5 * self.m * z_dot**2
            + 0.5 * self.k * z**2
            + 0.5 * self.I * theta_dot**2
            + 0.5 * self.kappa * theta**2
            + 0.5 * self.eps * z * theta
        )

    @property
    def translational_energy(self) -> float:
        """Translational (vertical) energy: 0.5*m*z_dot^2 + 0.5*k*z^2."""
        z, z_dot, _, _ = self._state
        return 0.5 * self.m * z_dot**2 + 0.5 * self.k * z**2

    @property
    def rotational_energy(self) -> float:
        """Rotational (torsional) energy: 0.5*I*theta_dot^2 + 0.5*kappa*theta^2."""
        _, _, theta, theta_dot = self._state
        return 0.5 * self.I * theta_dot**2 + 0.5 * self.kappa * theta**2

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state [z, z_dot, theta, theta_dot]."""
        self._state = np.array(
            [self.z_0, self.z_dot_0, self.theta_0, self.theta_dot_0],
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
        """Return current state [z, z_dot, theta, theta_dot]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute derivatives of the state vector.

        z' = z_dot
        z_dot' = -(k/m)*z - (eps/(2*m))*theta
        theta' = theta_dot
        theta_dot' = -(kappa/I)*theta - (eps/(2*I))*z
        """
        z, z_dot, theta, theta_dot = y
        z_ddot = -(self.k / self.m) * z - (self.eps / (2.0 * self.m)) * theta
        theta_ddot = (
            -(self.kappa / self.I) * theta - (self.eps / (2.0 * self.I)) * z
        )
        return np.array([z_dot, z_ddot, theta_dot, theta_ddot])
