"""Coupled harmonic oscillators simulation.

Two identical oscillators (mass m, spring constant k) connected by a weak
coupling spring (constant kc). Demonstrates normal mode splitting, beat
phenomena, and energy transfer between subsystems.

Target rediscoveries:
- Symmetric mode frequency: omega_s = sqrt(k/m)
- Antisymmetric mode frequency: omega_a = sqrt((k + 2*kc)/m)
- Beat frequency: omega_beat = omega_a - omega_s
- Energy transfer period: T_beat = 2*pi / omega_beat
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CoupledOscillators(SimulationEnvironment):
    """Two coupled harmonic oscillators.

    Equations of motion:
        m * x1'' = -k * x1 - kc * (x1 - x2)
        m * x2'' = -k * x2 - kc * (x2 - x1)

    State vector: [x1, v1, x2, v2]

    Parameters:
        k: individual spring constant (N/m), default 4.0
        m: mass of each oscillator (kg), default 1.0
        kc: coupling spring constant (N/m), default 0.5
        x1_0: initial displacement of oscillator 1, default 1.0
        v1_0: initial velocity of oscillator 1, default 0.0
        x2_0: initial displacement of oscillator 2, default 0.0
        v2_0: initial velocity of oscillator 2, default 0.0
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.k = p.get("k", 4.0)
        self.m = p.get("m", 1.0)
        self.kc = p.get("kc", 0.5)
        self.x1_0 = p.get("x1_0", 1.0)
        self.v1_0 = p.get("v1_0", 0.0)
        self.x2_0 = p.get("x2_0", 0.0)
        self.v2_0 = p.get("v2_0", 0.0)

    @property
    def omega_symmetric(self) -> float:
        """Symmetric normal mode frequency: omega_s = sqrt(k/m)."""
        return np.sqrt(self.k / self.m)

    @property
    def omega_antisymmetric(self) -> float:
        """Antisymmetric normal mode frequency: omega_a = sqrt((k + 2*kc)/m)."""
        return np.sqrt((self.k + 2.0 * self.kc) / self.m)

    @property
    def beat_frequency(self) -> float:
        """Beat frequency: omega_beat = omega_a - omega_s."""
        return self.omega_antisymmetric - self.omega_symmetric

    @property
    def beat_period(self) -> float:
        """Beat period: T_beat = 2*pi / omega_beat."""
        wb = self.beat_frequency
        if wb < 1e-15:
            return float("inf")
        return 2.0 * np.pi / wb

    @property
    def total_energy(self) -> float:
        """Total mechanical energy of the coupled system.

        E = 0.5*m*v1^2 + 0.5*m*v2^2
          + 0.5*k*x1^2 + 0.5*k*x2^2
          + 0.5*kc*(x1 - x2)^2
        """
        x1, v1, x2, v2 = self._state
        ke = 0.5 * self.m * (v1**2 + v2**2)
        pe_indiv = 0.5 * self.k * (x1**2 + x2**2)
        pe_coupling = 0.5 * self.kc * (x1 - x2)**2
        return ke + pe_indiv + pe_coupling

    def energy_oscillator_1(self, state: np.ndarray | None = None) -> float:
        """Energy stored in oscillator 1 (KE + half coupling PE).

        E1 = 0.5*m*v1^2 + 0.5*k*x1^2 + 0.25*kc*(x1 - x2)^2
        """
        s = state if state is not None else self._state
        x1, v1, x2, _ = s
        return 0.5 * self.m * v1**2 + 0.5 * self.k * x1**2

    def energy_oscillator_2(self, state: np.ndarray | None = None) -> float:
        """Energy stored in oscillator 2 (KE + half coupling PE).

        E2 = 0.5*m*v2^2 + 0.5*k*x2^2 + 0.25*kc*(x1 - x2)^2
        """
        s = state if state is not None else self._state
        _, _, x2, v2 = s
        return 0.5 * self.m * v2**2 + 0.5 * self.k * x2**2

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize positions and velocities of both oscillators."""
        self._state = np.array(
            [self.x1_0, self.v1_0, self.x2_0, self.v2_0],
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
        """Return current state [x1, v1, x2, v2]."""
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
        """Compute derivatives: [x1, v1, x2, v2] -> [v1, a1, v2, a2].

        a1 = -(k/m)*x1 - (kc/m)*(x1 - x2)
        a2 = -(k/m)*x2 - (kc/m)*(x2 - x1)
        """
        x1, v1, x2, v2 = y
        a1 = -(self.k * x1 + self.kc * (x1 - x2)) / self.m
        a2 = -(self.k * x2 + self.kc * (x2 - x1)) / self.m
        return np.array([v1, a1, v2, a2])

    def set_symmetric_mode(self, amplitude: float = 1.0) -> None:
        """Set initial conditions for symmetric normal mode (x1 = x2)."""
        self._state = np.array(
            [amplitude, 0.0, amplitude, 0.0], dtype=np.float64
        )
        self._step_count = 0

    def set_antisymmetric_mode(self, amplitude: float = 1.0) -> None:
        """Set initial conditions for antisymmetric normal mode (x1 = -x2)."""
        self._state = np.array(
            [amplitude, 0.0, -amplitude, 0.0], dtype=np.float64
        )
        self._step_count = 0
