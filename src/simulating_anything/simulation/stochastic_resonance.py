"""Stochastic resonance in a bistable double-well potential.

Demonstrates how optimal noise intensity amplifies a weak periodic signal
in a nonlinear bistable system. The phenomenon arises when the noise-induced
hopping rate between wells matches the signal frequency.

State: [x, v] (position and velocity in the double-well potential)
V(x) = x^4/4 - x^2/2 (quartic double-well)
Signal: A * sin(omega * t)
Noise: Gaussian white noise with intensity D

Key targets:
- SNR peaks at optimal noise D_opt (stochastic resonance)
- Kramers escape rate: r_K = (omega_0 / (2*pi)) * exp(-Delta_V / D)
- Residence time distribution: exponential with rate r_K
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class StochasticResonanceSimulation(SimulationEnvironment):
    """Stochastic resonance in a quartic double-well with periodic forcing."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.gamma = config.parameters.get("gamma", 0.5)
        self.signal_amplitude = config.parameters.get("signal_amplitude", 0.3)
        self.signal_omega = config.parameters.get("signal_omega", 0.1)
        self.noise_intensity = config.parameters.get("noise_intensity", 0.3)
        self.dt = config.dt
        self._state = np.zeros(2)
        self._time = 0.0
        self._rng = np.random.default_rng(42)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = np.array([
            self._rng.choice([-1.0, 1.0]),
            0.0,
        ])
        self._time = 0.0
        return self._state.copy()

    def step(self) -> np.ndarray:
        x, v = self._state
        dt = self.dt
        D = self.noise_intensity
        A = self.signal_amplitude
        omega = self.signal_omega
        gamma = self.gamma

        # Force from double-well: -dV/dx = x - x^3
        force = x - x**3
        # Periodic signal
        signal = A * np.sin(omega * self._time)
        # Noise
        noise = self._rng.normal() * np.sqrt(2 * D * dt)

        # Langevin dynamics (overdamped limit: dv/dt = force - gamma*v + signal + noise)
        v_new = v + (force - gamma * v + signal) * dt + noise
        x_new = x + v_new * dt

        self._state = np.array([x_new, v_new])
        self._time += dt
        return self._state.copy()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "StochasticResonance"
