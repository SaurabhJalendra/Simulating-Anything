"""Toda lattice simulation -- integrable nonlinear chain.

The Toda lattice is a one-dimensional lattice of particles with exponential
nearest-neighbor interactions. It is one of the few exactly integrable
nonlinear systems and supports soliton solutions.

Equations of motion (periodic boundary conditions):
    dx_i/dt = p_i
    dp_i/dt = exp(-(x_i - x_{i-1})) - exp(-(x_{i+1} - x_i))

where indices are taken modulo N (periodic).

Target rediscoveries:
- Energy conservation: E = sum(0.5*p_i^2 + exp(-(x_{i+1}-x_i))) = const
- Momentum conservation: sum(p_i) = const (for PBC)
- Small-amplitude limit reduces to harmonic spring-mass chain
- Soliton propagation: coherent nonlinear waves that maintain shape
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TodaLattice(SimulationEnvironment):
    """1D Toda lattice with periodic boundary conditions.

    State vector: [x_0, ..., x_{N-1}, p_0, ..., p_{N-1}]
    where x_i = displacement of particle i, p_i = momentum of particle i.

    The exponential interaction potential is V(r) = exp(-r) + r - 1,
    which is normalized so V(0) = 0 and V'(0) = 0 (equilibrium at r=0).

    Parameters:
        N: number of particles (default 8)
        a: coupling strength scaling (default 1.0)
        mode: initial excitation mode (default 1); 0 = small random
        amplitude: initial displacement amplitude (default 0.1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 8))
        self.a = p.get("a", 1.0)
        self._init_mode = int(p.get("mode", 1))
        self._init_amplitude = p.get("amplitude", 0.1)

    @property
    def positions(self) -> np.ndarray:
        """Particle displacements x_0, ..., x_{N-1}."""
        return self._state[:self.N].copy()

    @property
    def momenta(self) -> np.ndarray:
        """Particle momenta p_0, ..., p_{N-1}."""
        return self._state[self.N:].copy()

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy: sum of 0.5 * p_i^2."""
        p = self._state[self.N:]
        return float(0.5 * np.sum(p**2))

    @property
    def potential_energy(self) -> float:
        """Total potential energy from exponential interactions.

        V_total = sum_{i=0}^{N-1} [exp(-(x_{i+1} - x_i)) + (x_{i+1} - x_i) - 1]

        where x_N = x_0 (periodic boundary conditions). The normalization
        ensures V=0 at equilibrium (all x_i equal).
        """
        x = self._state[:self.N]
        dx = np.roll(x, -1) - x  # x_{i+1} - x_i, with PBC via roll
        return float(self.a * np.sum(np.exp(-dx) + dx - 1.0))

    @property
    def total_energy(self) -> float:
        """Total mechanical energy (KE + PE), a conserved quantity."""
        return self.kinetic_energy + self.potential_energy

    @property
    def total_momentum(self) -> float:
        """Total momentum sum(p_i), conserved under PBC."""
        return float(np.sum(self._state[self.N:]))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the lattice.

        If mode > 0, excite a single Fourier mode (sine wave displacements,
        zero momenta). If mode == 0, use small random displacements.
        """
        self._step_count = 0

        if self._init_mode > 0:
            # Single-mode sinusoidal excitation
            n = np.arange(self.N)
            x = self._init_amplitude * np.sin(
                2 * np.pi * self._init_mode * n / self.N
            )
            p = np.zeros(self.N, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed if seed is not None else 42)
            x = rng.normal(0.0, self._init_amplitude, size=self.N)
            # Zero total momentum for random init
            p = np.zeros(self.N, dtype=np.float64)

        self._state = np.concatenate([x, p]).astype(np.float64)
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [x_0,...,x_{N-1}, p_0,...,p_{N-1}]."""
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
        """Compute derivatives of [x, p] -> [dx/dt, dp/dt].

        dx_i/dt = p_i
        dp_i/dt = a * [exp(-(x_i - x_{i-1})) - exp(-(x_{i+1} - x_i))]

        Periodic boundary: x_N = x_0, x_{-1} = x_{N-1}.
        """
        N = self.N
        x = y[:N]
        p = y[N:]

        # Relative displacements with periodic boundary conditions
        dx_forward = np.roll(x, -1) - x    # x_{i+1} - x_i
        dx_backward = x - np.roll(x, 1)    # x_i - x_{i-1}

        # Forces: exp(-(x_i - x_{i-1})) - exp(-(x_{i+1} - x_i))
        dp_dt = self.a * (np.exp(-dx_backward) - np.exp(-dx_forward))

        return np.concatenate([p, dp_dt])

    def harmonic_frequencies(self) -> np.ndarray:
        """Analytical normal mode frequencies in the small-amplitude (harmonic) limit.

        In the harmonic limit, exp(-r) ~ 1 - r + r^2/2, so the Toda lattice
        reduces to a spring-mass chain with spring constant K = a (coupling).
        For periodic BC the normal mode frequencies are:

            omega_n = 2 * sqrt(a) * |sin(pi * n / N)|    for n = 0, 1, ..., N-1

        The n=0 mode has omega=0 (uniform translation).
        """
        n = np.arange(self.N)
        return 2.0 * np.sqrt(self.a) * np.abs(np.sin(np.pi * n / self.N))
