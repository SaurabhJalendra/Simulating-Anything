"""Fermi-Pasta-Ulam-Tsingou (FPUT) lattice simulation.

The FPUT lattice is the famous numerical experiment (1955) that led to chaos
theory and soliton physics. A chain of N masses connected by nonlinear springs:

Alpha model: F(delta) = k*delta + alpha*delta^2
Beta model:  F(delta) = k*delta + beta*delta^3

where delta = x_{i+1} - x_i is the displacement difference.

Equations of motion:
  d^2 x_i / dt^2 = F(x_{i+1} - x_i) - F(x_i - x_{i-1})

For the alpha model:
  d^2 x_i / dt^2 = (x_{i+1} - 2*x_i + x_{i-1})
                    + alpha*((x_{i+1}-x_i)^2 - (x_i-x_{i-1})^2)

Fixed boundary conditions: x_0 = x_{N+1} = 0.

Target rediscoveries:
- FPUT recurrence: energy returns to initial mode
- Energy conservation: symplectic integrator preserves Hamiltonian
- Mode energy sharing: energy does not thermalize
- Comparison of alpha (quadratic) vs beta (cubic) nonlinearity
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class FPUTSimulation(SimulationEnvironment):
    """1D FPUT lattice with fixed boundary conditions.

    State vector: [x_1, ..., x_N, v_1, ..., v_N]
    where x_i = displacement of particle i from equilibrium,
    v_i = velocity of particle i.

    Uses Stormer-Verlet (velocity Verlet) symplectic integration for
    excellent long-term energy conservation.

    Parameters:
        N: number of interior particles (default 32)
        k: linear spring constant (default 1.0)
        alpha: quadratic nonlinearity coefficient (default 0.25)
        beta: cubic nonlinearity coefficient (default 0.0)
        mode: initial normal mode to excite (default 1); 0 = small random
        amplitude: initial displacement amplitude (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 32))
        self.k = p.get("k", 1.0)
        self.alpha = p.get("alpha", 0.25)
        self.beta = p.get("beta", 0.0)
        self._init_mode = int(p.get("mode", 1))
        self._init_amplitude = p.get("amplitude", 1.0)

    @property
    def positions(self) -> np.ndarray:
        """Particle displacements x_1, ..., x_N."""
        return self._state[:self.N].copy()

    @property
    def velocities(self) -> np.ndarray:
        """Particle velocities v_1, ..., v_N."""
        return self._state[self.N:].copy()

    def _force_on_delta(self, delta: np.ndarray) -> np.ndarray:
        """Compute the spring force for a given displacement difference.

        F(delta) = k*delta + alpha*delta^2 + beta*delta^3

        Args:
            delta: array of displacement differences x_{i+1} - x_i.

        Returns:
            Force array of the same shape as delta.
        """
        return self.k * delta + self.alpha * delta**2 + self.beta * delta**3

    def _accelerations(self, x: np.ndarray) -> np.ndarray:
        """Compute accelerations for all N particles given positions.

        With fixed BC: x_0 = 0 (left wall), x_{N+1} = 0 (right wall).
        The force on particle i is F(x_{i+1}-x_i) - F(x_i-x_{i-1}).

        Args:
            x: displacement array of shape (N,).

        Returns:
            Acceleration array of shape (N,).
        """
        # Pad with fixed boundary zeros: x_0 = 0 on the left, x_{N+1} = 0 on the right
        x_padded = np.concatenate([[0.0], x, [0.0]])

        # delta_right[i] = x_{i+1} - x_i for i = 1..N (using padded indices)
        delta_right = x_padded[2:] - x_padded[1:-1]  # shape (N,)
        # delta_left[i] = x_i - x_{i-1} for i = 1..N
        delta_left = x_padded[1:-1] - x_padded[:-2]  # shape (N,)

        # a_i = F(delta_right_i) - F(delta_left_i)
        return self._force_on_delta(delta_right) - self._force_on_delta(delta_left)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the lattice with a single normal mode excitation.

        For fixed BC, normal modes are: phi_n(i) = sin(n*pi*i/(N+1))
        for n = 1, ..., N and i = 1, ..., N.

        If mode == 0, uses small random displacements.
        """
        self._step_count = 0

        if self._init_mode > 0:
            # Single normal mode excitation with fixed-end mode shapes
            i = np.arange(1, self.N + 1)
            x = self._init_amplitude * np.sin(
                self._init_mode * np.pi * i / (self.N + 1)
            )
            v = np.zeros(self.N, dtype=np.float64)
        else:
            rng = np.random.default_rng(seed if seed is not None else 42)
            x = rng.normal(0.0, self._init_amplitude * 0.01, size=self.N)
            v = np.zeros(self.N, dtype=np.float64)

        self._state = np.concatenate([x, v]).astype(np.float64)
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using Stormer-Verlet (velocity Verlet).

        The velocity Verlet algorithm is symplectic and time-reversible:
            v_{n+1/2} = v_n + (dt/2) * a(x_n)
            x_{n+1}   = x_n + dt * v_{n+1/2}
            v_{n+1}   = v_{n+1/2} + (dt/2) * a(x_{n+1})

        This preserves the Hamiltonian structure far better than RK4
        for long-time integrations (critical for observing FPUT recurrence).
        """
        dt = self.config.dt
        x = self._state[:self.N]
        v = self._state[self.N:]

        # Half-step velocity
        a_n = self._accelerations(x)
        v_half = v + 0.5 * dt * a_n

        # Full-step position
        x_new = x + dt * v_half

        # Half-step velocity with new accelerations
        a_new = self._accelerations(x_new)
        v_new = v_half + 0.5 * dt * a_new

        self._state = np.concatenate([x_new, v_new])
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [x_1,...,x_N, v_1,...,v_N]."""
        return self._state

    def compute_mode_energies(self) -> np.ndarray:
        """Compute energy in each Fourier normal mode.

        For fixed BC, the normal mode amplitudes are obtained via
        the discrete sine transform:
            Q_n = sqrt(2/(N+1)) * sum_{i=1}^{N} x_i * sin(n*pi*i/(N+1))
            P_n = sqrt(2/(N+1)) * sum_{i=1}^{N} v_i * sin(n*pi*i/(N+1))

        The mode energy (in the harmonic approximation) is:
            E_n = 0.5 * (P_n^2 + omega_n^2 * Q_n^2)

        where omega_n = 2*sqrt(k) * sin(n*pi / (2*(N+1))) for unit mass.

        Returns:
            Array of shape (N,) with energy in each mode n=1..N.
        """
        x = self._state[:self.N]
        v = self._state[self.N:]

        # Mode transform matrix: S[n, i] = sqrt(2/(N+1)) * sin(n*pi*i/(N+1))
        n_modes = np.arange(1, self.N + 1)
        i_particles = np.arange(1, self.N + 1)
        # Shape: (N_modes, N_particles)
        S = np.sqrt(2.0 / (self.N + 1)) * np.sin(
            np.outer(n_modes, i_particles) * np.pi / (self.N + 1)
        )

        Q = S @ x  # mode amplitudes
        P = S @ v  # mode momenta

        # Normal mode frequencies for fixed-end chain with unit mass
        omega = 2.0 * np.sqrt(self.k) * np.sin(
            n_modes * np.pi / (2 * (self.N + 1))
        )

        return 0.5 * (P**2 + omega**2 * Q**2)

    def compute_total_energy(self) -> float:
        """Compute total mechanical energy (kinetic + potential).

        KE = 0.5 * sum(v_i^2)
        PE = sum over bonds of V(delta) where delta = x_{i+1} - x_i
        V(delta) = 0.5*k*delta^2 + (alpha/3)*delta^3 + (beta/4)*delta^4

        Returns:
            Total energy as a float.
        """
        x = self._state[:self.N]
        v = self._state[self.N:]

        ke = 0.5 * np.sum(v**2)

        # Compute bond stretches (N+1 bonds for N interior particles)
        x_padded = np.concatenate([[0.0], x, [0.0]])
        delta = x_padded[1:] - x_padded[:-1]  # N+1 bond stretches

        pe = np.sum(
            0.5 * self.k * delta**2
            + (self.alpha / 3.0) * delta**3
            + (self.beta / 4.0) * delta**4
        )

        return float(ke + pe)

    def normal_mode_frequencies(self) -> np.ndarray:
        """Analytical normal mode frequencies for the linearized (harmonic) chain.

        omega_n = 2*sqrt(k) * sin(n*pi / (2*(N+1)))  for n = 1, ..., N.

        (Assumes unit mass.)
        """
        n = np.arange(1, self.N + 1)
        return 2.0 * np.sqrt(self.k) * np.sin(n * np.pi / (2 * (self.N + 1)))

    def recurrence_time_estimate(self) -> float:
        """Estimate the FPUT recurrence time from mode frequencies.

        The recurrence time is approximately 1 / (omega_2 - omega_1) * 2*pi,
        i.e., the beat period between the two lowest modes.

        Returns:
            Estimated recurrence time in simulation time units.
        """
        freqs = self.normal_mode_frequencies()
        if len(freqs) < 2:
            return float("inf")
        delta_omega = freqs[1] - freqs[0]
        if delta_omega < 1e-15:
            return float("inf")
        return 2.0 * np.pi / delta_omega
