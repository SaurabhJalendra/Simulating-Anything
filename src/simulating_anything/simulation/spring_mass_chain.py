"""1D coupled spring-mass chain simulation.

Demonstrates phonon physics, normal modes, and wave mechanics.

Target rediscoveries:
- Dispersion relation: omega(k) = 2*sqrt(K/m)*|sin(k*a/2)|
- Normal modes: u_i(n) = sin(n*pi*i/(N+1))
- Speed of sound: c = a*sqrt(K/m)
- Energy conservation in conservative system
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SpringMassChain(SimulationEnvironment):
    """1D chain of N masses connected by identical springs with fixed ends.

    Equation of motion for mass i:
        m * x_i'' = K * (x_{i+1} - 2*x_i + x_{i-1})

    with fixed boundary conditions: x_0 = x_{N+1} = 0 (pinned ends).

    State vector: [u_1, ..., u_N, v_1, ..., v_N]
    where u_i = displacement from equilibrium, v_i = velocity.

    Parameters:
        N: number of masses (default 20)
        K: spring constant (default 4.0)
        m: mass of each particle (default 1.0)
        a: equilibrium spacing between masses (default 1.0)
        mode: initial normal mode to excite (default 1); 0 = small random
        amplitude: initial displacement amplitude (default 0.1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 20))
        self.K = p.get("K", 4.0)
        self.m = p.get("m", 1.0)
        self.a = p.get("a", 1.0)
        self._init_mode = int(p.get("mode", 1))
        self._init_amplitude = p.get("amplitude", 0.1)

    @property
    def omega_max(self) -> float:
        """Maximum frequency in the chain: 2*sqrt(K/m)."""
        return 2.0 * np.sqrt(self.K / self.m)

    def compute_normal_mode_frequencies(self) -> np.ndarray:
        """Return analytical normal mode frequencies omega_n for n=1..N.

        omega_n = 2*sqrt(K/m) * sin(n*pi / (2*(N+1)))
        """
        n = np.arange(1, self.N + 1)
        return 2.0 * np.sqrt(self.K / self.m) * np.sin(
            n * np.pi / (2 * (self.N + 1))
        )

    def normal_mode_shape(self, mode_n: int) -> np.ndarray:
        """Return the displacement pattern for normal mode n.

        u_i(n) = sin(n * pi * i / (N+1)) for i = 1..N.
        """
        i = np.arange(1, self.N + 1)
        return np.sin(mode_n * np.pi * i / (self.N + 1))

    def excite_mode(self, mode_n: int, amplitude: float = 0.1) -> None:
        """Set initial conditions to a pure normal mode excitation.

        Displacements follow the mode shape; velocities are zero.
        """
        if mode_n < 1 or mode_n > self.N:
            raise ValueError(
                f"Mode number must be 1..{self.N}, got {mode_n}"
            )
        shape = self.normal_mode_shape(mode_n)
        # Normalize so max displacement = amplitude
        shape = amplitude * shape / np.max(np.abs(shape))
        u = shape
        v = np.zeros(self.N)
        self._state = np.concatenate([u, v])

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize the chain.

        If mode > 0, excite that normal mode. Otherwise use small random
        displacements seeded by the given seed.
        """
        self._step_count = 0
        if self._init_mode > 0:
            # Start with zero state, then excite
            self._state = np.zeros(2 * self.N, dtype=np.float64)
            self.excite_mode(self._init_mode, self._init_amplitude)
        else:
            rng = np.random.default_rng(seed if seed is not None else 42)
            u = rng.normal(0.0, self._init_amplitude, size=self.N)
            v = np.zeros(self.N)
            self._state = np.concatenate([u, v])
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [u_1,...,u_N, v_1,...,v_N]."""
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
        """Compute derivatives of [u, v] -> [v, a].

        Fixed boundary: u_0 = u_{N+1} = 0.
        a_i = (K/m) * (u_{i+1} - 2*u_i + u_{i-1})
        """
        N = self.N
        u = y[:N]
        v = y[N:]

        # Build extended displacement array with fixed boundary zeros
        u_ext = np.zeros(N + 2)
        u_ext[1:-1] = u

        # Discrete Laplacian: u_{i+1} - 2*u_i + u_{i-1}
        accel = (self.K / self.m) * (
            u_ext[2:] - 2.0 * u_ext[1:-1] + u_ext[:-2]
        )

        return np.concatenate([v, accel])

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy: sum of 0.5 * m * v_i^2."""
        v = self._state[self.N:]
        return 0.5 * self.m * np.sum(v**2)

    @property
    def potential_energy(self) -> float:
        """Total potential energy from spring stretching.

        PE = 0.5 * K * sum((u_{i+1} - u_i)^2) including boundary springs.
        """
        u = self._state[:self.N]
        u_ext = np.zeros(self.N + 2)
        u_ext[1:-1] = u
        du = np.diff(u_ext)
        return 0.5 * self.K * np.sum(du**2)

    @property
    def total_energy(self) -> float:
        """Total mechanical energy (KE + PE)."""
        return self.kinetic_energy + self.potential_energy

    def mode_amplitudes(self) -> np.ndarray:
        """Project current displacements onto normal mode basis.

        Returns amplitude of each mode n=1..N.
        The normal modes form an orthogonal basis with fixed boundary
        conditions: phi_n(i) = sin(n*pi*i/(N+1)).
        """
        u = self._state[:self.N]
        N = self.N
        amplitudes = np.zeros(N)
        for n in range(1, N + 1):
            shape = self.normal_mode_shape(n)
            # Projection with normalization: <u, phi_n> / <phi_n, phi_n>
            amplitudes[n - 1] = np.dot(u, shape) / np.dot(shape, shape)
        return amplitudes

    def speed_of_sound(self) -> float:
        """Theoretical speed of sound: c = a * sqrt(K/m)."""
        return self.a * np.sqrt(self.K / self.m)
