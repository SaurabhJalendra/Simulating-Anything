"""2D Ising model simulation -- statistical mechanics phase transition.

Target rediscoveries:
- Critical temperature: T_c = 2*J / ln(1 + sqrt(2)) ~ 2.269 for h=0
- Spontaneous magnetization: M(T) = (1 - sinh(2J/T)^{-4})^{1/8} for T < T_c
- Magnetic susceptibility divergence at T_c
- Energy per spin at criticality
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class IsingModel2D(SimulationEnvironment):
    """2D Ising model on a square lattice with periodic boundary conditions.

    Hamiltonian: H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
    Dynamics: Metropolis-Hastings single-spin-flip Monte Carlo.

    State vector: flattened spin lattice of shape (N*N,), values +1 or -1.

    Parameters:
        N: lattice side length (default 16)
        J: coupling constant (default 1.0, ferromagnetic when positive)
        h: external magnetic field (default 0.0)
        T: temperature in units of k_B=1 (default 2.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 16))
        self.J = p.get("J", 1.0)
        self.h = p.get("h", 0.0)
        self.T = p.get("T", 2.0)

        # Internal spin lattice stored as (N, N) array of +1/-1
        self._spins: np.ndarray = np.ones((self.N, self.N), dtype=np.int8)
        self._rng: np.random.Generator = np.random.default_rng(config.seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize spins randomly (+1 or -1) with equal probability."""
        self._rng = np.random.default_rng(seed)
        self._spins = self._rng.choice(
            np.array([1, -1], dtype=np.int8), size=(self.N, self.N)
        )
        self._step_count = 0
        return self.observe()

    def step(self) -> np.ndarray:
        """Perform one Monte Carlo sweep (N*N single-spin-flip attempts).

        Each attempt picks a random site, computes the energy change from
        flipping that spin, and accepts with Metropolis probability
        min(1, exp(-delta_E / T)).
        """
        n_sites = self.N * self.N
        beta = 1.0 / self.T if self.T > 0 else np.inf

        # Pre-generate random positions and acceptance thresholds
        rows = self._rng.integers(0, self.N, size=n_sites)
        cols = self._rng.integers(0, self.N, size=n_sites)
        randoms = self._rng.random(size=n_sites)

        for k in range(n_sites):
            i, j = rows[k], cols[k]
            s = self._spins[i, j]

            # Sum of nearest neighbors with periodic BC
            neighbors = (
                self._spins[(i + 1) % self.N, j]
                + self._spins[(i - 1) % self.N, j]
                + self._spins[i, (j + 1) % self.N]
                + self._spins[i, (j - 1) % self.N]
            )

            # Energy change from flipping spin s -> -s:
            # delta_E = 2*J*s*sum_neighbors + 2*h*s
            delta_E = 2.0 * self.J * s * neighbors + 2.0 * self.h * s

            # Metropolis acceptance
            if delta_E <= 0:
                self._spins[i, j] = -s
            elif randoms[k] < np.exp(-beta * delta_E):
                self._spins[i, j] = -s

        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return flattened spin lattice as float array of shape (N*N,)."""
        return self._spins.ravel().astype(np.float64)

    @property
    def spin_lattice(self) -> np.ndarray:
        """Return the (N, N) spin lattice view."""
        return self._spins.copy()

    @property
    def magnetization(self) -> float:
        """Mean magnetization per spin: m = (1/N^2) * sum_i s_i."""
        return float(np.mean(self._spins))

    @property
    def energy(self) -> float:
        """Total Hamiltonian energy.

        H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i

        Each nearest-neighbor pair is counted once.
        """
        spins = self._spins.astype(np.float64)
        # Interaction energy: sum over right and down neighbors (each pair once)
        e_interaction = -self.J * np.sum(
            spins * np.roll(spins, -1, axis=0)
            + spins * np.roll(spins, -1, axis=1)
        )
        # Field energy
        e_field = -self.h * np.sum(spins)
        return float(e_interaction + e_field)

    @property
    def energy_per_spin(self) -> float:
        """Energy per spin: E / N^2."""
        return self.energy / (self.N * self.N)

    @property
    def specific_heat(self) -> float:
        """Estimate specific heat from energy fluctuations.

        C = (<E^2> - <E>^2) / (N^2 * T^2)

        Note: This returns 0 unless called after collecting samples.
        Use measure_specific_heat() for proper estimation.
        """
        return 0.0

    @property
    def susceptibility(self) -> float:
        """Estimate magnetic susceptibility from magnetization fluctuations.

        chi = N^2 * (<m^2> - <|m|>^2) / T

        Note: This returns 0 unless called after collecting samples.
        Use measure_susceptibility() for proper estimation.
        """
        return 0.0

    def measure_equilibrium(
        self,
        n_equil_sweeps: int = 1000,
        n_measure_sweeps: int = 2000,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Run equilibration then measure thermodynamic quantities.

        Returns dict with magnetization, energy, specific_heat, susceptibility.
        """
        self.reset(seed=seed)

        # Equilibrate
        for _ in range(n_equil_sweeps):
            self.step()

        # Measure
        energies = []
        magnetizations = []
        for _ in range(n_measure_sweeps):
            self.step()
            energies.append(self.energy)
            magnetizations.append(self.magnetization)

        E = np.array(energies)
        M = np.array(magnetizations)
        n_spins = self.N * self.N

        mean_E = np.mean(E)
        mean_M = np.mean(np.abs(M))
        mean_M2 = np.mean(M ** 2)
        mean_E2 = np.mean(E ** 2)

        # Specific heat: C/k_B = (var(E)) / (T^2 * N^2)
        C = (mean_E2 - mean_E ** 2) / (self.T ** 2 * n_spins) if self.T > 0 else 0.0

        # Susceptibility: chi = N^2 * (<m^2> - <|m|>^2) / T
        chi = n_spins * (mean_M2 - mean_M ** 2) / self.T if self.T > 0 else 0.0

        return {
            "magnetization": float(mean_M),
            "energy_per_spin": float(mean_E / n_spins),
            "specific_heat": float(C),
            "susceptibility": float(chi),
        }

    @staticmethod
    def critical_temperature(J: float = 1.0) -> float:
        """Exact critical temperature for the 2D Ising model.

        T_c = 2*J / ln(1 + sqrt(2)) ~ 2.269 * J
        """
        return 2.0 * J / np.log(1.0 + np.sqrt(2.0))

    @staticmethod
    def onsager_magnetization(T: float, J: float = 1.0) -> float:
        """Onsager exact spontaneous magnetization for T < T_c.

        M(T) = (1 - sinh(2J/T)^{-4})^{1/8} for T < T_c
        M(T) = 0 for T >= T_c
        """
        T_c = IsingModel2D.critical_temperature(J)
        if T >= T_c:
            return 0.0
        sinh_val = np.sinh(2.0 * J / T)
        arg = 1.0 - sinh_val ** (-4)
        if arg <= 0:
            return 0.0
        return float(arg ** (1.0 / 8.0))
