"""Competitive Lotka-Volterra (4 species) simulation.

Implements the competitive Lotka-Volterra model for N species (default 4)
with all pairwise interactions:

    dN_i/dt = r_i * N_i * (1 - sum_j(alpha_ij * N_j / K_i))

where N_i = population of species i, r_i = intrinsic growth rate,
K_i = carrying capacity, alpha_ij = competition coefficient.
alpha_ii = 1 by convention (intraspecific competition).

Target rediscoveries:
- Competitive exclusion principle: strong competition reduces diversity
- Stable coexistence equilibrium: N* = alpha^(-1) @ K
- Community matrix eigenvalues determine local stability
- Shannon diversity as function of competition strength
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CompetitiveLVSimulation(SimulationEnvironment):
    """Competitive Lotka-Volterra model for n_species (default 4).

    State vector: [N_1, N_2, ..., N_n] where N_i is population of species i.

    Equations:
        dN_i/dt = r_i * N_i * (1 - sum_j(alpha_ij * N_j / K_i))

    Parameters (via config.parameters):
        n_species: number of species (default 4)
        r_0..r_{n-1}: intrinsic growth rates (default [1.0, 0.72, 1.53, 1.27])
        K_0..K_{n-1}: carrying capacities (default 100 for all)
        alpha_i_j: competition coefficient from species j on species i
            (default: diagonal=1, off-diagonal~0.5)
        N_0_0..N_0_{n-1}: initial populations (default K/2)
    """

    # Default parameters for 4-species coexistence
    DEFAULT_R = [1.0, 0.72, 1.53, 1.27]
    DEFAULT_K = [100.0, 100.0, 100.0, 100.0]
    DEFAULT_ALPHA = [
        [1.0, 0.5, 0.4, 0.3],
        [0.4, 1.0, 0.6, 0.3],
        [0.3, 0.4, 1.0, 0.5],
        [0.5, 0.3, 0.4, 1.0],
    ]

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.n_species = int(p.get("n_species", 4))
        n = self.n_species

        # Growth rates
        self.r = np.array([
            p.get(f"r_{i}", self.DEFAULT_R[i] if i < len(self.DEFAULT_R) else 1.0)
            for i in range(n)
        ], dtype=np.float64)

        # Carrying capacities
        self.K = np.array([
            p.get(f"K_{i}", self.DEFAULT_K[i] if i < len(self.DEFAULT_K) else 100.0)
            for i in range(n)
        ], dtype=np.float64)

        # Competition matrix (alpha_ij = effect of species j on species i)
        self.alpha = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                default_val = (
                    self.DEFAULT_ALPHA[i][j]
                    if i < len(self.DEFAULT_ALPHA) and j < len(self.DEFAULT_ALPHA[i])
                    else (1.0 if i == j else 0.5)
                )
                self.alpha[i, j] = p.get(f"alpha_{i}_{j}", default_val)

        # Initial populations (default: K/2)
        self.N_init = np.array([
            p.get(f"N_0_{i}", self.K[i] / 2.0)
            for i in range(n)
        ], dtype=np.float64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize all species populations."""
        self._state = self.N_init.copy()
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity enforcement."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [N_1, ..., N_n]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce non-negative populations
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Competitive Lotka-Volterra right-hand side.

        dN_i/dt = r_i * N_i * (1 - sum_j(alpha_ij * N_j / K_i))
        """
        # Interaction term: alpha @ N / K (element-wise division by K_i)
        interaction = self.alpha @ y / self.K
        return self.r * y * (1.0 - interaction)

    def equilibrium_point(self) -> np.ndarray:
        """Compute the interior equilibrium N* = alpha^{-1} @ K.

        The coexistence equilibrium satisfies:
            sum_j(alpha_ij * N_j* / K_i) = 1 for all i
        which gives: alpha @ N* = K (element-wise), i.e. N* = alpha^{-1} @ K.

        Returns:
            Equilibrium populations. May contain negative values if
            coexistence is not feasible.
        """
        try:
            return np.linalg.solve(self.alpha, self.K)
        except np.linalg.LinAlgError:
            return np.full(self.n_species, np.nan)

    def community_matrix(self) -> np.ndarray:
        """Compute the Jacobian (community matrix) at the equilibrium.

        For the competitive LV system, the Jacobian element J_ij at
        equilibrium N* is:
            J_ii = r_i * N_i* * (-alpha_ii / K_i)  (since 1 - sum = 0 at eq)
            J_ij = r_i * N_i* * (-alpha_ij / K_j)

        Simplified (at equilibrium where growth term = 0):
            J_ij = -r_i * N_i* * alpha_ij / K_i

        Returns:
            4x4 (or nxn) community matrix.
        """
        N_star = self.equilibrium_point()
        n = self.n_species
        J = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                J[i, j] = -self.r[i] * N_star[i] * self.alpha[i, j] / self.K[i]
        return J

    def stability_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the community matrix at equilibrium.

        All real parts negative => locally stable coexistence.

        Returns:
            Complex eigenvalue array.
        """
        J = self.community_matrix()
        return np.linalg.eigvals(J)

    def is_stable_coexistence(self) -> bool:
        """Check if the equilibrium is a feasible, stable coexistence.

        Feasible: all N_i* > 0.
        Stable: all eigenvalue real parts < 0.
        """
        N_star = self.equilibrium_point()
        if np.any(np.isnan(N_star)) or np.any(N_star <= 0):
            return False
        eigs = self.stability_eigenvalues()
        return bool(np.all(np.real(eigs) < 0))

    def diversity_index(self) -> float:
        """Shannon diversity index H = -sum(p_i * ln(p_i)).

        Computed from current population fractions.
        Returns 0 if total population is zero.
        """
        N = self._state
        total = np.sum(N)
        if total <= 0:
            return 0.0
        p = N / total
        # Avoid log(0) by filtering out zero-population species
        mask = p > 0
        return float(-np.sum(p[mask] * np.log(p[mask])))

    def n_surviving(self, threshold: float = 1e-3) -> int:
        """Count species with population above threshold."""
        return int(np.sum(self._state > threshold))

    def exclusion_sweep(
        self,
        alpha_12_values: np.ndarray | list[float],
        n_steps: int = 50000,
    ) -> dict[str, np.ndarray]:
        """Vary alpha_12 (competition of species 2 on species 1) and track survival.

        For each alpha_12 value, runs the simulation to steady state and
        records which species survive and the final diversity.

        Args:
            alpha_12_values: array of alpha_12 values to sweep.
            n_steps: simulation steps per alpha value.

        Returns:
            Dict with alpha_12 values, surviving counts, final populations,
            and diversity indices.
        """
        alpha_12_values = np.asarray(alpha_12_values)
        n_surviving_list = []
        final_pops_list = []
        diversity_list = []

        for alpha_12 in alpha_12_values:
            # Modify alpha_01 (effect of species 1 on species 0)
            self.alpha[0, 1] = alpha_12
            self.reset()
            for _ in range(n_steps):
                self.step()
            n_surviving_list.append(self.n_surviving())
            final_pops_list.append(self.observe().copy())
            diversity_list.append(self.diversity_index())

        return {
            "alpha_12": alpha_12_values,
            "n_surviving": np.array(n_surviving_list),
            "final_populations": np.array(final_pops_list),
            "diversity": np.array(diversity_list),
        }
