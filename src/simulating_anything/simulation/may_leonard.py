"""May-Leonard cyclic competition simulation.

Implements the N-species competitive Lotka-Volterra model with cyclic dominance
(May-Leonard dynamics). For 4 species this produces rock-paper-scissors-Spock
dynamics with heteroclinic cycles.

    dx_i/dt = r_i * x_i * (1 - sum_j(alpha_ij * x_j / K_j))

Competition matrix has circulant cyclic structure:
    alpha_ij = 1     if i == j  (intraspecific)
    alpha_ij = a     if j == (i+1) mod n  (strong: clockwise neighbor)
    alpha_ij = b     otherwise  (weak: all other species)

For 4 species with a > 1 and 0 < b < 1, orbits spiral along a heteroclinic
cycle visiting each single-species-dominant corner in sequence.

Target rediscoveries:
- Heteroclinic cycle period as function of a, b
- Interior fixed point x* = K / (1 + (n-1)*avg_alpha)
- Biodiversity (Shannon entropy) oscillations
- Cyclic dominance sequence detection
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MayLeonardSimulation(SimulationEnvironment):
    """May-Leonard cyclic competition model for n_species (default 4).

    State vector: [x_1, x_2, ..., x_n] where x_i is population of species i.

    Equations:
        dx_i/dt = r_i * x_i * (1 - sum_j(alpha_ij * x_j / K_j))

    The competition matrix has a cyclic structure encoding dominance.
    For 4 species, species i strongly competes with (i+1) mod n and
    weakly with (i+3) mod n, creating a 4-cycle.

    Parameters (via config.parameters):
        n_species: number of species (default 4)
        a: strong competition coefficient (default 1.5)
        b: weak competition coefficient (default 0.5)
        r: intrinsic growth rate for all species (default 1.0)
        K: carrying capacity for all species (default 1.0)
        x_0_i: initial population for species i (default 0.25)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.n_species = int(p.get("n_species", 4))
        self.a = p.get("a", 1.5)
        self.b_param = p.get("b", 0.5)
        self.r_val = p.get("r", 1.0)
        self.K_val = p.get("K", 1.0)

        n = self.n_species

        # Growth rates (uniform by default)
        self.r = np.full(n, self.r_val, dtype=np.float64)

        # Carrying capacities (uniform by default)
        self.K = np.full(n, self.K_val, dtype=np.float64)

        # Build the cyclic competition matrix
        self.alpha = self.build_competition_matrix()

        # Initial populations
        self.x_init = np.array([
            p.get(f"x_0_{i}", 1.0 / n)
            for i in range(n)
        ], dtype=np.float64)

    def build_competition_matrix(self) -> np.ndarray:
        """Construct the cyclic competition matrix.

        For n species, the matrix has circulant structure where the
        off-diagonal entries encode the dominance hierarchy. Each row
        is a cyclic shift of the first row.

        For n=4, a=1.5, b=0.5, the off-diagonal pattern per row is:
            [a, b, b] assigned to offsets [+1, +2, +3] mod n
        giving the circulant matrix:
            [[1, a, b, b],
             [b, 1, a, b],
             [b, b, 1, a],
             [a, b, b, 1]]

        Species i most strongly competes with species (i+1) mod n
        (coefficient a > 1) and has weaker interaction (coefficient b < 1)
        with all other species. This ensures all species interact and
        the interior fixed point is unstable, driving heteroclinic cycling.

        For n=3 this reduces to the classic May-Leonard model:
            [[1, a, b],
             [b, 1, a],
             [a, b, 1]]

        Returns:
            n_species x n_species competition matrix.
        """
        n = self.n_species
        alpha = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            alpha[i, i] = 1.0  # Intraspecific
            alpha[i, (i + 1) % n] = self.a  # Strong: dominates clockwise neighbor
            # Weak interaction with all other (non-self, non-strong) species
            for offset in range(2, n):
                alpha[i, (i + offset) % n] = self.b_param

        return alpha

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize all species populations."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Add small perturbation to break exact symmetry
            noise = rng.uniform(-0.01, 0.01, self.n_species)
            self._state = np.maximum(self.x_init + noise, 1e-10)
        else:
            self._state = self.x_init.copy()
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity enforcement."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current populations [x_1, ..., x_n]."""
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
        """May-Leonard competitive LV right-hand side.

        dx_i/dt = r_i * x_i * (1 - sum_j(alpha_ij * x_j / K_j))
        """
        interaction = self.alpha @ (y / self.K)
        return self.r * y * (1.0 - interaction)

    def compute_interior_fixed_point(self) -> np.ndarray:
        """Compute the interior (coexistence) fixed point.

        At the fixed point: sum_j(alpha_ij * x_j* / K_j) = 1 for all i.
        This gives: alpha @ (x* / K) = 1, so x* / K = alpha^{-1} @ 1,
        i.e., x* = K * (alpha^{-1} @ 1).

        For the symmetric case (all K equal), x* = K / row_sum(alpha^{-1}).

        Returns:
            Fixed point populations. May contain negative values if
            coexistence is not feasible.
        """
        try:
            ones = np.ones(self.n_species)
            x_over_K = np.linalg.solve(self.alpha, ones)
            return x_over_K * self.K
        except np.linalg.LinAlgError:
            return np.full(self.n_species, np.nan)

    def compute_total_population(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute total population N(t) = sum(x_i(t)) at each timestep.

        Args:
            trajectory: array of shape (n_timesteps, n_species).

        Returns:
            1D array of total population at each timestep.
        """
        return np.sum(trajectory, axis=1)

    def compute_dominance_index(self, trajectory: np.ndarray) -> np.ndarray:
        """Determine which species is dominant at each timestep.

        Args:
            trajectory: array of shape (n_timesteps, n_species).

        Returns:
            1D integer array of dominant species index at each timestep.
        """
        return np.argmax(trajectory, axis=1)

    def heteroclinic_cycle_period(self, n_steps: int = 50000) -> float:
        """Measure the period of the heteroclinic cycle.

        Runs the simulation and finds the period by detecting when the
        dominant species returns to the initial dominant species for the
        second time (completing one full cycle).

        Args:
            n_steps: number of simulation steps to run.

        Returns:
            Estimated period in time units. Returns np.inf if no cycle detected.
        """
        self.reset()
        dt = self.config.dt

        # Run a transient to let the system settle
        transient = min(n_steps // 5, 5000)
        for _ in range(transient):
            self.step()

        # Track dominant species changes
        initial_dominant = int(np.argmax(self._state))
        cycle_count = 0
        transition_times: list[float] = []
        prev_dominant = initial_dominant
        t = 0.0

        for _ in range(n_steps - transient):
            self.step()
            t += dt
            current_dominant = int(np.argmax(self._state))

            if current_dominant != prev_dominant:
                if current_dominant == initial_dominant:
                    cycle_count += 1
                    transition_times.append(t)
                prev_dominant = current_dominant

        if len(transition_times) >= 2:
            # Average period from consecutive returns
            periods = np.diff(transition_times)
            return float(np.mean(periods))
        elif len(transition_times) == 1:
            return float(transition_times[0])
        return np.inf

    def biodiversity_index(self, state: np.ndarray | None = None) -> float:
        """Compute Shannon entropy H = -sum(p_i * log(p_i)).

        Args:
            state: population vector. Uses current state if None.

        Returns:
            Shannon diversity index. Returns 0 if total population is zero.
        """
        if state is None:
            state = self._state
        total = np.sum(state)
        if total <= 0:
            return 0.0
        p = state / total
        mask = p > 0
        return float(-np.sum(p[mask] * np.log(p[mask])))

    def competition_parameter_sweep(
        self,
        a_values: np.ndarray | list[float],
        n_steps: int = 30000,
    ) -> dict[str, np.ndarray]:
        """Sweep the strong competition parameter a and measure dynamics.

        For each value of a, runs the simulation and records:
        - heteroclinic cycle period
        - final biodiversity index
        - final total population
        - whether cyclic dominance occurs

        Args:
            a_values: array of competition strengths to sweep.
            n_steps: simulation steps per parameter value.

        Returns:
            Dict with a_values, periods, biodiversity, total_pop, is_cyclic.
        """
        a_values = np.asarray(a_values)
        periods = []
        biodiversity = []
        total_pop = []
        is_cyclic = []

        original_a = self.a
        original_alpha = self.alpha.copy()

        for a_val in a_values:
            self.a = a_val
            self.alpha = self.build_competition_matrix()
            self.reset()

            states = [self.observe().copy()]
            for _ in range(n_steps):
                self.step()
                states.append(self.observe().copy())
            traj = np.array(states)

            # Measure period from dominance transitions
            dominance = self.compute_dominance_index(traj)
            changes = np.diff(dominance)
            n_transitions = int(np.sum(changes != 0))
            is_cyclic.append(n_transitions >= self.n_species)

            # Simple period estimate from total population oscillation
            total = self.compute_total_population(traj)
            # Find peaks in total population
            skip = len(total) // 5
            total_tail = total[skip:]
            peaks = []
            for j in range(1, len(total_tail) - 1):
                if total_tail[j] > total_tail[j - 1] and total_tail[j] > total_tail[j + 1]:
                    peaks.append(j)
            if len(peaks) >= 2:
                peak_diffs = np.diff(peaks) * self.config.dt
                periods.append(float(np.mean(peak_diffs)))
            else:
                periods.append(np.inf)

            biodiversity.append(self.biodiversity_index())
            total_pop.append(float(np.sum(self.observe())))

        # Restore original parameters
        self.a = original_a
        self.alpha = original_alpha

        return {
            "a_values": a_values,
            "periods": np.array(periods),
            "biodiversity": np.array(biodiversity),
            "total_population": np.array(total_pop),
            "is_cyclic": np.array(is_cyclic),
        }

    def n_surviving(self, threshold: float = 1e-3) -> int:
        """Count species with population above threshold."""
        return int(np.sum(self._state > threshold))

    def community_matrix(self) -> np.ndarray:
        """Compute the Jacobian at the interior fixed point.

        At the fixed point where growth term = 0:
            J_ij = -r_i * x_i* * alpha_ij / K_j

        Returns:
            n_species x n_species Jacobian matrix.
        """
        x_star = self.compute_interior_fixed_point()
        n = self.n_species
        J = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                J[i, j] = -self.r[i] * x_star[i] * self.alpha[i, j] / self.K[j]
        return J

    def stability_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the community matrix at the interior fixed point.

        For the cyclic May-Leonard system, the interior fixed point is
        typically unstable (positive real part eigenvalues), which drives
        the heteroclinic cycling.

        Returns:
            Complex eigenvalue array.
        """
        J = self.community_matrix()
        return np.linalg.eigvals(J)
