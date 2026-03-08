"""Replicator-mutator equation for evolutionary game dynamics.

The replicator-mutator equation models the evolution of N strategies in
a population under frequency-dependent selection and mutation:

    dx_i/dt = sum_j(x_j * Q_ji * f_j(x)) - phi(x) * x_i

where:
- x_i = frequency of strategy i (on the simplex, sum_i x_i = 1)
- f_j(x) = sum_k(A_jk * x_k) = fitness of strategy j (payoff matrix A)
- Q_ji = mutation probability from strategy j to strategy i
- phi(x) = sum_i(x_i * f_i(x)) = average population fitness

When Q = I (no mutation), this reduces to the standard replicator equation:
    dx_i/dt = x_i * (f_i(x) - phi(x))

Target rediscoveries:
- Rock-Paper-Scissors: heteroclinic cycles, interior fixed point at (1/3, 1/3, 1/3)
- Hawk-Dove: ESS frequency p* = (V - C) / (V + C) for V < C
- Prisoner's Dilemma: cooperation collapses (defect dominates)
- Mutation stabilizes interior equilibria (breaks heteroclinic cycles)
- SINDy ODE recovery on 3-strategy dynamics
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ReplicatorMutatorSimulation(SimulationEnvironment):
    """Replicator-mutator equation on the N-strategy simplex.

    State vector: [x_1, x_2, ..., x_N] where x_i is the frequency of
    strategy i and sum_i x_i = 1.

    Equations:
        dx_i/dt = sum_j(x_j * Q_ji * f_j(x)) - phi(x) * x_i
        f_j(x) = sum_k(A_jk * x_k)
        phi(x) = sum_i(x_i * f_i(x))

    Parameters (via config.parameters):
        n_strategies: number of strategies (default 3)
        A_i_j: payoff matrix entry (row i, col j)
            Default: Rock-Paper-Scissors [[0,-1,1],[1,0,-1],[-1,1,0]]
        mutation_rate: uniform mutation rate mu in [0, 1] (default 0.0)
            Q_ij = (1 - mu) * delta_ij + mu / N
        x_0_i: initial frequency of strategy i (default 1/N)
    """

    # Classic Rock-Paper-Scissors payoff matrix
    DEFAULT_PAYOFF_RPS = [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ]

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        self.n_strategies = int(p.get("n_strategies", 3))
        n = self.n_strategies

        # Build payoff matrix A
        self.payoff = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                default_val = (
                    self.DEFAULT_PAYOFF_RPS[i][j]
                    if i < len(self.DEFAULT_PAYOFF_RPS)
                    and j < len(self.DEFAULT_PAYOFF_RPS[i])
                    else 0.0
                )
                self.payoff[i, j] = p.get(f"A_{i}_{j}", default_val)

        # Mutation rate: Q_ji = (1 - mu) * delta_ji + mu / N
        self.mutation_rate = p.get("mutation_rate", 0.0)
        self.mutation_matrix = self._build_mutation_matrix()

        # Initial frequencies (default: uniform on simplex)
        x_init = np.array([
            p.get(f"x_0_{i}", 1.0 / n) for i in range(n)
        ], dtype=np.float64)
        # Normalize to ensure simplex constraint
        self.x_init = x_init / np.sum(x_init)

    def _build_mutation_matrix(self) -> np.ndarray:
        """Build mutation matrix Q where Q_ji = P(j mutates to i).

        With uniform mutation rate mu:
            Q_ji = (1 - mu) * delta_ji + mu / N

        Each row of Q sums to 1 (stochastic matrix).
        """
        n = self.n_strategies
        mu = self.mutation_rate
        Q = (1.0 - mu) * np.eye(n) + mu / n * np.ones((n, n))
        return Q.astype(np.float64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize strategy frequencies on the simplex."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Small perturbation from initial condition for seeded runs
            perturbation = rng.uniform(-0.01, 0.01, self.n_strategies)
            self._state = self.x_init + perturbation
            self._state = np.maximum(self._state, 1e-10)
            self._state /= np.sum(self._state)
        else:
            self._state = self.x_init.copy()
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with simplex projection."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current strategy frequencies [x_1, ..., x_N]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step with simplex projection."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(self._project_simplex(y + 0.5 * dt * k1))
        k3 = self._derivatives(self._project_simplex(y + 0.5 * dt * k2))
        k4 = self._derivatives(self._project_simplex(y + dt * k3))

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = self._project_simplex(self._state)

    def _project_simplex(self, x: np.ndarray) -> np.ndarray:
        """Project onto the probability simplex (non-negative, sum to 1)."""
        x_clipped = np.maximum(x, 0.0)
        total = np.sum(x_clipped)
        if total > 0:
            return x_clipped / total
        # Fallback to uniform if all zero
        return np.ones(self.n_strategies) / self.n_strategies

    def _derivatives(self, x: np.ndarray) -> np.ndarray:
        """Compute dx/dt for the replicator-mutator equation.

        dx_i/dt = sum_j(x_j * Q_ji * f_j(x)) - phi(x) * x_i

        Args:
            x: current strategy frequencies, shape (n_strategies,).

        Returns:
            Time derivatives, shape (n_strategies,).
        """
        # Fitness of each strategy: f_j = sum_k A_jk * x_k
        fitness = self.payoff @ x  # shape (n,)

        # Average fitness: phi = sum_j x_j * f_j
        avg_fitness = np.dot(x, fitness)

        # Replicator-mutator: dx_i/dt = sum_j(x_j * Q_ji * f_j) - phi * x_i
        # Q_ji = mutation_matrix[j, i], so we need sum over j of x_j * Q[j,i] * f_j
        # This is (Q^T) @ (x * f) for each i
        growth = self.mutation_matrix.T @ (x * fitness)

        return growth - avg_fitness * x

    def fitness(self, x: np.ndarray | None = None) -> np.ndarray:
        """Compute fitness of each strategy at current or given frequencies.

        Args:
            x: strategy frequencies. If None, uses current state.

        Returns:
            Fitness array, shape (n_strategies,).
        """
        if x is None:
            x = self._state
        return self.payoff @ x

    def average_fitness(self, x: np.ndarray | None = None) -> float:
        """Compute average population fitness phi(x) = sum_i x_i * f_i(x).

        Args:
            x: strategy frequencies. If None, uses current state.

        Returns:
            Scalar average fitness.
        """
        if x is None:
            x = self._state
        f = self.fitness(x)
        return float(np.dot(x, f))

    @property
    def interior_fixed_point(self) -> np.ndarray | None:
        """Compute the interior fixed point of the replicator equation (Q=I).

        For zero-sum antisymmetric games (like RPS), the interior fixed point
        is the uniform distribution (1/N, ..., 1/N).

        For general payoff matrices, the interior fixed point satisfies
        A @ x* = phi * 1, which gives x* proportional to the solution of
        A @ x = c * 1 on the simplex.

        Returns:
            Interior fixed point frequencies, or None if not found.
        """
        n = self.n_strategies
        # For antisymmetric zero-sum games, the interior FP is uniform
        if np.allclose(self.payoff, -self.payoff.T):
            return np.ones(n) / n

        # General case: solve (A - phi*I) x = 0 subject to sum(x) = 1
        # At interior FP, all fitnesses are equal: Ax = phi * 1
        # Augmented system: [A; 1^T] x = [phi*1; 1]
        # We solve A x = c * ones subject to sum(x) = 1
        try:
            # Add constraint sum(x) = 1 via augmented system
            A_aug = np.vstack([self.payoff, np.ones(n)])
            # We want Ax = c*1 and sum(x) = 1
            # Equivalent: [A - A_col_mean; 1^T] x = [0; 1]
            # Simpler approach: solve least-squares
            b_aug = np.zeros(n + 1)
            b_aug[-1] = 1.0
            # Replace fitness equality constraint with: A[0,:]-A[i,:] for i>0
            A_eq = np.zeros((n, n))
            for i in range(n - 1):
                A_eq[i] = self.payoff[0] - self.payoff[i + 1]
            A_eq[n - 1] = np.ones(n)
            b_eq = np.zeros(n)
            b_eq[n - 1] = 1.0
            x_star = np.linalg.solve(A_eq, b_eq)
            if np.all(x_star > -1e-10):
                x_star = np.maximum(x_star, 0.0)
                x_star /= np.sum(x_star)
                return x_star
        except np.linalg.LinAlgError:
            pass
        return None

    def is_ess(self, strategy_idx: int) -> bool:
        """Check if a pure strategy is an Evolutionary Stable Strategy (ESS).

        Strategy i is ESS if:
        1. A_ii >= A_ji for all j (best reply to itself)
        2. If A_ii = A_ji, then A_ij > A_jj (strict second condition)

        Args:
            strategy_idx: index of the pure strategy to check.

        Returns:
            True if strategy is an ESS.
        """
        i = strategy_idx
        n = self.n_strategies
        A = self.payoff

        for j in range(n):
            if j == i:
                continue
            # First condition: A_ii >= A_ji
            if A[i, i] < A[j, i]:
                return False
            # Second condition (if equality): A_ij > A_jj
            if np.isclose(A[i, i], A[j, i]) and A[i, j] <= A[j, j]:
                return False
        return True

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period of the first strategy frequency.

        Uses zero-crossing detection on x_0 - x_0_mean.

        Args:
            n_periods: number of periods to average over.

        Returns:
            Mean oscillation period, or inf if no oscillation detected.
        """
        dt = self.config.dt
        n = self.n_strategies
        x_mean = 1.0 / n  # expected mean for symmetric games

        # Transient
        transient_steps = int(100 / dt)
        for _ in range(transient_steps):
            self.step()

        # Detect upward crossings of x_0 through x_mean
        crossings = []
        prev_x = self._state[0]
        for _ in range(int(n_periods * 200 / dt)):
            self.step()
            x = self._state[0]
            if prev_x < x_mean and x >= x_mean:
                t_cross = (self._step_count - 1) * dt + dt * (
                    x_mean - prev_x
                ) / (x - prev_x + 1e-30)
                crossings.append(t_cross)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))

    def hawk_dove_ess(self) -> float | None:
        """Compute the theoretical ESS frequency for Hawk-Dove game.

        For the Hawk-Dove (Chicken) game with payoffs:
            A = [[V_hd/2 - C_hd/2, V_hd], [0, V_hd/2]]
        or equivalently in the symmetric 2-strategy form with
            A = [[(V-C)/2, V], [0, V/2]]

        The mixed ESS is p* = V/C (probability of Hawk).

        This method extracts V and C from a 2x2 payoff matrix matching
        the Hawk-Dove structure.

        Returns:
            ESS frequency of the first strategy (Hawk), or None if not
            a valid Hawk-Dove game.
        """
        if self.n_strategies != 2:
            return None

        A = self.payoff
        # Standard Hawk-Dove: A = [[(V-C)/2, V], [0, V/2]]
        # From A[0,1] = V and A[1,1] = V/2, so V = 2*A[1,1]
        # From A[0,0] = (V-C)/2, so C = V - 2*A[0,0]
        V = 2.0 * A[1, 1]
        C = V - 2.0 * A[0, 0]

        if C <= 0 or V <= 0:
            return None
        if V >= C:
            # Hawk is ESS (pure strategy dominates)
            return 1.0

        # Mixed ESS: p* = V/C
        return float(V / C)

    def simulate_to_steady_state(
        self,
        max_steps: int = 100000,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """Run until steady state or max_steps reached.

        Args:
            max_steps: maximum number of steps.
            tol: convergence tolerance on max |dx/dt|.

        Returns:
            Final state frequencies.
        """
        for _ in range(max_steps):
            dx = self._derivatives(self._state)
            if np.max(np.abs(dx)) < tol:
                break
            self.step()
        return self._state
