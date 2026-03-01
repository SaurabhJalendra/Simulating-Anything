"""Network SIS epidemic model simulation.

SIS (Susceptible-Infected-Susceptible) model on a contact network.
Mean-field ODE approximation: dp_i/dt = -gamma*p_i + beta*(1-p_i)*sum_j(A_ij*p_j)

Key properties:
- Epidemic threshold: beta_c/gamma = 1/lambda_max(A) where lambda_max is spectral radius
- Below threshold: disease-free equilibrium (all infection dies out)
- Above threshold: endemic equilibrium persists
- Network topology determines threshold:
  - Complete graph: beta_c = gamma/N
  - Erdos-Renyi: beta_c ~ gamma/<k>
  - Regular lattice: threshold depends on dimension and degree
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NetworkSISSimulation(SimulationEnvironment):
    """Network SIS epidemic model with mean-field ODE dynamics.

    State vector: [p_1, ..., p_N] infection probabilities for N nodes.

    Parameters:
        N: number of nodes in the network
        beta: infection rate per contact
        gamma: recovery rate
        network_type: 'erdos_renyi', 'complete', or 'regular'
        mean_degree: average number of connections per node
        initial_fraction: fraction of initially infected nodes
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 50))
        self.beta = p.get("beta", 0.3)
        self.gamma = p.get("gamma", 0.1)
        self.mean_degree = p.get("mean_degree", 6.0)
        self.initial_fraction = p.get("initial_fraction", 0.1)
        # String parameters cannot go in parameters dict; use class attribute
        self.network_type = "erdos_renyi"
        self._adjacency: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize infection probabilities and generate network."""
        rng = np.random.default_rng(seed)
        self._adjacency = self.generate_network(
            self.network_type, self.N, self.mean_degree, rng,
        )
        # Initialize: a fraction of nodes start infected
        n_infected = max(1, int(self.N * self.initial_fraction))
        self._state = np.zeros(self.N, dtype=np.float64)
        infected_indices = rng.choice(self.N, size=n_infected, replace=False)
        self._state[infected_indices] = 1.0
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current infection probabilities."""
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
        # Clamp probabilities to [0, 1]
        self._state = np.clip(self._state, 0.0, 1.0)

    def _derivatives(self, p: np.ndarray) -> np.ndarray:
        """Compute dp/dt for all nodes.

        dp_i/dt = -gamma*p_i + beta*(1-p_i)*sum_j(A_ij*p_j)
        """
        # Neighbor infection pressure: A @ p gives sum of infected neighbors
        neighbor_pressure = self._adjacency @ p
        return -self.gamma * p + self.beta * (1.0 - p) * neighbor_pressure

    @staticmethod
    def generate_network(
        network_type: str,
        N: int,
        mean_degree: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a symmetric adjacency matrix for the given network type.

        Args:
            network_type: 'erdos_renyi', 'complete', or 'regular'
            N: number of nodes
            mean_degree: average degree (edges per node)
            rng: random number generator

        Returns:
            NxN symmetric binary adjacency matrix (no self-loops).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        if network_type == "complete":
            adj = np.ones((N, N), dtype=np.float64) - np.eye(N, dtype=np.float64)
        elif network_type == "regular":
            adj = _generate_regular_lattice(N, int(round(mean_degree)))
        else:
            # Erdos-Renyi: p = mean_degree / (N-1)
            p_edge = min(mean_degree / max(N - 1, 1), 1.0)
            upper = rng.random((N, N)) < p_edge
            # Make symmetric, remove self-loops
            upper = np.triu(upper, k=1)
            adj = (upper | upper.T).astype(np.float64)

        return adj

    def compute_epidemic_threshold(self) -> float:
        """Compute epidemic threshold tau_c = beta_c/gamma = 1/lambda_max(A).

        Returns:
            Critical ratio beta_c/gamma.
        """
        lam = self.spectral_radius()
        if lam < 1e-12:
            return float("inf")
        return 1.0 / lam

    def compute_prevalence(self) -> float:
        """Compute current prevalence: mean infection probability."""
        return float(np.mean(self._state))

    def endemic_equilibrium(self, n_transient: int = 10000) -> float:
        """Run to steady state and return endemic prevalence."""
        for _ in range(n_transient):
            self.step()
        # Average over additional steps
        prevalences = []
        for _ in range(2000):
            self.step()
            prevalences.append(self.compute_prevalence())
        return float(np.mean(prevalences))

    def infection_sweep(
        self,
        beta_values: np.ndarray,
        n_transient: int = 8000,
        n_measure: int = 2000,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Sweep beta values and measure steady-state prevalence.

        Args:
            beta_values: array of beta values to test
            n_transient: steps to skip for transient
            n_measure: steps to average over
            seed: random seed for network and initial conditions

        Returns:
            Dict with 'beta', 'prevalence', 'tau' (beta/gamma) arrays.
        """
        prevalences = []
        for b in beta_values:
            self.beta = b
            self.reset(seed=seed)
            for _ in range(n_transient):
                self.step()
            prev_vals = []
            for _ in range(n_measure):
                self.step()
                prev_vals.append(self.compute_prevalence())
            prevalences.append(float(np.mean(prev_vals)))

        return {
            "beta": np.array(beta_values),
            "prevalence": np.array(prevalences),
            "tau": np.array(beta_values) / self.gamma,
        }

    def degree_distribution(self) -> np.ndarray:
        """Return degree of each node (number of neighbors)."""
        return np.sum(self._adjacency, axis=1)

    def spectral_radius(self) -> float:
        """Largest eigenvalue of the adjacency matrix."""
        eigenvalues = np.linalg.eigvalsh(self._adjacency)
        return float(np.max(np.abs(eigenvalues)))


def _generate_regular_lattice(N: int, degree: int) -> np.ndarray:
    """Generate a regular ring lattice where each node connects to 'degree' neighbors.

    Each node connects to degree/2 nearest neighbors on each side (ring topology).
    """
    adj = np.zeros((N, N), dtype=np.float64)
    half = degree // 2
    for i in range(N):
        for d in range(1, half + 1):
            j = (i + d) % N
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj
