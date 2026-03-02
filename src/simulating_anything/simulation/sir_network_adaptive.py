"""SIR dynamics on an adaptive network with rewiring.

Susceptible nodes rewire away from infected neighbors, creating protective
clustering that slows epidemic spread. This couples network topology evolution
with disease dynamics.

Model rules (discrete time, per step):
1. Infection: each S-I edge transmits with probability beta
2. Recovery: each I node recovers with probability gamma
3. Rewiring: each S node with an I neighbor rewires that edge to a random
   S node with probability w (per S-I edge)

Key phenomena:
- Rewiring suppresses epidemic final size
- Bistability: for intermediate w, both endemic and disease-free states coexist
- Network fragmentation at high rewiring rates
- Degree-degree correlations emerge from rewiring
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

# Node states
_S = 0  # Susceptible
_I = 1  # Infected
_R = 2  # Recovered


class SIRNetworkAdaptiveSimulation(SimulationEnvironment):
    """SIR epidemic on an adaptive network with susceptible rewiring.

    State vector: [S_frac, I_frac, R_frac] (fractions, sum = 1)

    Parameters:
        N_nodes: number of nodes in the network (default 200)
        k_avg: average degree of initial Erdos-Renyi network (default 6)
        beta: per-link infection probability per timestep (default 0.1)
        gamma: per-node recovery probability per timestep (default 0.05)
        w: rewiring probability per S-I edge per timestep (default 0.3)
        I_0: number of initially infected nodes (default 5)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N_nodes = int(p.get("N_nodes", 200))
        self.k_avg = p.get("k_avg", 6.0)
        self.beta = p.get("beta", 0.1)
        self.gamma = p.get("gamma", 0.05)
        self.w = p.get("w", 0.3)
        self.I_0 = int(p.get("I_0", 5))

        # Internal state: node statuses and adjacency list
        self._node_status: np.ndarray | None = None
        self._adj: list[set[int]] | None = None
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Erdos-Renyi network and seed initial infected nodes."""
        self._rng = np.random.default_rng(seed)
        N = self.N_nodes

        # Build Erdos-Renyi adjacency list
        p_edge = min(self.k_avg / max(N - 1, 1), 1.0)
        self._adj = [set() for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if self._rng.random() < p_edge:
                    self._adj[i].add(j)
                    self._adj[j].add(i)

        # Initialize node statuses: all S except I_0 random infected
        self._node_status = np.full(N, _S, dtype=np.int32)
        n_infected = min(self.I_0, N)
        infected_nodes = self._rng.choice(N, size=n_infected, replace=False)
        self._node_status[infected_nodes] = _I

        self._step_count = 0
        self._state = self._compute_fractions()
        return self._state

    def step(self) -> np.ndarray:
        """Advance one discrete timestep: infect, recover, rewire."""
        N = self.N_nodes
        rng = self._rng
        status = self._node_status
        adj = self._adj

        # Collect S-I edges for infection and rewiring
        new_infected = set()
        rewire_pairs = []  # (s_node, i_node) pairs for rewiring

        for node in range(N):
            if status[node] != _S:
                continue
            for nbr in list(adj[node]):
                if status[nbr] == _I:
                    # Infection attempt
                    if rng.random() < self.beta:
                        new_infected.add(node)
                    # Rewiring attempt (only if node is still S)
                    if rng.random() < self.w:
                        rewire_pairs.append((node, nbr))

        # Recovery: each I node recovers with probability gamma
        new_recovered = set()
        for node in range(N):
            if status[node] == _I:
                if rng.random() < self.gamma:
                    new_recovered.add(node)

        # Apply rewiring: S node breaks link to I neighbor, connects to random S
        s_nodes = [n for n in range(N) if status[n] == _S and n not in new_infected]
        for s_node, i_node in rewire_pairs:
            # Only rewire if s_node is still S (not newly infected)
            if s_node in new_infected:
                continue
            if i_node not in adj[s_node]:
                continue
            # Find candidate S nodes to rewire to (excluding self and current neighbors)
            candidates = [
                n for n in s_nodes
                if n != s_node and n not in adj[s_node]
            ]
            if not candidates:
                continue
            # Break S-I edge
            adj[s_node].discard(i_node)
            adj[i_node].discard(s_node)
            # Add S-S edge
            target = candidates[rng.integers(len(candidates))]
            adj[s_node].add(target)
            adj[target].add(s_node)

        # Apply state transitions
        for node in new_infected:
            status[node] = _I
        for node in new_recovered:
            status[node] = _R

        self._step_count += 1
        self._state = self._compute_fractions()
        return self._state

    def observe(self) -> np.ndarray:
        """Return current [S_frac, I_frac, R_frac]."""
        return self._state

    def _compute_fractions(self) -> np.ndarray:
        """Compute S, I, R fractions from node statuses."""
        N = self.N_nodes
        n_s = int(np.sum(self._node_status == _S))
        n_i = int(np.sum(self._node_status == _I))
        n_r = int(np.sum(self._node_status == _R))
        return np.array([n_s / N, n_i / N, n_r / N], dtype=np.float64)

    def get_edge_count(self) -> int:
        """Return total number of undirected edges in the network."""
        return sum(len(nbrs) for nbrs in self._adj) // 2

    def get_mean_degree(self) -> float:
        """Return current mean degree of the network."""
        total_degree = sum(len(nbrs) for nbrs in self._adj)
        return total_degree / self.N_nodes

    def get_clustering_coefficient(self) -> float:
        """Compute global clustering coefficient of the current network.

        C = 3 * triangles / connected_triples
        """
        triangles = 0
        triples = 0
        for node in range(self.N_nodes):
            nbrs = list(self._adj[node])
            k = len(nbrs)
            if k < 2:
                continue
            triples += k * (k - 1) // 2
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    if nbrs[j] in self._adj[nbrs[i]]:
                        triangles += 1
        if triples == 0:
            return 0.0
        return triangles / triples

    def get_si_edge_count(self) -> int:
        """Count edges between susceptible and infected nodes."""
        status = self._node_status
        count = 0
        for node in range(self.N_nodes):
            if status[node] == _S:
                for nbr in self._adj[node]:
                    if status[nbr] == _I:
                        count += 1
        return count

    def get_ss_edge_fraction(self) -> float:
        """Fraction of edges that connect two susceptible nodes."""
        status = self._node_status
        ss_count = 0
        total = 0
        for node in range(self.N_nodes):
            for nbr in self._adj[node]:
                if nbr > node:
                    total += 1
                    if status[node] == _S and status[nbr] == _S:
                        ss_count += 1
        if total == 0:
            return 0.0
        return ss_count / total

    def final_size(self) -> float:
        """Return fraction of population that was infected (I + R)."""
        return float(self._state[1] + self._state[2])
