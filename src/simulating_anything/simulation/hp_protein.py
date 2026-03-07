"""HP lattice protein folding simulation.

The HP (Hydrophobic-Polar) model is the simplest model that captures
the essential physics of protein folding: the hydrophobic collapse.
Residues on a 2D lattice are either H (hydrophobic) or P (polar).
The native state minimizes energy by maximizing H-H contacts.

Discovery targets:
- Ground state energy for known sequences
- Folding temperature T_f (specific heat peak)
- Free energy landscape and folding funnel
- Contact order vs folding rate correlation
- Designability: some structures are far more common as ground states
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class HPProteinSimulation(SimulationEnvironment):
    """HP lattice protein folding via Monte Carlo on 2D square lattice.

    A chain of N residues (H or P) is placed on a 2D lattice.
    Moves: end moves, corner moves, crankshaft moves.
    Energy: E = -epsilon * (number of non-bonded H-H contacts).

    State vector: [x1, y1, x2, y2, ..., xN, yN, energy, n_contacts,
                   radius_of_gyration, T] of shape (2*N + 3,).

    Parameters:
        sequence: encoded as list of 0s (H) and 1s (P) via "sequence_len"
            and "h_fraction" for random generation, or fixed known sequences.
        N: chain length (default: 20)
        h_fraction: fraction of hydrophobic residues (default: 0.5)
        T: temperature (default: 1.0, in units of epsilon/k_B)
        epsilon: H-H contact energy (default: 1.0)
        seed: random seed
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 20))
        self.h_fraction = p.get("h_fraction", 0.5)
        self.T = p.get("T", 1.0)
        self.epsilon = p.get("epsilon", 1.0)
        self._seed = config.seed
        self._rng = np.random.RandomState(self._seed)

        # Generate sequence: 0 = H (hydrophobic), 1 = P (polar)
        self._sequence = self._generate_sequence()

        # Positions on 2D lattice
        self._positions: np.ndarray = np.zeros((self.N, 2), dtype=np.int32)
        self._energy = 0.0
        self._n_contacts = 0
        self._best_energy = 0.0

    def _generate_sequence(self) -> np.ndarray:
        """Generate HP sequence."""
        n_h = int(self.N * self.h_fraction)
        seq = np.zeros(self.N, dtype=np.int32)
        seq[n_h:] = 1  # P residues
        self._rng.shuffle(seq)
        return seq

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._step_count = 0

        # Initialize as straight chain along x-axis
        self._positions = np.zeros((self.N, 2), dtype=np.int32)
        for i in range(self.N):
            self._positions[i] = [i, 0]

        self._energy = self._compute_energy()
        self._best_energy = self._energy
        self._n_contacts = self._count_hh_contacts()
        self._state = self._build_state()
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Perform one Monte Carlo sweep (N attempted moves)."""
        for _ in range(self.N):
            self._mc_move()
        self._step_count += 1
        self._state = self._build_state()
        return self._state.copy()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    def _build_state(self) -> np.ndarray:
        """Build flat state vector."""
        pos_flat = self._positions.ravel().astype(np.float64)
        rg = self._radius_of_gyration()
        extras = np.array([self._energy, float(self._n_contacts), rg, self.T])
        return np.concatenate([pos_flat, extras])

    def _compute_energy(self) -> float:
        """Compute total energy: -epsilon per non-bonded H-H contact."""
        return -self.epsilon * self._count_hh_contacts()

    def _count_hh_contacts(self) -> int:
        """Count non-bonded H-H nearest-neighbor contacts."""
        contacts = 0
        pos_set = {}
        for i in range(self.N):
            pos_set[tuple(self._positions[i])] = i

        for i in range(self.N):
            if self._sequence[i] != 0:  # Not H
                continue
            x, y = self._positions[i]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (x + dx, y + dy)
                if neighbor in pos_set:
                    j = pos_set[neighbor]
                    if j > i + 1 and self._sequence[j] == 0:
                        # Non-bonded (not sequential) H-H contact
                        contacts += 1
                    elif j < i - 1 and self._sequence[j] == 0:
                        pass  # Already counted when i was smaller

        return contacts

    def _mc_move(self) -> None:
        """Attempt a single Monte Carlo move."""
        # Pick random residue
        idx = self._rng.randint(0, self.N)

        # Try a move: end move, corner move, or crankshaft
        new_pos = self._propose_move(idx)
        if new_pos is None:
            return

        # Check self-avoidance
        old_pos = self._positions[idx].copy()
        pos_set = set()
        for i in range(self.N):
            if i != idx:
                pos_set.add(tuple(self._positions[i]))

        if tuple(new_pos) in pos_set:
            return  # Overlap, reject

        # Check connectivity
        self._positions[idx] = new_pos
        if not self._is_connected():
            self._positions[idx] = old_pos
            return

        # Compute energy change
        new_energy = self._compute_energy()
        dE = new_energy - self._energy

        # Metropolis criterion
        if dE <= 0 or self._rng.random() < np.exp(-dE / max(self.T, 1e-10)):
            self._energy = new_energy
            self._n_contacts = self._count_hh_contacts()
            if self._energy < self._best_energy:
                self._best_energy = self._energy
        else:
            self._positions[idx] = old_pos

    def _propose_move(self, idx: int) -> np.ndarray | None:
        """Propose a new position for residue idx."""
        x, y = self._positions[idx]
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        if idx == 0 or idx == self.N - 1:
            # End move: move to any adjacent position of neighbor
            neighbor_idx = 1 if idx == 0 else self.N - 2
            nx, ny = self._positions[neighbor_idx]
            candidates = directions + np.array([nx, ny])
            valid = [c for c in candidates if not np.array_equal(c, [x, y])]
            if valid:
                return valid[self._rng.randint(len(valid))]
            return None
        else:
            # Corner move: if residues i-1 and i+1 are diagonal
            prev = self._positions[idx - 1]
            nxt = self._positions[idx + 1]
            diff = nxt - prev
            if abs(diff[0]) == 1 and abs(diff[1]) == 1:
                # Two possible corner positions
                c1 = np.array([prev[0], nxt[1]])
                c2 = np.array([nxt[0], prev[1]])
                choice = c1 if not np.array_equal(c1, [x, y]) else c2
                return choice
            return None

    def _is_connected(self) -> bool:
        """Check that the chain is connected (sequential neighbors are lattice neighbors)."""
        for i in range(self.N - 1):
            diff = np.abs(self._positions[i + 1] - self._positions[i])
            if diff.sum() != 1:
                return False
        return True

    def _radius_of_gyration(self) -> float:
        """Compute radius of gyration."""
        com = np.mean(self._positions, axis=0).astype(np.float64)
        diff = self._positions.astype(np.float64) - com
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    def specific_heat(self, energies: np.ndarray) -> float:
        """Compute specific heat from energy time series: C = var(E) / T^2."""
        if self.T < 1e-10 or len(energies) < 2:
            return 0.0
        return float(np.var(energies) / self.T ** 2)

    def get_contact_map(self) -> np.ndarray:
        """Return N x N contact map (1 if residues are lattice neighbors)."""
        cmap = np.zeros((self.N, self.N), dtype=np.int32)
        for i in range(self.N):
            for j in range(i + 2, self.N):
                diff = np.abs(self._positions[i] - self._positions[j])
                if diff.sum() == 1:
                    cmap[i, j] = 1
                    cmap[j, i] = 1
        return cmap
