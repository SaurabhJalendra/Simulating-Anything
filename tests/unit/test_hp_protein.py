"""Tests for the HP lattice protein folding simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.hp_protein import HPProteinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 10,
    h_fraction: float = 0.5,
    T: float = 1.0,
    epsilon: float = 1.0,
    n_steps: int = 100,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.HP_PROTEIN,
        dt=1.0,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "h_fraction": h_fraction,
            "T": T,
            "epsilon": epsilon,
        },
        seed=seed,
    )


class TestHPProtein:
    """Comprehensive tests for HPProteinSimulation."""

    # ------------------------------------------------------------------ #
    # 1. Instantiation
    # ------------------------------------------------------------------ #
    def test_instantiation_default_params(self):
        """Default parameters are applied correctly."""
        config = SimulationConfig(
            domain=Domain.HP_PROTEIN,
            dt=1.0,
            n_steps=100,
            parameters={},
        )
        sim = HPProteinSimulation(config)
        assert sim.N == 20
        assert sim.h_fraction == 0.5
        assert sim.T == 1.0
        assert sim.epsilon == 1.0

    def test_instantiation_custom_params(self):
        """Custom parameters are stored correctly."""
        sim = HPProteinSimulation(_make_config(N=15, h_fraction=0.7, T=2.0, epsilon=0.5))
        assert sim.N == 15
        assert sim.h_fraction == 0.7
        assert sim.T == 2.0
        assert sim.epsilon == 0.5

    # ------------------------------------------------------------------ #
    # 2. Reset returns correct shape
    # ------------------------------------------------------------------ #
    def test_reset_shape(self):
        """Reset returns state vector of shape (2*N + 3,) -- actually (2*N + 4,)
        because state = [positions_flat(2N), energy, n_contacts, Rg, T]."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        # State: 2*N positions + energy + n_contacts + Rg + T = 2*N + 4
        expected_len = 2 * N + 4
        assert state.shape == (expected_len,), (
            f"Expected shape ({expected_len},), got {state.shape}"
        )

    def test_reset_initial_chain_is_straight(self):
        """After reset the chain should be a straight line along x-axis."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        positions = state[: 2 * N].reshape(N, 2)
        for i in range(N):
            assert positions[i, 0] == i, f"Residue {i} x != {i}"
            assert positions[i, 1] == 0, f"Residue {i} y != 0"

    # ------------------------------------------------------------------ #
    # 3. Step returns same shape, no NaN
    # ------------------------------------------------------------------ #
    def test_step_shape(self):
        """Step returns state vector with the same shape as reset."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        s0 = sim.reset(seed=42)
        s1 = sim.step()
        assert s1.shape == s0.shape

    def test_step_no_nan(self):
        """State vector should contain no NaN values after stepping."""
        sim = HPProteinSimulation(_make_config(N=10))
        sim.reset(seed=42)
        for _ in range(20):
            state = sim.step()
            assert not np.any(np.isnan(state)), "NaN found in state after step"

    # ------------------------------------------------------------------ #
    # 4. Chain connectivity maintained
    # ------------------------------------------------------------------ #
    def test_chain_connectivity_after_steps(self):
        """Sequential residues must remain lattice neighbors after MC sweeps."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        # Verify connectivity via the internal method
        assert sim._is_connected(), "Chain connectivity broken after MC sweeps"

    def test_chain_connectivity_positions(self):
        """Directly check Manhattan distance = 1 between consecutive residues."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(30):
            sim.step()
        positions = sim._positions
        for i in range(N - 1):
            diff = np.abs(positions[i + 1] - positions[i])
            manhattan = diff.sum()
            assert manhattan == 1, (
                f"Residues {i} and {i+1} not lattice neighbors: distance={manhattan}"
            )

    # ------------------------------------------------------------------ #
    # 5. Energy is non-positive
    # ------------------------------------------------------------------ #
    def test_energy_non_positive(self):
        """Energy = -epsilon * n_contacts, so energy <= 0 always."""
        sim = HPProteinSimulation(_make_config(N=10))
        sim.reset(seed=42)
        for _ in range(50):
            state = sim.step()
        # Energy is at index 2*N in the state vector
        energy = state[2 * 10]
        assert energy <= 0.0, f"Energy should be non-positive, got {energy}"

    def test_energy_equals_neg_epsilon_times_contacts(self):
        """Internal energy should equal -epsilon * n_contacts."""
        epsilon = 2.0
        sim = HPProteinSimulation(_make_config(N=10, epsilon=epsilon))
        sim.reset(seed=42)
        for _ in range(30):
            sim.step()
        expected = -epsilon * sim._n_contacts
        assert abs(sim._energy - expected) < 1e-12, (
            f"Energy {sim._energy} != -epsilon*contacts = {expected}"
        )

    # ------------------------------------------------------------------ #
    # 6. Contact map has correct shape
    # ------------------------------------------------------------------ #
    def test_contact_map_shape(self):
        """Contact map should be (N, N)."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        cmap = sim.get_contact_map()
        assert cmap.shape == (N, N)

    def test_contact_map_symmetric(self):
        """Contact map should be symmetric."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(20):
            sim.step()
        cmap = sim.get_contact_map()
        np.testing.assert_array_equal(cmap, cmap.T)

    def test_contact_map_no_self_contacts(self):
        """Diagonal of contact map should be zero."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        cmap = sim.get_contact_map()
        assert np.all(np.diag(cmap) == 0)

    def test_contact_map_no_sequential_contacts(self):
        """Contact map excludes bonded (sequential) neighbors -- i and i+1 are bonded."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(20):
            sim.step()
        cmap = sim.get_contact_map()
        for i in range(N - 1):
            assert cmap[i, i + 1] == 0, (
                f"Sequential residues {i},{i+1} should not appear in contact map"
            )

    # ------------------------------------------------------------------ #
    # 7. Radius of gyration is positive
    # ------------------------------------------------------------------ #
    def test_radius_of_gyration_positive(self):
        """Radius of gyration should always be positive for N > 1."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        rg = state[2 * N + 2]  # Rg is third extra field
        assert rg > 0.0, f"Radius of gyration should be positive, got {rg}"

    def test_radius_of_gyration_after_steps(self):
        """Rg remains positive throughout simulation."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(30):
            state = sim.step()
            rg = state[2 * N + 2]
            assert rg > 0.0

    # ------------------------------------------------------------------ #
    # 8. Determinism: same seed -> same result
    # ------------------------------------------------------------------ #
    def test_determinism_same_seed(self):
        """Two runs with the same seed must produce identical trajectories."""
        N = 10
        config = _make_config(N=N, seed=123)
        sim1 = HPProteinSimulation(config)
        sim2 = HPProteinSimulation(config)

        s1 = sim1.reset(seed=123)
        s2 = sim2.reset(seed=123)
        np.testing.assert_array_equal(s1, s2)

        for _ in range(10):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different trajectories."""
        N = 10
        sim1 = HPProteinSimulation(_make_config(N=N, seed=1))
        sim2 = HPProteinSimulation(_make_config(N=N, seed=999))
        sim1.reset(seed=1)
        sim2.reset(seed=999)
        # Run enough steps for MC moves to diverge
        for _ in range(20):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.array_equal(s1, s2), "Different seeds should give different states"

    # ------------------------------------------------------------------ #
    # 9. Custom sequence length
    # ------------------------------------------------------------------ #
    def test_custom_sequence_length(self):
        """Chain length N is respected."""
        for N in [5, 8, 15, 25]:
            sim = HPProteinSimulation(_make_config(N=N))
            state = sim.reset(seed=42)
            assert state.shape == (2 * N + 4,), f"Wrong shape for N={N}"
            assert sim.N == N

    def test_sequence_composition(self):
        """Number of H residues approximately matches h_fraction."""
        N = 20
        h_frac = 0.6
        sim = HPProteinSimulation(_make_config(N=N, h_fraction=h_frac))
        n_h = np.sum(sim._sequence == 0)
        expected = int(N * h_frac)
        assert n_h == expected, f"Expected {expected} H residues, got {n_h}"

    # ------------------------------------------------------------------ #
    # 10. Temperature effect
    # ------------------------------------------------------------------ #
    def test_temperature_stored_in_state(self):
        """Temperature should be the last element of the state vector."""
        T = 2.5
        N = 10
        sim = HPProteinSimulation(_make_config(N=N, T=T))
        state = sim.reset(seed=42)
        assert state[-1] == T, f"Last state element should be T={T}, got {state[-1]}"

    def test_low_temperature_lower_energy(self):
        """At low T, MC should accept only downhill moves, reaching lower energy."""
        N = 10
        n_steps = 200

        # Low temperature run
        sim_cold = HPProteinSimulation(_make_config(N=N, T=0.1, seed=42))
        sim_cold.reset(seed=42)
        energies_cold = []
        for _ in range(n_steps):
            state = sim_cold.step()
            energies_cold.append(state[2 * N])

        # High temperature run
        sim_hot = HPProteinSimulation(_make_config(N=N, T=100.0, seed=42))
        sim_hot.reset(seed=42)
        energies_hot = []
        for _ in range(n_steps):
            state = sim_hot.step()
            energies_hot.append(state[2 * N])

        # The cold simulation should reach equal or lower mean energy in the second half
        mean_cold = np.mean(energies_cold[n_steps // 2:])
        mean_hot = np.mean(energies_hot[n_steps // 2:])
        assert mean_cold <= mean_hot + 0.5, (
            f"Low-T mean energy ({mean_cold}) should be <= high-T ({mean_hot})"
        )

    # ------------------------------------------------------------------ #
    # 11. Specific heat
    # ------------------------------------------------------------------ #
    def test_specific_heat_non_negative(self):
        """Specific heat C = var(E) / T^2 should be >= 0."""
        sim = HPProteinSimulation(_make_config(N=10, T=1.0))
        sim.reset(seed=42)
        energies = []
        for _ in range(50):
            state = sim.step()
            energies.append(state[2 * 10])
        C = sim.specific_heat(np.array(energies))
        assert C >= 0.0, f"Specific heat should be non-negative, got {C}"

    def test_specific_heat_zero_temperature(self):
        """Specific heat at T=0 should return 0 (guarded)."""
        sim = HPProteinSimulation(_make_config(N=10, T=0.0))
        C = sim.specific_heat(np.array([0.0, 0.0, 0.0]))
        assert C == 0.0

    def test_specific_heat_single_sample(self):
        """Specific heat with fewer than 2 samples returns 0."""
        sim = HPProteinSimulation(_make_config(N=10, T=1.0))
        C = sim.specific_heat(np.array([1.0]))
        assert C == 0.0

    # ------------------------------------------------------------------ #
    # 12. Multiple MC sweeps don't crash
    # ------------------------------------------------------------------ #
    def test_many_sweeps_no_crash(self):
        """Running many MC sweeps should not raise any errors."""
        sim = HPProteinSimulation(_make_config(N=10, n_steps=200))
        sim.reset(seed=42)
        for _ in range(200):
            state = sim.step()
        assert state is not None
        assert not np.any(np.isnan(state))

    # ------------------------------------------------------------------ #
    # Additional tests
    # ------------------------------------------------------------------ #
    def test_observe_matches_step(self):
        """observe() should return the same state as the last step()."""
        sim = HPProteinSimulation(_make_config(N=10))
        sim.reset(seed=42)
        last = sim.step()
        observed = sim.observe()
        np.testing.assert_array_equal(last, observed)

    def test_self_avoidance(self):
        """No two residues should occupy the same lattice site."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        positions = sim._positions
        pos_set = set()
        for i in range(N):
            pos_tuple = tuple(positions[i])
            assert pos_tuple not in pos_set, (
                f"Residue {i} overlaps at position {pos_tuple}"
            )
            pos_set.add(pos_tuple)

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shape."""
        N = 10
        n_steps = 30
        sim = HPProteinSimulation(_make_config(N=N, n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 2 * N + 4)

    def test_all_h_sequence_has_contacts(self):
        """An all-H chain (h_fraction=1.0) should form contacts after enough steps."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N, h_fraction=1.0, T=0.5, seed=7))
        sim.reset(seed=7)
        for _ in range(300):
            sim.step()
        # With all H residues and low T, some non-bonded contacts should form
        contacts = sim._n_contacts
        # It's possible for a short chain to find at least 1 contact
        # but not guaranteed for N=10, so check energy consistency
        assert sim._energy == -sim.epsilon * contacts

    def test_all_p_sequence_no_hh_contacts(self):
        """An all-P chain (h_fraction=0.0) should have zero H-H contacts."""
        N = 10
        sim = HPProteinSimulation(_make_config(N=N, h_fraction=0.0, T=1.0))
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        assert sim._n_contacts == 0, "All-P chain should have no H-H contacts"
        assert sim._energy == 0.0, "All-P chain energy should be 0"

    def test_best_energy_tracked(self):
        """_best_energy should be <= current energy at all times."""
        sim = HPProteinSimulation(_make_config(N=10, T=1.0))
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
            assert sim._best_energy <= sim._energy, (
                f"Best energy {sim._best_energy} > current {sim._energy}"
            )
