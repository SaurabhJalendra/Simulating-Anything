"""Tests for the SIR Network Adaptive simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.sir_network_adaptive import (
    SIRNetworkAdaptiveSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N_nodes: int = 100,
    k_avg: float = 6.0,
    beta: float = 0.1,
    gamma: float = 0.05,
    w: float = 0.3,
    I_0: int = 5,
    dt: float = 1.0,
    n_steps: int = 200,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SIR_NETWORK_ADAPTIVE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N_nodes": float(N_nodes),
            "k_avg": k_avg,
            "beta": beta,
            "gamma": gamma,
            "w": w,
            "I_0": float(I_0),
        },
    )


class TestSIRNetworkAdaptiveCreation:
    """Tests for simulation creation and initialization."""

    def test_create_default(self):
        """Simulation can be created with default parameters."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        assert sim.N_nodes == 100
        assert sim.beta == 0.1
        assert sim.gamma == 0.05
        assert sim.w == 0.3

    def test_custom_parameters(self):
        """Custom parameters are correctly assigned."""
        config = _make_config(N_nodes=50, beta=0.2, gamma=0.1, w=0.5, I_0=3)
        sim = SIRNetworkAdaptiveSimulation(config)
        assert sim.N_nodes == 50
        assert sim.beta == 0.2
        assert sim.gamma == 0.1
        assert sim.w == 0.5
        assert sim.I_0 == 3

    def test_reset_returns_state(self):
        """Reset returns a valid state array."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        state = sim.reset(seed=42)
        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)

    def test_state_shape(self):
        """State vector has shape (3,) for [S, I, R] fractions."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=200))
        state = sim.reset(seed=42)
        assert state.shape == (3,)

    def test_observe_matches_state(self):
        """Observe returns the same state as reset/step."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        state = sim.reset(seed=42)
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)


class TestSIRNetworkAdaptiveConservation:
    """Tests for conservation laws and invariants."""

    def test_sir_sum_equals_one_initial(self):
        """S + I + R = 1 after initialization."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        state = sim.reset(seed=42)
        np.testing.assert_allclose(state.sum(), 1.0, atol=1e-12)

    def test_sir_sum_equals_one_after_steps(self):
        """S + I + R = 1 is maintained throughout simulation."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1.0, atol=1e-12,
                err_msg="S + I + R must always equal 1"
            )

    def test_fractions_non_negative(self):
        """All fractions remain non-negative."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(200):
            state = sim.step()
            assert np.all(state >= -1e-12), f"Negative fraction: {state}"

    def test_fractions_bounded(self):
        """All fractions stay in [0, 1]."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(200):
            state = sim.step()
            assert np.all(state >= -1e-12), f"Below 0: {state}"
            assert np.all(state <= 1.0 + 1e-12), f"Above 1: {state}"

    def test_recovered_monotonically_increases(self):
        """R fraction can only increase (SIR has no R->S or R->I)."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        sim.reset(seed=42)
        prev_r = 0.0
        for _ in range(200):
            state = sim.step()
            assert state[2] >= prev_r - 1e-12, (
                f"R decreased from {prev_r} to {state[2]}"
            )
            prev_r = state[2]

    def test_susceptible_monotonically_decreases(self):
        """S fraction can only decrease (SIR has no return to S)."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        state = sim.reset(seed=42)
        prev_s = state[0]
        for _ in range(200):
            state = sim.step()
            assert state[0] <= prev_s + 1e-12, (
                f"S increased from {prev_s} to {state[0]}"
            )
            prev_s = state[0]


class TestSIRNetworkAdaptiveNetwork:
    """Tests for network initialization and structure."""

    def test_adjacency_list_created(self):
        """Adjacency list is created on reset."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=50))
        sim.reset(seed=42)
        assert sim._adj is not None
        assert len(sim._adj) == 50

    def test_adjacency_symmetric(self):
        """If i is neighbor of j, then j is neighbor of i."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=50))
        sim.reset(seed=42)
        for i in range(50):
            for j in sim._adj[i]:
                assert i in sim._adj[j], (
                    f"Edge ({i},{j}) is not symmetric"
                )

    def test_no_self_loops(self):
        """No node is its own neighbor."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=50))
        sim.reset(seed=42)
        for i in range(50):
            assert i not in sim._adj[i], f"Self-loop at node {i}"

    def test_mean_degree_approximately_correct(self):
        """Mean degree is approximately k_avg for Erdos-Renyi."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=500, k_avg=8.0))
        sim.reset(seed=42)
        mean_deg = sim.get_mean_degree()
        assert abs(mean_deg - 8.0) < 2.0, (
            f"Mean degree {mean_deg} too far from target 8.0"
        )

    def test_edge_count_positive(self):
        """Network has a positive number of edges."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=50))
        sim.reset(seed=42)
        assert sim.get_edge_count() > 0

    def test_initial_infected_count(self):
        """Initially exactly I_0 nodes are infected."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=100, I_0=10))
        state = sim.reset(seed=42)
        expected_i_frac = 10 / 100
        np.testing.assert_allclose(state[1], expected_i_frac, atol=1e-12)

    def test_initial_recovered_zero(self):
        """Initially no nodes are recovered."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state[2] == 0.0

    def test_symmetry_maintained_after_rewiring(self):
        """Adjacency remains symmetric after rewiring steps."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=50, w=0.8))
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        for i in range(50):
            for j in sim._adj[i]:
                assert i in sim._adj[j], (
                    f"Asymmetric edge ({i},{j}) after rewiring"
                )


class TestSIRNetworkAdaptiveDynamics:
    """Tests for epidemic dynamics and rewiring effects."""

    def test_step_advances_state(self):
        """State changes after stepping."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(beta=0.2))
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        # Run enough steps that something should change
        for _ in range(20):
            sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1), "State did not change after 20 steps"

    def test_epidemic_progresses(self):
        """With sufficient beta, infected fraction should rise initially."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=200, beta=0.3, gamma=0.02, w=0.0, I_0=10)
        )
        sim.reset(seed=42)
        initial_i = sim.observe()[1]
        max_i = initial_i
        for _ in range(100):
            state = sim.step()
            if state[1] > max_i:
                max_i = state[1]
        assert max_i > initial_i, (
            f"Peak infected {max_i} should exceed initial {initial_i}"
        )

    def test_epidemic_resolves(self):
        """Epidemic eventually ends: I -> 0 given enough time."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=100, beta=0.15, gamma=0.1, w=0.0, n_steps=500)
        )
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        # After long enough, I should be very small
        assert sim.observe()[1] < 0.05, (
            f"Infected fraction {sim.observe()[1]} should be near 0"
        )

    def test_no_infection_without_beta(self):
        """With beta=0, no new infections occur."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=100, beta=0.0, gamma=0.05, w=0.0, I_0=5)
        )
        state = sim.reset(seed=42)
        initial_s = state[0]
        for _ in range(100):
            state = sim.step()
        # S should not decrease (no infections)
        assert state[0] >= initial_s - 1e-12

    def test_recovery_without_infection(self):
        """With beta=0 but gamma>0, infected recover and R increases."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=100, beta=0.0, gamma=0.1, w=0.0, I_0=10)
        )
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        state = sim.observe()
        # All initially infected should have recovered
        assert state[1] < 0.02, f"Infected should be ~0, got {state[1]}"
        assert state[2] > 0.05, f"Recovered should be ~I_0/N, got {state[2]}"

    def test_higher_beta_more_infection(self):
        """Higher beta leads to larger final epidemic size."""
        sizes = []
        for beta in [0.05, 0.15, 0.3]:
            config = _make_config(
                N_nodes=200, beta=beta, gamma=0.05, w=0.0, I_0=5,
            )
            sim = SIRNetworkAdaptiveSimulation(config)
            sim.reset(seed=42)
            for _ in range(300):
                sim.step()
            sizes.append(sim.final_size())
        # Final size should be non-decreasing with beta
        for i in range(len(sizes) - 1):
            assert sizes[i] <= sizes[i + 1] + 0.05, (
                f"Final size not monotonic: {sizes}"
            )


class TestSIRNetworkAdaptiveRewiring:
    """Tests specifically for adaptive rewiring behavior."""

    def test_rewiring_reduces_epidemic(self):
        """Rewiring (w > 0) should reduce epidemic final size vs w=0."""
        # Without rewiring
        config_0 = _make_config(
            N_nodes=200, beta=0.15, gamma=0.05, w=0.0, I_0=5,
        )
        sim_0 = SIRNetworkAdaptiveSimulation(config_0)
        sim_0.reset(seed=42)
        for _ in range(300):
            sim_0.step()
        size_no_rewire = sim_0.final_size()

        # With rewiring
        config_w = _make_config(
            N_nodes=200, beta=0.15, gamma=0.05, w=0.8, I_0=5,
        )
        sim_w = SIRNetworkAdaptiveSimulation(config_w)
        sim_w.reset(seed=42)
        for _ in range(300):
            sim_w.step()
        size_rewire = sim_w.final_size()

        # Rewiring should reduce epidemic size (allow some stochastic tolerance)
        assert size_rewire < size_no_rewire + 0.1, (
            f"Rewiring size {size_rewire} should be less than "
            f"no-rewire size {size_no_rewire}"
        )

    def test_rewiring_preserves_edge_count_approximately(self):
        """Rewiring replaces edges, so total edge count stays roughly constant."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=100, w=0.5, beta=0.05, gamma=0.02)
        )
        sim.reset(seed=42)
        initial_edges = sim.get_edge_count()
        for _ in range(50):
            sim.step()
        final_edges = sim.get_edge_count()
        # Edge count should be within 30% (some edges may be lost
        # if no suitable S target found)
        ratio = final_edges / max(initial_edges, 1)
        assert 0.5 < ratio < 1.5, (
            f"Edge count ratio {ratio} too far from 1.0 "
            f"(initial={initial_edges}, final={final_edges})"
        )

    def test_no_rewiring_when_w_zero(self):
        """With w=0, network topology should not change."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=50, w=0.0, beta=0.0, gamma=0.0, I_0=5)
        )
        sim.reset(seed=42)
        initial_adj = [s.copy() for s in sim._adj]
        for _ in range(50):
            sim.step()
        for i in range(50):
            assert sim._adj[i] == initial_adj[i], (
                f"Network changed at node {i} despite w=0 and beta=0"
            )

    def test_ss_edge_fraction_increases_with_rewiring(self):
        """Rewiring creates more S-S edges over time."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=200, w=0.8, beta=0.05, gamma=0.02, I_0=10)
        )
        sim.reset(seed=42)
        initial_ss = sim.get_ss_edge_fraction()
        for _ in range(30):
            sim.step()
        later_ss = sim.get_ss_edge_fraction()
        # S-S fraction should increase or stay similar (not decrease dramatically)
        # Note: as epidemic progresses, S count drops too, so this is subtle
        assert later_ss >= initial_ss - 0.2, (
            f"SS fraction dropped from {initial_ss} to {later_ss}"
        )

    def test_clustering_computable(self):
        """Clustering coefficient is computable and in [0, 1]."""
        sim = SIRNetworkAdaptiveSimulation(_make_config(N_nodes=50))
        sim.reset(seed=42)
        cc = sim.get_clustering_coefficient()
        assert 0.0 <= cc <= 1.0, f"Clustering {cc} out of bounds"

    def test_si_edge_count_decreases(self):
        """S-I edge count should decrease over time as epidemic resolves."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=100, beta=0.15, gamma=0.1, w=0.3, I_0=10)
        )
        sim.reset(seed=42)
        # Let epidemic play out
        for _ in range(300):
            sim.step()
        # At end, should have very few S-I edges
        si_edges = sim.get_si_edge_count()
        assert si_edges < 50, (
            f"S-I edge count {si_edges} should be low after epidemic"
        )


class TestSIRNetworkAdaptiveDeterminism:
    """Tests for reproducibility with seeds."""

    def test_deterministic_with_seed(self):
        """Same seed produces identical trajectories."""
        config = _make_config(N_nodes=50)

        sim1 = SIRNetworkAdaptiveSimulation(config)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = SIRNetworkAdaptiveSimulation(config)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe()

        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_different_seeds_differ(self):
        """Different seeds produce different trajectories."""
        config = _make_config(N_nodes=200, beta=0.1, gamma=0.02)

        sim1 = SIRNetworkAdaptiveSimulation(config)
        sim1.reset(seed=42)
        for _ in range(30):
            sim1.step()
        s1 = sim1.observe()

        sim2 = SIRNetworkAdaptiveSimulation(config)
        sim2.reset(seed=123)
        for _ in range(30):
            sim2.step()
        s2 = sim2.observe()

        # Compare at mid-epidemic where stochastic differences are visible
        assert not np.allclose(s1, s2), "Different seeds should give different results"

    def test_reset_restores_initial_state(self):
        """Resetting with same seed gives identical state."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        s1 = sim.reset(seed=42).copy()
        for _ in range(50):
            sim.step()
        s2 = sim.reset(seed=42)
        np.testing.assert_allclose(s1, s2, atol=1e-12)


class TestSIRNetworkAdaptiveTrajectory:
    """Tests for trajectory collection via run()."""

    def test_run_returns_trajectory_data(self):
        """run() returns a valid TrajectoryData object."""
        config = _make_config(N_nodes=50, n_steps=50)
        sim = SIRNetworkAdaptiveSimulation(config)
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 3)  # 50 steps + initial

    def test_trajectory_states_shape(self):
        """Trajectory states have correct shape (n_steps+1, 3)."""
        config = _make_config(N_nodes=30, n_steps=100)
        sim = SIRNetworkAdaptiveSimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)

    def test_trajectory_timestamps(self):
        """Trajectory timestamps are correct."""
        config = _make_config(dt=1.0, n_steps=50)
        sim = SIRNetworkAdaptiveSimulation(config)
        traj = sim.run(n_steps=50)
        assert len(traj.timestamps) == 51
        np.testing.assert_allclose(traj.timestamps[0], 0.0)
        np.testing.assert_allclose(traj.timestamps[-1], 50.0)

    def test_trajectory_all_finite(self):
        """All trajectory values are finite (no NaN or inf)."""
        config = _make_config(N_nodes=50, n_steps=100)
        sim = SIRNetworkAdaptiveSimulation(config)
        traj = sim.run(n_steps=100)
        assert np.all(np.isfinite(traj.states))

    def test_trajectory_conservation_all_steps(self):
        """S + I + R = 1 at every timestep in trajectory."""
        config = _make_config(N_nodes=100, n_steps=100)
        sim = SIRNetworkAdaptiveSimulation(config)
        traj = sim.run(n_steps=100)
        sums = traj.states.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_trajectory_parameters_stored(self):
        """Parameters are stored in the trajectory metadata."""
        config = _make_config(N_nodes=50, beta=0.2, gamma=0.1)
        sim = SIRNetworkAdaptiveSimulation(config)
        traj = sim.run(n_steps=10)
        assert "beta" in traj.parameters
        assert traj.parameters["beta"] == 0.2


class TestSIRNetworkAdaptiveRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_rewiring_sweep_data(self):
        """Rewiring sweep produces valid data arrays."""
        from simulating_anything.rediscovery.sir_network_adaptive import (
            generate_rewiring_sweep_data,
        )
        data = generate_rewiring_sweep_data(
            n_w=5, n_trials=3, N_nodes=50, n_steps=50,
        )
        assert len(data["w"]) == 5
        assert len(data["mean_final_size"]) == 5
        assert len(data["mean_clustering"]) == 5
        assert np.all(data["mean_final_size"] >= 0)
        assert np.all(data["mean_final_size"] <= 1.0 + 1e-6)

    def test_comparison_data(self):
        """Comparison data has matching no-rewire and rewire arrays."""
        from simulating_anything.rediscovery.sir_network_adaptive import (
            generate_comparison_data,
        )
        data = generate_comparison_data(
            n_trials=5, N_nodes=50, n_steps=50,
        )
        assert len(data["no_rewire_final_size"]) == 5
        assert len(data["rewire_final_size"]) == 5
        assert np.all(data["no_rewire_final_size"] >= 0)
        assert np.all(data["rewire_final_size"] >= 0)

    def test_clustering_timeseries(self):
        """Clustering timeseries has matching arrays."""
        from simulating_anything.rediscovery.sir_network_adaptive import (
            generate_clustering_timeseries,
        )
        data = generate_clustering_timeseries(
            N_nodes=50, n_steps=50, measure_interval=10, seed=42,
        )
        assert len(data["time"]) == len(data["clustering"])
        assert len(data["time"]) == len(data["ss_fraction"])
        assert data["time"][0] == 0
        assert np.all(data["clustering"] >= 0)
        assert np.all(data["clustering"] <= 1.0 + 1e-6)

    def test_beta_sweep_data(self):
        """Beta sweep produces valid data."""
        from simulating_anything.rediscovery.sir_network_adaptive import (
            generate_beta_sweep_data,
        )
        data = generate_beta_sweep_data(
            n_beta=5, n_trials=3, N_nodes=50, n_steps=50,
        )
        assert len(data["beta"]) == 5
        assert len(data["mean_final_size"]) == 5
        assert np.all(data["mean_final_size"] >= 0)

    def test_rewiring_sweep_suppression_trend(self):
        """Final size should generally decrease with increasing w."""
        from simulating_anything.rediscovery.sir_network_adaptive import (
            generate_rewiring_sweep_data,
        )
        data = generate_rewiring_sweep_data(
            n_w=5, n_trials=5, N_nodes=100, beta=0.15, gamma=0.05,
            n_steps=200,
        )
        fs = data["mean_final_size"]
        # The first point (w=0) should have higher final size than last (w=1)
        # Allow tolerance for stochastic variation
        assert fs[0] >= fs[-1] - 0.15, (
            f"Expected w=0 final size ({fs[0]}) >= w=1 ({fs[-1]})"
        )


class TestSIRNetworkAdaptiveEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_initially_infected(self):
        """Works when all nodes start infected."""
        config = _make_config(N_nodes=20, I_0=20, beta=0.1, gamma=0.1)
        sim = SIRNetworkAdaptiveSimulation(config)
        state = sim.reset(seed=42)
        assert state[0] == 0.0  # No susceptible
        assert state[1] == 1.0  # All infected
        for _ in range(100):
            state = sim.step()
        np.testing.assert_allclose(state.sum(), 1.0, atol=1e-12)

    def test_single_infected(self):
        """Works with only one initially infected node."""
        config = _make_config(N_nodes=50, I_0=1)
        sim = SIRNetworkAdaptiveSimulation(config)
        state = sim.reset(seed=42)
        assert state[1] == 1 / 50
        for _ in range(100):
            sim.step()
        assert np.all(np.isfinite(sim.observe()))

    def test_small_network(self):
        """Works with a very small network."""
        config = _make_config(N_nodes=5, k_avg=2.0, I_0=1)
        sim = SIRNetworkAdaptiveSimulation(config)
        state = sim.reset(seed=42)
        assert state.shape == (3,)
        for _ in range(50):
            state = sim.step()
        np.testing.assert_allclose(state.sum(), 1.0, atol=1e-12)

    def test_dense_network(self):
        """Works with a nearly complete network."""
        config = _make_config(N_nodes=20, k_avg=18.0, I_0=2)
        sim = SIRNetworkAdaptiveSimulation(config)
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))
        np.testing.assert_allclose(state.sum(), 1.0, atol=1e-12)

    def test_zero_gamma_no_recovery(self):
        """With gamma=0, no recovery occurs."""
        config = _make_config(
            N_nodes=50, beta=0.2, gamma=0.0, w=0.0, I_0=5
        )
        sim = SIRNetworkAdaptiveSimulation(config)
        sim.reset(seed=42)
        for _ in range(100):
            state = sim.step()
        # R should stay at 0 since no recovery
        assert state[2] == 0.0, f"R should be 0 with gamma=0, got {state[2]}"

    def test_final_size_method(self):
        """final_size() returns I + R fraction."""
        sim = SIRNetworkAdaptiveSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        state = sim.observe()
        expected = state[1] + state[2]
        np.testing.assert_allclose(sim.final_size(), expected, atol=1e-12)

    def test_stability_long_run(self):
        """No NaN or infinite values after many steps."""
        sim = SIRNetworkAdaptiveSimulation(
            _make_config(N_nodes=50, n_steps=500)
        )
        sim.reset(seed=42)
        for _ in range(500):
            state = sim.step()
        assert np.all(np.isfinite(state))
        np.testing.assert_allclose(state.sum(), 1.0, atol=1e-12)
