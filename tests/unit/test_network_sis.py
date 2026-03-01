"""Tests for the Network SIS epidemic simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.network_sis import (
    NetworkSISSimulation,
    _generate_regular_lattice,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 50,
    beta: float = 0.3,
    gamma: float = 0.1,
    mean_degree: float = 6.0,
    dt: float = 0.01,
    initial_fraction: float = 0.1,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.NETWORK_SIS,
        dt=dt,
        n_steps=1000,
        parameters={
            "N": float(N),
            "beta": beta,
            "gamma": gamma,
            "mean_degree": mean_degree,
            "initial_fraction": initial_fraction,
        },
    )


class TestNetworkSISSimulation:
    def test_reset_shape(self):
        """State vector has N elements after reset."""
        sim = NetworkSISSimulation(_make_config(N=50))
        state = sim.reset(seed=42)
        assert state.shape == (50,)

    def test_observe_shape(self):
        """Observe returns N-element array."""
        sim = NetworkSISSimulation(_make_config(N=30))
        sim.reset(seed=42)
        obs = sim.observe()
        assert obs.shape == (30,)

    def test_step_advances(self):
        """State should change after a step."""
        sim = NetworkSISSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same seed produces identical trajectories."""
        sim1 = NetworkSISSimulation(_make_config())
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = NetworkSISSimulation(_make_config())
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe()

        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_stability_no_nan(self):
        """No NaN after many steps."""
        sim = NetworkSISSimulation(_make_config(N=30, dt=0.01))
        sim.reset(seed=42)
        for _ in range(10000):
            sim.step()
        assert np.all(np.isfinite(sim.observe()))

    def test_probabilities_bounded(self):
        """All p_i must remain in [0, 1] throughout simulation."""
        sim = NetworkSISSimulation(_make_config(beta=0.5, gamma=0.05))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= -1e-12), f"Negative probability: {state.min()}"
            assert np.all(state <= 1.0 + 1e-12), f"Probability > 1: {state.max()}"

    def test_disease_free_below_threshold(self):
        """Below epidemic threshold, infection should die out."""
        config = _make_config(N=50, beta=0.001, gamma=0.5, mean_degree=6.0)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        # Verify we are below threshold: tau = beta/gamma = 0.002 << tau_c
        tau = sim.beta / sim.gamma
        tau_c = sim.compute_epidemic_threshold()
        assert tau < tau_c, f"tau={tau} should be < tau_c={tau_c}"
        for _ in range(20000):
            sim.step()
        prevalence = sim.compute_prevalence()
        assert prevalence < 0.05, f"Prevalence {prevalence} should be near 0 below threshold"

    def test_endemic_above_threshold(self):
        """Above epidemic threshold, steady-state prevalence should be > 0."""
        config = _make_config(N=50, beta=0.5, gamma=0.05, mean_degree=8.0)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        tau = sim.beta / sim.gamma
        tau_c = sim.compute_epidemic_threshold()
        assert tau > tau_c, f"tau={tau} should be > tau_c={tau_c}"
        for _ in range(15000):
            sim.step()
        prevalence = sim.compute_prevalence()
        assert prevalence > 0.1, (
            f"Prevalence {prevalence} should be significant above threshold"
        )

    def test_epidemic_threshold_complete_graph(self):
        """For a complete graph, tau_c = 1/(N-1)."""
        N = 30
        config = _make_config(N=N, mean_degree=float(N - 1))
        sim = NetworkSISSimulation(config)
        sim.network_type = "complete"
        sim.reset(seed=42)
        tau_c = sim.compute_epidemic_threshold()
        expected = 1.0 / (N - 1)
        np.testing.assert_allclose(tau_c, expected, rtol=0.01)

    def test_epidemic_threshold_er_graph(self):
        """For Erdos-Renyi, tau_c ~ 1/<k> (approximately)."""
        N = 200
        mean_degree = 10.0
        config = _make_config(N=N, mean_degree=mean_degree)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        tau_c = sim.compute_epidemic_threshold()
        # For large ER graphs, lambda_max ~ <k>, so tau_c ~ 1/<k>
        expected_approx = 1.0 / mean_degree
        # Allow generous tolerance due to finite size effects
        assert tau_c < expected_approx * 2.0, (
            f"tau_c={tau_c} should be near 1/<k>={expected_approx}"
        )
        assert tau_c > expected_approx * 0.3, (
            f"tau_c={tau_c} should be near 1/<k>={expected_approx}"
        )

    def test_network_generation_shape_and_symmetry(self):
        """Adjacency matrix should be NxN, symmetric, with no self-loops."""
        N = 40
        config = _make_config(N=N)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        A = sim._adjacency
        assert A.shape == (N, N)
        np.testing.assert_allclose(A, A.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(A), 0.0)

    def test_adjacency_symmetric(self):
        """Adjacency matrix must be symmetric for all network types."""
        for net_type in ["erdos_renyi", "regular", "complete"]:
            config = _make_config(N=30)
            sim = NetworkSISSimulation(config)
            sim.network_type = net_type
            sim.reset(seed=42)
            A = sim._adjacency
            np.testing.assert_allclose(A, A.T, atol=1e-12, err_msg=net_type)

    def test_spectral_radius_positive(self):
        """Spectral radius of a connected graph must be positive."""
        config = _make_config(N=30, mean_degree=6.0)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        lam = sim.spectral_radius()
        assert lam > 0, f"Spectral radius should be positive, got {lam}"

    def test_prevalence_increases_with_beta(self):
        """Higher beta should lead to higher prevalence (monotonic trend)."""
        config = _make_config(N=50, mean_degree=8.0)
        sim = NetworkSISSimulation(config)

        prev_values = []
        beta_values = [0.05, 0.2, 0.5, 1.0]
        for beta in beta_values:
            sim.beta = beta
            sim.gamma = 0.1
            sim.reset(seed=42)
            for _ in range(10000):
                sim.step()
            prev_vals = []
            for _ in range(2000):
                sim.step()
                prev_vals.append(sim.compute_prevalence())
            prev_values.append(float(np.mean(prev_vals)))

        # Prevalence should be non-decreasing (allow small tolerance)
        for i in range(len(prev_values) - 1):
            assert prev_values[i] <= prev_values[i + 1] + 0.05, (
                f"Prevalence not monotonic: {prev_values}"
            )

    def test_infection_sweep(self):
        """Infection sweep should produce valid data."""
        config = _make_config(N=30, mean_degree=6.0)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        beta_vals = np.linspace(0.01, 0.5, 5)
        sweep = sim.infection_sweep(
            beta_vals, n_transient=1000, n_measure=500, seed=42,
        )
        assert len(sweep["beta"]) == 5
        assert len(sweep["prevalence"]) == 5
        assert len(sweep["tau"]) == 5
        assert np.all(sweep["prevalence"] >= 0)
        assert np.all(sweep["prevalence"] <= 1.0 + 1e-6)

    def test_higher_degree_lower_threshold(self):
        """Denser network (higher mean degree) should have lower epidemic threshold."""
        thresholds = []
        for deg in [4.0, 8.0, 16.0]:
            config = _make_config(N=100, mean_degree=deg)
            sim = NetworkSISSimulation(config)
            sim.reset(seed=42)
            thresholds.append(sim.compute_epidemic_threshold())

        # Higher degree -> higher lambda_max -> lower threshold
        for i in range(len(thresholds) - 1):
            assert thresholds[i] > thresholds[i + 1] - 0.01, (
                f"Threshold not decreasing with degree: {thresholds}"
            )

    def test_degree_distribution(self):
        """Degree distribution should be reasonable for ER graph."""
        config = _make_config(N=100, mean_degree=10.0)
        sim = NetworkSISSimulation(config)
        sim.reset(seed=42)
        degrees = sim.degree_distribution()
        assert len(degrees) == 100
        assert np.all(degrees >= 0)
        # Mean degree should be close to target
        actual_mean = float(np.mean(degrees))
        assert abs(actual_mean - 10.0) < 3.0, (
            f"Mean degree {actual_mean} too far from 10.0"
        )

    def test_regular_lattice_degree(self):
        """Regular lattice should give uniform degree."""
        adj = _generate_regular_lattice(30, 6)
        degrees = np.sum(adj, axis=1)
        np.testing.assert_allclose(degrees, 6.0)

    def test_complete_graph_degree(self):
        """Complete graph has degree N-1 for all nodes."""
        config = _make_config(N=20)
        sim = NetworkSISSimulation(config)
        sim.network_type = "complete"
        sim.reset(seed=42)
        degrees = sim.degree_distribution()
        np.testing.assert_allclose(degrees, 19.0)

    def test_run_trajectory(self):
        """Run method should produce a valid TrajectoryData."""
        config = _make_config(N=20, dt=0.01)
        config = config.model_copy(update={"n_steps": 100})
        sim = NetworkSISSimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 20)
        assert np.all(np.isfinite(traj.states))


class TestNetworkSISRediscovery:
    def test_threshold_sweep_data(self):
        """Threshold sweep should produce valid data arrays."""
        from simulating_anything.rediscovery.network_sis import (
            generate_threshold_sweep_data,
        )
        data = generate_threshold_sweep_data(
            N=20, mean_degree=4.0, n_beta=5,
            n_transient=500, n_measure=200, dt=0.05,
        )
        assert len(data["tau"]) == 5
        assert len(data["prevalence"]) == 5
        assert data["tau_c_theory"] > 0
        assert data["lambda_max"] > 0
        assert np.all(data["prevalence"] >= 0)

    def test_topology_comparison_data(self):
        """Topology comparison should have data for all three network types."""
        from simulating_anything.rediscovery.network_sis import (
            generate_topology_comparison_data,
        )
        data = generate_topology_comparison_data(
            N=20, mean_degree=4.0, n_beta=3,
            n_transient=500, n_measure=200, dt=0.05,
        )
        assert "erdos_renyi" in data
        assert "regular" in data
        assert "complete" in data
        for net_type in data:
            assert data[net_type]["tau_c"] > 0
            assert data[net_type]["lambda_max"] > 0
