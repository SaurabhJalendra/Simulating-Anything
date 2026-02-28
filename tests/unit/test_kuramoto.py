"""Tests for the Kuramoto coupled oscillators simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.kuramoto import KuramotoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(K: float = 1.0, N: int = 50, dt: float = 0.01) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.KURAMOTO,
        dt=dt,
        n_steps=1000,
        parameters={"N": float(N), "K": K, "omega_std": 1.0},
    )


class TestKuramotoSimulation:
    def test_initial_state_shape(self):
        sim = KuramotoSimulation(_make_config(N=50))
        state = sim.reset(seed=42)
        assert state.shape == (50,)

    def test_step_advances(self):
        sim = KuramotoSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_phases_stay_finite(self):
        sim = KuramotoSimulation(_make_config(K=3.0))
        sim.reset(seed=42)
        for _ in range(1000):
            sim.step()
        assert np.all(np.isfinite(sim.observe()))

    def test_order_parameter_range(self):
        sim = KuramotoSimulation(_make_config(K=2.0))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        r, psi = sim.order_parameter
        assert 0 <= r <= 1.0 + 1e-10
        assert -np.pi <= psi <= np.pi + 1e-10

    def test_zero_coupling_desynchronized(self):
        """With K=0, oscillators should be desynchronized (r ~ 1/sqrt(N))."""
        sim = KuramotoSimulation(_make_config(K=0.0, N=200))
        sim.reset(seed=42)
        for _ in range(2000):
            sim.step()
        r = sim.order_parameter_r
        # For random phases, r ~ 1/sqrt(N) ~ 0.07, should be small
        assert r < 0.3, f"r={r} too high for K=0"

    def test_strong_coupling_synchronized(self):
        """With large K, oscillators should synchronize (r ~ 1)."""
        sim = KuramotoSimulation(_make_config(K=5.0, N=100))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
        r = sim.order_parameter_r
        assert r > 0.7, f"r={r} too low for K=5.0"

    def test_critical_coupling_formula(self):
        """K_c = 4*omega_std/pi for uniform distribution."""
        sim = KuramotoSimulation(_make_config())
        K_c = sim.critical_coupling
        # omega_std=1.0, uniform: K_c = 4/pi ~ 1.273
        np.testing.assert_allclose(K_c, 4 / np.pi, atol=0.01)

    def test_different_N_sizes(self):
        """Should work for different system sizes."""
        for N in [10, 50, 100]:
            sim = KuramotoSimulation(_make_config(N=N))
            state = sim.reset(seed=42)
            assert state.shape == (N,)
            sim.step()
            assert sim.observe().shape == (N,)

    def test_natural_frequencies_generated(self):
        sim = KuramotoSimulation(_make_config(N=100))
        sim.reset(seed=42)
        assert hasattr(sim, "omega")
        assert len(sim.omega) == 100
        # For uniform on [-1, 1], should be in range
        assert np.all(sim.omega >= -1.0 - 1e-10)
        assert np.all(sim.omega <= 1.0 + 1e-10)

    def test_measure_steady_state_r(self):
        sim = KuramotoSimulation(_make_config(K=3.0, N=50))
        r = sim.measure_steady_state_r(
            n_transient_steps=1000,
            n_measure_steps=500,
            seed=42,
        )
        assert 0 <= r <= 1.0 + 1e-10


class TestKuramotoRediscovery:
    def test_sync_transition_data(self):
        from simulating_anything.rediscovery.kuramoto import generate_sync_transition_data
        data = generate_sync_transition_data(
            n_K=5, N=30, n_trials=2, n_transient=500, n_measure=200, dt=0.05,
        )
        assert len(data["K"]) == 5
        assert len(data["r_mean"]) == 5
        assert len(data["r_std"]) == 5
        assert np.all(data["r_mean"] >= 0)
        assert np.all(data["r_mean"] <= 1.0 + 0.1)

    def test_transition_occurs(self):
        """r should increase with K."""
        from simulating_anything.rediscovery.kuramoto import generate_sync_transition_data
        data = generate_sync_transition_data(
            n_K=10, N=50, n_trials=3, n_transient=2000, n_measure=1000, dt=0.02,
        )
        # r at highest K should be larger than at lowest K
        assert data["r_mean"][-1] > data["r_mean"][0], (
            f"No transition: r_low={data['r_mean'][0]}, r_high={data['r_mean'][-1]}"
        )

    def test_finite_size_data(self):
        from simulating_anything.rediscovery.kuramoto import generate_finite_size_data
        data = generate_finite_size_data(
            N_values=[10, 30], K=3.0, n_trials=2, dt=0.02,
        )
        assert len(data["N"]) == 2
        assert len(data["r"]) == 2
