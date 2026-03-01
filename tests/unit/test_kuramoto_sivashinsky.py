"""Tests for the Kuramoto-Sivashinsky equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.kuramoto_sivashinsky import KuramotoSivashinsky
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    L: float = 32.0 * np.pi,
    N: int = 128,
    dt: float = 0.05,
    n_steps: int = 100,
    viscosity: float = 1.0,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.KURAMOTO_SIVASHINSKY,
        dt=dt,
        n_steps=n_steps,
        parameters={"L": L, "N": float(N), "viscosity": viscosity},
        seed=seed,
    )


class TestKuramotoSivashinskyCreation:
    def test_default_parameters(self):
        sim = KuramotoSivashinsky(_make_config())
        assert sim.L == pytest.approx(32.0 * np.pi)
        assert sim.N == 128
        assert sim.viscosity == 1.0

    def test_custom_parameters(self):
        sim = KuramotoSivashinsky(_make_config(L=20.0, N=64, viscosity=2.0))
        assert sim.L == pytest.approx(20.0)
        assert sim.N == 64
        assert sim.viscosity == pytest.approx(2.0)

    def test_grid_spacing(self):
        sim = KuramotoSivashinsky(_make_config(L=10.0, N=50))
        assert sim.dx == pytest.approx(10.0 / 50)

    def test_grid_points(self):
        sim = KuramotoSivashinsky(_make_config(N=64))
        assert len(sim.x) == 64


class TestKuramotoSivashinskyState:
    def test_initial_state_shape(self):
        sim = KuramotoSivashinsky(_make_config(N=128))
        state = sim.reset()
        assert state.shape == (128,)

    def test_initial_state_shape_custom_N(self):
        sim = KuramotoSivashinsky(_make_config(N=256))
        state = sim.reset()
        assert state.shape == (256,)

    def test_step_advances_state(self):
        sim = KuramotoSivashinsky(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = KuramotoSivashinsky(_make_config())
        state = sim.reset()
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)

    def test_random_init(self):
        sim = KuramotoSivashinsky(_make_config())
        sim.init_type = "random"
        state = sim.reset()
        # Random init should not be all zeros
        assert np.max(np.abs(state)) > 0

    def test_sine_init(self):
        sim = KuramotoSivashinsky(_make_config())
        sim.init_type = "sine"
        state = sim.reset()
        assert np.max(np.abs(state)) > 0


class TestKuramotoSivashinskyPhysics:
    def test_energy_bounded_no_blowup(self):
        """Energy should remain bounded for well-resolved chaotic regime."""
        sim = KuramotoSivashinsky(_make_config(
            L=32.0 * np.pi, N=128, dt=0.05, n_steps=500,
        ))
        sim.reset()
        for _ in range(500):
            sim.step()
            assert np.isfinite(sim.energy), "Energy blew up"
            assert sim.energy < 1e6, f"Energy too large: {sim.energy}"

    def test_small_L_decays_to_zero(self):
        """For L < 2*pi*sqrt(2), trivial solution u=0 is stable."""
        L_small = 2 * np.pi * 0.9  # Below critical
        sim = KuramotoSivashinsky(_make_config(
            L=L_small, N=64, dt=0.01, n_steps=2000,
        ))
        sim.init_type = "sine"
        sim.reset()
        for _ in range(2000):
            sim.step()
        # Should decay to near zero
        assert sim.energy < 1e-6, f"Energy did not decay: {sim.energy}"

    def test_spatial_mean_conserved(self):
        """KS equation conserves the spatial mean of u."""
        sim = KuramotoSivashinsky(_make_config(
            L=32.0 * np.pi, N=128, dt=0.05,
        ))
        sim.reset()
        mean_0 = sim.spatial_mean
        for _ in range(200):
            sim.step()
        mean_f = sim.spatial_mean
        # Spatial mean should be conserved to high precision
        np.testing.assert_allclose(mean_0, mean_f, atol=1e-8)

    def test_large_L_develops_chaos(self):
        """For large L, the system should develop spatiotemporal chaos."""
        sim = KuramotoSivashinsky(_make_config(
            L=32.0 * np.pi, N=128, dt=0.05,
        ))
        sim.reset()
        # Run past transient
        for _ in range(500):
            sim.step()
        # Energy should be non-trivial (not zero)
        assert sim.energy > 1e-6, "System did not develop chaos"

    def test_trajectory_reproducibility(self):
        """Same seed should give identical trajectories."""
        config = _make_config(seed=42)
        sim1 = KuramotoSivashinsky(config)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = KuramotoSivashinsky(config)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_different_seeds_differ(self):
        """Different seeds should give different trajectories."""
        sim1 = KuramotoSivashinsky(_make_config(seed=42))
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()

        sim2 = KuramotoSivashinsky(_make_config(seed=99))
        sim2.reset(seed=99)
        for _ in range(100):
            sim2.step()

        assert not np.allclose(sim1.observe(), sim2.observe())

    def test_run_trajectory(self):
        """The run() method should produce a valid TrajectoryData."""
        sim = KuramotoSivashinsky(_make_config(n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 128)  # 50 steps + initial
        assert len(traj.timestamps) == 51
        assert np.all(np.isfinite(traj.states))


class TestKuramotoSivashinskyProperties:
    def test_energy_property(self):
        sim = KuramotoSivashinsky(_make_config())
        sim.reset()
        energy = sim.energy
        assert isinstance(energy, float)
        assert energy >= 0

    def test_spatial_mean_property(self):
        sim = KuramotoSivashinsky(_make_config())
        sim.reset()
        mean = sim.spatial_mean
        assert isinstance(mean, float)

    def test_max_amplitude_property(self):
        sim = KuramotoSivashinsky(_make_config())
        sim.reset()
        amp = sim.max_amplitude
        assert isinstance(amp, float)
        assert amp >= 0

    def test_energy_spectrum(self):
        sim = KuramotoSivashinsky(_make_config(N=128))
        sim.reset()
        k_vals, spectrum = sim.energy_spectrum()
        assert len(k_vals) == 64  # N/2
        assert len(spectrum) == 64
        assert np.all(spectrum >= 0)

    def test_correlation_length(self):
        sim = KuramotoSivashinsky(_make_config(
            L=32.0 * np.pi, N=128, dt=0.05,
        ))
        sim.reset()
        # Run to develop structure
        for _ in range(300):
            sim.step()
        cl = sim.correlation_length()
        assert isinstance(cl, float)
        assert cl > 0
        assert cl <= sim.L / 2


class TestKuramotoSivashinskyRediscovery:
    def test_energy_evolution_data(self):
        from simulating_anything.rediscovery.kuramoto_sivashinsky import (
            generate_energy_evolution_data,
        )
        data = generate_energy_evolution_data(
            L=20 * np.pi, N=64, n_steps=100, dt=0.05,
        )
        assert len(data["time"]) == 101
        assert len(data["energy"]) == 101
        assert len(data["spatial_mean"]) == 101
        assert np.all(np.isfinite(data["energy"]))

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.kuramoto_sivashinsky import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(
            L=32 * np.pi, N=64, n_steps=800, dt=0.05, n_trials=2,
        )
        assert len(data["lyapunov_estimates"]) == 2
        assert data["lyapunov_mean"] > 0  # Should be chaotic

    def test_spatial_correlation_data(self):
        from simulating_anything.rediscovery.kuramoto_sivashinsky import (
            generate_spatial_correlation_data,
        )
        data = generate_spatial_correlation_data(
            L_values=np.array([20 * np.pi, 32 * np.pi]),
            N=64,
            n_steps=2200,
            dt=0.05,
        )
        assert len(data["L"]) >= 1
        assert len(data["correlation_length"]) >= 1
        assert np.all(data["correlation_length"] > 0)
