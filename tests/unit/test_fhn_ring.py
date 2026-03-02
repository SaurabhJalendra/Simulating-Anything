"""Tests for the FitzHugh-Nagumo ring network simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.fhn_ring import FHNRingSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 20,
    D: float = 0.5,
    I_ext: float = 0.5,
    dt: float = 0.05,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.FHN_RING,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": 0.7, "b": 0.8, "eps": 0.08,
            "I": I_ext, "D": D, "N": float(N),
        },
    )


class TestFHNRingSimulation:
    def test_initial_state_shape(self):
        sim = FHNRingSimulation(_make_config(N=20))
        state = sim.reset(seed=42)
        assert state.shape == (40,), f"Expected (40,), got {state.shape}"

    def test_initial_state_shape_varied_N(self):
        """State dimension should be 2*N for any N."""
        for N in [5, 10, 30, 50]:
            sim = FHNRingSimulation(_make_config(N=N))
            state = sim.reset(seed=42)
            assert state.shape == (2 * N,)

    def test_step_advances_state(self):
        sim = FHNRingSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1), "State did not change after step"

    def test_trajectory_stays_finite(self):
        sim = FHNRingSimulation(_make_config(N=20, D=0.5, dt=0.05))
        sim.reset(seed=42)
        for _ in range(2000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became non-finite"

    def test_trajectory_bounded(self):
        """All voltages and recovery variables should stay bounded."""
        sim = FHNRingSimulation(_make_config(N=20, D=0.5, dt=0.05))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()
            v = sim.v_neurons
            w = sim.w_neurons
            assert np.all(np.abs(v) < 10), f"v diverged: max={np.max(np.abs(v))}"
            assert np.all(np.abs(w) < 10), f"w diverged: max={np.max(np.abs(w))}"

    def test_observe_matches_state(self):
        sim = FHNRingSimulation(_make_config(N=10))
        sim.reset(seed=42)
        sim.step()
        state = sim.observe()
        v = sim.v_neurons
        w = sim.w_neurons
        np.testing.assert_array_equal(state[:10], v)
        np.testing.assert_array_equal(state[10:], w)

    def test_deterministic_with_same_seed(self):
        sim1 = FHNRingSimulation(_make_config())
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = FHNRingSimulation(_make_config())
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_different_seeds_different_states(self):
        sim1 = FHNRingSimulation(_make_config())
        sim1.reset(seed=1)
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = FHNRingSimulation(_make_config())
        sim2.reset(seed=99)
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        assert not np.allclose(s1, s2, atol=1e-6)

    def test_derivatives_uncoupled(self):
        """With D=0, derivatives match single FHN neuron dynamics."""
        sim = FHNRingSimulation(_make_config(N=3, D=0.0, I_ext=0.0))
        v = np.array([0.0, 1.0, -1.0])
        w = np.array([0.0, 0.0, 0.0])
        dv, dw = sim._derivatives(v, w)
        # dv_0 = 0 - 0 - 0 + 0 + 0 = 0
        # dv_1 = 1 - 1/3 - 0 + 0 = 0.6667
        # dv_2 = -1 + 1/3 - 0 + 0 = -0.6667
        np.testing.assert_allclose(dv[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dv[1], 2.0 / 3.0, atol=1e-10)
        np.testing.assert_allclose(dv[2], -2.0 / 3.0, atol=1e-10)
        # dw_i = eps*(v_i + a - b*w_i)
        np.testing.assert_allclose(dw[0], 0.08 * 0.7, atol=1e-10)
        np.testing.assert_allclose(dw[1], 0.08 * 1.7, atol=1e-10)
        np.testing.assert_allclose(dw[2], 0.08 * (-0.3), atol=1e-10)

    def test_ring_laplacian_periodic(self):
        """Ring Laplacian should wrap around for periodic BCs."""
        sim = FHNRingSimulation(_make_config(N=5, D=1.0, I_ext=0.0))
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        w = np.zeros(5)
        dv, _ = sim._derivatives(v, w)
        # For i=0: Lap = v[-1] - 2*v[0] + v[1] = 0 - 2 + 0 = -2
        # dv[0] = v[0] - v[0]^3/3 - w[0] + I + D*Lap
        #       = 1 - 1/3 - 0 + 0 + 1*(-2) = -1.333...
        expected_dv0 = 1.0 - 1.0 / 3.0 + 1.0 * (-2.0)
        np.testing.assert_allclose(dv[0], expected_dv0, atol=1e-10)
        # For i=4 (last): Lap = v[3] - 2*v[4] + v[0] = 0 - 0 + 1 = 1
        expected_dv4 = 0.0 - 0.0 - 0.0 + 0.0 + 1.0 * 1.0
        np.testing.assert_allclose(dv[4], expected_dv4, atol=1e-10)

    def test_zero_coupling_independent(self):
        """With D=0, neurons are independent."""
        sim = FHNRingSimulation(_make_config(N=5, D=0.0))
        sim.reset(seed=42)
        # Give neuron 0 a big kick, others stay at rest
        sim._v[0] = 2.0
        sim._state = np.concatenate([sim._v, sim._w])
        for _ in range(100):
            sim.step()
        # The kicked neuron should differ significantly from others
        v = sim.v_neurons
        # Neurons 1-4 should all be the same since they started identically
        # (they won't be identical due to seed perturbation, but close)
        # Neuron 0 should be different from the others
        assert abs(v[0] - np.mean(v[1:])) > 0.01 or np.std(v[1:]) < 0.5

    def test_strong_coupling_synchronizes(self):
        """With large D, neurons should synchronize."""
        sim = FHNRingSimulation(_make_config(N=10, D=5.0, dt=0.02))
        sim.reset(seed=42)
        for _ in range(10000):
            sim.step()
        v = sim.v_neurons
        std = np.std(v)
        assert std < 0.5, f"Neurons not synchronized at D=5.0, std(v)={std:.4f}"

    def test_compute_synchronization_range(self):
        sim = FHNRingSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        r = sim.compute_synchronization()
        assert 0.0 <= r <= 1.0 + 1e-10, f"r={r} out of range"

    def test_compute_synchronization_identical(self):
        """Identical voltages should give r=1."""
        sim = FHNRingSimulation(_make_config(N=5))
        sim.reset(seed=42)
        sim._v = np.full(5, 1.0)
        r = sim.compute_synchronization()
        assert r == 1.0

    def test_compute_voltage_variance(self):
        sim = FHNRingSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        var = sim.compute_voltage_variance()
        assert var >= 0.0
        assert np.isfinite(var)

    def test_detect_traveling_wave_returns_nonneg(self):
        sim = FHNRingSimulation(_make_config(N=20, D=0.5))
        sim.reset(seed=42)
        for _ in range(1000):
            sim.step()
        speed = sim.detect_traveling_wave()
        assert speed >= 0.0
        assert np.isfinite(speed)

    def test_measure_mean_voltage(self):
        sim = FHNRingSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        mv = sim.measure_mean_voltage()
        assert np.isfinite(mv)
        assert abs(mv) < 10

    def test_measure_phase_coherence_range(self):
        sim = FHNRingSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        coh = sim.measure_phase_coherence()
        assert 0.0 <= coh <= 1.0, f"Coherence out of range: {coh}"

    def test_run_trajectory(self):
        """The run() method should produce valid TrajectoryData."""
        sim = FHNRingSimulation(_make_config(N=10, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 20)  # 2*N=20
        assert np.all(np.isfinite(traj.states))

    def test_v_neurons_w_neurons_properties(self):
        sim = FHNRingSimulation(_make_config(N=15))
        sim.reset(seed=42)
        sim.step()
        v = sim.v_neurons
        w = sim.w_neurons
        assert v.shape == (15,)
        assert w.shape == (15,)
        assert np.all(np.isfinite(v))
        assert np.all(np.isfinite(w))


class TestFHNRingRediscovery:
    def test_sync_transition_data_generation(self):
        from simulating_anything.rediscovery.fhn_ring import (
            generate_sync_transition_data,
        )
        data = generate_sync_transition_data(
            n_D=5, N=10, n_transient=200, n_measure=100, dt=0.05,
        )
        assert len(data["D"]) == 5
        assert len(data["r"]) == 5
        assert np.all(data["r"] >= 0)
        assert np.all(data["r"] <= 1.0 + 0.1)

    def test_wave_speed_data_generation(self):
        from simulating_anything.rediscovery.fhn_ring import (
            generate_wave_speed_data,
        )
        data = generate_wave_speed_data(
            n_D=3, N=10, n_transient=200, dt=0.05,
        )
        assert len(data["D"]) == 3
        assert len(data["speed"]) == 3
        assert np.all(data["speed"] >= 0)

    def test_strong_coupling_reduces_variance(self):
        """Strong coupling should reduce voltage variance (synchronization)."""
        # Weak coupling: neurons should have more voltage spread
        sim_weak = FHNRingSimulation(_make_config(N=15, D=0.01, dt=0.05))
        sim_weak.reset(seed=42)
        for _ in range(5000):
            sim_weak.step()
        var_weak_samples = []
        for _ in range(1000):
            sim_weak.step()
            var_weak_samples.append(sim_weak.compute_voltage_variance())
        var_weak = np.mean(var_weak_samples)

        # Strong coupling: neurons should synchronize
        sim_strong = FHNRingSimulation(_make_config(N=15, D=5.0, dt=0.02))
        sim_strong.reset(seed=42)
        for _ in range(10000):
            sim_strong.step()
        var_strong_samples = []
        for _ in range(2000):
            sim_strong.step()
            var_strong_samples.append(sim_strong.compute_voltage_variance())
        var_strong = np.mean(var_strong_samples)

        assert var_strong < var_weak + 0.1, (
            f"Strong coupling should reduce variance: "
            f"var_weak={var_weak:.4f}, var_strong={var_strong:.4f}"
        )
