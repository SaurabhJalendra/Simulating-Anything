"""Tests for the FitzHugh-Nagumo 2D lattice simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.fhn_lattice import FHNLattice
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(
    a: float = 0.7,
    b: float = 0.8,
    eps: float = 0.08,
    I_ext: float = 0.5,
    D: float = 1.0,
    N: int = 8,
    dt: float = 0.02,
    n_steps: int = 50,
) -> FHNLattice:
    """Helper to create an FHNLattice simulation with sensible test defaults."""
    config = SimulationConfig(
        domain=Domain.FHN_SPATIAL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "eps": eps,
            "I": I_ext,
            "D": D,
            "N": float(N),
        },
    )
    return FHNLattice(config)


class TestFHNLatticeCreation:
    def test_create_simulation(self):
        sim = _make_sim()
        assert sim.N == 8
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.I_ext == 0.5
        assert sim.D == 1.0

    def test_custom_parameters(self):
        sim = _make_sim(a=0.5, eps=0.1, D=2.0, I_ext=0.3, N=16, dt=0.01)
        assert sim.a == 0.5
        assert sim.eps == 0.1
        assert sim.D == 2.0
        assert sim.I_ext == 0.3
        assert sim.N == 16

    def test_default_parameters_from_empty_dict(self):
        config = SimulationConfig(
            domain=Domain.FHN_SPATIAL,
            dt=0.02,
            n_steps=10,
            parameters={},
        )
        sim = FHNLattice(config)
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.I_ext == 0.5
        assert sim.D == 1.0
        assert sim.N == 32

    def test_cfl_violation_raises(self):
        """dt exceeding CFL limit for forward Euler should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            _make_sim(D=5.0, dt=1.0)


class TestFHNLatticeState:
    def test_initial_state_shape(self):
        sim = _make_sim(N=8)
        state = sim.reset()
        assert state.shape == (2 * 8 * 8,)  # 2 * N * N = 128

    def test_state_dim(self):
        """State dimension should be exactly 2*N*N."""
        for N in [4, 8, 16]:
            sim = _make_sim(N=N, dt=0.02)
            state = sim.reset()
            assert state.shape == (2 * N * N,)

    def test_step_returns_correct_shape(self):
        sim = _make_sim(N=8)
        sim.reset()
        state = sim.step()
        assert state.shape == (2 * 8 * 8,)

    def test_observe_matches_state(self):
        sim = _make_sim(N=8)
        sim.reset()
        sim.step()
        obs = sim.observe()
        np.testing.assert_array_equal(obs, sim._state)

    def test_v_w_fields_from_state(self):
        """First N*N entries are v, next N*N are w."""
        sim = _make_sim(N=8)
        state = sim.reset()
        v_flat = state[:64]
        w_flat = state[64:]
        np.testing.assert_array_equal(v_flat, sim._v.ravel())
        np.testing.assert_array_equal(w_flat, sim._w.ravel())


class TestFHNLatticePeriodicBCs:
    def test_periodic_boundary_conditions(self):
        """The Laplacian should use periodic wrapping: site (0,j) neighbors (N-1,j)."""
        sim = _make_sim(N=8, D=1.0, dt=0.01)
        sim.reset()

        # Create a state with a sharp spike at (0, 0)
        sim._v[:] = 0.0
        sim._v[0, 0] = 1.0
        sim._w[:] = 0.0

        lap = sim._laplacian_periodic(sim._v)

        # At (0,0): neighbors are (1,0), (N-1,0), (0,1), (0,N-1)
        # All are 0 except the center which is 1, so Lap = 0+0+0+0 - 4*1 = -4
        assert lap[0, 0] == pytest.approx(-4.0)

        # At (1,0): neighbors include (0,0)=1, so Lap = 1+0+0+0 - 4*0 = 1
        assert lap[1, 0] == pytest.approx(1.0)

        # At (N-1,0): periodic wrap, neighbor is (0,0)=1
        assert lap[sim.N - 1, 0] == pytest.approx(1.0)

        # At (0,1): neighbor is (0,0)=1
        assert lap[0, 1] == pytest.approx(1.0)

        # At (0,N-1): periodic wrap, neighbor is (0,0)=1
        assert lap[0, sim.N - 1] == pytest.approx(1.0)

    def test_laplacian_of_uniform_is_zero(self):
        """Laplacian of a uniform field should be zero everywhere."""
        sim = _make_sim(N=8)
        uniform = np.ones((8, 8)) * 3.5
        lap = sim._laplacian_periodic(uniform)
        np.testing.assert_allclose(lap, 0.0, atol=1e-12)


class TestFHNLatticeDynamics:
    def test_step_advances_state(self):
        sim = _make_sim()
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_state_bounded(self):
        """State should not blow up over a moderate simulation."""
        sim = _make_sim(N=8, n_steps=200, dt=0.02)
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), "State contains non-finite values"
        # FHN v should stay bounded roughly in [-3, 3]
        v_flat = state[:sim.N * sim.N]
        assert np.all(np.abs(v_flat) < 10.0), f"v out of bounds: max={np.max(np.abs(v_flat))}"

    def test_reset_deterministic(self):
        """Same seed should produce identical initial states."""
        sim1 = _make_sim(N=8)
        s1 = sim1.reset(seed=123)

        sim2 = _make_sim(N=8)
        s2 = sim2.reset(seed=123)

        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different initial states."""
        sim1 = _make_sim(N=8)
        s1 = sim1.reset(seed=1)

        sim2 = _make_sim(N=8)
        s2 = sim2.reset(seed=2)

        assert not np.allclose(s1, s2)

    def test_trajectory_reproducibility(self):
        """Same seed should produce identical trajectories."""
        sim1 = _make_sim(N=8, dt=0.02, n_steps=50)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = _make_sim(N=8, dt=0.02, n_steps=50)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_array_equal(s1, s2)

    def test_diffusion_smooths_perturbation(self):
        """With high D and no reaction, a sharp feature should smooth out."""
        # Use large D, zero reaction by setting I and eps to suppress dynamics
        sim = _make_sim(N=16, D=2.0, dt=0.01, eps=0.0, I_ext=0.0, n_steps=100)
        sim.reset(seed=42)

        # Set a sharp spike
        sim._v[:] = 0.0
        sim._w[:] = 0.0
        sim._v[8, 8] = 5.0
        sim._state = sim._pack_state()

        std_initial = np.std(sim._v)

        # With eps=0, dw/dt=0 so w stays zero
        # dv/dt = v - v^3/3 - w + I + D*Lap(v)
        # For small v near 0: dv/dt ~ v + D*Lap(v), but with D=2 the diffusion
        # dominates the reaction for a localized spike
        for _ in range(50):
            sim.step()

        std_final = np.std(sim._v)
        # The spike should have spread out, reducing the standard deviation
        assert std_final < std_initial

    def test_uniform_steady_state(self):
        """Starting from the rest state with no noise, the system should remain uniform."""
        sim = _make_sim(N=8, D=1.0, dt=0.02, n_steps=100)
        # Manually initialize to exact rest state without noise
        v_rest = sim._find_rest_v()
        w_rest = (v_rest + sim.a) / sim.b_param
        sim._v = np.full((sim.N, sim.N), v_rest, dtype=np.float64)
        sim._w = np.full((sim.N, sim.N), w_rest, dtype=np.float64)
        sim._state = sim._pack_state()
        sim._step_count = 0

        for _ in range(100):
            sim.step()

        # Should remain very close to uniform
        assert np.std(sim._v) < 1e-6, f"v std = {np.std(sim._v)} (expected near 0)"


class TestFHNLatticeProperties:
    def test_v_field_property(self):
        sim = _make_sim(N=8)
        sim.reset()
        v = sim.v_field
        assert v.shape == (8, 8)

    def test_w_field_property(self):
        sim = _make_sim(N=8)
        sim.reset()
        w = sim.w_field
        assert w.shape == (8, 8)

    def test_v_field_returns_copy(self):
        """Modifying the returned field should not affect internal state."""
        sim = _make_sim(N=8)
        sim.reset()
        v = sim.v_field
        v[:] = 999.0
        assert not np.any(sim._v == 999.0)

    def test_mean_v(self):
        sim = _make_sim(N=8)
        sim.reset()
        expected = float(np.mean(sim._v))
        assert sim.mean_v == pytest.approx(expected)

    def test_std_v(self):
        sim = _make_sim(N=8)
        sim.reset()
        expected = float(np.std(sim._v))
        assert sim.std_v == pytest.approx(expected)

    def test_sync_order_parameter_range(self):
        """Order parameter should be between 0 and 1."""
        sim = _make_sim(N=8)
        sim.reset()
        order = sim.synchronization_order_parameter
        assert 0.0 <= order <= 1.0

    def test_sync_order_parameter_uniform(self):
        """Perfectly uniform v should give order parameter close to 1."""
        sim = _make_sim(N=8)
        sim.reset()
        sim._v[:] = 1.5  # uniform
        order = sim.synchronization_order_parameter
        assert order == pytest.approx(1.0)


class TestFHNLatticePatternClassification:
    def test_classify_uniform(self):
        """Uniform state should be classified as 'uniform'."""
        sim = _make_sim(N=8)
        sim.reset()
        sim._v[:] = 1.0
        assert sim.classify_pattern() == "uniform"

    def test_classify_returns_valid_label(self):
        """Pattern label should be one of the known categories."""
        sim = _make_sim(N=8)
        sim.reset()
        for _ in range(50):
            sim.step()
        pattern = sim.classify_pattern()
        assert pattern in {"uniform", "wave", "spiral", "turbulent"}

    def test_spatial_pattern_formation(self):
        """With coupling D>0 and perturbation, patterns should evolve."""
        sim = _make_sim(N=16, D=0.5, dt=0.02, n_steps=200)
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        # After evolving, the field should not be exactly uniform
        # (the initial perturbation drives pattern formation)
        assert np.std(sim._v) > 1e-4


class TestFHNLatticeTrajectory:
    def test_run_trajectory(self):
        """The run() method should produce a valid TrajectoryData."""
        sim = _make_sim(N=8, n_steps=20, dt=0.02)
        traj = sim.run(n_steps=20)
        expected_dim = 2 * 8 * 8  # 128
        assert traj.states.shape == (21, expected_dim)  # 20 steps + initial
        assert np.all(np.isfinite(traj.states))

    def test_different_parameters_different_dynamics(self):
        """Changing D should produce different dynamics."""
        sim1 = _make_sim(N=8, D=0.1, dt=0.02)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = _make_sim(N=8, D=5.0, dt=0.01)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe().copy()

        assert not np.allclose(s1, s2)


class TestFHNLatticeRediscovery:
    def test_sync_data_generation(self):
        from simulating_anything.rediscovery.fhn_lattice import generate_sync_data

        data = generate_sync_data(n_D=3, n_steps=50, dt=0.02, N=8)
        assert len(data["D"]) >= 1
        assert len(data["order_parameter"]) == len(data["D"])
        assert np.all(data["order_parameter"] >= 0)
        assert np.all(data["order_parameter"] <= 1.0)

    def test_pattern_data_generation(self):
        from simulating_anything.rediscovery.fhn_lattice import generate_pattern_data

        data = generate_pattern_data(n_D=2, n_I=2, n_steps=30, dt=0.02, N=8)
        n_expected = len(data["D"])
        assert n_expected >= 1
        assert len(data["pattern"]) == n_expected
        for p in data["pattern"]:
            assert p in {"uniform", "wave", "spiral", "turbulent"}

    def test_wave_speed_data_generation(self):
        from simulating_anything.rediscovery.fhn_lattice import generate_wave_speed_data

        data = generate_wave_speed_data(
            n_D=3, n_steps=30, n_measure=30, dt=0.02, N=8,
        )
        assert len(data["D"]) >= 1
        assert len(data["wave_speed"]) == len(data["D"])
        assert np.all(data["wave_speed"] >= 0)

    def test_rediscovery_runs_without_pysr(self):
        """Full rediscovery pipeline should work without PySR installed."""
        import tempfile

        from simulating_anything.rediscovery.fhn_lattice import (
            run_fhn_lattice_rediscovery,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_fhn_lattice_rediscovery(output_dir=tmpdir, n_iterations=1)

        assert results["domain"] == "fhn_lattice"
        assert "sync_transition" in results
        assert "pattern_classification" in results
        assert "wave_speed_data" in results
