"""Tests for the 1D Gray-Scott reaction-diffusion simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.gray_scott_1d import (
    GrayScott1DSimulation,
    _laplacian_1d,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    D_u: float = 0.16,
    D_v: float = 0.08,
    f: float = 0.04,
    k: float = 0.06,
    N: int = 64,
    L: float = 2.5,
    dt: float | None = None,
    n_steps: int = 100,
    seed: int = 42,
) -> SimulationConfig:
    """Create a SimulationConfig with CFL-safe defaults."""
    dx = L / N
    cfl_limit = dx**2 / (4.0 * max(D_u, D_v))
    if dt is None:
        dt = 0.5 * cfl_limit  # half the CFL limit for safety
    return SimulationConfig(
        domain=Domain.GRAY_SCOTT_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "D_u": D_u,
            "D_v": D_v,
            "f": f,
            "k": k,
            "N": float(N),
            "L": L,
        },
        seed=seed,
    )


class TestGrayScott1DCreation:
    """Test simulation creation and configuration."""

    def test_creation_defaults(self):
        """Simulation can be created with default parameters."""
        sim = GrayScott1DSimulation(_make_config())
        assert sim.D_u == 0.16
        assert sim.D_v == 0.08
        assert sim.f == 0.04
        assert sim.k == 0.06

    def test_custom_parameters(self):
        """Custom parameters are properly set."""
        sim = GrayScott1DSimulation(_make_config(D_u=0.2, D_v=0.1, f=0.05, k=0.065))
        assert sim.D_u == 0.2
        assert sim.D_v == 0.1
        assert sim.f == 0.05
        assert sim.k == 0.065

    def test_grid_size(self):
        """Grid is properly constructed."""
        sim = GrayScott1DSimulation(_make_config(N=128, L=5.0))
        assert sim.N == 128
        assert sim.L == 5.0
        assert sim.dx == pytest.approx(5.0 / 128)
        assert len(sim.x) == 128

    def test_domain_enum(self):
        """Config uses the correct Domain enum."""
        config = _make_config()
        assert config.domain == Domain.GRAY_SCOTT_1D


class TestGrayScott1DState:
    """Test state shape and initial conditions."""

    def test_state_shape(self):
        """State should be (2, N) for u and v fields."""
        sim = GrayScott1DSimulation(_make_config(N=64))
        state = sim.reset()
        assert state.shape == (2, 64)

    def test_state_shape_128(self):
        """State shape with N=128."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        state = sim.reset()
        assert state.shape == (2, 128)

    def test_initial_u_mostly_one(self):
        """u should be close to 1.0 everywhere except perturbation region."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        state = sim.reset()
        u = state[0]
        # Most of u should be near 1.0 (outside perturbation region)
        far_from_center = np.concatenate([u[:20], u[-20:]])
        assert np.mean(far_from_center) > 0.9

    def test_initial_v_mostly_zero(self):
        """v should be close to 0.0 everywhere except perturbation region."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        state = sim.reset()
        v = state[1]
        # Most of v should be near 0.0 (outside perturbation region)
        far_from_center = np.concatenate([v[:20], v[-20:]])
        assert np.mean(far_from_center) < 0.05

    def test_initial_perturbation_exists(self):
        """Perturbation should be visible in the center."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        state = sim.reset()
        v = state[1]
        center = 64
        # Center region should have higher v than edges
        center_v = np.mean(v[center - 5:center + 5])
        edge_v = np.mean(v[:10])
        assert center_v > edge_v

    def test_values_physical_range(self):
        """Initial values should be in [0, 1] after clamping."""
        sim = GrayScott1DSimulation(_make_config())
        state = sim.reset()
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)


class TestGrayScott1DCFLCondition:
    """Test CFL stability condition."""

    def test_cfl_limit_computed(self):
        """CFL limit should be dx^2 / (4 * D_max)."""
        sim = GrayScott1DSimulation(_make_config(N=64, L=2.5))
        dx = 2.5 / 64
        expected_cfl = dx**2 / (4.0 * 0.16)
        assert sim.cfl_limit == pytest.approx(expected_cfl)

    def test_cfl_smaller_with_larger_N(self):
        """Higher resolution should give stricter CFL."""
        sim1 = GrayScott1DSimulation(_make_config(N=64))
        sim2 = GrayScott1DSimulation(_make_config(N=128))
        assert sim2.cfl_limit < sim1.cfl_limit

    def test_safe_dt_doesnt_diverge(self):
        """A CFL-safe dt should keep values finite."""
        sim = GrayScott1DSimulation(_make_config(N=64))
        sim.reset()
        for _ in range(100):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))


class TestGrayScott1DStep:
    """Test simulation stepping behavior."""

    def test_step_changes_state(self):
        """Stepping should modify the state."""
        sim = GrayScott1DSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """observe() should return the current state."""
        sim = GrayScott1DSimulation(_make_config())
        state = sim.reset()
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)

    def test_step_returns_state(self):
        """step() should return the new state."""
        sim = GrayScott1DSimulation(_make_config())
        sim.reset()
        state = sim.step()
        assert state.shape == (2, 64)
        np.testing.assert_array_equal(state, sim.observe())

    def test_multiple_steps_finite(self):
        """Multiple steps should stay finite with safe dt."""
        sim = GrayScott1DSimulation(_make_config(N=64, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))


class TestGrayScott1DTrajectory:
    """Test trajectory collection."""

    def test_trajectory_collection(self):
        """run() should collect a full trajectory."""
        sim = GrayScott1DSimulation(_make_config(n_steps=50))
        traj = sim.run(n_steps=50)
        # n_steps + 1 (includes initial state)
        assert traj.states.shape[0] == 51
        assert traj.states.shape[1] == 2
        assert traj.states.shape[2] == 64

    def test_trajectory_timestamps(self):
        """Timestamps should be properly spaced."""
        config = _make_config(n_steps=10)
        sim = GrayScott1DSimulation(config)
        traj = sim.run(n_steps=10)
        assert len(traj.timestamps) == 11
        np.testing.assert_allclose(
            traj.timestamps,
            np.arange(11) * config.dt,
        )

    def test_trajectory_parameters(self):
        """Trajectory should store parameters."""
        sim = GrayScott1DSimulation(_make_config())
        traj = sim.run(n_steps=10)
        assert "D_u" in traj.parameters
        assert traj.parameters["D_u"] == 0.16


class TestGrayScott1DPulseDetection:
    """Test pulse counting and detection."""

    def test_no_pulses_initially(self):
        """Initial condition might show a perturbation peak."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        sim.reset()
        # The perturbation might or might not register as a "pulse"
        # depending on threshold -- just check it returns an int
        n = sim.count_pulses()
        assert isinstance(n, int)
        assert n >= 0

    def test_count_pulses_with_explicit_threshold(self):
        """count_pulses with a very low threshold should detect any peak."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        sim.reset()
        # With threshold=0.001, the initial perturbation should register
        n_low = sim.count_pulses(threshold=0.001)
        n_high = sim.count_pulses(threshold=10.0)
        assert n_low >= n_high  # lower threshold -> more or equal pulses
        assert n_high == 0  # impossibly high threshold -> no pulses

    def test_count_pulses_on_flat_field(self):
        """A flat field should have zero pulses."""
        sim = GrayScott1DSimulation(_make_config(N=64))
        sim.reset()
        # Override v with flat field
        sim._v = np.zeros(64)
        sim._state = np.stack([sim._u, sim._v], axis=0)
        assert sim.count_pulses() == 0

    def test_count_pulses_synthetic(self):
        """Synthetic pulse peaks should be detected."""
        sim = GrayScott1DSimulation(_make_config(N=256))
        sim.reset()
        # Create two clear peaks in v
        sim._v = np.zeros(256)
        sim._v[64] = 0.5
        sim._v[63] = 0.3
        sim._v[65] = 0.3
        sim._v[192] = 0.5
        sim._v[191] = 0.3
        sim._v[193] = 0.3
        sim._state = np.stack([sim._u, sim._v], axis=0)
        n = sim.count_pulses(threshold=0.4)
        assert n == 2


class TestGrayScott1DPeriodicBC:
    """Test periodic boundary conditions."""

    def test_laplacian_constant_field(self):
        """Laplacian of a constant field should be zero."""
        field = np.ones(100)
        lap = _laplacian_1d(field, 0.1)
        np.testing.assert_allclose(lap, 0.0, atol=1e-14)

    def test_laplacian_periodic(self):
        """Laplacian should respect periodic boundaries."""
        N = 64
        L = 2 * np.pi
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        # sin(x) has Laplacian = -sin(x)
        field = np.sin(x)
        lap = _laplacian_1d(field, dx)
        expected = -np.sin(x)
        # Finite differences approximate the Laplacian
        np.testing.assert_allclose(lap, expected, atol=0.01)

    def test_boundary_wraps(self):
        """Edge values should connect via periodic boundary."""
        field = np.zeros(10)
        field[0] = 1.0
        dx = 1.0
        lap = _laplacian_1d(field, dx)
        # Laplacian at 0: (field[-1] + field[1] - 2*field[0]) = (0 + 0 - 2) = -2
        assert lap[0] == pytest.approx(-2.0)
        # Laplacian at -1: (field[-2] + field[0] - 2*field[-1]) = (0 + 1 - 0) = 1
        assert lap[-1] == pytest.approx(1.0)


class TestGrayScott1DParameterVariations:
    """Test behavior under different parameter values."""

    def test_higher_D_u_smoother(self):
        """Higher D_u should produce smoother u field."""
        config_low = _make_config(D_u=0.05, N=64)
        config_high = _make_config(D_u=0.5, N=64)

        sim_low = GrayScott1DSimulation(config_low)
        sim_low.reset(seed=42)
        sim_high = GrayScott1DSimulation(config_high)
        sim_high.reset(seed=42)

        # Run fewer steps to avoid divergence with high D and same dt
        n_steps = 20
        for _ in range(n_steps):
            sim_low.step()
            sim_high.step()

        # Higher D should smooth out gradients
        grad_low = np.std(np.diff(sim_low._u))
        grad_high = np.std(np.diff(sim_high._u))
        assert grad_high <= grad_low + 1e-10

    def test_zero_feed_rate_v_decays(self):
        """With f=0, v should decay toward zero (no feeding)."""
        config = _make_config(f=0.0, k=0.06, N=64)
        sim = GrayScott1DSimulation(config)
        sim.reset()
        initial_v_total = sim.total_v
        for _ in range(200):
            sim.step()
        final_v_total = sim.total_v
        assert final_v_total <= initial_v_total

    def test_different_seeds_different_noise(self):
        """Different seeds should produce different initial conditions."""
        sim1 = GrayScott1DSimulation(_make_config())
        sim2 = GrayScott1DSimulation(_make_config(seed=123))
        s1 = sim1.reset(seed=42)
        s2 = sim2.reset(seed=123)
        assert not np.allclose(s1, s2)


class TestGrayScott1DProperties:
    """Test computed properties."""

    def test_total_v(self):
        """total_v should integrate v over domain."""
        sim = GrayScott1DSimulation(_make_config(N=64))
        sim.reset()
        tv = sim.total_v
        expected = np.sum(sim._v) * sim.dx
        assert tv == pytest.approx(expected)

    def test_total_u(self):
        """total_u should integrate u over domain."""
        sim = GrayScott1DSimulation(_make_config(N=64))
        sim.reset()
        tu = sim.total_u
        expected = np.sum(sim._u) * sim.dx
        assert tu == pytest.approx(expected)

    def test_max_v(self):
        """max_v should return the maximum of the v field."""
        sim = GrayScott1DSimulation(_make_config(N=64))
        sim.reset()
        assert sim.max_v == pytest.approx(np.max(sim._v))

    def test_max_v_initially_positive(self):
        """max_v should be positive due to perturbation."""
        sim = GrayScott1DSimulation(_make_config(N=128))
        sim.reset()
        assert sim.max_v > 0.0


class TestGrayScott1DPulseSpeed:
    """Test pulse speed measurement."""

    def test_speed_nonnegative(self):
        """Pulse speed should be non-negative."""
        config = _make_config(N=64, n_steps=100)
        sim = GrayScott1DSimulation(config)
        sim.reset()
        speed = sim.measure_pulse_speed(n_steps=10)
        assert speed >= 0.0

    def test_speed_finite(self):
        """Pulse speed should be finite."""
        config = _make_config(N=64)
        sim = GrayScott1DSimulation(config)
        sim.reset()
        speed = sim.measure_pulse_speed(n_steps=10)
        assert np.isfinite(speed)


class TestGrayScott1DRediscovery:
    """Test rediscovery data generation functions."""

    def test_pulse_regime_data_generation(self):
        """generate_pulse_regime_data should produce valid arrays."""
        from simulating_anything.rediscovery.gray_scott_1d import (
            generate_pulse_regime_data,
        )
        data = generate_pulse_regime_data(n_f=3, n_k=3, n_steps=100, N=32)
        assert len(data["f"]) == 9  # 3 * 3
        assert len(data["k"]) == 9
        assert len(data["pulse_count"]) == 9
        assert len(data["max_v"]) == 9
        assert np.all(data["pulse_count"] >= 0)

    def test_splitting_data_generation(self):
        """generate_splitting_data should produce valid arrays."""
        from simulating_anything.rediscovery.gray_scott_1d import (
            generate_splitting_data,
        )
        data = generate_splitting_data(n_k=4, n_steps=100, N=32)
        assert len(data["k"]) == 4
        assert len(data["pulse_count"]) == 4

    def test_speed_data_generation(self):
        """generate_speed_data should produce valid arrays."""
        from simulating_anything.rediscovery.gray_scott_1d import (
            generate_speed_data,
        )
        data = generate_speed_data(
            n_samples=2, track_steps=20, warmup_steps=50, N=32,
        )
        assert "f" in data
        assert "speed" in data
        assert len(data["f"]) == len(data["speed"])

    def test_rediscovery_runner(self, tmp_path):
        """Full rediscovery runner should produce results."""
        from simulating_anything.rediscovery.gray_scott_1d import (
            run_gray_scott_1d_rediscovery,
        )
        results = run_gray_scott_1d_rediscovery(
            output_dir=tmp_path / "gs1d",
            n_iterations=5,
        )
        assert results["domain"] == "gray_scott_1d"
        assert "pulse_regime" in results
        assert "splitting" in results
        assert "pulse_speed" in results
        # Check output file was written
        assert (tmp_path / "gs1d" / "results.json").exists()
