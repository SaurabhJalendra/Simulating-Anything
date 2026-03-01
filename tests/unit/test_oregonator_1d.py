"""Tests for the 1D Oregonator reaction-diffusion simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.oregonator_1d import (
    Oregonator1DSimulation,
    _laplacian_1d,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    eps: float = 0.1,
    f: float = 1.0,
    q: float = 0.002,
    D_u: float = 1.0,
    D_v: float = 0.6,
    N: int = 64,
    L: float = 50.0,
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
        domain=Domain.OREGONATOR_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "eps": eps,
            "f": f,
            "q": q,
            "D_u": D_u,
            "D_v": D_v,
            "N": float(N),
            "L": L,
        },
        seed=seed,
    )


class TestOregonator1DCreation:
    """Test simulation creation and configuration."""

    def test_creation_defaults(self):
        """Simulation can be created with default parameters."""
        sim = Oregonator1DSimulation(_make_config())
        assert sim.eps == 0.1
        assert sim.f == 1.0
        assert sim.q == 0.002
        assert sim.D_u == 1.0
        assert sim.D_v == 0.6

    def test_custom_parameters(self):
        """Custom parameters are properly set."""
        sim = Oregonator1DSimulation(_make_config(eps=0.05, f=1.5, q=0.005))
        assert sim.eps == 0.05
        assert sim.f == 1.5
        assert sim.q == 0.005

    def test_grid_size(self):
        """Grid is properly constructed."""
        sim = Oregonator1DSimulation(_make_config(N=128, L=80.0))
        assert sim.N == 128
        assert sim.L == 80.0
        assert sim.dx == pytest.approx(80.0 / 128)
        assert len(sim.x) == 128

    def test_domain_enum(self):
        """Config uses the correct Domain enum."""
        config = _make_config()
        assert config.domain == Domain.OREGONATOR_1D


class TestOregonator1DState:
    """Test state shape and initial conditions."""

    def test_state_shape(self):
        """State should be (2, N) for u and v fields."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        state = sim.reset()
        assert state.shape == (2, 64)

    def test_state_shape_200(self):
        """State shape with N=200 (default Oregonator)."""
        sim = Oregonator1DSimulation(_make_config(N=200))
        state = sim.reset()
        assert state.shape == (2, 200)

    def test_initial_u_mostly_quiescent(self):
        """u should be near q everywhere except stimulus region."""
        sim = Oregonator1DSimulation(_make_config(N=128, L=100.0))
        state = sim.reset()
        u = state[0]
        # Far from stimulus (right half), u should be close to q
        far_region = u[64:]
        assert np.mean(far_region) < 0.1

    def test_initial_v_mostly_zero(self):
        """v should be close to 0.0 everywhere."""
        sim = Oregonator1DSimulation(_make_config(N=128))
        state = sim.reset()
        v = state[1]
        # Most of v should be near 0 (small noise only)
        assert np.mean(v) < 0.05

    def test_initial_stimulus_exists(self):
        """A localized stimulus should be present in u near the left edge."""
        sim = Oregonator1DSimulation(_make_config(N=128))
        state = sim.reset()
        u = state[0]
        # Stimulus region near left edge should have higher u than right half
        stim_region = u[:20]
        far_region = u[64:]
        assert np.max(stim_region) > np.max(far_region)

    def test_values_nonnegative(self):
        """Initial values should be non-negative (concentrations)."""
        sim = Oregonator1DSimulation(_make_config())
        state = sim.reset()
        assert np.all(state >= 0.0)


class TestOregonator1DCFLCondition:
    """Test CFL stability condition."""

    def test_cfl_limit_computed(self):
        """CFL limit should be dx^2 / (4 * D_max)."""
        sim = Oregonator1DSimulation(_make_config(N=64, L=50.0, D_u=1.0, D_v=0.6))
        dx = 50.0 / 64
        expected_cfl = dx**2 / (4.0 * 1.0)  # D_max = D_u = 1.0
        assert sim.cfl_limit == pytest.approx(expected_cfl)

    def test_cfl_smaller_with_larger_N(self):
        """Higher resolution should give stricter CFL."""
        sim1 = Oregonator1DSimulation(_make_config(N=64))
        sim2 = Oregonator1DSimulation(_make_config(N=128))
        assert sim2.cfl_limit < sim1.cfl_limit

    def test_safe_dt_doesnt_diverge(self):
        """A CFL-safe dt should keep values finite."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        for _ in range(100):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))

    def test_cfl_with_higher_diffusion(self):
        """Higher diffusion coefficient should give smaller CFL limit."""
        sim_low = Oregonator1DSimulation(_make_config(D_u=0.5, D_v=0.3))
        sim_high = Oregonator1DSimulation(_make_config(D_u=2.0, D_v=0.6))
        assert sim_high.cfl_limit < sim_low.cfl_limit


class TestOregonator1DStep:
    """Test simulation stepping behavior."""

    def test_step_changes_state(self):
        """Stepping should modify the state."""
        sim = Oregonator1DSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """observe() should return the current state."""
        sim = Oregonator1DSimulation(_make_config())
        state = sim.reset()
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)

    def test_step_returns_state(self):
        """step() should return the new state."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        state = sim.step()
        assert state.shape == (2, 64)
        np.testing.assert_array_equal(state, sim.observe())

    def test_multiple_steps_finite(self):
        """Multiple steps should stay finite with safe dt."""
        sim = Oregonator1DSimulation(_make_config(N=64, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))

    def test_values_stay_nonnegative(self):
        """Concentrations should remain non-negative after stepping."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        for _ in range(200):
            sim.step()
        state = sim.observe()
        assert np.all(state >= 0.0)


class TestOregonator1DTrajectory:
    """Test trajectory collection."""

    def test_trajectory_collection(self):
        """run() should collect a full trajectory."""
        sim = Oregonator1DSimulation(_make_config(N=64, n_steps=50))
        traj = sim.run(n_steps=50)
        # n_steps + 1 (includes initial state)
        assert traj.states.shape[0] == 51
        assert traj.states.shape[1] == 2
        assert traj.states.shape[2] == 64

    def test_trajectory_timestamps(self):
        """Timestamps should be properly spaced."""
        config = _make_config(n_steps=10)
        sim = Oregonator1DSimulation(config)
        traj = sim.run(n_steps=10)
        assert len(traj.timestamps) == 11
        np.testing.assert_allclose(
            traj.timestamps,
            np.arange(11) * config.dt,
        )

    def test_trajectory_parameters(self):
        """Trajectory should store parameters."""
        sim = Oregonator1DSimulation(_make_config())
        traj = sim.run(n_steps=10)
        assert "eps" in traj.parameters
        assert traj.parameters["eps"] == 0.1
        assert "D_u" in traj.parameters
        assert traj.parameters["D_u"] == 1.0


class TestOregonator1DPulseDetection:
    """Test pulse counting and detection."""

    def test_count_pulses_returns_int(self):
        """count_pulses should return a non-negative integer."""
        sim = Oregonator1DSimulation(_make_config(N=128))
        sim.reset()
        n = sim.count_pulses()
        assert isinstance(n, int)
        assert n >= 0

    def test_count_pulses_with_explicit_threshold(self):
        """count_pulses with very high threshold should return 0."""
        sim = Oregonator1DSimulation(_make_config(N=128))
        sim.reset()
        n = sim.count_pulses(threshold=100.0)
        assert n == 0

    def test_count_pulses_on_flat_field(self):
        """A flat u field should have zero pulses."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        sim._u = np.full(64, 0.01)
        sim._state = np.stack([sim._u, sim._v], axis=0)
        assert sim.count_pulses() == 0

    def test_count_pulses_synthetic(self):
        """Synthetic pulse peaks in u should be detected."""
        sim = Oregonator1DSimulation(_make_config(N=256, L=100.0))
        sim.reset()
        # Create two clear peaks in u
        sim._u = np.full(256, 0.01)
        sim._u[64] = 0.8
        sim._u[63] = 0.4
        sim._u[65] = 0.4
        sim._u[192] = 0.8
        sim._u[191] = 0.4
        sim._u[193] = 0.4
        sim._state = np.stack([sim._u, sim._v], axis=0)
        n = sim.count_pulses(threshold=0.5)
        assert n == 2


class TestOregonator1DPeriodicBC:
    """Test periodic boundary conditions."""

    def test_laplacian_constant_field(self):
        """Laplacian of a constant field should be zero."""
        field = np.ones(100) * 0.5
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
        np.testing.assert_allclose(lap, expected, atol=0.01)

    def test_boundary_wraps(self):
        """Edge values should connect via periodic boundary."""
        field = np.zeros(10)
        field[0] = 1.0
        dx = 1.0
        lap = _laplacian_1d(field, dx)
        # At index 0: (field[-1] + field[1] - 2*field[0]) = (0 + 0 - 2) = -2
        assert lap[0] == pytest.approx(-2.0)
        # At index -1: (field[-2] + field[0] - 2*field[-1]) = (0 + 1 - 0) = 1
        assert lap[-1] == pytest.approx(1.0)


class TestOregonator1DProperties:
    """Test computed properties."""

    def test_total_u(self):
        """total_u should integrate u over domain."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        tu = sim.total_u
        expected = np.sum(sim._u) * sim.dx
        assert tu == pytest.approx(expected)

    def test_total_v(self):
        """total_v should integrate v over domain."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        tv = sim.total_v
        expected = np.sum(sim._v) * sim.dx
        assert tv == pytest.approx(expected)

    def test_max_u(self):
        """max_u should return the maximum of the u field."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        assert sim.max_u == pytest.approx(np.max(sim._u))

    def test_max_v(self):
        """max_v should return the maximum of the v field."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        assert sim.max_v == pytest.approx(np.max(sim._v))

    def test_max_u_positive_from_stimulus(self):
        """max_u should be positive due to stimulus."""
        sim = Oregonator1DSimulation(_make_config(N=128))
        sim.reset()
        assert sim.max_u > 0.05  # Stimulus pushes u well above q

    def test_pulse_width_zero_on_flat(self):
        """Pulse width should be 0 when u is flat and low."""
        sim = Oregonator1DSimulation(_make_config(N=64))
        sim.reset()
        sim._u = np.full(64, 0.01)
        sim._state = np.stack([sim._u, sim._v], axis=0)
        assert sim.pulse_width == 0.0

    def test_pulse_width_positive_with_pulse(self):
        """Pulse width should be positive when a pulse exists."""
        sim = Oregonator1DSimulation(_make_config(N=128, L=50.0))
        sim.reset()
        # Create a synthetic pulse
        sim._u = np.full(128, 0.01)
        center = 64
        for i in range(center - 5, center + 6):
            sim._u[i] = 0.5
        sim._state = np.stack([sim._u, sim._v], axis=0)
        width = sim.pulse_width
        assert width > 0.0
        # Width should be approximately 11 * dx
        expected = 11 * (50.0 / 128)
        assert width == pytest.approx(expected, rel=0.01)


class TestOregonator1DPulseSpeed:
    """Test pulse speed measurement."""

    def test_speed_nonnegative(self):
        """Pulse speed should be non-negative."""
        config = _make_config(N=64, n_steps=100)
        sim = Oregonator1DSimulation(config)
        sim.reset()
        speed = sim.measure_pulse_speed(n_steps=10)
        assert speed >= 0.0

    def test_speed_finite(self):
        """Pulse speed should be finite."""
        config = _make_config(N=64)
        sim = Oregonator1DSimulation(config)
        sim.reset()
        speed = sim.measure_pulse_speed(n_steps=10)
        assert np.isfinite(speed)


class TestOregonator1DParameterVariations:
    """Test behavior under different parameter values."""

    def test_different_seeds_different_noise(self):
        """Different seeds should produce different initial conditions."""
        sim1 = Oregonator1DSimulation(_make_config())
        sim2 = Oregonator1DSimulation(_make_config(seed=123))
        s1 = sim1.reset(seed=42)
        s2 = sim2.reset(seed=123)
        assert not np.allclose(s1, s2)

    def test_smaller_eps_faster_dynamics(self):
        """Smaller eps should produce faster u dynamics (larger du/dt)."""
        config_slow = _make_config(eps=0.2, N=64)
        config_fast = _make_config(eps=0.05, N=64)
        sim_slow = Oregonator1DSimulation(config_slow)
        sim_fast = Oregonator1DSimulation(config_fast)

        sim_slow.reset(seed=42)
        sim_fast.reset(seed=42)

        # Step once and check that the fast system changes more
        s0_slow = sim_slow.observe().copy()
        s0_fast = sim_fast.observe().copy()

        sim_slow.step()
        sim_fast.step()

        change_slow = np.max(np.abs(sim_slow.observe() - s0_slow))
        change_fast = np.max(np.abs(sim_fast.observe() - s0_fast))

        # Faster timescale should cause more change per step
        assert change_fast > change_slow

    def test_higher_D_u_smoother(self):
        """Higher D_u should produce smoother u field after evolution."""
        config_low = _make_config(D_u=0.3, D_v=0.2, N=64)
        config_high = _make_config(D_u=2.0, D_v=0.6, N=64)

        sim_low = Oregonator1DSimulation(config_low)
        sim_low.reset(seed=42)
        sim_high = Oregonator1DSimulation(config_high)
        sim_high.reset(seed=42)

        n_steps = 20
        for _ in range(n_steps):
            sim_low.step()
            sim_high.step()

        # Higher D should smooth gradients
        grad_low = np.std(np.diff(sim_low._u))
        grad_high = np.std(np.diff(sim_high._u))
        assert grad_high <= grad_low + 1e-10


class TestOregonator1DRediscovery:
    """Test rediscovery data generation functions."""

    def test_pulse_speed_vs_eps_data(self):
        """generate_pulse_speed_vs_eps should produce valid arrays."""
        from simulating_anything.rediscovery.oregonator_1d import (
            generate_pulse_speed_vs_eps,
        )
        data = generate_pulse_speed_vs_eps(
            n_eps=3, N=50, L=40.0, warmup_steps=100, track_steps=50,
        )
        assert "eps" in data
        assert "speed" in data
        assert len(data["eps"]) == len(data["speed"])
        # All speeds should be non-negative
        assert np.all(data["speed"] >= 0)

    def test_pulse_speed_vs_D_u_data(self):
        """generate_pulse_speed_vs_D_u should produce valid arrays."""
        from simulating_anything.rediscovery.oregonator_1d import (
            generate_pulse_speed_vs_D_u,
        )
        data = generate_pulse_speed_vs_D_u(
            n_D=3, N=50, L=40.0, warmup_steps=100, track_steps=50,
        )
        assert "D_u" in data
        assert "speed" in data
        assert len(data["D_u"]) == len(data["speed"])

    def test_wave_formation_data(self):
        """generate_wave_formation_data should produce snapshot arrays."""
        from simulating_anything.rediscovery.oregonator_1d import (
            generate_wave_formation_data,
        )
        data = generate_wave_formation_data(
            N=50, L=40.0, n_snapshots=5, total_steps=100,
        )
        assert "snapshots" in data
        assert "times" in data
        assert "x" in data
        assert data["snapshots"].shape[1] == 2  # u and v
        assert data["snapshots"].shape[2] == 50  # N grid points
        assert len(data["times"]) == len(data["snapshots"])

    def test_rediscovery_runner(self, tmp_path):
        """Full rediscovery runner should produce results."""
        from simulating_anything.rediscovery.oregonator_1d import (
            run_oregonator_1d_rediscovery,
        )
        results = run_oregonator_1d_rediscovery(
            output_dir=tmp_path / "oreg1d",
            n_iterations=5,
        )
        assert results["domain"] == "oregonator_1d"
        assert "wave_formation" in results
        # Check output file was written
        assert (tmp_path / "oreg1d" / "results.json").exists()
