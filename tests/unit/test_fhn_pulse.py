"""Tests for the 1D FitzHugh-Nagumo traveling pulse simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.fhn_pulse import FHNPulseSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 0.7,
    b: float = 0.8,
    eps: float = 0.08,
    D_v: float = 1.0,
    D_w: float = 0.0,
    I_stim: float = 0.0,
    N: int = 128,
    L: float = 50.0,
    dt: float | None = None,
    n_steps: int = 100,
) -> SimulationConfig:
    """Create a CFL-safe SimulationConfig for FHN Pulse."""
    dx = L / N
    D_max = max(D_v, D_w)
    cfl_limit = dx ** 2 / (2.0 * D_max) if D_max > 0 else 1.0
    if dt is None:
        dt = 0.4 * cfl_limit  # safe default
    return SimulationConfig(
        domain=Domain.FHN_PULSE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "eps": eps,
            "D_v": D_v,
            "D_w": D_w,
            "I_stim": I_stim,
            "N": float(N),
            "L": L,
        },
    )


class TestFHNPulseCreation:
    """Tests for simulation creation and parameter handling."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = FHNPulseSimulation(_make_config())
        assert sim.N == 128
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.D_v == 1.0
        assert sim.D_w == 0.0
        assert sim.L == 50.0

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = FHNPulseSimulation(_make_config(a=0.5, eps=0.1, D_v=2.0, D_w=0.1))
        assert sim.a == 0.5
        assert sim.eps == 0.1
        assert sim.D_v == 2.0
        assert sim.D_w == 0.1

    def test_grid_setup(self):
        """Grid should be set up correctly."""
        sim = FHNPulseSimulation(_make_config(N=64, L=20.0))
        assert sim.N == 64
        assert sim.dx == pytest.approx(20.0 / 64)
        assert len(sim.x) == 64
        assert sim.x[0] == pytest.approx(0.0)

    def test_cfl_violation_raises(self):
        """Extremely large dt should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            FHNPulseSimulation(_make_config(dt=100.0, D_v=5.0, N=32, L=10.0))

    def test_cfl_limit_computed(self):
        """CFL limit should be dx^2 / (2 * D_max)."""
        sim = FHNPulseSimulation(_make_config(N=64, L=32.0, D_v=2.0))
        dx = 32.0 / 64
        expected = dx ** 2 / (2.0 * 2.0)
        assert sim.cfl_limit == pytest.approx(expected)

    def test_zero_diffusion_no_cfl(self):
        """With D_v=0 and D_w=0, CFL limit should be inf."""
        sim = FHNPulseSimulation(_make_config(D_v=0.0, D_w=0.0, dt=1.0))
        assert sim.cfl_limit == np.inf

    def test_stimulus_parameter(self):
        """I_stim should be stored correctly."""
        sim = FHNPulseSimulation(_make_config(I_stim=0.5))
        assert sim.I_stim == 0.5


class TestFHNPulseState:
    """Tests for state shape and initialization."""

    def test_initial_state_shape(self):
        """State should be [v_1..v_N, w_1..w_N] with shape (2*N,)."""
        sim = FHNPulseSimulation(_make_config(N=64))
        state = sim.reset()
        assert state.shape == (128,)  # 2 * N = 2 * 64

    def test_step_returns_correct_shape(self):
        """Step should return state of same shape."""
        sim = FHNPulseSimulation(_make_config(N=64))
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_observe_returns_correct_shape(self):
        """Observe should return state of correct shape."""
        sim = FHNPulseSimulation(_make_config(N=64))
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)

    def test_initial_pulse_present(self):
        """Reset should produce a localized pulse in v above rest state."""
        sim = FHNPulseSimulation(_make_config(N=128))
        sim.reset()
        v = sim.v_field
        v_rest = sim._find_rest_v()
        # The Gaussian bump should raise v significantly above rest
        assert np.max(v) > v_rest + 0.5

    def test_initial_v_field_has_bump_at_left(self):
        """The initial pulse should be near the left end of the domain."""
        sim = FHNPulseSimulation(_make_config(N=256, L=100.0))
        sim.reset()
        v = sim.v_field
        peak_idx = np.argmax(v)
        # Pulse center is at L * 0.1 = 10.0, so peak should be in the left 30%
        assert peak_idx < 0.3 * sim.N

    def test_initial_w_field_uniform(self):
        """Initially w should be approximately uniform at the rest value."""
        sim = FHNPulseSimulation(_make_config(N=128))
        sim.reset()
        w = sim.w_field
        w_rest = (sim._find_rest_v() + sim.a) / sim.b_param
        # w should be nearly uniform (only v gets the pulse)
        assert np.std(w) < 0.01
        assert np.mean(w) == pytest.approx(w_rest, abs=0.01)


class TestFHNPulseDynamics:
    """Tests for time evolution and physical correctness."""

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = FHNPulseSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_state_bounded_short_run(self):
        """State should remain bounded over moderate simulation."""
        sim = FHNPulseSimulation(_make_config(n_steps=300))
        sim.reset()
        for _ in range(300):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))
        v = state[:sim.N]
        assert np.all(np.abs(v) < 10.0)

    def test_state_bounded_long_run(self):
        """State should remain bounded over a long simulation."""
        sim = FHNPulseSimulation(_make_config(N=64, L=30.0, n_steps=1000))
        sim.reset()
        for _ in range(1000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "NaN or Inf detected"
        v = state[:sim.N]
        w = state[sim.N:]
        assert np.all(np.abs(v) < 10.0)
        assert np.all(np.abs(w) < 10.0)

    def test_pulse_propagation(self):
        """The v peak should move in space as the pulse propagates."""
        sim = FHNPulseSimulation(_make_config(N=256, L=100.0, n_steps=1000))
        sim.reset()
        peak_pos_0 = np.argmax(sim.v_field) * sim.dx

        for _ in range(500):
            sim.step()
        peak_pos_1 = np.argmax(sim.v_field) * sim.dx

        # Handle periodic wrap
        delta = abs(peak_pos_1 - peak_pos_0)
        if delta > sim.L / 2:
            delta = sim.L - delta
        assert delta > 1.0  # Should have moved at least 1 spatial unit

    def test_no_diffusion_local_dynamics(self):
        """With D_v=0, each grid point evolves as a local FHN oscillator."""
        config = _make_config(D_v=0.0, D_w=0.0, N=16, L=10.0, dt=0.02, n_steps=50)
        sim = FHNPulseSimulation(config)
        sim.reset(seed=42)

        # Store initial values at a grid point away from the pulse
        idx = 12  # far from the left-end pulse
        v0 = sim._v_field[idx]
        w0 = sim._w_field[idx]

        # Run PDE simulation
        for _ in range(50):
            sim.step()
        v_pde = sim._v_field[idx]
        w_pde = sim._w_field[idx]

        # Run local FHN ODE (RK4) for same initial conditions
        dt = 0.02
        v, w = v0, w0
        a, b, eps = sim.a, sim.b_param, sim.eps
        I_stim = sim.I_stim
        for _ in range(50):
            def f_v(vi, wi):
                return vi - vi ** 3 / 3 - wi + I_stim

            def f_w(vi, wi):
                return eps * (vi + a - b * wi)

            kv1 = f_v(v, w)
            kw1 = f_w(v, w)
            kv2 = f_v(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
            kw2 = f_w(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
            kv3 = f_v(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
            kw3 = f_w(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
            kv4 = f_v(v + dt * kv3, w + dt * kw3)
            kw4 = f_w(v + dt * kv3, w + dt * kw3)
            v = v + (dt / 6) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
            w = w + (dt / 6) * (kw1 + 2 * kw2 + 2 * kw3 + kw4)

        np.testing.assert_allclose(v_pde, v, rtol=0.01)
        np.testing.assert_allclose(w_pde, w, rtol=0.01)

    def test_higher_D_v_faster_pulse(self):
        """Larger D_v should produce a faster traveling pulse."""
        speeds = []
        for D_v in [0.5, 2.0]:
            config = _make_config(N=128, L=80.0, D_v=D_v, n_steps=2000)
            sim = FHNPulseSimulation(config)
            sim.reset(seed=42)
            # Warmup
            for _ in range(500):
                sim.step()
            speed = sim.measure_wave_speed(n_steps=500)
            speeds.append(speed)
        # D_v=2.0 should give faster pulse than D_v=0.5
        assert speeds[1] > speeds[0], (
            f"Speed at D_v=2.0 ({speeds[1]:.4f}) should exceed "
            f"speed at D_v=0.5 ({speeds[0]:.4f})"
        )


class TestFHNPulseReproducibility:
    """Tests for deterministic behavior."""

    def test_same_seed_identical_trajectory(self):
        """Same seed should produce identical trajectories."""
        cfg = _make_config(N=64, n_steps=100)
        sim1 = FHNPulseSimulation(cfg)
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = FHNPulseSimulation(cfg)
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds with I_stim should produce (slightly) different results.

        Note: without noise, seeds only affect the config seed parameter,
        but reset is deterministic. For the FHN pulse, reset is entirely
        deterministic (no random component), so we verify that different
        parameters produce different results instead.
        """
        cfg1 = _make_config(N=64, D_v=1.0, n_steps=50)
        cfg2 = _make_config(N=64, D_v=1.5, n_steps=50)

        sim1 = FHNPulseSimulation(cfg1)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()

        sim2 = FHNPulseSimulation(cfg2)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()

        assert not np.allclose(sim1.observe(), sim2.observe())


class TestFHNPulseProperties:
    """Tests for property accessors."""

    def test_v_field_property(self):
        """v_field should return a copy of the v portion."""
        sim = FHNPulseSimulation(_make_config(N=64))
        sim.reset()
        v = sim.v_field
        assert v.shape == (64,)
        # Modifying the copy should not affect the simulation
        v[0] = 999.0
        assert sim.v_field[0] != 999.0

    def test_w_field_property(self):
        """w_field should return a copy of the w portion."""
        sim = FHNPulseSimulation(_make_config(N=64))
        sim.reset()
        w = sim.w_field
        assert w.shape == (64,)

    def test_mean_v(self):
        """mean_v should match np.mean of v_field."""
        sim = FHNPulseSimulation(_make_config(N=64))
        sim.reset()
        expected = float(np.mean(sim._v_field))
        assert sim.mean_v == pytest.approx(expected)

    def test_max_v(self):
        """max_v should match np.max of v_field."""
        sim = FHNPulseSimulation(_make_config(N=64))
        sim.reset()
        expected = float(np.max(sim._v_field))
        assert sim.max_v == pytest.approx(expected)

    def test_v_max_position(self):
        """v_max_position should be in the left part of the domain."""
        sim = FHNPulseSimulation(_make_config(N=128, L=50.0))
        sim.reset()
        pos = sim.v_max_position
        # Pulse center is near L * 0.1 = 5.0
        assert 0.0 <= pos < 50.0
        assert pos < 20.0  # Should be in the left portion

    def test_pulse_count_initial(self):
        """Initially there should be one pulse from the IC."""
        sim = FHNPulseSimulation(_make_config(N=128))
        sim.reset()
        assert sim.pulse_count >= 1

    def test_pulse_width_positive(self):
        """Pulse width should be positive when a pulse is present."""
        sim = FHNPulseSimulation(_make_config(N=128))
        sim.reset()
        assert sim.pulse_width > 0.0

    def test_total_excitation(self):
        """total_excitation should be the integral of v over the domain."""
        sim = FHNPulseSimulation(_make_config(N=64, L=20.0))
        sim.reset()
        expected = float(np.sum(sim._v_field) * sim.dx)
        assert sim.total_excitation == pytest.approx(expected)


class TestFHNPulseTrajectory:
    """Tests for run() and trajectory collection."""

    def test_run_trajectory_shape(self):
        """run() should produce a TrajectoryData with correct shapes."""
        sim = FHNPulseSimulation(_make_config(N=32, n_steps=20))
        traj = sim.run(n_steps=20)
        assert traj.states.shape == (21, 64)  # 20 steps + initial, 2*32
        assert np.all(np.isfinite(traj.states))

    def test_run_trajectory_timestamps(self):
        """Timestamps should match expected values."""
        cfg = _make_config(N=32, n_steps=10)
        sim = FHNPulseSimulation(cfg)
        traj = sim.run(n_steps=10)
        assert len(traj.timestamps) == 11
        assert traj.timestamps[0] == pytest.approx(0.0)
        assert traj.timestamps[-1] == pytest.approx(10 * cfg.dt)

    def test_run_trajectory_parameters(self):
        """TrajectoryData should contain the simulation parameters."""
        sim = FHNPulseSimulation(_make_config(N=32, n_steps=5))
        traj = sim.run(n_steps=5)
        assert "D_v" in traj.parameters
        assert "eps" in traj.parameters


class TestFHNPulseRestState:
    """Tests for the rest state computation."""

    def test_rest_state_default(self):
        """Rest state should be a valid nullcline intersection."""
        sim = FHNPulseSimulation(_make_config())
        v_rest = sim._find_rest_v()
        w_rest = (v_rest + sim.a) / sim.b_param
        # Check nullcline: v - v^3/3 - w + I_stim = 0
        residual = v_rest - v_rest ** 3 / 3.0 - w_rest + sim.I_stim
        assert abs(residual) < 1e-10

    def test_rest_state_with_stimulus(self):
        """Rest state should shift with I_stim."""
        sim_no_stim = FHNPulseSimulation(_make_config(I_stim=0.0))
        sim_stim = FHNPulseSimulation(_make_config(I_stim=0.3))
        v_rest_0 = sim_no_stim._find_rest_v()
        v_rest_stim = sim_stim._find_rest_v()
        # Positive I_stim should shift the rest state to higher v
        assert v_rest_stim > v_rest_0

    def test_rest_state_satisfies_w_nullcline(self):
        """Rest state should satisfy w = (v + a) / b."""
        sim = FHNPulseSimulation(_make_config())
        v_rest = sim._find_rest_v()
        w_rest = (v_rest + sim.a) / sim.b_param
        # Check w nullcline: eps * (v + a - b*w) = 0
        residual = sim.eps * (v_rest + sim.a - sim.b_param * w_rest)
        assert abs(residual) < 1e-10


class TestFHNPulseWaveSpeed:
    """Tests for wave speed measurement."""

    def test_wave_speed_positive(self):
        """Wave speed should be positive for a propagating pulse."""
        sim = FHNPulseSimulation(_make_config(N=128, L=80.0, n_steps=2000))
        sim.reset(seed=42)
        # Warmup to let pulse form
        for _ in range(400):
            sim.step()
        speed = sim.measure_wave_speed(n_steps=400)
        assert speed > 0.0, f"Expected positive wave speed, got {speed}"

    def test_wave_speed_not_nan(self):
        """Wave speed should not be NaN."""
        sim = FHNPulseSimulation(_make_config(N=128, L=80.0, n_steps=1000))
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        speed = sim.measure_wave_speed(n_steps=200)
        assert np.isfinite(speed)

    def test_wave_speed_reasonable_range(self):
        """Wave speed should be in a physically reasonable range."""
        sim = FHNPulseSimulation(_make_config(N=256, L=100.0, D_v=1.0, n_steps=3000))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        speed = sim.measure_wave_speed(n_steps=800)
        # For D_v=1.0, eps=0.08, speed is typically O(1)
        assert 0.0 <= speed < 50.0, f"Speed {speed} out of expected range"


class TestFHNPulseDiffusion:
    """Tests for diffusion behavior."""

    def test_with_inhibitor_diffusion(self):
        """Simulation should work with D_w > 0."""
        sim = FHNPulseSimulation(_make_config(D_v=1.0, D_w=0.1, N=64, L=30.0))
        sim.reset()
        for _ in range(100):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_pure_reaction_no_spreading(self):
        """With D_v=0, initial perturbation should not spread spatially."""
        sim = FHNPulseSimulation(_make_config(D_v=0.0, D_w=0.0, N=64, L=30.0, dt=0.01))
        sim.reset(seed=42)
        v0 = sim.v_field.copy()
        # Identify initially perturbed region
        v_rest = sim._find_rest_v()
        perturbed_0 = np.abs(v0 - v_rest) > 0.01
        n_perturbed_0 = np.sum(perturbed_0)

        for _ in range(50):
            sim.step()

        v1 = sim.v_field
        perturbed_1 = np.abs(v1 - v_rest) > 0.01
        n_perturbed_1 = np.sum(perturbed_1)

        # Without diffusion, the number of perturbed points should not grow much
        # (reaction can modify local values but not spread to new grid points)
        assert n_perturbed_1 <= n_perturbed_0 + 2  # small tolerance


class TestFHNPulseRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_speed_vs_D_v_data_generation(self):
        """Speed vs D_v sweep should produce valid data."""
        from simulating_anything.rediscovery.fhn_pulse import (
            generate_speed_vs_D_v_data,
        )
        data = generate_speed_vs_D_v_data(
            n_D=3, n_steps=300, N=64, L=50.0,
        )
        assert "D_v" in data
        assert "wave_speed" in data
        assert "sqrt_D_v" in data
        assert len(data["D_v"]) >= 1
        assert np.all(data["wave_speed"] >= 0)
        assert np.all(np.isfinite(data["wave_speed"]))

    def test_speed_vs_eps_data_generation(self):
        """Speed vs eps sweep should produce valid data."""
        from simulating_anything.rediscovery.fhn_pulse import (
            generate_speed_vs_eps_data,
        )
        data = generate_speed_vs_eps_data(
            n_eps=3, n_steps=300, N=64, L=50.0,
        )
        assert "eps" in data
        assert "wave_speed" in data
        assert len(data["eps"]) >= 1
        assert np.all(data["wave_speed"] >= 0)

    def test_pulse_propagation_data(self):
        """Propagation data should contain snapshots and positions."""
        from simulating_anything.rediscovery.fhn_pulse import (
            generate_pulse_propagation_data,
        )
        data = generate_pulse_propagation_data(
            D_v=1.0, n_steps=200, N=64, L=50.0, n_snapshots=5,
        )
        assert "snapshots_v" in data
        assert "snapshots_w" in data
        assert "peak_positions" in data
        assert "times" in data
        assert len(data["times"]) >= 1
        assert data["snapshots_v"].shape[1] == 64  # N grid points

    def test_pulse_shape_data(self):
        """Pulse shape data should contain profile and metrics."""
        from simulating_anything.rediscovery.fhn_pulse import (
            generate_pulse_shape_data,
        )
        data = generate_pulse_shape_data(
            D_v=1.0, n_steps=200, N=64, L=50.0,
        )
        assert "v" in data
        assert "w" in data
        assert "v_rest" in data
        assert "v_max" in data
        assert "pulse_width" in data
        assert "pulse_count" in data
        assert len(data["v"]) == 64

    def test_speed_vs_D_v_sqrt_scaling(self):
        """sqrt_D_v array should match sqrt of D_v values."""
        from simulating_anything.rediscovery.fhn_pulse import (
            generate_speed_vs_D_v_data,
        )
        data = generate_speed_vs_D_v_data(
            n_D=3, n_steps=200, N=64, L=50.0,
        )
        np.testing.assert_allclose(
            data["sqrt_D_v"], np.sqrt(data["D_v"]), rtol=1e-10,
        )

    def test_make_config_cfl_safe(self):
        """_make_config should produce CFL-safe dt when dt=None."""
        from simulating_anything.rediscovery.fhn_pulse import _make_config
        config = _make_config(N=64, L=20.0, D_v=5.0)
        dx = 20.0 / 64
        cfl_limit = dx ** 2 / (2.0 * 5.0)
        assert config.dt < cfl_limit

    def test_make_config_domain(self):
        """_make_config should use FHN_PULSE domain."""
        from simulating_anything.rediscovery.fhn_pulse import _make_config
        config = _make_config()
        assert config.domain == Domain.FHN_PULSE
