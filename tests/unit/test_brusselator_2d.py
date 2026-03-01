"""Tests for the 2D Brusselator reaction-diffusion simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.brusselator_2d import Brusselator2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 4.5,
    b: float = 7.0,
    D_u: float = 1.0,
    D_v: float = 8.0,
    N_grid: int = 32,
    L_domain: float = 32.0,
    dt: float = 0.01,
    n_steps: int = 100,
) -> SimulationConfig:
    """Build a config for 2D Brusselator tests."""
    return SimulationConfig(
        domain=Domain.BRUSSELATOR_2D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "D_u": D_u,
            "D_v": D_v,
            "N_grid": float(N_grid),
            "L_domain": L_domain,
        },
    )


class TestBrusselator2DSimulation:
    """Core simulation tests."""

    def test_initial_state_shape(self):
        """State should have shape (2*N*N,) for u and v fields."""
        N = 32
        sim = Brusselator2DSimulation(_make_config(N_grid=N))
        state = sim.reset()
        assert state.shape == (2 * N * N,)

    def test_initial_state_shape_different_grid(self):
        """Verify state shape scales with grid resolution."""
        N = 16
        sim = Brusselator2DSimulation(_make_config(N_grid=N, L_domain=16.0))
        state = sim.reset()
        assert state.shape == (2 * N * N,)

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = Brusselator2DSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """observe() should return the same state as the last step."""
        sim = Brusselator2DSimulation(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (2 * 32 * 32,)

    def test_run_trajectory(self):
        """run() should return a trajectory with correct number of steps."""
        n_steps = 50
        sim = Brusselator2DSimulation(_make_config(n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        # n_steps + 1 because initial state is included
        assert traj.states.shape[0] == n_steps + 1
        assert traj.states.shape[1] == 2 * 32 * 32

    def test_fields_accessible(self):
        """u_field and v_field should have shape (N, N)."""
        N = 32
        sim = Brusselator2DSimulation(_make_config(N_grid=N))
        sim.reset()
        assert sim.u_field.shape == (N, N)
        assert sim.v_field.shape == (N, N)

    def test_initial_values_near_fixed_point(self):
        """Initial fields should be near the homogeneous steady state."""
        a, b = 4.5, 7.0
        sim = Brusselator2DSimulation(_make_config(a=a, b=b))
        sim.reset()
        u_star, v_star = a, b / a
        np.testing.assert_allclose(sim.mean_u, u_star, atol=0.1)
        np.testing.assert_allclose(sim.mean_v, v_star, atol=0.1)

    def test_positive_concentrations(self):
        """Concentrations should remain non-negative after steps."""
        sim = Brusselator2DSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        assert np.all(sim.u_field >= 0.0)
        assert np.all(sim.v_field >= 0.0)


class TestBrusselator2DPhysics:
    """Physical property tests."""

    def test_fixed_point(self):
        """Fixed point should be (a, b/a)."""
        a, b = 4.5, 7.0
        sim = Brusselator2DSimulation(_make_config(a=a, b=b))
        u_star, v_star = sim.fixed_point
        assert u_star == pytest.approx(a)
        assert v_star == pytest.approx(b / a)

    def test_turing_threshold_b(self):
        """Turing threshold: b_c = 1 + a^2."""
        sim = Brusselator2DSimulation(_make_config(a=4.5))
        assert sim.turing_threshold_b == pytest.approx(1.0 + 4.5**2)

        sim2 = Brusselator2DSimulation(_make_config(a=2.0))
        assert sim2.turing_threshold_b == pytest.approx(5.0)

    def test_turing_threshold_diffusion_ratio(self):
        """Diffusion ratio threshold: D_v/D_u > ((a+1)/(a-1))^2."""
        sim = Brusselator2DSimulation(_make_config(a=4.5))
        expected = ((4.5 + 1) / (4.5 - 1)) ** 2
        assert sim.turing_threshold_diffusion_ratio == pytest.approx(expected)

    def test_turing_threshold_diffusion_ratio_a_le_1(self):
        """For a <= 1, diffusion ratio threshold should be infinity."""
        sim = Brusselator2DSimulation(_make_config(a=1.0, b=3.0))
        assert sim.turing_threshold_diffusion_ratio == float("inf")

    def test_is_turing_unstable_true(self):
        """Default parameters (a=4.5, b=7.0 < 1+a^2=21.25) -> not unstable."""
        # b=7 < 1 + 4.5^2 = 21.25, so NOT Turing unstable
        sim = Brusselator2DSimulation(_make_config(a=4.5, b=7.0))
        assert not sim.is_turing_unstable

    def test_is_turing_unstable_when_above_threshold(self):
        """When b > 1+a^2 and D_v/D_u large enough, should be Turing unstable."""
        a = 4.5
        b = 25.0  # > 1 + 4.5^2 = 21.25
        D_u = 1.0
        D_v = 20.0  # D_v/D_u = 20 >> ((4.5+1)/(4.5-1))^2 = 2.47
        sim = Brusselator2DSimulation(_make_config(a=a, b=b, D_u=D_u, D_v=D_v))
        assert sim.is_turing_unstable

    def test_theoretical_wavelength(self):
        """Theoretical wavelength: lambda = 2*pi*sqrt(D_v/(b-1-a^2))."""
        a, b, D_v = 4.5, 25.0, 8.0
        sim = Brusselator2DSimulation(_make_config(a=a, b=b, D_v=D_v))
        gap = b - 1.0 - a ** 2
        expected = 2.0 * np.pi * np.sqrt(D_v / gap)
        assert sim.theoretical_wavelength == pytest.approx(expected)

    def test_theoretical_wavelength_below_threshold(self):
        """Below Turing threshold, theoretical wavelength is infinity."""
        sim = Brusselator2DSimulation(_make_config(a=4.5, b=5.0))
        assert sim.theoretical_wavelength == float("inf")


class TestBrusselator2DCFL:
    """CFL stability condition tests."""

    def test_cfl_limit_computed(self):
        """CFL limit should be dx^2 / (4*D_max)."""
        N, L, D_v = 32, 32.0, 8.0
        dx = L / N
        expected_cfl = dx ** 2 / (4.0 * D_v)
        sim = Brusselator2DSimulation(_make_config(
            N_grid=N, L_domain=L, D_v=D_v, dt=expected_cfl * 0.5
        ))
        assert sim.dt_cfl == pytest.approx(expected_cfl)

    def test_cfl_violation_raises(self):
        """dt exceeding CFL limit should raise ValueError."""
        N, L, D_v = 32, 32.0, 8.0
        dx = L / N
        dt_cfl = dx ** 2 / (4.0 * D_v)
        with pytest.raises(ValueError, match="exceeds CFL limit"):
            Brusselator2DSimulation(_make_config(
                N_grid=N, L_domain=L, D_v=D_v, dt=dt_cfl * 1.5
            ))

    def test_cfl_just_below_limit_ok(self):
        """dt just below CFL limit should be accepted."""
        N, L, D_v = 32, 32.0, 8.0
        dx = L / N
        dt_cfl = dx ** 2 / (4.0 * D_v)
        dt = dt_cfl * 0.99
        sim = Brusselator2DSimulation(_make_config(
            N_grid=N, L_domain=L, D_v=D_v, dt=dt
        ))
        assert sim._step_count == 0  # Created successfully


class TestBrusselator2DWavelength:
    """Pattern wavelength measurement tests."""

    def test_compute_pattern_wavelength_uniform(self):
        """For uniform field (no pattern), wavelength should be infinity."""
        sim = Brusselator2DSimulation(_make_config())
        # Reset with seed for reproducibility, but use very small noise
        sim._u = np.full((32, 32), 4.5, dtype=np.float64)
        sim._v = np.full((32, 32), 7.0 / 4.5, dtype=np.float64)
        sim._state = np.concatenate([sim._u.ravel(), sim._v.ravel()])

        wl = sim.compute_pattern_wavelength()
        assert wl == float("inf")

    def test_compute_pattern_wavelength_sinusoidal(self):
        """For a known sinusoidal pattern, wavelength should match."""
        N = 64
        L = 64.0
        dx = L / N
        sim = Brusselator2DSimulation(_make_config(N_grid=N, L_domain=L))
        sim.reset()

        # Create a sinusoidal pattern with known wavelength
        x = np.arange(N) * dx
        target_wl = L / 4.0  # 4 wavelengths in the domain
        k_target = 2.0 * np.pi / target_wl
        pattern = np.sin(k_target * x)
        sim._u = np.outer(pattern, np.ones(N)) + 4.5
        sim._state = np.concatenate([sim._u.ravel(), sim._v.ravel()])

        measured_wl = sim.compute_pattern_wavelength()
        np.testing.assert_allclose(measured_wl, target_wl, rtol=0.01)

    def test_radial_power_spectrum_shape(self):
        """Radial power spectrum should return arrays of correct length."""
        N = 32
        sim = Brusselator2DSimulation(_make_config(N_grid=N))
        sim.reset()
        k_bins, power = sim.compute_radial_power_spectrum()
        assert len(k_bins) == N // 2
        assert len(power) == N // 2


class TestBrusselator2DSpatialHeterogeneity:
    """Spatial heterogeneity (pattern detection) tests."""

    def test_heterogeneity_uniform_field(self):
        """Uniform field should have zero heterogeneity."""
        sim = Brusselator2DSimulation(_make_config())
        sim._u = np.full((32, 32), 4.5, dtype=np.float64)
        assert sim._spatial_heterogeneity(sim._u) == pytest.approx(0.0, abs=1e-15)

    def test_heterogeneity_noisy_field(self):
        """Noisy field should have positive heterogeneity."""
        sim = Brusselator2DSimulation(_make_config())
        rng = np.random.default_rng(42)
        sim._u = 4.5 + 0.5 * rng.standard_normal((32, 32))
        het = sim._spatial_heterogeneity(sim._u)
        assert het > 0.01

    def test_spatial_heterogeneity_u_v_accessible(self):
        """spatial_heterogeneity_u and _v should be accessible after reset."""
        sim = Brusselator2DSimulation(_make_config())
        sim.reset()
        assert isinstance(sim.spatial_heterogeneity_u, float)
        assert isinstance(sim.spatial_heterogeneity_v, float)

    def test_heterogeneity_near_zero_mean(self):
        """Field with near-zero mean should return 0 heterogeneity."""
        sim = Brusselator2DSimulation(_make_config())
        sim._u = np.full((32, 32), 1e-20, dtype=np.float64)
        assert sim._spatial_heterogeneity(sim._u) == 0.0


class TestBrusselator2DStability:
    """Numerical stability tests."""

    def test_bounded_after_many_steps(self):
        """Solution should remain bounded after many time steps."""
        sim = Brusselator2DSimulation(_make_config(n_steps=500))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        assert np.all(np.isfinite(sim.u_field))
        assert np.all(np.isfinite(sim.v_field))
        assert np.max(sim.u_field) < 1000.0
        assert np.max(sim.v_field) < 1000.0

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical trajectories."""
        config = _make_config()
        sim1 = Brusselator2DSimulation(config)
        sim1.reset(seed=123)
        for _ in range(50):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = Brusselator2DSimulation(config)
        sim2.reset(seed=123)
        for _ in range(50):
            sim2.step()
        state2 = sim2.observe()

        np.testing.assert_array_equal(state1, state2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different trajectories."""
        config = _make_config()
        sim1 = Brusselator2DSimulation(config)
        sim1.reset(seed=1)
        for _ in range(50):
            sim1.step()

        sim2 = Brusselator2DSimulation(config)
        sim2.reset(seed=2)
        for _ in range(50):
            sim2.step()

        assert not np.allclose(sim1.observe(), sim2.observe())


class TestBrusselator2DRediscovery:
    """Rediscovery data generation tests."""

    def test_pattern_data_generation(self):
        """Pattern data generation should return valid results."""
        from simulating_anything.rediscovery.brusselator_2d import (
            generate_pattern_data,
        )
        data = generate_pattern_data(
            n_b=3, n_Dv=2, n_steps=100, N_grid=16, L_domain=16.0,
        )
        assert len(data["b"]) == 6  # 3 * 2
        assert len(data["D_v"]) == 6
        assert len(data["heterogeneity_u"]) == 6
        assert len(data["patterned"]) == 6
        assert data["b_c_theory"] == pytest.approx(1.0 + 4.5**2)

    def test_wavelength_data_generation(self):
        """Wavelength data generation should return valid results."""
        from simulating_anything.rediscovery.brusselator_2d import (
            generate_wavelength_data,
        )
        data = generate_wavelength_data(
            n_Dv=3, n_steps=100, N_grid=32, L_domain=32.0,
            a=4.5, b=25.0,
        )
        assert len(data["D_v"]) == 3
        assert len(data["wavelength"]) == 3
        assert len(data["wavelength_theory"]) == 3
        assert data["a"] == 4.5
        assert data["b"] == 25.0

    def test_wavelength_data_rejects_subcritical(self):
        """Should reject b below Turing threshold."""
        from simulating_anything.rediscovery.brusselator_2d import (
            generate_wavelength_data,
        )
        with pytest.raises(ValueError, match="must be"):
            generate_wavelength_data(
                n_Dv=3, n_steps=100, N_grid=16, L_domain=16.0,
                a=4.5, b=5.0,  # 5.0 < 1 + 4.5^2 = 21.25
            )
