"""Tests for the 1D Brusselator reaction-diffusion PDE simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.brusselator_1d import Brusselator1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    a: float = 1.0,
    b: float = 3.0,
    D_u: float = 0.01,
    D_v: float = 0.1,
    N: int = 64,
    L: float = 20.0,
    dt: float = 0.005,
    n_steps: int = 100,
) -> SimulationConfig:
    """Build a SimulationConfig for the 1D Brusselator."""
    return SimulationConfig(
        domain=Domain.BRUSSELATOR_1D,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "D_u": D_u,
            "D_v": D_v,
            "N": float(N),
            "L": L,
        },
    )


# ---------------------------------------------------------------------------
# Creation and defaults
# ---------------------------------------------------------------------------

class TestBrusselator1DCreation:
    def test_create_simulation(self):
        sim = Brusselator1DSimulation(_make_config())
        assert sim.N == 64
        assert sim.a == 1.0
        assert sim.b == 3.0
        assert sim.D_u == 0.01
        assert sim.D_v == 0.1
        assert sim.L == 20.0

    def test_default_parameters(self):
        """Defaults match specification when parameters dict is empty."""
        config = SimulationConfig(
            domain=Domain.BRUSSELATOR_1D,
            dt=0.001,
            n_steps=10,
            parameters={},
        )
        sim = Brusselator1DSimulation(config)
        assert sim.a == 1.0
        assert sim.b == 3.0
        assert sim.D_u == 0.01
        assert sim.D_v == 0.1
        assert sim.N == 256
        assert sim.L == 20.0

    def test_custom_parameters(self):
        sim = Brusselator1DSimulation(_make_config(a=2.0, b=6.0, D_u=0.02, D_v=0.2))
        assert sim.a == 2.0
        assert sim.b == 6.0
        assert sim.D_u == 0.02
        assert sim.D_v == 0.2

    def test_grid_spacing(self):
        sim = Brusselator1DSimulation(_make_config(N=100, L=10.0))
        assert sim.dx == pytest.approx(0.1)
        assert len(sim.x) == 100

    def test_cfl_limit_computed(self):
        sim = Brusselator1DSimulation(_make_config(N=64, L=20.0, D_v=0.1))
        dx = 20.0 / 64
        expected_cfl = dx ** 2 / (2.0 * 0.1)
        assert sim.cfl_limit == pytest.approx(expected_cfl)


# ---------------------------------------------------------------------------
# CFL stability
# ---------------------------------------------------------------------------

class TestBrusselator1DCFL:
    def test_cfl_violation_raises(self):
        """dt too large for grid should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            Brusselator1DSimulation(_make_config(dt=10.0, N=64, D_v=0.1))

    def test_just_below_cfl_ok(self):
        """dt just below the CFL limit should not raise."""
        N = 64
        L = 20.0
        D_v = 0.1
        dx = L / N
        dt_cfl = dx ** 2 / (2.0 * D_v)
        sim = Brusselator1DSimulation(_make_config(dt=0.99 * dt_cfl, N=N, D_v=D_v))
        assert sim is not None

    def test_zero_diffusion_no_cfl(self):
        """With D_u=D_v=0, CFL limit should be infinite."""
        sim = Brusselator1DSimulation(_make_config(D_u=0.0, D_v=0.0, dt=1.0))
        assert sim.cfl_limit == float("inf")


# ---------------------------------------------------------------------------
# State shape and initialization
# ---------------------------------------------------------------------------

class TestBrusselator1DState:
    def test_initial_state_shape(self):
        sim = Brusselator1DSimulation(_make_config(N=64))
        state = sim.reset()
        assert state.shape == (128,)  # 2 * N

    def test_initial_state_shape_256(self):
        sim = Brusselator1DSimulation(_make_config(N=256, dt=0.001))
        state = sim.reset()
        assert state.shape == (512,)  # 2 * 256

    def test_step_returns_correct_shape(self):
        sim = Brusselator1DSimulation(_make_config(N=64))
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_observe_matches_state(self):
        sim = Brusselator1DSimulation(_make_config(N=64))
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)
        # First N values are u, last N are v
        assert np.all(obs[:64] >= 0)
        assert np.all(obs[64:] >= 0)

    def test_state_u_v_split(self):
        """State vector has u in first half, v in second half."""
        N = 32
        sim = Brusselator1DSimulation(_make_config(N=N))
        state = sim.reset()
        u_part = state[:N]
        v_part = state[N:]
        np.testing.assert_array_equal(u_part, sim._u)
        np.testing.assert_array_equal(v_part, sim._v)

    def test_initial_near_steady_state(self):
        """Initial u should be near a, initial v should be near b/a."""
        sim = Brusselator1DSimulation(_make_config(a=1.0, b=3.0, N=64))
        sim.reset(seed=42)
        assert np.mean(sim._u) == pytest.approx(1.0, abs=0.1)
        assert np.mean(sim._v) == pytest.approx(3.0, abs=0.1)

    def test_concentrations_positive_after_reset(self):
        sim = Brusselator1DSimulation(_make_config())
        sim.reset()
        assert np.all(sim._u >= 0.01)
        assert np.all(sim._v >= 0.01)


# ---------------------------------------------------------------------------
# Steady state and properties
# ---------------------------------------------------------------------------

class TestBrusselator1DProperties:
    def test_fixed_point(self):
        sim = Brusselator1DSimulation(_make_config(a=2.0, b=5.0))
        u_star, v_star = sim.fixed_point
        assert u_star == pytest.approx(2.0)
        assert v_star == pytest.approx(2.5)  # b/a = 5/2

    def test_fixed_point_a_zero(self):
        sim = Brusselator1DSimulation(_make_config(a=0.0, b=0.0, D_u=0.0, D_v=0.0))
        assert sim.fixed_point == (0.0, 0.0)

    def test_turing_threshold_a1(self):
        sim = Brusselator1DSimulation(_make_config(a=1.0))
        assert sim.turing_threshold == pytest.approx(2.0)

    def test_turing_threshold_a2(self):
        sim = Brusselator1DSimulation(_make_config(a=2.0))
        assert sim.turing_threshold == pytest.approx(5.0)

    def test_is_turing_unstable_true(self):
        sim = Brusselator1DSimulation(_make_config(a=1.0, b=3.0))
        assert sim.is_turing_unstable  # 3.0 > 1 + 1^2 = 2.0

    def test_is_turing_unstable_false(self):
        sim = Brusselator1DSimulation(_make_config(a=1.0, b=1.5))
        assert not sim.is_turing_unstable  # 1.5 < 2.0

    def test_total_u_positive(self):
        sim = Brusselator1DSimulation(_make_config())
        sim.reset()
        assert sim.total_u > 0

    def test_total_u_integral(self):
        sim = Brusselator1DSimulation(_make_config(N=64))
        sim.reset()
        expected = float(np.sum(sim._u) * sim.dx)
        assert sim.total_u == pytest.approx(expected)

    def test_total_v_positive(self):
        sim = Brusselator1DSimulation(_make_config())
        sim.reset()
        assert sim.total_v > 0

    def test_mean_u_near_a(self):
        """After moderate evolution, mean u stays near steady state a."""
        sim = Brusselator1DSimulation(_make_config(a=1.0, b=3.0, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        assert 0.3 < sim.mean_u < 3.0

    def test_mean_v_near_b_over_a(self):
        """After moderate evolution, mean v stays near b/a."""
        sim = Brusselator1DSimulation(_make_config(a=1.0, b=3.0, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        assert 0.5 < sim.mean_v < 10.0

    def test_spatial_heterogeneity_nonneg(self):
        sim = Brusselator1DSimulation(_make_config())
        sim.reset()
        assert sim.spatial_heterogeneity_u >= 0.0
        assert sim.spatial_heterogeneity_v >= 0.0

    def test_u_field_returns_copy(self):
        sim = Brusselator1DSimulation(_make_config())
        sim.reset()
        field = sim.u_field
        field[:] = 999.0
        assert not np.any(sim._u == 999.0)

    def test_v_field_returns_copy(self):
        sim = Brusselator1DSimulation(_make_config())
        sim.reset()
        field = sim.v_field
        field[:] = 999.0
        assert not np.any(sim._v == 999.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestBrusselator1DDeterminism:
    def test_deterministic_reset(self):
        """Same seed produces identical initial states."""
        sim1 = Brusselator1DSimulation(_make_config(N=32))
        sim2 = Brusselator1DSimulation(_make_config(N=32))
        s1 = sim1.reset(seed=123)
        s2 = sim2.reset(seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_deterministic_trajectory(self):
        """Same seed produces identical trajectories."""
        cfg = _make_config(N=32, n_steps=50)
        sim1 = Brusselator1DSimulation(cfg)
        sim2 = Brusselator1DSimulation(cfg)
        t1 = sim1.run(n_steps=50)
        t2 = sim2.run(n_steps=50)
        np.testing.assert_array_equal(t1.states, t2.states)

    def test_different_seeds_differ(self):
        sim = Brusselator1DSimulation(_make_config(N=32))
        s1 = sim.reset(seed=1)
        s2 = sim.reset(seed=2)
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Stability / boundedness
# ---------------------------------------------------------------------------

class TestBrusselator1DStability:
    def test_concentrations_stay_positive(self):
        """u and v remain non-negative throughout the run."""
        sim = Brusselator1DSimulation(_make_config(n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        assert np.all(sim._u >= 0)
        assert np.all(sim._v >= 0)

    def test_trajectory_bounded(self):
        """Concentrations do not blow up over a long run."""
        sim = Brusselator1DSimulation(_make_config(n_steps=2000))
        sim.reset()
        for _ in range(2000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "state contains non-finite values"
            assert np.max(np.abs(state)) < 200, "state diverged"

    def test_no_nan(self):
        """No NaN values appear during simulation."""
        sim = Brusselator1DSimulation(_make_config(n_steps=1000))
        sim.reset()
        for _ in range(1000):
            sim.step()
        assert not np.any(np.isnan(sim._u))
        assert not np.any(np.isnan(sim._v))


# ---------------------------------------------------------------------------
# Turing instability / pattern formation
# ---------------------------------------------------------------------------

class TestBrusselator1DTuring:
    def test_below_turing_stays_homogeneous(self):
        """Below b_c = 1+a^2, patterns should not form."""
        sim = Brusselator1DSimulation(_make_config(
            a=1.0, b=1.5, D_u=0.01, D_v=0.1,
            N=64, n_steps=3000, dt=0.005,
        ))
        sim.reset(seed=42)
        for _ in range(3000):
            sim.step()
        assert sim.spatial_heterogeneity_u < 0.1

    def test_above_turing_patterns_form(self):
        """Above b_c with large D_v/D_u, Turing patterns should appear."""
        sim = Brusselator1DSimulation(_make_config(
            a=1.0, b=4.0, D_u=0.005, D_v=0.2,
            N=128, L=30.0, n_steps=20000, dt=0.002,
        ))
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        # Expect clear spatial pattern (high heterogeneity)
        assert sim.spatial_heterogeneity_u > 0.01

    def test_pattern_has_finite_wavelength(self):
        """Turing patterns should produce a measurable dominant wavelength."""
        sim = Brusselator1DSimulation(_make_config(
            a=1.0, b=4.0, D_u=0.005, D_v=0.2,
            N=128, L=30.0, n_steps=20000, dt=0.002,
        ))
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        wl = sim.dominant_wavelength()
        assert np.isfinite(wl)
        assert 0 < wl < sim.L


# ---------------------------------------------------------------------------
# Wavelength and peak detection
# ---------------------------------------------------------------------------

class TestBrusselator1DWavelength:
    def test_dominant_wavelength_synthetic(self):
        """Injecting a sinusoidal pattern should give the correct wavelength."""
        sim = Brusselator1DSimulation(_make_config(N=128, L=20.0))
        sim.reset()
        k_target = 4
        sim._u = sim.a + 0.5 * np.sin(2 * np.pi * k_target * sim.x / sim.L)
        sim._state = np.concatenate([sim._u, sim._v])
        wl = sim.dominant_wavelength()
        expected = sim.L / k_target
        assert wl == pytest.approx(expected, rel=0.01)

    def test_dominant_wavelength_uniform_is_inf(self):
        """Uniform field should return infinite wavelength."""
        sim = Brusselator1DSimulation(_make_config(N=64))
        sim.reset()
        sim._u = np.ones(64) * sim.a
        sim._state = np.concatenate([sim._u, sim._v])
        wl = sim.dominant_wavelength()
        assert wl == float("inf")

    def test_count_peaks_synthetic(self):
        """Injecting a known number of peaks."""
        sim = Brusselator1DSimulation(_make_config(N=256, L=20.0))
        sim.reset()
        n_peaks = 5
        sim._u = sim.a + 0.5 * np.sin(2 * np.pi * n_peaks * sim.x / sim.L)
        # The sin function produces exactly n_peaks peaks in [0, L)
        count = sim.count_peaks("u", threshold=sim.a)
        assert count == n_peaks


# ---------------------------------------------------------------------------
# Diffusion behavior
# ---------------------------------------------------------------------------

class TestBrusselator1DDiffusion:
    def test_no_diffusion_reduces_to_ode(self):
        """With D_u=D_v=0, each grid point evolves as independent ODE."""
        config = _make_config(
            D_u=0.0, D_v=0.0, dt=0.005, n_steps=50, N=8,
            a=1.0, b=3.0,
        )
        sim = Brusselator1DSimulation(config)
        sim.reset(seed=42)

        u0 = sim._u[0]
        v0 = sim._v[0]

        for _ in range(50):
            sim.step()

        u_pde = sim._u[0]
        v_pde = sim._v[0]

        # Manual Euler integration of Brusselator ODE
        dt = 0.005
        u, v = u0, v0
        a, b = sim.a, sim.b
        for _ in range(50):
            u2v = u ** 2 * v
            du = a - (b + 1) * u + u2v
            dv = b * u - u2v
            u = max(u + dt * du, 0.0)
            v = max(v + dt * dv, 0.0)

        np.testing.assert_allclose(u_pde, u, rtol=0.02)
        np.testing.assert_allclose(v_pde, v, rtol=0.02)

    def test_diffusion_smooths_sharp_gradient(self):
        """High diffusion should smooth a step function."""
        config = _make_config(
            a=0.0, b=0.0, D_u=0.5, D_v=0.5,
            dt=0.0005, n_steps=100, N=64, L=20.0,
        )
        sim = Brusselator1DSimulation(config)
        sim.reset(seed=42)

        # Set a sharp step
        sim._u[:32] = 5.0
        sim._u[32:] = 1.0
        sim._state = np.concatenate([sim._u, sim._v])

        grad_initial = np.max(np.abs(np.diff(sim._u)))

        for _ in range(100):
            sim.step()

        grad_final = np.max(np.abs(np.diff(sim._u)))
        assert grad_final < grad_initial


# ---------------------------------------------------------------------------
# TrajectoryData
# ---------------------------------------------------------------------------

class TestBrusselator1DTrajectory:
    def test_run_trajectory_shape(self):
        sim = Brusselator1DSimulation(_make_config(N=32, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 64)  # 50+1 timesteps, 2*32

    def test_run_trajectory_finite(self):
        sim = Brusselator1DSimulation(_make_config(N=32, n_steps=100))
        traj = sim.run(n_steps=100)
        assert np.all(np.isfinite(traj.states))

    def test_trajectory_timestamps(self):
        sim = Brusselator1DSimulation(_make_config(N=32, n_steps=10, dt=0.005))
        traj = sim.run(n_steps=10)
        assert len(traj.timestamps) == 11
        assert traj.timestamps[0] == pytest.approx(0.0)
        assert traj.timestamps[-1] == pytest.approx(10 * 0.005)

    def test_trajectory_parameters_stored(self):
        sim = Brusselator1DSimulation(_make_config(a=1.5, b=4.0))
        traj = sim.run(n_steps=10)
        assert traj.parameters["a"] == pytest.approx(1.5)
        assert traj.parameters["b"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------

class TestBrusselator1DRediscoveryData:
    def test_wavelength_data_generation(self):
        from simulating_anything.rediscovery.brusselator_1d import (
            generate_wavelength_data,
        )
        data = generate_wavelength_data(
            n_Dv=3, n_steps=100, dt=0.005, N=64, L=20.0,
            a=1.0, b=3.5,
        )
        assert len(data["D_v"]) == 3
        assert len(data["wavelength"]) == 3
        assert len(data["wavelength_theory"]) == 3
        assert data["D_u"] == 0.01

    def test_instability_onset_data_generation(self):
        from simulating_anything.rediscovery.brusselator_1d import (
            generate_instability_onset_data,
        )
        data = generate_instability_onset_data(
            n_b=5, n_steps=100, dt=0.005, N=32, L=10.0,
        )
        assert len(data["b"]) == 5
        assert len(data["heterogeneity"]) == 5
        assert len(data["patterned"]) == 5
        assert data["b_c_theory"] == pytest.approx(2.0)

    def test_phase_diagram_data_generation(self):
        from simulating_anything.rediscovery.brusselator_1d import (
            generate_pattern_phase_data,
        )
        data = generate_pattern_phase_data(
            n_b=3, n_Dv=2, n_steps=100, dt=0.005, N=32, L=10.0,
        )
        assert len(data["b"]) == 6  # 3 * 2
        assert len(data["heterogeneity"]) == 6
        assert data["b_c_theory"] == pytest.approx(2.0)

    def test_wavelength_data_requires_turing(self):
        """b below threshold should raise ValueError."""
        from simulating_anything.rediscovery.brusselator_1d import (
            generate_wavelength_data,
        )
        with pytest.raises(ValueError, match="must be"):
            generate_wavelength_data(b=1.5, a=1.0)  # b < 1+a^2=2


# ---------------------------------------------------------------------------
# Parameter sweep edge cases
# ---------------------------------------------------------------------------

class TestBrusselator1DParameterEdges:
    def test_small_a(self):
        """Small a should still produce a valid simulation."""
        sim = Brusselator1DSimulation(_make_config(a=0.1, b=1.5, N=32))
        traj = sim.run(n_steps=50)
        assert np.all(np.isfinite(traj.states))

    def test_large_b(self):
        """Large b should not blow up (within CFL)."""
        sim = Brusselator1DSimulation(_make_config(
            a=1.0, b=10.0, N=64, dt=0.005, n_steps=200,
        ))
        sim.reset()
        for _ in range(200):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_equal_diffusion_below_hopf(self):
        """D_u == D_v below Hopf threshold: no Turing patterns, stays homogeneous."""
        # b=1.5 < b_c=2.0 so no oscillation AND no Turing instability
        sim = Brusselator1DSimulation(_make_config(
            a=1.0, b=1.5, D_u=0.05, D_v=0.05,
            N=64, n_steps=5000, dt=0.002,
        ))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
        # Equal diffusion below Hopf: fully stable homogeneous state
        assert sim.spatial_heterogeneity_u < 0.1
