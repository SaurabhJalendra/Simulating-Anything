"""Tests for the 1D Brusselator-Diffusion PDE simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.brusselator_diffusion import BrusselatorDiffusion
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 1.0,
    b: float = 3.0,
    D_u: float = 0.01,
    D_v: float = 0.1,
    N_grid: int = 64,
    L_domain: float = 20.0,
    dt: float = 0.01,
    n_steps: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BRUSSELATOR_DIFFUSION,
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


class TestBrusselatorDiffusionCreation:
    def test_create_simulation(self):
        sim = BrusselatorDiffusion(_make_config())
        assert sim.N == 64
        assert sim.a == 1.0
        assert sim.b == 3.0
        assert sim.D_u == 0.01
        assert sim.D_v == 0.1

    def test_default_parameters(self):
        """Defaults match the specification."""
        config = SimulationConfig(
            domain=Domain.BRUSSELATOR_DIFFUSION,
            dt=0.001,
            n_steps=10,
            parameters={},
        )
        sim = BrusselatorDiffusion(config)
        assert sim.a == 1.0
        assert sim.b == 3.0
        assert sim.D_u == 0.01
        assert sim.D_v == 0.1
        assert sim.N == 128
        assert sim.L == 20.0

    def test_cfl_violation_raises(self):
        """dt too large for grid spacing should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            # dt=1.0 far exceeds CFL for dx=20/64~0.3125, D_v=0.1
            # CFL: dt < dx^2/(4*D_v) ~ 0.244
            BrusselatorDiffusion(_make_config(dt=1.0))

    def test_fixed_point(self):
        sim = BrusselatorDiffusion(_make_config(a=2.0, b=5.0))
        u_star, v_star = sim.fixed_point
        assert u_star == pytest.approx(2.0)
        assert v_star == pytest.approx(2.5)  # b/a = 5/2


class TestBrusselatorDiffusionState:
    def test_initial_state_shape(self):
        sim = BrusselatorDiffusion(_make_config(N_grid=64))
        state = sim.reset()
        assert state.shape == (128,)  # 2 * N_grid = 2 * 64

    def test_step_returns_correct_shape(self):
        sim = BrusselatorDiffusion(_make_config(N_grid=64))
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_observe_matches_state(self):
        sim = BrusselatorDiffusion(_make_config(N_grid=64))
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)
        # First N values are u, last N are v
        assert np.all(obs[:64] >= 0)
        assert np.all(obs[64:] >= 0)

    def test_concentrations_positive(self):
        """Concentrations should remain non-negative."""
        sim = BrusselatorDiffusion(_make_config(n_steps=200))
        sim.reset()
        for _ in range(200):
            sim.step()
        assert np.all(sim._u >= 0)
        assert np.all(sim._v >= 0)


class TestBrusselatorDiffusionProperties:
    def test_total_u_property(self):
        sim = BrusselatorDiffusion(_make_config())
        sim.reset()
        assert sim.total_u > 0
        expected = float(np.sum(sim._u) * sim.dx)
        assert sim.total_u == pytest.approx(expected)

    def test_total_v_property(self):
        sim = BrusselatorDiffusion(_make_config())
        sim.reset()
        assert sim.total_v > 0

    def test_mean_u_near_steady_state(self):
        """Mean u should stay near a for moderate runs."""
        sim = BrusselatorDiffusion(_make_config(a=1.0, b=3.0, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        # Mean u should be approximately a=1.0 (within 50%)
        assert 0.5 < sim.mean_u < 2.0

    def test_mean_v_near_steady_state(self):
        """Mean v should stay near b/a for moderate runs."""
        sim = BrusselatorDiffusion(_make_config(a=1.0, b=3.0, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        # Mean v should be approximately b/a=3.0 (within factor of 2)
        assert 1.0 < sim.mean_v < 6.0

    def test_spatial_heterogeneity_u(self):
        sim = BrusselatorDiffusion(_make_config())
        sim.reset()
        het = sim.spatial_heterogeneity_u
        assert het >= 0.0

    def test_spatial_heterogeneity_v(self):
        sim = BrusselatorDiffusion(_make_config())
        sim.reset()
        het = sim.spatial_heterogeneity_v
        assert het >= 0.0

    def test_u_field_returns_copy(self):
        sim = BrusselatorDiffusion(_make_config())
        sim.reset()
        field = sim.u_field
        field[:] = 999.0
        # Original should be unchanged
        assert np.all(sim._u != 999.0)

    def test_turing_threshold(self):
        sim1 = BrusselatorDiffusion(_make_config(a=1.0))
        assert sim1.turing_threshold == pytest.approx(2.0)

        sim2 = BrusselatorDiffusion(_make_config(a=2.0))
        assert sim2.turing_threshold == pytest.approx(5.0)

    def test_is_turing_unstable_flag(self):
        # b=3.0 > 1+1^2=2.0: unstable
        sim_unstable = BrusselatorDiffusion(_make_config(a=1.0, b=3.0))
        assert sim_unstable.is_turing_unstable

        # b=1.5 < 1+1^2=2.0: stable
        sim_stable = BrusselatorDiffusion(_make_config(a=1.0, b=1.5))
        assert not sim_stable.is_turing_unstable


class TestBrusselatorDiffusionStability:
    def test_below_turing_stays_homogeneous(self):
        """Below Turing threshold (b < 1+a^2), patterns should not form."""
        sim = BrusselatorDiffusion(_make_config(
            a=1.0, b=1.5, D_u=0.01, D_v=0.1,
            N_grid=64, n_steps=2000, dt=0.01,
        ))
        sim.reset(seed=42)
        for _ in range(2000):
            sim.step()
        # Heterogeneity should remain low
        assert sim.spatial_heterogeneity_u < 0.1

    def test_above_turing_patterns_form(self):
        """Above Turing threshold with large D_v/D_u, patterns should form."""
        sim = BrusselatorDiffusion(_make_config(
            a=1.0, b=4.0, D_u=0.005, D_v=0.2,
            N_grid=128, L_domain=30.0, n_steps=15000, dt=0.005,
        ))
        sim.reset(seed=42)
        for _ in range(15000):
            sim.step()
        # With b=4 >> b_c=2 and D_v/D_u=40, strong Turing patterns expected
        assert sim.spatial_heterogeneity_u > 0.01

    def test_trajectory_bounded(self):
        """Concentrations should not blow up."""
        sim = BrusselatorDiffusion(_make_config(
            a=1.0, b=3.0, n_steps=1000, dt=0.01,
        ))
        sim.reset()
        for _ in range(1000):
            sim.step()
            assert np.all(np.isfinite(sim._u)), "u contains non-finite values"
            assert np.all(np.isfinite(sim._v)), "v contains non-finite values"
            assert np.max(sim._u) < 100, f"u diverged: max={np.max(sim._u)}"
            assert np.max(sim._v) < 100, f"v diverged: max={np.max(sim._v)}"


class TestBrusselatorDiffusionDiffusion:
    def test_no_diffusion_reduces_to_ode(self):
        """With D_u=D_v=0, each grid point evolves as an independent Brusselator."""
        config = _make_config(
            D_u=0.0, D_v=0.0, dt=0.005, n_steps=50, N_grid=8,
            a=1.0, b=3.0,
        )
        sim = BrusselatorDiffusion(config)
        sim.reset(seed=42)

        # Store initial values at point 0
        u0 = sim._u[0]
        v0 = sim._v[0]

        # Run the PDE simulation
        for _ in range(50):
            sim.step()

        u_pde = sim._u[0]
        v_pde = sim._v[0]

        # Run a pure ODE (Euler) for same initial conditions
        dt = 0.005
        u, v = u0, v0
        a, b = sim.a, sim.b
        for _ in range(50):
            u2v = u ** 2 * v
            du = a - (b + 1) * u + u2v
            dv = b * u - u2v
            u = max(u + dt * du, 0.0)
            v = max(v + dt * dv, 0.0)

        # Should match within small tolerance (both using Euler)
        np.testing.assert_allclose(u_pde, u, rtol=0.02)
        np.testing.assert_allclose(v_pde, v, rtol=0.02)

    def test_diffusion_smooths_gradients(self):
        """With high diffusion and no reaction, gradients should decrease."""
        config = _make_config(
            a=0.0, b=0.0, D_u=0.5, D_v=0.5,
            dt=0.001, n_steps=100, N_grid=64,
        )
        sim = BrusselatorDiffusion(config)
        sim.reset(seed=42)

        # Set a sharp step function
        sim._u[:32] = 5.0
        sim._u[32:] = 1.0
        sim._state = np.concatenate([sim._u, sim._v])

        grad_initial = np.max(np.abs(np.diff(sim._u)))

        for _ in range(100):
            sim.step()

        grad_final = np.max(np.abs(np.diff(sim._u)))
        assert grad_final < grad_initial


class TestBrusselatorDiffusionTrajectory:
    def test_run_trajectory(self):
        """The run() method should produce a valid TrajectoryData."""
        sim = BrusselatorDiffusion(_make_config(N_grid=32, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 64)  # 50 steps + initial, 2*32=64
        assert np.all(np.isfinite(traj.states))

    def test_dominant_wavelength(self):
        """dominant_wavelength should return a finite positive value."""
        sim = BrusselatorDiffusion(_make_config(N_grid=64))
        sim.reset()
        # Inject a sinusoidal pattern
        k_target = 3  # 3 full wavelengths in domain
        sim._u = sim.a + 0.5 * np.sin(2 * np.pi * k_target * sim.x / sim.L)
        sim._state = np.concatenate([sim._u, sim._v])
        wl = sim.dominant_wavelength()
        expected_wl = sim.L / k_target
        assert wl == pytest.approx(expected_wl, rel=0.01)


class TestBrusselatorDiffusionRediscovery:
    def test_pattern_data_generation(self):
        from simulating_anything.rediscovery.brusselator_diffusion import (
            generate_pattern_data,
        )

        data = generate_pattern_data(
            n_b=3, n_Dv=2, n_steps=100, dt=0.01, N_grid=32, L_domain=10.0,
        )
        assert len(data["b"]) == 6  # 3 * 2
        assert len(data["heterogeneity_u"]) == 6
        assert np.all(data["heterogeneity_u"] >= 0)
        assert data["b_c_theory"] == pytest.approx(2.0)

    def test_wavelength_data_generation(self):
        from simulating_anything.rediscovery.brusselator_diffusion import (
            generate_wavelength_data,
        )

        data = generate_wavelength_data(
            n_Dv=3, n_steps=100, dt=0.005, N_grid=64, L_domain=20.0,
            a=1.0, b=3.5,
        )
        assert len(data["D_v"]) == 3
        assert len(data["wavelength"]) == 3
        assert len(data["wavelength_theory"]) == 3
        assert data["D_u"] == 0.01
