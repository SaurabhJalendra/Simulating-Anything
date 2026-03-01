"""Tests for the 1D diffusive Lotka-Volterra simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.diffusive_lv import DiffusiveLotkaVolterra
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    delta: float = 0.2,
    D_u: float = 0.1,
    D_v: float = 0.05,
    N_grid: int = 64,
    L_domain: float = 20.0,
    dt: float = 0.005,
    n_steps: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.DIFFUSIVE_LV,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "D_u": D_u,
            "D_v": D_v,
            "N_grid": float(N_grid),
            "L_domain": L_domain,
        },
    )


class TestDiffusiveLVCreation:
    def test_create_simulation(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        assert sim.N == 64
        assert sim.alpha == 1.0
        assert sim.D_u == 0.1

    def test_cfl_violation_raises(self):
        """dt too large for grid spacing should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            # dt=1.0 is way too large for dx=20/64~0.3125, D_u=0.1
            # CFL: dt < dx^2/(4*D_u) = 0.244
            DiffusiveLotkaVolterra(_make_config(dt=1.0))


class TestDiffusiveLVDynamics:
    def test_initial_state_shape(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        state = sim.reset()
        assert state.shape == (128,)  # 2 * N_grid = 2 * 64

    def test_step_returns_correct_shape(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_observe_matches_state(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)
        # First N values are prey, last N are predator
        assert np.all(obs[:64] >= 0)
        assert np.all(obs[64:] >= 0)

    def test_prey_predator_nonnegative(self):
        """Populations should remain non-negative."""
        sim = DiffusiveLotkaVolterra(_make_config(n_steps=200))
        sim.reset()
        for _ in range(200):
            sim.step()
        assert np.all(sim._u >= 0)
        assert np.all(sim._v >= 0)

    def test_total_biomass_bounded(self):
        """Total biomass should remain bounded (not blow up)."""
        sim = DiffusiveLotkaVolterra(_make_config(n_steps=500))
        sim.reset()
        initial_biomass = sim.total_biomass
        for _ in range(500):
            sim.step()
        final_biomass = sim.total_biomass
        # Biomass should not grow by more than 10x (sanity check)
        assert final_biomass < 10 * initial_biomass + 1.0

    def test_total_prey_property(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        sim.reset()
        assert sim.total_prey > 0
        # total_prey = sum(u) * dx, should be positive
        expected = float(np.sum(sim._u) * sim.dx)
        assert sim.total_prey == pytest.approx(expected)

    def test_total_predator_property(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        sim.reset()
        assert sim.total_predator > 0

    def test_total_biomass_is_sum(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        sim.reset()
        assert sim.total_biomass == pytest.approx(
            sim.total_prey + sim.total_predator
        )


class TestDiffusiveLVDiffusion:
    def test_diffusion_smooths_gradients(self):
        """With high diffusion and no reaction, gradients should decrease."""
        # Use very high diffusion, very small reaction
        config = _make_config(
            alpha=0.0, beta=0.0, gamma=0.0, delta=0.0,
            D_u=1.0, D_v=1.0, dt=0.001, n_steps=100,
        )
        sim = DiffusiveLotkaVolterra(config)
        sim.reset(seed=42)

        # Set a sharp step function as initial prey
        sim._u[:32] = 5.0
        sim._u[32:] = 1.0
        sim._state = np.concatenate([sim._u, sim._v])

        grad_initial = np.max(np.abs(np.diff(sim._u)))

        for _ in range(100):
            sim.step()

        grad_final = np.max(np.abs(np.diff(sim._u)))
        assert grad_final < grad_initial

    def test_no_diffusion_reduces_to_ode(self):
        """With D_u=D_v=0, each grid point evolves as independent LV ODE."""
        config_pde = _make_config(
            D_u=0.0, D_v=0.0, dt=0.005, n_steps=50, N_grid=8,
        )
        sim = DiffusiveLotkaVolterra(config_pde)
        sim.reset(seed=42)

        # Store initial values at point 0
        u0 = sim._u[0]
        v0 = sim._v[0]

        # Run the PDE simulation
        for _ in range(50):
            sim.step()

        u_pde = sim._u[0]
        v_pde = sim._v[0]

        # Run a pure ODE (manual RK4-like) for same initial conditions
        # Using Euler steps to match the PDE solver
        dt = 0.005
        u, v = u0, v0
        alpha, beta, gamma, delta = sim.alpha, sim.beta, sim.gamma, sim.delta
        for _ in range(50):
            du = alpha * u - beta * u * v
            dv = -gamma * v + delta * u * v
            u = max(u + dt * du, 0.0)
            v = max(v + dt * dv, 0.0)

        # Should match within ~1% (Euler vs split step)
        np.testing.assert_allclose(u_pde, u, rtol=0.02)
        np.testing.assert_allclose(v_pde, v, rtol=0.02)

    def test_periodic_boundary(self):
        """FFT-based solver should handle periodic boundaries."""
        config = _make_config(D_u=0.5, D_v=0.5, dt=0.001, n_steps=50, N_grid=32)
        sim = DiffusiveLotkaVolterra(config)
        sim.reset(seed=42)

        # Put all prey concentration at the last grid point
        sim._u[:] = 0.1
        sim._u[-1] = 10.0
        sim._state = np.concatenate([sim._u, sim._v])

        for _ in range(50):
            sim.step()

        # Due to periodic BCs, the concentration should have spread
        # to the first grid point (wrapping around)
        assert sim._u[0] > 0.2  # Should have received diffusion from u[-1]


class TestDiffusiveLVPatterns:
    def test_spatial_heterogeneity_method(self):
        sim = DiffusiveLotkaVolterra(_make_config())
        sim.reset()
        het = sim.spatial_heterogeneity(sim._u)
        assert het >= 0.0
        # Uniform field should have low heterogeneity
        assert het < 0.5

    def test_run_trajectory(self):
        """The run() method should produce a valid TrajectoryData."""
        sim = DiffusiveLotkaVolterra(_make_config(n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 128)  # 50 steps + initial
        assert np.all(np.isfinite(traj.states))


class TestDiffusiveLVRediscovery:
    def test_spatial_pattern_data_generation(self):
        from simulating_anything.rediscovery.diffusive_lv import (
            generate_spatial_pattern_data,
        )

        data = generate_spatial_pattern_data(
            n_runs=3, n_steps=100, dt=0.005, N_grid=32, L_domain=10.0,
        )
        assert len(data["D_u"]) == 3
        assert len(data["prey_heterogeneity"]) == 3
        assert np.all(data["prey_heterogeneity"] >= 0)
        assert np.all(data["total_prey"] > 0)
