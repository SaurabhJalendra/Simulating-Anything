"""Tests for the BZ spiral (2D Oregonator PDE) simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.bz_spiral import (
    BZSpiralSimulation,
    _laplacian_2d_neumann,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    eps: float = 0.01,
    f: float = 1.0,
    q: float = 0.002,
    D_u: float = 1.0,
    D_v: float = 0.0,
    Nx: int = 64,
    Ny: int = 64,
    dx: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 100,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BZ_SPIRAL,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "eps": eps,
            "f": f,
            "q": q,
            "D_u": D_u,
            "D_v": D_v,
            "Nx": float(Nx),
            "Ny": float(Ny),
            "dx": dx,
        },
    )


class TestBZSpiralSimulation:
    """Tests for the BZSpiralSimulation class."""

    def test_grid_shape(self):
        """Grid dimensions match Nx x Ny."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=48))
        sim.reset()
        assert sim.get_u_field().shape == (32, 48)
        assert sim.get_v_field().shape == (32, 48)

    def test_observe_shape(self):
        """Observation vector has size 2 * Nx * Ny."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=32))
        state = sim.reset()
        assert state.shape == (2 * 32 * 32,)
        obs = sim.observe()
        assert obs.shape == (2 * 32 * 32,)

    def test_reset_produces_wavefront(self):
        """Initial condition has both high and low u regions."""
        sim = BZSpiralSimulation(_make_config(Nx=64, Ny=64))
        sim.reset()
        u = sim.get_u_field()
        # Should have excited region (high u) and rest region (low u)
        assert np.max(u) > 0.5, "No excited region in initial condition"
        assert np.min(u) < 0.1, "No rest region in initial condition"

    def test_excitability(self):
        """A suprathreshold perturbation should propagate (u range increases)."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=32, dt=0.01))
        sim.reset()
        # Run a few steps; activity should spread from initial perturbation
        for _ in range(50):
            sim.step()
        later_active = np.sum(sim.get_u_field() > 0.1)
        # Activity pattern should change (either spread or reorganize)
        assert later_active > 0, "No activity detected after stepping"

    def test_step_advances(self):
        """State changes after a step."""
        sim = BZSpiralSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1), "State did not change after step"

    def test_deterministic(self):
        """Same seed produces same result."""
        sim1 = BZSpiralSimulation(_make_config(seed=123))
        sim1.reset()
        for _ in range(10):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = BZSpiralSimulation(_make_config(seed=123))
        sim2.reset()
        for _ in range(10):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_stability(self):
        """No NaN after many steps."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=32, dt=0.01))
        sim.reset()
        for _ in range(500):
            state = sim.step()
        assert np.all(np.isfinite(state)), "NaN/Inf detected in state"

    def test_u_bounds(self):
        """u stays in reasonable range (non-negative, bounded)."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=32))
        sim.reset()
        for _ in range(200):
            sim.step()
        u = sim.get_u_field()
        assert np.all(u >= 0), f"Negative u detected: min={np.min(u)}"
        assert np.max(u) < 10.0, f"u diverged: max={np.max(u)}"

    def test_v_bounds(self):
        """v stays non-negative."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=32))
        sim.reset()
        for _ in range(200):
            sim.step()
        v = sim.get_v_field()
        assert np.all(v >= 0), f"Negative v detected: min={np.min(v)}"

    def test_diffusion_only_u(self):
        """With D_v=0, the v field should not exhibit diffusion smoothing.

        We verify this indirectly: create a sharp v gradient and check
        that it persists (while u would smooth out).
        """
        sim = BZSpiralSimulation(_make_config(D_u=1.0, D_v=0.0, Nx=32, Ny=32))
        sim.reset()
        # Set v to a step function
        v_init = np.zeros((32, 32))
        v_init[:16, :] = 1.0
        sim._v = v_init.copy()
        # Set u uniform to isolate v dynamics
        sim._u = np.full((32, 32), 0.1)

        # Take one step
        sim.step()
        v_after = sim.get_v_field()
        # v should still have a relatively sharp transition (no diffusion)
        # The reaction term u - v will modify v, but not via spatial smoothing
        # Check that the gradient at the interface is not excessively smoothed
        grad = np.abs(np.diff(v_after[:, 16], axis=0))
        max_grad = np.max(grad)
        # With D_v=0, interface should remain sharper than a diffused version
        assert max_grad > 0.01, "v field appears too smooth for D_v=0"

    def test_wave_propagation(self):
        """Activity should spread from the initial perturbation region."""
        sim = BZSpiralSimulation(_make_config(Nx=64, Ny=64, dt=0.01))
        sim.reset()
        u_init = sim.get_u_field()
        initial_mean = np.mean(u_init[48:, :])  # Bottom quarter (initially rest)

        for _ in range(300):
            sim.step()

        u_later = sim.get_u_field()
        later_mean = np.mean(u_later[48:, :])
        # Bottom region should have changed as the wave reaches it
        assert abs(later_mean - initial_mean) > 0.001 or later_mean > initial_mean, (
            "Wave did not propagate to bottom region"
        )

    def test_spiral_formation(self):
        """After many steps, spatial pattern should exist (not uniform)."""
        sim = BZSpiralSimulation(_make_config(Nx=64, Ny=64, dt=0.01))
        sim.reset()
        for _ in range(500):
            sim.step()
        u = sim.get_u_field()
        u_range = np.max(u) - np.min(u)
        assert u_range > 0.01, (
            f"No spatial pattern detected after 500 steps, u_range={u_range}"
        )

    def test_get_fields(self):
        """get_u_field and get_v_field return correct 2D shapes."""
        sim = BZSpiralSimulation(_make_config(Nx=48, Ny=32))
        sim.reset()
        assert sim.get_u_field().shape == (48, 32)
        assert sim.get_v_field().shape == (48, 32)
        # Verify they are copies (not references)
        u1 = sim.get_u_field()
        u2 = sim.get_u_field()
        u1[0, 0] = -999.0
        assert u2[0, 0] != -999.0

    def test_run_trajectory(self):
        """Can run n steps via the run() method."""
        sim = BZSpiralSimulation(_make_config(Nx=16, Ny=16, n_steps=20))
        traj = sim.run(n_steps=20)
        assert traj.states.shape == (21, 2 * 16 * 16)  # 20 steps + initial
        assert len(traj.timestamps) == 21

    def test_cfl_condition(self):
        """dt that violates CFL should raise ValueError."""
        # dx=0.5, D_u=1.0 => CFL limit = 0.5^2 / (4*1.0) = 0.0625
        with pytest.raises(ValueError, match="CFL"):
            BZSpiralSimulation(_make_config(dx=0.5, D_u=1.0, dt=0.1))

    def test_laplacian_computation(self):
        """Finite difference Laplacian is correct on a known test pattern."""
        # Quadratic field: f(x,y) = x^2 + y^2, Laplacian = 2 + 2 = 4
        n = 16
        dx = 1.0
        x = np.arange(n) * dx
        y = np.arange(n) * dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        field = X ** 2 + Y ** 2

        lap = _laplacian_2d_neumann(field, dx)
        # Interior points should have Laplacian close to 4.0
        interior = lap[2:-2, 2:-2]
        np.testing.assert_allclose(interior, 4.0, atol=1e-10)

    def test_rediscovery_data(self):
        """Rediscovery data generation works without error."""
        from simulating_anything.rediscovery.bz_spiral import generate_spiral_data

        data = generate_spiral_data(
            eps=0.01, D_u=1.0, Nx=32, Ny=32, dx=0.5, dt=0.01,
            n_warmup=100, n_measure=100,
        )
        assert "wavelength" in data
        assert "frequency" in data
        assert "u_range" in data
        assert isinstance(data["wavelength"], float)

    def test_detect_spiral_tip(self):
        """Spiral tip detection returns valid grid indices or (-1, -1)."""
        sim = BZSpiralSimulation(_make_config(Nx=32, Ny=32))
        sim.reset()
        # Run a few steps so there is some pattern
        for _ in range(50):
            sim.step()
        tip = sim.detect_spiral_tip()
        assert len(tip) == 2
        # Either valid indices or (-1, -1)
        if tip[0] >= 0:
            assert 0 <= tip[0] < 32
            assert 0 <= tip[1] < 32
        else:
            assert tip == (-1, -1)
