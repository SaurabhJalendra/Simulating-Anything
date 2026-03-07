"""Tests for the restricted three-body gravitational simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.three_body import ThreeBodySimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _config(**params) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CUSTOM, dt=0.001, n_steps=10000,
        parameters=params,
    )


class TestThreeBodySimulation:
    """Test CR3BP simulation."""

    def test_init(self):
        config = _config(mu=0.01, x0=0.5, y0=0.5)
        sim = ThreeBodySimulation(config)
        assert sim.mu == 0.01

    def test_reset(self):
        config = _config(mu=0.01, x0=1.0, y0=0.0, vx0=0.0, vy0=0.5)
        sim = ThreeBodySimulation(config)
        state = sim.reset()
        assert state.shape == (4,)
        assert state[0] == pytest.approx(1.0)
        assert state[3] == pytest.approx(0.5)

    def test_step_finite(self):
        config = _config(mu=0.01, x0=0.5, y0=0.866, vx0=0.0, vy0=0.0)
        sim = ThreeBodySimulation(config)
        sim.reset()
        for _ in range(100):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_jacobi_constant_conservation(self):
        """Jacobi constant should be conserved in CR3BP."""
        config = _config(mu=0.01, x0=0.5, y0=0.5, vx0=0.1, vy0=-0.2)
        sim = ThreeBodySimulation(config)
        sim.reset()
        cj_initial = sim.jacobi_constant()

        for _ in range(5000):
            sim.step()

        cj_final = sim.jacobi_constant()
        # Should be conserved to within RK4 accuracy
        assert cj_final == pytest.approx(cj_initial, rel=1e-4)

    def test_lagrange_l4_location(self):
        """L4 point should be at (0.5 - mu, sqrt(3)/2)."""
        config = _config(mu=0.01)
        sim = ThreeBodySimulation(config)
        lps = sim.lagrange_points()

        l4 = lps["L4"]
        assert l4[0] == pytest.approx(0.5 - 0.01, abs=1e-6)
        assert l4[1] == pytest.approx(np.sqrt(3) / 2, abs=1e-6)

    def test_lagrange_l5_location(self):
        """L5 point should be at (0.5 - mu, -sqrt(3)/2)."""
        config = _config(mu=0.01)
        sim = ThreeBodySimulation(config)
        lps = sim.lagrange_points()

        l5 = lps["L5"]
        assert l5[0] == pytest.approx(0.5 - 0.01, abs=1e-6)
        assert l5[1] == pytest.approx(-np.sqrt(3) / 2, abs=1e-6)

    def test_lagrange_collinear_points_exist(self):
        """L1, L2, L3 should lie on the x-axis."""
        config = _config(mu=0.01)
        sim = ThreeBodySimulation(config)
        lps = sim.lagrange_points()

        for name in ["L1", "L2", "L3"]:
            assert lps[name][1] == pytest.approx(0.0, abs=1e-6)

    def test_l1_between_primaries(self):
        """L1 should be between the two primary masses."""
        config = _config(mu=0.01)
        sim = ThreeBodySimulation(config)
        lps = sim.lagrange_points()
        l1_x = lps["L1"][0]
        # Should be between m1 at -mu and m2 at 1-mu
        assert -0.01 < l1_x < (1 - 0.01)

    def test_orbit_near_l4_stable(self):
        """Small displacement from L4 should remain bounded for small mu."""
        mu = 0.01
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.001, n_steps=20000,
            parameters={
                "mu": mu,
                "x0": 0.5 - mu + 0.01,  # Small offset from L4
                "y0": np.sqrt(3) / 2 + 0.01,
                "vx0": 0.0, "vy0": 0.0,
            },
        )
        sim = ThreeBodySimulation(config)
        sim.reset()
        max_r = 0.0
        for _ in range(20000):
            state = sim.step()
            r = np.sqrt(
                (state[0] - (0.5 - mu)) ** 2
                + (state[1] - np.sqrt(3) / 2) ** 2
            )
            max_r = max(max_r, r)

        # For mu=0.01 (well below Gascheau limit 0.0385), orbit should stay bounded
        assert max_r < 1.0

    def test_run_trajectory(self):
        config = _config(mu=0.01, x0=0.5, y0=0.5, vx0=0.1, vy0=-0.1)
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=100,
            parameters={"mu": 0.01, "x0": 0.5, "y0": 0.5, "vx0": 0.1, "vy0": -0.1},
        )
        sim = ThreeBodySimulation(config)
        traj = sim.run(100)
        assert traj.states.shape == (101, 4)

    def test_lyapunov_computation(self):
        """Lyapunov exponent should be computable and finite."""
        config = _config(mu=0.01, x0=0.5, y0=0.5, vx0=0.3, vy0=-0.3)
        sim = ThreeBodySimulation(config)
        lyap = sim.compute_lyapunov(n_steps=1000, dt=0.01)
        assert np.isfinite(lyap)

    def test_different_mass_ratios(self):
        """Simulation should work for various mass ratios."""
        for mu in [0.001, 0.01, 0.1, 0.3]:
            config = _config(mu=mu, x0=0.5, y0=0.5)
            sim = ThreeBodySimulation(config)
            sim.reset()
            for _ in range(100):
                state = sim.step()
            assert np.all(np.isfinite(state))
