"""Tests for the Lorenz system simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.lorenz import LorenzSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestLorenzSimulation:
    """Tests for the Lorenz system simulation."""

    def _make_sim(self, **kwargs) -> LorenzSimulation:
        defaults = {
            "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzSimulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 1.0)
        assert np.isclose(state[2], 1.0)

    def test_custom_initial_conditions(self):
        sim = self._make_sim(x_0=5.0, y_0=-3.0, z_0=20.0)
        state = sim.reset()
        assert np.isclose(state[0], 5.0)
        assert np.isclose(state[1], -3.0)
        assert np.isclose(state[2], 20.0)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_derivatives_at_origin(self):
        """At origin with rho < 1, all derivatives should be zero (fixed point)."""
        sim = self._make_sim(rho=0.5)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_classic(self):
        """Test derivatives at a known point."""
        sim = self._make_sim(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        sim.reset()
        # At state [1, 1, 1]:
        # dx = 10*(1-1) = 0
        # dy = 1*(28-1) - 1 = 26
        # dz = 1*1 - (8/3)*1 = 1 - 8/3 = -5/3
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 26.0)
        assert np.isclose(derivs[2], 1.0 - 8.0 / 3.0)

    def test_trajectory_bounded(self):
        """Lorenz trajectories should remain bounded for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_z_stays_positive(self):
        """z coordinate should stay positive after initial transient on the attractor."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(1000):
            sim.step()
        # z should be positive on the attractor
        for _ in range(5000):
            state = sim.step()
            assert state[2] > -1.0, f"z went below -1: {state[2]}"

    def test_attractor_statistics(self):
        """Time-averaged z should be close to rho - 1 for chaotic attractor."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        z_vals = []
        for _ in range(20000):
            state = sim.step()
            z_vals.append(state[2])
        z_mean = np.mean(z_vals)
        # For rho=28, time-averaged z ~ rho - 1 = 27
        assert abs(z_mean - 25.0) < 5.0, f"z_mean={z_mean:.1f}, expected ~25"


class TestLorenzFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, rho=28.0) -> LorenzSimulation:
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=0.01,
            n_steps=1000,
            parameters={"sigma": 10.0, "rho": rho, "beta": 8.0 / 3.0},
        )
        return LorenzSimulation(config)

    def test_three_fixed_points_for_rho_gt_1(self):
        sim = self._make_sim(rho=28.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_one_fixed_point_for_rho_lt_1(self):
        sim = self._make_sim(rho=0.5)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_fixed_point_symmetry(self):
        """The two non-origin fixed points should be symmetric."""
        sim = self._make_sim(rho=28.0)
        sim.reset()
        fps = sim.fixed_points
        # C+ = (c, c, rho-1), C- = (-c, -c, rho-1)
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], -fps[2][1])
        assert np.isclose(fps[1][2], fps[2][2])
        assert np.isclose(fps[1][2], 27.0)  # rho - 1

    def test_fixed_point_values(self):
        """Check exact fixed point values."""
        sim = self._make_sim(rho=28.0)
        sim.reset()
        fps = sim.fixed_points
        beta = 8.0 / 3.0
        c = np.sqrt(beta * 27.0)
        np.testing.assert_array_almost_equal(fps[1], [c, c, 27.0])

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at fixed points."""
        sim = self._make_sim(rho=28.0)
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestLorenzLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """Lorenz at rho=28 should have positive largest Lyapunov exponent."""
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=0.01,
            n_steps=20000,
            parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        )
        sim = LorenzSimulation(config)
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.5, f"Lyapunov {lam:.3f} too small for chaotic regime"
        assert lam < 2.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_negative_lyapunov_stable(self):
        """Lorenz at rho=10 (below critical) should have negative Lyapunov."""
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=0.01,
            n_steps=20000,
            parameters={"sigma": 10.0, "rho": 10.0, "beta": 8.0 / 3.0},
        )
        sim = LorenzSimulation(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam < 0.5, f"Lyapunov {lam:.3f} too large for stable regime"


class TestLorenzRediscovery:
    """Tests for Lorenz data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.lorenz import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["sigma"] == 10.0
        assert data["rho"] == 28.0

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.lorenz import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        from simulating_anything.rediscovery.lorenz import generate_chaos_transition_data

        data = generate_chaos_transition_data(n_rho=5, n_steps=2000, dt=0.01)
        assert len(data["rho"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_chaos_transition_contains_types(self):
        """Sweep from 0.5 to 35 should find both stable and chaotic regimes."""
        from simulating_anything.rediscovery.lorenz import generate_chaos_transition_data

        data = generate_chaos_transition_data(n_rho=10, n_steps=5000, dt=0.01)
        types = set(data["attractor_type"])
        # Should have at least origin or fixed_point AND chaotic
        has_stable = "origin" in types or "fixed_point" in types
        has_chaotic = "chaotic" in types
        assert has_stable, f"No stable regime found in types: {types}"
        assert has_chaotic, f"No chaotic regime found in types: {types}"
