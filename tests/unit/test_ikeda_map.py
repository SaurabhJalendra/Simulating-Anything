"""Tests for the Ikeda map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.ikeda_map import IkedaMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    u: float = 0.9,
    x_0: float = 0.0,
    y_0: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.IKEDA_MAP,
        dt=1.0,
        n_steps=100,
        parameters={"u": u, "x_0": x_0, "y_0": y_0},
    )


class TestIkedaMapSimulation:
    def test_reset(self):
        """Reset returns initial state."""
        sim = IkedaMapSimulation(_make_config(x_0=1.5, y_0=-0.3))
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == pytest.approx(1.5)
        assert state[1] == pytest.approx(-0.3)

    def test_observe_shape(self):
        """Observe returns a 2-element array."""
        sim = IkedaMapSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_step_advances(self):
        """State changes after one step."""
        sim = IkedaMapSimulation(_make_config(x_0=0.5, y_0=0.5))
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same parameters produce the same orbit."""
        sim1 = IkedaMapSimulation(_make_config(u=0.7, x_0=0.1, y_0=0.1))
        sim2 = IkedaMapSimulation(_make_config(u=0.7, x_0=0.1, y_0=0.1))
        sim1.reset()
        sim2.reset()
        for _ in range(50):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)

    def test_stability(self):
        """No NaN after 50000 iterations at chaotic parameters."""
        sim = IkedaMapSimulation(_make_config(u=0.9))
        sim.reset()
        for _ in range(50000):
            state = sim.step()
        assert np.all(np.isfinite(state)), "NaN detected after 50000 steps"

    def test_bounded(self):
        """Attractor stays bounded at chaotic parameters."""
        sim = IkedaMapSimulation(_make_config(u=0.9))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            x, y = state
            assert abs(x) < 5.0, f"x={x} out of bounds"
            assert abs(y) < 5.0, f"y={y} out of bounds"

    def test_fixed_point_low_u(self):
        """For u=0.3, the orbit converges to a fixed point."""
        sim = IkedaMapSimulation(_make_config(u=0.3))
        sim.reset()
        # Iterate long enough to converge
        for _ in range(2000):
            sim.step()
        s1 = sim.observe().copy()
        sim.step()
        s2 = sim.observe()
        np.testing.assert_allclose(s1, s2, atol=1e-8)

    def test_fixed_point_accuracy(self):
        """Verify that found fixed points satisfy the map equations."""
        sim = IkedaMapSimulation(_make_config(u=0.5))
        fps = sim.find_fixed_points()
        assert len(fps) >= 1, "Should find at least one fixed point"
        for fp in fps:
            x, y = fp
            t = sim.compute_t(x, y)
            x_new = 1.0 + sim.u * (x * np.cos(t) - y * np.sin(t))
            y_new = sim.u * (x * np.sin(t) + y * np.cos(t))
            assert x_new == pytest.approx(x, abs=1e-8)
            assert y_new == pytest.approx(y, abs=1e-8)

    def test_jacobian_shape(self):
        """Jacobian is a 2x2 matrix."""
        sim = IkedaMapSimulation(_make_config())
        jac = sim.compute_jacobian(0.5, 0.3)
        assert jac.shape == (2, 2)

    def test_dissipative(self):
        """|det(J)| = u^2 < 1 at any point on the attractor."""
        u = 0.7
        sim = IkedaMapSimulation(_make_config(u=u))
        sim.reset()
        # Iterate to reach attractor
        for _ in range(1000):
            sim.step()
        x, y = sim.observe()
        jac = sim.compute_jacobian(x, y)
        det_j = abs(np.linalg.det(jac))
        np.testing.assert_allclose(det_j, u**2, rtol=1e-10)

    def test_chaotic_lyapunov(self):
        """Positive Lyapunov exponent at u=0.9 (chaotic regime)."""
        sim = IkedaMapSimulation(_make_config(u=0.9))
        sim.reset()
        lam = sim.compute_lyapunov(n_steps=10000, n_transient=500)
        assert lam > 0, f"Lyapunov={lam} should be positive for chaotic u=0.9"

    def test_stable_lyapunov(self):
        """Negative Lyapunov exponent at u=0.3 (stable fixed point)."""
        sim = IkedaMapSimulation(_make_config(u=0.3))
        sim.reset()
        lam = sim.compute_lyapunov(n_steps=5000, n_transient=500)
        assert lam < 0, f"Lyapunov={lam} should be negative for stable u=0.3"

    def test_bifurcation_sweep(self):
        """Bifurcation sweep produces valid data."""
        sim = IkedaMapSimulation(_make_config())
        u_vals = np.linspace(0.1, 1.0, 10)
        data = sim.bifurcation_sweep(u_vals, n_transient=100, n_record=10)
        assert len(data["u"]) == 100  # 10 u * 10 record points
        assert len(data["x"]) == 100
        assert len(data["y"]) == 100

    def test_attractor_bounded(self):
        """All orbit points from bifurcation sweep stay within [-15, 15]."""
        sim = IkedaMapSimulation(_make_config())
        u_vals = np.linspace(0.1, 1.0, 20)
        data = sim.bifurcation_sweep(u_vals, n_transient=500, n_record=50)
        # The Ikeda attractor at u=1.0 can reach ~10 in x and y
        assert np.all(np.abs(data["x"]) < 15.0), "x values out of [-15, 15]"
        assert np.all(np.abs(data["y"]) < 15.0), "y values out of [-15, 15]"

    def test_period_doubling(self):
        """Detect period-2 at u=0.5 (period-doubling region)."""
        sim = IkedaMapSimulation(_make_config(u=0.5))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=5000)
        assert period == 2, f"Period={period} should be 2 at u=0.5"

    def test_t_function(self):
        """Verify t computation: t = 0.4 - 6/(1 + x^2 + y^2)."""
        t = IkedaMapSimulation.compute_t(0.0, 0.0)
        # t = 0.4 - 6/(1+0+0) = 0.4 - 6 = -5.6
        assert t == pytest.approx(-5.6)

        t2 = IkedaMapSimulation.compute_t(1.0, 1.0)
        # t = 0.4 - 6/(1+1+1) = 0.4 - 2 = -1.6
        assert t2 == pytest.approx(-1.6)

    def test_run_trajectory_shape(self):
        """Run method produces correct trajectory shape."""
        sim = IkedaMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 2)

    def test_default_parameters(self):
        """Config with no parameters uses defaults."""
        config = SimulationConfig(
            domain=Domain.IKEDA_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = IkedaMapSimulation(config)
        assert sim.u == 0.9
        assert sim.x_0 == 0.0
        assert sim.y_0 == 0.0

    def test_step_matches_manual(self):
        """Verify one step against manual calculation."""
        sim = IkedaMapSimulation(_make_config(u=0.9, x_0=1.0, y_0=0.0))
        sim.reset()
        state = sim.step()
        # t = 0.4 - 6/(1 + 1 + 0) = 0.4 - 3 = -2.6
        t = -2.6
        x_expected = 1.0 + 0.9 * (1.0 * np.cos(t) - 0.0 * np.sin(t))
        y_expected = 0.9 * (1.0 * np.sin(t) + 0.0 * np.cos(t))
        assert state[0] == pytest.approx(x_expected)
        assert state[1] == pytest.approx(y_expected)


class TestIkedaMapRediscovery:
    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.ikeda_map import generate_bifurcation_data
        data = generate_bifurcation_data(n_u=10, u_min=0.1, u_max=1.0)
        assert len(data["u_values"]) == 10
        assert len(data["periods"]) == 10

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.ikeda_map import generate_lyapunov_data
        data = generate_lyapunov_data(n_u=10, u_min=0.1, u_max=1.0)
        assert len(data["u"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_fixed_point_data(self):
        from simulating_anything.rediscovery.ikeda_map import generate_fixed_point_data
        data = generate_fixed_point_data(n_u=5, u_min=0.1, u_max=0.5)
        assert len(data["fixed_points"]) == 5
        for entry in data["fixed_points"]:
            assert "u" in entry
            assert "n_fixed_points" in entry
