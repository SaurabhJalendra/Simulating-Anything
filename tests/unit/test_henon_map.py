"""Tests for the Henon map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.henon_map import HenonMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 1.4,
    b: float = 0.3,
    x_0: float = 0.0,
    y_0: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.HENON_MAP,
        dt=1.0,
        n_steps=100,
        parameters={"a": a, "b": b, "x_0": x_0, "y_0": y_0},
    )


class TestHenonMapSimulation:
    def test_creation_and_params(self):
        sim = HenonMapSimulation(_make_config())
        assert sim.a == 1.4
        assert sim.b == 0.3

    def test_initial_state_shape(self):
        sim = HenonMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == pytest.approx(0.0)
        assert state[1] == pytest.approx(0.0)

    def test_step_applies_map(self):
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3, x_0=0.5, y_0=0.1))
        sim.reset()
        sim.step()
        # x_1 = 1 - 1.4*0.5^2 + 0.1 = 1 - 0.35 + 0.1 = 0.75
        # y_1 = 0.3 * 0.5 = 0.15
        assert sim.observe()[0] == pytest.approx(0.75)
        assert sim.observe()[1] == pytest.approx(0.15)

    def test_fixed_point_computation(self):
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3))
        fps = sim.fixed_points
        assert len(fps) == 2
        # Both should be 2D arrays
        for fp in fps:
            assert fp.shape == (2,)
            assert fp[1] == pytest.approx(sim.b * fp[0])

    def test_fixed_point_is_equilibrium(self):
        """Verify that f(x*) = x* for each fixed point."""
        sim = HenonMapSimulation(_make_config(a=1.0, b=0.3))
        fps = sim.fixed_points
        for fp in fps:
            x, y = fp
            x_new = 1.0 - sim.a * x**2 + y
            y_new = sim.b * x
            assert x_new == pytest.approx(x, abs=1e-10)
            assert y_new == pytest.approx(y, abs=1e-10)

    def test_jacobian_determinant_equals_neg_b(self):
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3))
        assert sim.jacobian_determinant == pytest.approx(-0.3)

    def test_jacobian_determinant_area_contraction(self):
        """Area contracts by |b| each iteration."""
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3))
        assert abs(sim.jacobian_determinant) == pytest.approx(0.3)

    def test_chaotic_positive_lyapunov(self):
        """At a=1.4, b=0.3 the largest Lyapunov exponent should be positive."""
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3))
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=500)
        assert lam > 0.3, f"Lyapunov={lam} should be > 0.3 for chaotic regime"

    def test_lyapunov_at_classic_params(self):
        """At a=1.4, b=0.3 the Lyapunov exponent is approximately 0.42."""
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3))
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=50000, n_transient=1000)
        np.testing.assert_allclose(lam, 0.42, atol=0.05)

    def test_non_chaotic_negative_lyapunov(self):
        """At small a (e.g. a=0.3), the orbit is periodic/stable."""
        sim = HenonMapSimulation(_make_config(a=0.3, b=0.3))
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        assert lam < 0, f"Lyapunov={lam} should be negative for non-chaotic a=0.3"

    def test_periodic_orbit_detection(self):
        """At a=0.3, b=0.3 the orbit should be periodic (period 1 = fixed point)."""
        sim = HenonMapSimulation(_make_config(a=0.3, b=0.3))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=1000)
        assert period == 1

    def test_trajectory_bounded(self):
        """Trajectory stays bounded for standard parameters."""
        sim = HenonMapSimulation(_make_config(a=1.4, b=0.3))
        sim.reset()
        for _ in range(5000):
            sim.step()
            x, y = sim.observe()
            assert abs(x) < 10, f"x={x} unbounded"
            assert abs(y) < 10, f"y={y} unbounded"

    def test_b_zero_reduces_to_1d(self):
        """When b=0, the y coordinate is always 0, and the x iteration
        becomes x_{n+1} = 1 - a*x_n^2 (a 1D quadratic map)."""
        sim = HenonMapSimulation(_make_config(a=1.0, b=0.0, x_0=0.5, y_0=0.0))
        sim.reset()
        x = 0.5
        for _ in range(10):
            sim.step()
            x = 1.0 - 1.0 * x**2  # 1D quadratic map
            assert sim.observe()[0] == pytest.approx(x, abs=1e-12)
            assert sim.observe()[1] == pytest.approx(0.0, abs=1e-12)

    def test_run_trajectory_shape(self):
        sim = HenonMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 2)

    def test_bifurcation_diagram(self):
        sim = HenonMapSimulation(_make_config())
        a_vals = np.linspace(0.0, 1.4, 10)
        data = sim.bifurcation_diagram(a_vals, n_transient=100, n_plot=10)
        assert len(data["a"]) == 100  # 10 a * 10 plot points
        assert len(data["x"]) == 100

    def test_observe_returns_current_state(self):
        sim = HenonMapSimulation(_make_config(x_0=0.5, y_0=0.1))
        sim.reset()
        state = sim.observe()
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.1)

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.HENON_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = HenonMapSimulation(config)
        assert sim.a == 1.4
        assert sim.b == 0.3
        assert sim.x_0 == 0.0
        assert sim.y_0 == 0.0


class TestHenonMapRediscovery:
    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.henon_map import generate_bifurcation_data
        data = generate_bifurcation_data(n_a=20, a_min=0.0, a_max=1.4)
        assert len(data["a_values"]) == 20
        assert len(data["periods"]) == 20

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.henon_map import generate_lyapunov_data
        data = generate_lyapunov_data(n_a=10, a_min=0.0, a_max=1.4)
        assert len(data["a"]) == 10
        assert len(data["lyapunov"]) == 10
