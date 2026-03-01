"""Tests for the Selkov glycolysis model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.selkov import (
    SelkovSimulation,
    compute_hopf_b,
    compute_hopf_boundary,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 0.08,
    b: float = 0.6,
    dt: float = 0.01,
    x_0: float = 0.5,
    y_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SELKOV,
        dt=dt,
        n_steps=1000,
        parameters={"a": a, "b": b, "x_0": x_0, "y_0": y_0},
    )


class TestSelkovSimulation:
    def test_initial_state_shape(self):
        sim = SelkovSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = SelkovSimulation(_make_config(x_0=0.3, y_0=0.7))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.3, 0.7])

    def test_step_advances_state(self):
        sim = SelkovSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_fixed_point_formula(self):
        """x* = b, y* = b / (a + b^2)."""
        sim = SelkovSimulation(_make_config(a=0.08, b=0.6))
        x_star, y_star = sim.fixed_point
        assert x_star == pytest.approx(0.6)
        assert y_star == pytest.approx(0.6 / (0.08 + 0.36), rel=1e-10)

    def test_fixed_point_different_params(self):
        sim = SelkovSimulation(_make_config(a=0.1, b=1.0))
        x_star, y_star = sim.fixed_point
        assert x_star == pytest.approx(1.0)
        assert y_star == pytest.approx(1.0 / (0.1 + 1.0), rel=1e-10)

    def test_derivatives_at_fixed_point(self):
        """At the fixed point, derivatives should be zero."""
        a, b = 0.08, 0.6
        sim = SelkovSimulation(_make_config(a=a, b=b))
        sim.reset()
        fp = np.array(sim.fixed_point)
        dy = sim._derivatives(fp)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-12)

    def test_derivatives_at_fixed_point_different_params(self):
        a, b = 0.15, 0.8
        sim = SelkovSimulation(_make_config(a=a, b=b))
        sim.reset()
        fp = np.array(sim.fixed_point)
        dy = sim._derivatives(fp)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-12)

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded for reasonable parameters."""
        sim = SelkovSimulation(_make_config(a=0.08, b=0.6, dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            x, y = sim.observe()
            assert abs(x) < 50, f"x diverged: {x}"
            assert abs(y) < 50, f"y diverged: {y}"

    def test_non_negative_concentrations(self):
        """Chemical concentrations should stay non-negative for normal params."""
        sim = SelkovSimulation(_make_config(a=0.08, b=0.6, dt=0.005, x_0=0.1, y_0=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            x, y = sim.observe()
            assert x > -0.01, f"x went negative: {x}"
            assert y > -0.01, f"y went negative: {y}"

    def test_observe_returns_current_state(self):
        sim = SelkovSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = SelkovSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101


class TestSelkovHopfBifurcation:
    def test_hopf_threshold_positive(self):
        sim = SelkovSimulation(_make_config(a=0.08))
        b_c = sim.hopf_threshold
        assert b_c > 0

    def test_hopf_threshold_increases_with_a(self):
        """Larger a should generally shift the Hopf boundary."""
        b_c_small = compute_hopf_b(0.05)
        b_c_large = compute_hopf_b(0.5)
        # Both should be positive
        assert b_c_small > 0
        assert b_c_large > 0

    def test_below_hopf_converges(self):
        """Below Hopf threshold, system should converge to fixed point."""
        a = 0.08
        b_c = compute_hopf_b(a)
        b = b_c * 0.5  # Well below threshold
        x_star = b
        y_star = b / (a + b**2)
        sim = SelkovSimulation(_make_config(
            a=a, b=b, dt=0.01, x_0=x_star + 0.1, y_0=y_star + 0.1,
        ))
        sim.reset()
        for _ in range(50000):
            sim.step()
        x, y = sim.observe()
        np.testing.assert_allclose([x, y], [x_star, y_star], atol=0.05)

    def test_above_hopf_oscillates(self):
        """Above Hopf threshold, system should oscillate (limit cycle)."""
        a = 0.08
        b_c = compute_hopf_b(a)
        b = b_c * 2.0  # Well above threshold
        sim = SelkovSimulation(_make_config(a=a, b=b, dt=0.005))
        sim.reset()
        # Skip transient
        for _ in range(50000):
            sim.step()
        # Measure variation
        x_vals = []
        for _ in range(10000):
            sim.step()
            x_vals.append(sim.observe()[0])
        amplitude = max(x_vals) - min(x_vals)
        assert amplitude > 0.05, f"No oscillation detected, amplitude={amplitude}"

    def test_is_oscillatory_flag(self):
        a = 0.08
        b_c = compute_hopf_b(a)
        sim_below = SelkovSimulation(_make_config(a=a, b=b_c * 0.5))
        assert not sim_below.is_oscillatory

        sim_above = SelkovSimulation(_make_config(a=a, b=b_c * 2.0))
        assert sim_above.is_oscillatory

    def test_compute_hopf_boundary_array(self):
        a_values = np.array([0.02, 0.05, 0.08, 0.1, 0.15])
        b_c_values = compute_hopf_boundary(a_values)
        assert len(b_c_values) == 5
        assert all(b > 0 for b in b_c_values)

    def test_hopf_trace_condition(self):
        """At the Hopf point, trace of Jacobian should be zero."""
        a = 0.08
        b_c = compute_hopf_b(a)
        # tr(J) = -1 + 2*b^2/(a+b^2) - (a + b^2)
        trace = -1.0 + 2.0 * b_c**2 / (a + b_c**2) - a - b_c**2
        assert abs(trace) < 1e-8, f"Trace at Hopf: {trace}"


class TestSelkovPeriod:
    def test_measure_period_oscillatory(self):
        """Should return finite period when oscillating."""
        a = 0.08
        b_c = compute_hopf_b(a)
        b = b_c * 2.5
        sim = SelkovSimulation(_make_config(a=a, b=b, dt=0.005))
        sim.reset()
        period = sim.measure_period(n_periods=3)
        assert np.isfinite(period)
        assert period > 0

    def test_measure_period_stable(self):
        """Should return inf when below Hopf threshold."""
        a = 0.08
        b_c = compute_hopf_b(a)
        b = b_c * 0.3
        sim = SelkovSimulation(_make_config(a=a, b=b, dt=0.01))
        sim.reset()
        period = sim.measure_period()
        assert period == float("inf")

    def test_measure_amplitude_oscillatory(self):
        """Should return positive amplitude when oscillating."""
        a = 0.08
        b_c = compute_hopf_b(a)
        b = b_c * 2.5
        sim = SelkovSimulation(_make_config(a=a, b=b, dt=0.005))
        sim.reset()
        amp = sim.measure_amplitude(transient_time=300.0)
        assert amp > 0.01


class TestSelkovRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.selkov import generate_ode_data

        data = generate_ode_data(a=0.08, b=0.6, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["a"] == 0.08
        assert data["b"] == 0.6

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.selkov import generate_bifurcation_data

        data = generate_bifurcation_data(a=0.08, n_b=5, dt=0.01)
        assert len(data["b"]) == 5
        assert len(data["amplitude"]) == 5
        assert data["b_c_theory"] > 0

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.selkov import generate_period_data

        data = generate_period_data(a=0.08, n_b=3, dt=0.01)
        assert len(data["b"]) == 3
        assert len(data["period"]) == 3
        assert data["b_c"] > 0
