"""Tests for the logistic map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.logistic_map import LogisticMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(r: float = 3.5, x_0: float = 0.5) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.LOGISTIC_MAP,
        dt=1.0,
        n_steps=100,
        parameters={"r": r, "x_0": x_0},
    )


class TestLogisticMapSimulation:
    def test_initial_state(self):
        sim = LogisticMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state[0] == pytest.approx(0.5)

    def test_step_applies_map(self):
        sim = LogisticMapSimulation(_make_config(r=3.0, x_0=0.5))
        sim.reset()
        sim.step()
        # x_1 = r * x_0 * (1 - x_0) = 3.0 * 0.5 * 0.5 = 0.75
        assert sim.observe()[0] == pytest.approx(0.75)

    def test_fixed_point_r_less_3(self):
        """For r < 3, should converge to fixed point x* = 1 - 1/r."""
        sim = LogisticMapSimulation(_make_config(r=2.5, x_0=0.5))
        sim.reset()
        for _ in range(1000):
            sim.step()
        x_star = 1 - 1 / 2.5
        assert sim.observe()[0] == pytest.approx(x_star, abs=1e-6)

    def test_period_2_r_3_2(self):
        """For r=3.2, should have period-2 orbit."""
        sim = LogisticMapSimulation(_make_config(r=3.2))
        sim.reset()
        period = sim.detect_period()
        assert period == 2

    def test_period_1_r_2_5(self):
        """For r=2.5, should have period-1 (fixed point)."""
        sim = LogisticMapSimulation(_make_config(r=2.5))
        sim.reset()
        period = sim.detect_period()
        assert period == 1

    def test_bounded_in_unit_interval(self):
        """Iterates should stay in [0, 1] for r <= 4."""
        sim = LogisticMapSimulation(_make_config(r=3.9, x_0=0.1))
        sim.reset()
        for _ in range(1000):
            sim.step()
            x = sim.observe()[0]
            assert 0 <= x <= 1, f"x={x} outside [0,1]"

    def test_lyapunov_negative_for_stable(self):
        """Lyapunov exponent should be negative for r < 3 (stable fixed point)."""
        sim = LogisticMapSimulation(_make_config(r=2.5))
        lam = sim.lyapunov_exponent(n_iterations=5000)
        assert lam < 0, f"Lyapunov={lam} should be negative"

    def test_lyapunov_positive_for_chaos(self):
        """Lyapunov exponent should be positive for r=4 (full chaos)."""
        sim = LogisticMapSimulation(_make_config(r=4.0))
        lam = sim.lyapunov_exponent(n_iterations=5000)
        # At r=4, lambda = ln(2) ~ 0.693
        assert lam > 0.5, f"Lyapunov={lam} too low for r=4"

    def test_lyapunov_at_r4_is_ln2(self):
        """At r=4, the exact Lyapunov exponent is ln(2)."""
        # x_0=0.5 is a special case at r=4 (maps to 1->0), use 0.3 instead
        sim = LogisticMapSimulation(_make_config(r=4.0, x_0=0.3))
        lam = sim.lyapunov_exponent(n_iterations=50000)
        np.testing.assert_allclose(lam, np.log(2), atol=0.05)

    def test_bifurcation_diagram(self):
        sim = LogisticMapSimulation(_make_config())
        r_vals = np.linspace(2.5, 4.0, 10)
        data = sim.bifurcation_diagram(r_vals, n_transient=100, n_plot=10)
        assert len(data["r"]) == 100  # 10 r * 10 plot points
        assert len(data["x"]) == 100


class TestLogisticMapRediscovery:
    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.logistic_map import generate_bifurcation_data
        data = generate_bifurcation_data(n_r=20, r_min=2.5, r_max=4.0)
        assert len(data["r_values"]) == 20
        assert len(data["periods"]) == 20

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.logistic_map import generate_lyapunov_data
        data = generate_lyapunov_data(n_r=10, r_min=2.5, r_max=4.0)
        assert len(data["r"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_feigenbaum_estimation(self):
        from simulating_anything.rediscovery.logistic_map import (
            estimate_feigenbaum,
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_r=500, r_min=2.5, r_max=4.0)
        result = estimate_feigenbaum(data["periods"], data["r_values"])
        assert "bifurcation_points" in result
        assert len(result["bifurcation_points"]) >= 2
