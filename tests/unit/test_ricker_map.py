"""Tests for the Ricker map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.ricker_map import RickerMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 2.0,
    K: float = 1.0,
    x_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.RICKER_MAP,
        dt=1.0,
        n_steps=100,
        parameters={"r": r, "K": K, "x_0": x_0},
    )


class TestRickerMapSimulation:
    def test_creation_and_params(self):
        sim = RickerMapSimulation(_make_config())
        assert sim.r == 2.0
        assert sim.K == 1.0
        assert sim.x_0 == 0.5

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.RICKER_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = RickerMapSimulation(config)
        assert sim.r == 2.0
        assert sim.K == 1.0
        assert sim.x_0 == 0.5

    def test_initial_state_shape(self):
        sim = RickerMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state[0] == pytest.approx(0.5)

    def test_step_applies_map(self):
        sim = RickerMapSimulation(_make_config(r=1.0, K=1.0, x_0=0.5))
        sim.reset()
        sim.step()
        # x_1 = 0.5 * exp(1.0 * (1 - 0.5/1.0)) = 0.5 * exp(0.5)
        expected = 0.5 * np.exp(0.5)
        assert sim.observe()[0] == pytest.approx(expected)

    def test_observe_returns_current_state(self):
        sim = RickerMapSimulation(_make_config(x_0=0.7))
        sim.reset()
        assert sim.observe()[0] == pytest.approx(0.7)

    def test_fixed_point_convergence_r_small(self):
        """For r < 2, should converge to x* = K."""
        sim = RickerMapSimulation(_make_config(r=1.5, K=1.0, x_0=0.3))
        sim.reset()
        for _ in range(1000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(1.0, abs=1e-6)

    def test_fixed_point_convergence_r_1(self):
        """For r=1.0, should converge to x* = K=2.0."""
        sim = RickerMapSimulation(_make_config(r=1.0, K=2.0, x_0=0.5))
        sim.reset()
        for _ in range(500):
            sim.step()
        assert sim.observe()[0] == pytest.approx(2.0, abs=1e-4)

    def test_fixed_point_is_equilibrium(self):
        """Verify x* = K is a fixed point: f(K) = K * exp(r*(1-K/K)) = K."""
        K = 2.5
        r = 1.0
        # f(K) = K * exp(r * 0) = K * 1 = K
        x_next = K * np.exp(r * (1.0 - K / K))
        assert x_next == pytest.approx(K, abs=1e-12)

    def test_find_fixed_points(self):
        sim = RickerMapSimulation(_make_config(r=1.5, K=1.0))
        fps = sim.find_fixed_points()
        assert len(fps) == 2
        # Trivial fixed point x=0
        assert fps[0]["x"] == pytest.approx(0.0)
        assert fps[0]["stability"] == "unstable"
        # Nontrivial fixed point x=K
        assert fps[1]["x"] == pytest.approx(1.0)
        assert fps[1]["stability"] == "stable"

    def test_fixed_point_stability_r_gt_2(self):
        """For r > 2, nontrivial fixed point x*=K is unstable."""
        sim = RickerMapSimulation(_make_config(r=2.5, K=1.0))
        fps = sim.find_fixed_points()
        assert fps[1]["stability"] == "unstable"
        # eigenvalue = 1 - r = 1 - 2.5 = -1.5, |eigenvalue| > 1
        assert fps[1]["eigenvalue"] == pytest.approx(-1.5)

    def test_period_1_r_small(self):
        """For r=1.5 (< 2), orbit has period 1 (fixed point)."""
        sim = RickerMapSimulation(_make_config(r=1.5))
        sim.reset()
        period = sim.detect_period()
        assert period == 1

    def test_period_2_r_above_2(self):
        """For r slightly above 2 (e.g., 2.2), orbit should have period 2."""
        sim = RickerMapSimulation(_make_config(r=2.2, x_0=0.5))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=2000)
        assert period == 2

    def test_overcompensation(self):
        """Population should overshoot K when r > 1."""
        sim = RickerMapSimulation(_make_config(r=2.5, K=1.0, x_0=0.1))
        sim.reset()
        max_x = 0.0
        for _ in range(500):
            sim.step()
            x = sim.observe()[0]
            max_x = max(max_x, x)
        # With r=2.5, population should exceed K=1.0
        assert max_x > 1.0, f"Max x={max_x} should exceed K=1.0"

    def test_population_stays_positive(self):
        """Population should remain positive for positive initial conditions."""
        sim = RickerMapSimulation(_make_config(r=2.5, K=1.0, x_0=0.5))
        sim.reset()
        for _ in range(1000):
            sim.step()
            x = sim.observe()[0]
            assert x > 0, f"Population x={x} should be positive"

    def test_lyapunov_negative_for_stable(self):
        """Lyapunov exponent should be negative for r < 2 (stable fixed point)."""
        sim = RickerMapSimulation(_make_config(r=1.5))
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        assert lam < 0, f"Lyapunov={lam} should be negative for r=1.5"

    def test_lyapunov_positive_for_chaos(self):
        """Lyapunov exponent should be positive for r > 2 in chaotic regime."""
        sim = RickerMapSimulation(_make_config(r=3.0))
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=1000)
        assert lam > 0, f"Lyapunov={lam} should be positive for r=3.0"

    def test_lyapunov_at_fixed_point_matches_theory(self):
        """At x*=K, the theoretical Lyapunov is ln|1-r|.

        For r=1.5: ln|1-1.5| = ln(0.5) ~ -0.693
        """
        sim = RickerMapSimulation(_make_config(r=1.5, x_0=0.9))
        lam = sim.compute_lyapunov(n_iterations=20000, n_transient=1000)
        theory = np.log(abs(1.0 - 1.5))
        np.testing.assert_allclose(lam, theory, atol=0.05)

    def test_bifurcation_diagram(self):
        sim = RickerMapSimulation(_make_config())
        r_vals = np.linspace(0.5, 3.0, 10)
        data = sim.bifurcation_diagram(r_vals, n_transient=100, n_plot=10)
        assert len(data["r"]) == 100  # 10 r * 10 plot points
        assert len(data["x"]) == 100

    def test_run_trajectory_shape(self):
        sim = RickerMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 1)

    def test_carrying_capacity_scaling(self):
        """Fixed point should scale with K."""
        for K in [0.5, 1.0, 2.0, 5.0]:
            sim = RickerMapSimulation(_make_config(r=1.0, K=K, x_0=K * 0.3))
            sim.reset()
            for _ in range(500):
                sim.step()
            assert sim.observe()[0] == pytest.approx(K, abs=1e-3)


class TestRickerMapRediscovery:
    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.ricker_map import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_r=20, r_min=0.5, r_max=3.0)
        assert len(data["r_values"]) == 20
        assert len(data["periods"]) == 20

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.ricker_map import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_r=10, r_min=0.5, r_max=3.0)
        assert len(data["r"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_feigenbaum_estimation(self):
        from simulating_anything.rediscovery.ricker_map import (
            estimate_feigenbaum,
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_r=500, r_min=0.5, r_max=3.5)
        result = estimate_feigenbaum(data["periods"], data["r_values"])
        assert "bifurcation_points" in result
        assert len(result["bifurcation_points"]) >= 2

    def test_fixed_point_in_rediscovery(self):
        from simulating_anything.rediscovery.ricker_map import _make_config
        from simulating_anything.simulation.ricker_map import RickerMapSimulation
        config = _make_config(r=1.5, K=2.0)
        sim = RickerMapSimulation(config)
        fps = sim.find_fixed_points()
        # Should find x*=0 and x*=K=2.0
        x_values = [fp["x"] for fp in fps]
        assert 0.0 in x_values
        assert 2.0 in x_values
