"""Tests for the Lozi map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.lozi_map import LoziMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 1.7,
    b: float = 0.5,
    x_0: float = 0.0,
    y_0: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.LOZI_MAP,  
        dt=1.0,
        n_steps=100,
        parameters={"a": a, "b": b, "x_0": x_0, "y_0": y_0},
    )


class TestLoziMapConstruction:
    def test_creation_and_params(self):
        sim = LoziMapSimulation(_make_config())
        assert sim.a == 1.7
        assert sim.b == 0.5

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.LOZI_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = LoziMapSimulation(config)
        assert sim.a == 1.7
        assert sim.b == 0.5
        assert sim.x_0 == 0.0
        assert sim.y_0 == 0.0

    def test_custom_parameters(self):
        sim = LoziMapSimulation(_make_config(a=1.5, b=0.3, x_0=0.1, y_0=-0.2))
        assert sim.a == 1.5
        assert sim.b == 0.3
        assert sim.x_0 == 0.1
        assert sim.y_0 == -0.2


class TestLoziMapState:
    def test_initial_state_shape(self):
        sim = LoziMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == pytest.approx(0.0)
        assert state[1] == pytest.approx(0.0)

    def test_observe_returns_current_state(self):
        sim = LoziMapSimulation(_make_config(x_0=0.5, y_0=0.1))
        sim.reset()
        state = sim.observe()
        assert state[0] == pytest.approx(0.5)
        assert state[1] == pytest.approx(0.1)

    def test_observe_shape(self):
        sim = LoziMapSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_reset_restores_initial(self):
        sim = LoziMapSimulation(_make_config(x_0=0.3, y_0=-0.1))
        sim.reset()
        sim.step()
        sim.step()
        state = sim.reset()
        assert state[0] == pytest.approx(0.3)
        assert state[1] == pytest.approx(-0.1)


class TestLoziMapStep:
    def test_step_applies_map_positive_x(self):
        """Test map iteration with x > 0."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5, x_0=0.5, y_0=0.1))
        sim.reset()
        sim.step()
        # x_1 = 1 - 1.7*|0.5| + 0.1 = 1 - 0.85 + 0.1 = 0.25
        # y_1 = 0.5 * 0.5 = 0.25
        assert sim.observe()[0] == pytest.approx(0.25)
        assert sim.observe()[1] == pytest.approx(0.25)

    def test_step_applies_map_negative_x(self):
        """Test map iteration with x < 0."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5, x_0=-0.5, y_0=0.1))
        sim.reset()
        sim.step()
        # x_1 = 1 - 1.7*|-0.5| + 0.1 = 1 - 0.85 + 0.1 = 0.25
        # y_1 = 0.5 * (-0.5) = -0.25
        assert sim.observe()[0] == pytest.approx(0.25)
        assert sim.observe()[1] == pytest.approx(-0.25)

    def test_step_applies_map_zero_x(self):
        """Test map iteration with x = 0."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5, x_0=0.0, y_0=0.3))
        sim.reset()
        sim.step()
        # x_1 = 1 - 1.7*|0| + 0.3 = 1.3
        # y_1 = 0.5 * 0 = 0
        assert sim.observe()[0] == pytest.approx(1.3)
        assert sim.observe()[1] == pytest.approx(0.0)

    def test_piecewise_linearity(self):
        """Verify |x| nonlinearity: same |x| with different signs gives same |x_new|
        but different y_new."""
        sim_pos = LoziMapSimulation(_make_config(a=1.7, b=0.5, x_0=0.4, y_0=0.0))
        sim_neg = LoziMapSimulation(_make_config(a=1.7, b=0.5, x_0=-0.4, y_0=0.0))
        sim_pos.reset()
        sim_neg.reset()
        s_pos = sim_pos.step()
        s_neg = sim_neg.step()
        # x_new should be the same (both use |0.4|)
        assert s_pos[0] == pytest.approx(s_neg[0])
        # y_new should differ in sign (b*x)
        assert s_pos[1] == pytest.approx(-s_neg[1])

    def test_deterministic(self):
        """Same parameters produce the same orbit."""
        sim1 = LoziMapSimulation(_make_config(a=1.5, b=0.3, x_0=0.1, y_0=0.1))
        sim2 = LoziMapSimulation(_make_config(a=1.5, b=0.3, x_0=0.1, y_0=0.1))
        sim1.reset()
        sim2.reset()
        for _ in range(50):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)


class TestLoziMapFixedPoints:
    def test_fixed_point_positive_branch(self):
        """Positive fixed point x* = 1/(1+a-b)."""
        a, b = 1.7, 0.5
        sim = LoziMapSimulation(_make_config(a=a, b=b))
        fps = sim.fixed_points
        # Should have at least one fixed point from positive branch
        x_expected = 1.0 / (1.0 + a - b)
        y_expected = b * x_expected
        found = False
        for fp in fps:
            if abs(fp[0] - x_expected) < 1e-10:
                assert fp[1] == pytest.approx(y_expected, abs=1e-10)
                found = True
        assert found, f"Positive fixed point ({x_expected}, {y_expected}) not found"

    def test_fixed_point_is_equilibrium(self):
        """Verify that f(x*) = x* for each fixed point."""
        sim = LoziMapSimulation(_make_config(a=1.5, b=0.4))
        fps = sim.fixed_points
        assert len(fps) >= 1
        for fp in fps:
            x, y = fp
            x_new = 1.0 - sim.a * abs(x) + y
            y_new = sim.b * x
            assert x_new == pytest.approx(x, abs=1e-10)
            assert y_new == pytest.approx(y, abs=1e-10)

    def test_fixed_point_y_equals_bx(self):
        """At a fixed point, y* = b*x*."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        fps = sim.fixed_points
        for fp in fps:
            assert fp[1] == pytest.approx(sim.b * fp[0], abs=1e-10)

    def test_negative_branch_fixed_point(self):
        """When 1 - a - b < 0 and a + b > 1, negative branch gives x* < 0."""
        # a=1.7, b=0.5: 1-a-b = -1.2, x* = 1/(-1.2) < 0
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        fps = sim.fixed_points
        neg_fps = [fp for fp in fps if fp[0] < 0]
        assert len(neg_fps) == 1
        x_expected = 1.0 / (1.0 - 1.7 - 0.5)
        assert neg_fps[0][0] == pytest.approx(x_expected, abs=1e-10)


class TestLoziMapJacobian:
    def test_jacobian_determinant_equals_neg_b(self):
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        assert sim.jacobian_determinant == pytest.approx(-0.5)

    def test_jacobian_determinant_area_contraction(self):
        """Area contracts by |b| each iteration."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        assert abs(sim.jacobian_determinant) == pytest.approx(0.5)

    def test_jacobian_at_positive_x(self):
        """Jacobian for x > 0: [[-a, 1], [b, 0]]."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        jac = sim.jacobian_at(0.5)
        expected = np.array([[-1.7, 1.0], [0.5, 0.0]])
        np.testing.assert_allclose(jac, expected)

    def test_jacobian_at_negative_x(self):
        """Jacobian for x < 0: [[a, 1], [b, 0]]."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        jac = sim.jacobian_at(-0.5)
        expected = np.array([[1.7, 1.0], [0.5, 0.0]])
        np.testing.assert_allclose(jac, expected)

    def test_jacobian_det_from_matrix(self):
        """det(J) computed from the matrix matches the property."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        jac = sim.jacobian_at(0.3)
        det = np.linalg.det(jac)
        assert det == pytest.approx(sim.jacobian_determinant, abs=1e-12)

    def test_jacobian_det_constant_along_orbit(self):
        """det(J) is constant regardless of orbit point."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        sim.reset()
        for _ in range(100):
            sim.step()
            x = sim.observe()[0]
            if abs(x) > 1e-10:
                jac = sim.jacobian_at(x)
                det = np.linalg.det(jac)
                assert det == pytest.approx(-0.5, abs=1e-12)


class TestLoziMapChaos:
    def test_chaotic_positive_lyapunov(self):
        """At a=1.7, b=0.5 the largest Lyapunov exponent should be positive."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=500)
        assert lam > 0.2, f"Lyapunov={lam} should be > 0.2 for chaotic regime"

    def test_non_chaotic_negative_lyapunov(self):
        """At small a, the orbit should be stable (negative Lyapunov)."""
        sim = LoziMapSimulation(_make_config(a=0.3, b=0.3))
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        assert lam < 0, f"Lyapunov={lam} should be negative for non-chaotic a=0.3"

    def test_lyapunov_sum_equals_ln_b(self):
        """Sum of Lyapunov exponents equals ln|b| for 2D dissipative map."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        sim.reset()
        spectrum = sim.compute_lyapunov_spectrum(
            n_iterations=50000, n_transient=1000,
        )
        lyap_sum = spectrum[0] + spectrum[1]
        ln_b = np.log(abs(sim.b))
        np.testing.assert_allclose(lyap_sum, ln_b, atol=0.05)


class TestLoziMapTrajectory:
    def test_trajectory_bounded(self):
        """Trajectory stays bounded for standard parameters."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        sim.reset()
        for _ in range(5000):
            sim.step()
            x, y = sim.observe()
            assert abs(x) < 10, f"x={x} unbounded"
            assert abs(y) < 10, f"y={y} unbounded"

    def test_trajectory_no_nan(self):
        """No NaN after many iterations at chaotic parameters."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        sim.reset()
        for _ in range(50000):
            state = sim.step()
        assert np.all(np.isfinite(state)), "NaN detected after 50000 steps"

    def test_run_trajectory_shape(self):
        sim = LoziMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 2)

    def test_b_zero_reduces_to_1d(self):
        """When b=0, y is always 0 and x becomes a tent-like 1D map."""
        sim = LoziMapSimulation(_make_config(a=1.0, b=0.0, x_0=0.5, y_0=0.0))
        sim.reset()
        x = 0.5
        for _ in range(10):
            sim.step()
            x = 1.0 - 1.0 * abs(x)
            assert sim.observe()[0] == pytest.approx(x, abs=1e-12)
            assert sim.observe()[1] == pytest.approx(0.0, abs=1e-12)

    def test_periodic_orbit_detection_stable(self):
        """At small a, the orbit should converge to a fixed point (period 1)."""
        sim = LoziMapSimulation(_make_config(a=0.5, b=0.3))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=1000)
        assert period == 1

    def test_chaotic_period_detection(self):
        """At chaotic parameters, detect_period should return -1."""
        sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=1000)
        assert period == -1


class TestLoziMapBifurcation:
    def test_bifurcation_diagram(self):
        sim = LoziMapSimulation(_make_config())
        a_vals = np.linspace(0.0, 1.7, 10)
        data = sim.bifurcation_diagram(a_vals, n_transient=100, n_plot=10)
        assert len(data["a"]) == 100  # 10 a * 10 plot points
        assert len(data["x"]) == 100

    def test_bifurcation_data_bounded(self):
        """All bifurcation diagram points should be finite."""
        sim = LoziMapSimulation(_make_config())
        a_vals = np.linspace(0.1, 1.7, 20)
        data = sim.bifurcation_diagram(a_vals, n_transient=500, n_plot=50)
        assert np.all(np.isfinite(data["x"]))
        assert np.all(np.isfinite(data["a"]))


class TestLoziMapRediscovery:
    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.lozi_map import generate_bifurcation_data
        data = generate_bifurcation_data(n_a=20, a_min=0.0, a_max=1.7)
        assert len(data["a_values"]) == 20
        assert len(data["periods"]) == 20

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.lozi_map import generate_lyapunov_data
        data = generate_lyapunov_data(n_a=10, a_min=0.0, a_max=1.7)
        assert len(data["a"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_fixed_point_data(self):
        from simulating_anything.rediscovery.lozi_map import generate_fixed_point_data
        data = generate_fixed_point_data(n_a=5, a_min=0.5, a_max=1.5)
        assert len(data["fixed_points"]) == 5
        for entry in data["fixed_points"]:
            assert "a" in entry
            assert "n_fixed_points" in entry
