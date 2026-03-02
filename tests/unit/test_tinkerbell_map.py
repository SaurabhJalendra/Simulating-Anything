"""Tests for the Tinkerbell map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.tinkerbell_map import TinkerbellMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 0.9,
    b: float = -0.6013,
    c: float = 2.0,
    d: float = 0.5,
    x_0: float = -0.72,
    y_0: float = -0.64,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.TINKERBELL_MAP,
        dt=1.0,
        n_steps=100,
        parameters={
            "a": a, "b": b, "c": c, "d": d,
            "x_0": x_0, "y_0": y_0,
        },
    )


class TestTinkerbellMapConstruction:
    """Tests for construction and parameter handling."""

    def test_creation_and_params(self):
        sim = TinkerbellMapSimulation(_make_config())
        assert sim.a == 0.9
        assert sim.b == -0.6013
        assert sim.c == 2.0
        assert sim.d == 0.5

    def test_default_parameters(self):
        """Config with no parameters uses defaults."""
        config = SimulationConfig(
            domain=Domain.TINKERBELL_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = TinkerbellMapSimulation(config)
        assert sim.a == 0.9
        assert sim.b == -0.6013
        assert sim.c == 2.0
        assert sim.d == 0.5
        assert sim.x_0 == -0.72
        assert sim.y_0 == -0.64

    def test_custom_parameters(self):
        config = _make_config(a=0.5, b=-0.5, c=1.5, d=0.3)
        sim = TinkerbellMapSimulation(config)
        assert sim.a == 0.5
        assert sim.b == -0.5
        assert sim.c == 1.5
        assert sim.d == 0.3


class TestTinkerbellMapState:
    """Tests for state initialization and shape."""

    def test_initial_state_shape(self):
        sim = TinkerbellMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = TinkerbellMapSimulation(_make_config(x_0=-0.72, y_0=-0.64))
        state = sim.reset()
        assert state[0] == pytest.approx(-0.72)
        assert state[1] == pytest.approx(-0.64)

    def test_observe_returns_current_state(self):
        sim = TinkerbellMapSimulation(_make_config(x_0=0.1, y_0=0.2))
        sim.reset()
        state = sim.observe()
        assert state[0] == pytest.approx(0.1)
        assert state[1] == pytest.approx(0.2)


class TestTinkerbellMapDynamics:
    """Tests for the map iteration."""

    def test_step_applies_map_correctly(self):
        """Verify one step of the map with known values."""
        sim = TinkerbellMapSimulation(
            _make_config(a=0.9, b=-0.6013, c=2.0, d=0.5, x_0=0.1, y_0=0.2)
        )
        sim.reset()
        sim.step()
        # x_1 = 0.1^2 - 0.2^2 + 0.9*0.1 + (-0.6013)*0.2
        #      = 0.01 - 0.04 + 0.09 - 0.12026 = -0.06026
        # y_1 = 2*0.1*0.2 + 2.0*0.1 + 0.5*0.2
        #      = 0.04 + 0.2 + 0.1 = 0.34
        assert sim.observe()[0] == pytest.approx(-0.06026, abs=1e-10)
        assert sim.observe()[1] == pytest.approx(0.34, abs=1e-10)

    def test_step_from_origin(self):
        """Starting from origin: x_1 = 0, y_1 = 0 (trivial fixed point if origin is FP)."""
        sim = TinkerbellMapSimulation(
            _make_config(a=0.9, b=-0.6013, c=2.0, d=0.5, x_0=0.0, y_0=0.0)
        )
        sim.reset()
        sim.step()
        # x_1 = 0 + 0 + 0 + 0 = 0
        # y_1 = 0 + 0 + 0 = 0
        assert sim.observe()[0] == pytest.approx(0.0, abs=1e-15)
        assert sim.observe()[1] == pytest.approx(0.0, abs=1e-15)

    def test_step_count_increments(self):
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_origin_is_fixed_point(self):
        """The origin (0,0) is always a fixed point of the Tinkerbell map."""
        sim = TinkerbellMapSimulation(
            _make_config(x_0=0.0, y_0=0.0)
        )
        sim.reset()
        for _ in range(10):
            sim.step()
        assert sim.observe()[0] == pytest.approx(0.0, abs=1e-12)
        assert sim.observe()[1] == pytest.approx(0.0, abs=1e-12)


class TestTinkerbellMapJacobian:
    """Tests for the Jacobian computation."""

    def test_jacobian_shape(self):
        sim = TinkerbellMapSimulation(_make_config())
        jac = sim.compute_jacobian(0.0, 0.0)
        assert jac.shape == (2, 2)

    def test_jacobian_at_origin(self):
        """At origin: J = [[a, b], [c, d]]."""
        sim = TinkerbellMapSimulation(
            _make_config(a=0.9, b=-0.6013, c=2.0, d=0.5)
        )
        jac = sim.compute_jacobian(0.0, 0.0)
        assert jac[0, 0] == pytest.approx(0.9)
        assert jac[0, 1] == pytest.approx(-0.6013)
        assert jac[1, 0] == pytest.approx(2.0)
        assert jac[1, 1] == pytest.approx(0.5)

    def test_jacobian_at_nonzero_point(self):
        """J at (x,y) = [[2x+a, -2y+b], [2y+c, 2x+d]]."""
        sim = TinkerbellMapSimulation(
            _make_config(a=0.9, b=-0.6013, c=2.0, d=0.5)
        )
        x, y = 0.5, -0.3
        jac = sim.compute_jacobian(x, y)
        assert jac[0, 0] == pytest.approx(2 * 0.5 + 0.9)
        assert jac[0, 1] == pytest.approx(-2 * (-0.3) + (-0.6013))
        assert jac[1, 0] == pytest.approx(2 * (-0.3) + 2.0)
        assert jac[1, 1] == pytest.approx(2 * 0.5 + 0.5)

    def test_jacobian_determinant_at_origin(self):
        """det(J) at origin = a*d - b*c."""
        sim = TinkerbellMapSimulation(
            _make_config(a=0.9, b=-0.6013, c=2.0, d=0.5)
        )
        det = sim.jacobian_determinant_at(0.0, 0.0)
        expected = 0.9 * 0.5 - (-0.6013) * 2.0  # 0.45 + 1.2026 = 1.6526
        assert det == pytest.approx(expected, abs=1e-10)

    def test_jacobian_determinant_consistency(self):
        """det(J) computed via method matches np.linalg.det."""
        sim = TinkerbellMapSimulation(_make_config())
        x, y = 0.3, -0.5
        det_method = sim.jacobian_determinant_at(x, y)
        jac = sim.compute_jacobian(x, y)
        det_numpy = np.linalg.det(jac)
        assert det_method == pytest.approx(det_numpy, abs=1e-10)


class TestTinkerbellMapFixedPoints:
    """Tests for fixed point computation."""

    def test_origin_in_fixed_points(self):
        """The origin (0,0) should always be a fixed point."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        origin_found = any(
            np.linalg.norm(fp) < 1e-6 for fp in fps
        )
        assert origin_found, "Origin should be a fixed point"

    def test_fixed_points_satisfy_map(self):
        """Each fixed point should satisfy f(z*) = z*."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        assert len(fps) >= 1
        for fp in fps:
            x, y = fp
            x_new = x**2 - y**2 + sim.a * x + sim.b * y
            y_new = 2.0 * x * y + sim.c * x + sim.d * y
            assert x_new == pytest.approx(x, abs=1e-8)
            assert y_new == pytest.approx(y, abs=1e-8)

    def test_fixed_points_are_distinct(self):
        """All returned fixed points should be distinct."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                dist = np.linalg.norm(fps[i] - fps[j])
                assert dist > 1e-6, (
                    f"FP {i} and FP {j} too close: dist={dist}"
                )

    def test_at_least_two_fixed_points(self):
        """The Tinkerbell map should have at least two fixed points
        (origin and one nontrivial)."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        assert len(fps) >= 2


class TestTinkerbellMapChaos:
    """Tests for chaotic behavior and Lyapunov exponents."""

    def test_positive_lyapunov_classic(self):
        """At classic parameters the orbit should be chaotic (positive Lyapunov)."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=1000)
        assert lam > 0.0, f"Lyapunov={lam} should be positive for chaotic regime"

    def test_lyapunov_spectrum_shape(self):
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        spectrum = sim.compute_lyapunov_spectrum(
            n_iterations=5000, n_transient=500,
        )
        assert spectrum.shape == (2,)

    def test_lyapunov_spectrum_ordering(self):
        """Spectrum should be sorted descending: lambda_1 >= lambda_2."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        spectrum = sim.compute_lyapunov_spectrum(
            n_iterations=5000, n_transient=500,
        )
        assert spectrum[0] >= spectrum[1]

    def test_lyapunov_spectrum_sum_negative(self):
        """For a dissipative attractor, sum of Lyapunov exponents should be
        negative (volume contracts on average)."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        spectrum = sim.compute_lyapunov_spectrum(
            n_iterations=10000, n_transient=1000,
        )
        # For strange attractors in dissipative maps, sum < 0
        assert spectrum[0] + spectrum[1] < 0, (
            f"Sum of Lyapunov exponents {spectrum[0] + spectrum[1]:.4f} "
            f"should be negative for dissipative attractor"
        )


class TestTinkerbellMapAttractor:
    """Tests for the strange attractor."""

    def test_trajectory_bounded(self):
        """Trajectory at classic parameters should remain bounded."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            sim.step()
            x, y = sim.observe()
            assert abs(x) < 5, f"x={x} unbounded"
            assert abs(y) < 5, f"y={y} unbounded"

    def test_attractor_spans_nonzero_area(self):
        """The attractor should span a nonzero region of the plane."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        points = sim.collect_attractor(n_steps=5000, n_transient=500)
        x_range = np.ptp(points[:, 0])
        y_range = np.ptp(points[:, 1])
        assert x_range > 0.1, f"x range {x_range} too small"
        assert y_range > 0.1, f"y range {y_range} too small"

    def test_attractor_points_bounded(self):
        """All attractor points should lie within expected bounds."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        points = sim.collect_attractor(n_steps=5000, n_transient=500)
        assert np.all(np.abs(points) < 5), "Attractor points exceed bounds"

    def test_collect_attractor_shape(self):
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        points = sim.collect_attractor(n_steps=100, n_transient=100)
        assert points.shape == (100, 2)


class TestTinkerbellMapBifurcation:
    """Tests for bifurcation diagram and period detection."""

    def test_bifurcation_diagram_returns_data(self):
        sim = TinkerbellMapSimulation(_make_config())
        a_vals = np.linspace(-0.3, 0.9, 10)
        data = sim.bifurcation_diagram(a_vals, n_transient=100, n_plot=10)
        assert "a" in data
        assert "x" in data
        assert len(data["a"]) > 0
        assert len(data["a"]) == len(data["x"])

    def test_period_detection_chaotic(self):
        """At classic chaotic parameters, period should be -1 (chaotic)."""
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=1000)
        assert period == -1, (
            f"Period={period}, expected -1 for chaotic regime"
        )

    def test_period_detection_fixed_point(self):
        """At the origin (a trivial fixed point), period should be 1."""
        sim = TinkerbellMapSimulation(
            _make_config(x_0=0.0, y_0=0.0)
        )
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=100)
        assert period == 1


class TestTinkerbellMapTrajectory:
    """Tests for trajectory output."""

    def test_run_trajectory_shape(self):
        sim = TinkerbellMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 2)

    def test_run_trajectory_timestamps(self):
        sim = TinkerbellMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        assert len(traj.timestamps) == 51
        assert traj.timestamps[0] == pytest.approx(0.0)

    def test_run_trajectory_parameters(self):
        sim = TinkerbellMapSimulation(_make_config(a=0.8))
        traj = sim.run(n_steps=10)
        assert traj.parameters["a"] == pytest.approx(0.8)


class TestTinkerbellMapReproducibility:
    """Tests for reproducibility and determinism."""

    def test_deterministic_evolution(self):
        """Two runs with same IC should produce identical trajectories."""
        sim1 = TinkerbellMapSimulation(_make_config())
        sim2 = TinkerbellMapSimulation(_make_config())
        t1 = sim1.run(n_steps=100)
        t2 = sim2.run(n_steps=100)
        np.testing.assert_array_equal(t1.states, t2.states)

    def test_reset_restores_initial(self):
        sim = TinkerbellMapSimulation(_make_config())
        sim.reset()
        for _ in range(50):
            sim.step()
        sim.reset()
        assert sim.observe()[0] == pytest.approx(-0.72)
        assert sim.observe()[1] == pytest.approx(-0.64)


class TestTinkerbellMapRediscovery:
    """Tests for the rediscovery module."""

    def test_lyapunov_data_generation(self):
        from simulating_anything.rediscovery.tinkerbell_map import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_a=10, a_min=0.0, a_max=1.0)
        assert len(data["a"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.tinkerbell_map import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_a=10, a_min=0.0, a_max=1.0)
        assert len(data["a_values"]) == 10
        assert len(data["periods"]) == 10
