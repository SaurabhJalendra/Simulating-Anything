"""Tests for the Arneodo attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.arneodo import ArneodoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestArneodoSimulation:
    """Tests for the Arneodo system simulation basics."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {
            "a": 5.5, "b": 3.5, "d": 1.0,
            "x_0": 0.2, "y_0": 0.2, "z_0": 0.2,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 5.5
        assert sim.b == 3.5
        assert sim.d == 1.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=6.0, b=4.0, d=2.0)
        assert sim.a == 6.0
        assert sim.b == 4.0
        assert sim.d == 2.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=-0.5, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], -0.5)
        assert np.isclose(state[2], 3.0)

    def test_step_advances_state(self):
        """A single step changes the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_current_state(self):
        """observe() returns current state with correct shape."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_deterministic(self):
        """Same parameters produce the same trajectory."""
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_trajectory_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))


class TestArneodoDerivatives:
    """Tests for the Arneodo ODE derivative computation."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {
            "a": 5.5, "b": 3.5, "d": 1.0,
            "x_0": 0.2, "y_0": 0.2, "z_0": 0.2,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=5.5, b=3.5, d=1.0:
            dx = 1  (= y)
            dy = 1  (= z)
            dz = -5.5*1 - 3.5*1 - 1 + 1.0*1^3 = -5.5 - 3.5 - 1 + 1 = -9.0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], -9.0)

    def test_derivatives_cubic_nonlinearity(self):
        """Test that the cubic nonlinearity x^3 is correctly computed.

        At state [2, 0, 0] with a=5.5, b=3.5, d=1.0:
            dx = 0  (= y)
            dy = 0  (= z)
            dz = -5.5*2 - 3.5*0 - 0 + 1.0*2^3 = -11.0 + 8.0 = -3.0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], -3.0)

    def test_jerk_form_consistency(self):
        """The jerk computed from compute_jerk matches dz/dt from _derivatives."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([1.5, -0.3, 0.7])
        derivs = sim._derivatives(state)
        jerk = sim.compute_jerk(state)
        assert np.isclose(derivs[2], jerk)


class TestArneodoFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {"a": 5.5, "b": 3.5, "d": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_three_fixed_points(self):
        """Standard parameters (a/d > 0) should give three fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_origin_is_fixed_point(self):
        """First fixed point should be the origin."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_fixed_point_symmetry(self):
        """The two non-origin fixed points should be symmetric in x."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], 0.0)
        assert np.isclose(fps[2][1], 0.0)

    def test_fixed_point_x_value(self):
        """x-coordinate should be +/-sqrt(a/d).

        For a=5.5, d=1.0: x = sqrt(5.5) ~ 2.345
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        x_expected = np.sqrt(5.5 / 1.0)
        assert np.isclose(abs(fps[1][0]), x_expected)
        assert np.isclose(abs(fps[2][0]), x_expected)

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_only_origin_when_a_negative_d_positive(self):
        """When a/d < 0, only the origin exists as a fixed point."""
        sim = self._make_sim(a=-1.0, d=1.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])


class TestArneodoJacobian:
    """Tests for Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {"a": 5.5, "b": 3.5, "d": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be a 3x3 matrix."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at the origin should have standard jerk structure.

        J = [[0, 1, 0], [0, 0, 1], [-a, -b, -1]]
        """
        sim = self._make_sim(a=5.5, b=3.5, d=1.0)
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-5.5, -3.5, -1.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_trace_always_minus_one(self):
        """Trace of Jacobian should be -1 at any state (constant dissipation)."""
        sim = self._make_sim()
        sim.reset()
        for state in [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-5.0, 10.0, -7.0]),
            np.array([100.0, -50.0, 25.0]),
        ]:
            J = sim.jacobian(state)
            assert np.isclose(np.trace(J), -1.0), (
                f"Trace should be -1 at state {state}, got {np.trace(J)}"
            )


class TestArneodoDivergence:
    """Tests for the constant divergence property."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {"a": 5.5, "b": 3.5, "d": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_divergence_equals_minus_one(self):
        """Divergence should be exactly -1 everywhere."""
        sim = self._make_sim()
        sim.reset()
        assert sim.compute_divergence(np.array([0.0, 0.0, 0.0])) == -1.0
        assert sim.compute_divergence(np.array([10.0, -5.0, 3.0])) == -1.0

    def test_divergence_independent_of_parameters(self):
        """Divergence = -1 regardless of a, b, d parameter values."""
        for a, b, d in [(1.0, 1.0, 1.0), (10.0, 5.0, 3.0), (0.1, 0.5, 10.0)]:
            sim = self._make_sim(a=a, b=b, d=d)
            sim.reset()
            div = sim.compute_divergence(np.array([1.0, 2.0, 3.0]))
            assert div == -1.0, f"Divergence should be -1 for a={a}, b={b}, d={d}"


class TestArneodoTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {
            "a": 5.5, "b": 3.5, "d": 1.0,
            "x_0": 0.2, "y_0": 0.2, "z_0": 0.2,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Arneodo trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(500):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, (
                f"Trajectory diverged: {state}"
            )

    def test_different_d_gives_different_trajectory(self):
        """Changing d should change the trajectory behavior."""
        sim1 = self._make_sim(d=1.0)
        sim2 = self._make_sim(d=3.0)
        sim1.reset()
        sim2.reset()
        for _ in range(500):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.01)


class TestArneodoRK4Convergence:
    """Tests for RK4 integration accuracy."""

    def test_rk4_convergence_order(self):
        """RK4 should show 4th-order convergence when dt is halved.

        Run the same trajectory with dt and dt/2, compare the final states.
        The error ratio should be approximately 2^4 = 16.
        """
        dt1 = 0.01
        dt2 = dt1 / 2
        n_steps1 = 200
        n_steps2 = 400  # Same total time

        config1 = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=dt1,
            n_steps=n_steps1,
            parameters={"a": 5.5, "b": 3.5, "d": 1.0},
        )
        config2 = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=dt2,
            n_steps=n_steps2,
            parameters={"a": 5.5, "b": 3.5, "d": 1.0},
        )

        sim1 = ArneodoSimulation(config1)
        sim2 = ArneodoSimulation(config2)
        traj1 = sim1.run(n_steps=n_steps1)
        traj2 = sim2.run(n_steps=n_steps2)

        # Compare with a reference at dt/4
        dt3 = dt1 / 4
        n_steps3 = 800
        config3 = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=dt3,
            n_steps=n_steps3,
            parameters={"a": 5.5, "b": 3.5, "d": 1.0},
        )
        sim3 = ArneodoSimulation(config3)
        traj3 = sim3.run(n_steps=n_steps3)

        err1 = np.linalg.norm(traj1.states[-1] - traj3.states[-1])
        err2 = np.linalg.norm(traj2.states[-1] - traj3.states[-1])

        # For RK4, error ratio should be around 16 (2^4)
        if err2 > 1e-14:  # Avoid division by near-zero
            ratio = err1 / err2
            assert ratio > 5.0, (
                f"RK4 convergence ratio {ratio:.1f} too low (expected ~16)"
            )


class TestArneodoChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Arneodo at standard parameters should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=20000,
            parameters={"a": 5.5, "b": 3.5, "d": 1.0},
        )
        sim = ArneodoSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, f"Lyapunov {lam:.3f} should be positive for chaos"
        assert lam < 10.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_d(self):
        """Lyapunov exponent should change as d varies."""
        lyap_d1 = _compute_lyapunov_at_d(1.0)
        lyap_d05 = _compute_lyapunov_at_d(0.5)
        assert lyap_d1 != lyap_d05, (
            "Lyapunov should differ for different d values"
        )

    def test_bifurcation_sweep(self):
        """Bifurcation sweep should produce valid data."""
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters={"a": 5.5, "b": 3.5, "d": 1.0},
        )
        sim = ArneodoSimulation(config)
        sim.reset()
        d_values = np.array([0.5, 1.0, 2.0])
        result = sim.bifurcation_sweep(
            d_values, n_transient=1000, n_measure=5000
        )
        assert len(result["d"]) == 3
        assert len(result["lyapunov_exponent"]) == 3
        assert len(result["attractor_type"]) == 3


class TestArneodoJerkStructure:
    """Tests verifying the jerk (third-order ODE) structure."""

    def _make_sim(self, **kwargs) -> ArneodoSimulation:
        defaults = {"a": 5.5, "b": 3.5, "d": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ARNEODO,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ArneodoSimulation(config)

    def test_dx_equals_y(self):
        """First equation: dx/dt = y (jerk chain rule)."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([3.0, 7.0, -2.0])
        derivs = sim._derivatives(state)
        assert np.isclose(derivs[0], state[1])

    def test_dy_equals_z(self):
        """Second equation: dy/dt = z (jerk chain rule)."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([3.0, 7.0, -2.0])
        derivs = sim._derivatives(state)
        assert np.isclose(derivs[1], state[2])

    def test_jerk_form_matches_dz(self):
        """Third equation: dz/dt = jerk = -a*x - b*y - z + d*x^3."""
        sim = self._make_sim(a=5.5, b=3.5, d=1.0)
        sim.reset()
        state = np.array([2.0, 1.0, -3.0])
        derivs = sim._derivatives(state)
        # dz = -5.5*2 - 3.5*1 - (-3) + 1.0*8 = -11 - 3.5 + 3 + 8 = -3.5
        expected_dz = -5.5 * 2.0 - 3.5 * 1.0 - (-3.0) + 1.0 * 2.0**3
        assert np.isclose(derivs[2], expected_dz)


class TestArneodoRediscovery:
    """Tests for Arneodo data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.arneodo import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 5.5
        assert data["b"] == 3.5
        assert data["d"] == 1.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.arneodo import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.arneodo import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(n_d=5, n_steps=2000, dt=0.01)
        assert len(data["d"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.arneodo import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_d_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.arneodo import (
            generate_lyapunov_vs_d_data,
        )

        data = generate_lyapunov_vs_d_data(n_d=5, n_steps=3000, dt=0.01)
        assert len(data["d"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_d(d: float) -> float:
    """Helper to compute Lyapunov exponent at a given d."""
    config = SimulationConfig(
        domain=Domain.ARNEODO,
        dt=0.01,
        n_steps=20000,
        parameters={
            "a": 5.5, "b": 3.5, "d": d,
            "x_0": 0.2, "y_0": 0.2, "z_0": 0.2,
        },
    )
    sim = ArneodoSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.01)
