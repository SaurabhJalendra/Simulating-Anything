"""Tests for the Tigan (T-system) simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.tigan import TiganSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestTiganSimulation:
    """Tests for the Tigan system simulation basics."""

    def _make_sim(self, **kwargs) -> TiganSimulation:
        defaults = {
            "a": 2.1, "b": 0.6, "c": 30.0,
            "x_0": 0.1, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return TiganSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 2.1
        assert sim.b == 0.6
        assert sim.c == 30.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=3.0, b=1.0, c=25.0)
        assert sim.a == 3.0
        assert sim.b == 1.0
        assert sim.c == 25.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=5.0, y_0=-3.0, z_0=20.0)
        state = sim.reset()
        assert np.isclose(state[0], 5.0)
        assert np.isclose(state[1], -3.0)
        assert np.isclose(state[2], 20.0)

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
        """Same parameters should produce the same trajectory."""
        sim1 = self._make_sim(a=2.1, b=0.6, c=30.0)
        sim2 = self._make_sim(a=2.1, b=0.6, c=30.0)
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


class TestTiganDerivatives:
    """Tests for the Tigan ODE derivative computation."""

    def _make_sim(self, **kwargs) -> TiganSimulation:
        defaults = {
            "a": 2.1, "b": 0.6, "c": 30.0,
            "x_0": 0.1, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return TiganSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=2.1, b=0.6, c=30:
            dx = 2.1*(1 - 1) = 0
            dy = (30 - 2.1)*1 - 2.1*1*1 = 27.9 - 2.1 = 25.8
            dz = -0.6*1 + 1*1 = 0.4
        """
        sim = self._make_sim(a=2.1, b=0.6, c=30.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 25.8)
        assert np.isclose(derivs[2], 0.4)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 3, 1] with a=2.1, b=0.6, c=30.

            dx = 2.1*(3 - 2) = 2.1
            dy = (30 - 2.1)*2 - 2.1*2*1 = 55.8 - 4.2 = 51.6
            dz = -0.6*1 + 2*3 = -0.6 + 6 = 5.4
        """
        sim = self._make_sim(a=2.1, b=0.6, c=30.0)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 3.0, 1.0]))
        assert np.isclose(derivs[0], 2.1)
        assert np.isclose(derivs[1], 51.6)
        assert np.isclose(derivs[2], 5.4)


class TestTiganFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> TiganSimulation:
        defaults = {"a": 2.1, "b": 0.6, "c": 30.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return TiganSimulation(config)

    def test_three_fixed_points(self):
        """Standard parameters should give three fixed points."""
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
        """The two non-origin fixed points should be symmetric in x, y."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], -fps[2][1])
        assert np.isclose(fps[1][2], fps[2][2])

    def test_fixed_point_z_value(self):
        """z-coordinate of symmetric fixed points should be (c - a)/a.

        For a=2.1, b=0.6, c=30: z = (30 - 2.1)/2.1 = 27.9/2.1
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        z_expected = (30.0 - 2.1) / 2.1
        assert np.isclose(fps[1][2], z_expected)
        assert np.isclose(fps[2][2], z_expected)

    def test_fixed_point_x_value(self):
        """x-coordinate should be +/-sqrt(b*(c-a)/a).

        For a=2.1, b=0.6, c=30: x = sqrt(0.6*(30-2.1)/2.1)
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        x_expected = np.sqrt(0.6 * (30.0 - 2.1) / 2.1)
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


class TestTiganJacobian:
    """Tests for Jacobian computation."""

    def _make_sim(self, **kwargs) -> TiganSimulation:
        defaults = {"a": 2.1, "b": 0.6, "c": 30.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return TiganSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be a 3x3 matrix."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([1.0, 1.0, 1.0]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should have known structure.

        J(0,0,0) = [[-a, a, 0], [(c-a), 0, 0], [0, 0, -b]]
        """
        sim = self._make_sim(a=2.1, b=0.6, c=30.0)
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-2.1, 2.1, 0.0],
            [27.9, 0.0, 0.0],
            [0.0, 0.0, -0.6],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_trace_equals_divergence(self):
        """Trace of Jacobian should equal -(a + b) at any point."""
        sim = self._make_sim()
        sim.reset()
        for pt in [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-5.0, 10.0, 7.0]),
        ]:
            J = sim.jacobian(pt)
            trace = np.trace(J)
            assert np.isclose(trace, -(sim.a + sim.b))


class TestTiganDivergence:
    """Tests for divergence computation."""

    def _make_sim(self, **kwargs) -> TiganSimulation:
        defaults = {"a": 2.1, "b": 0.6, "c": 30.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return TiganSimulation(config)

    def test_divergence_standard_params(self):
        """Divergence should be -(a + b) = -(2.1 + 0.6) = -2.7."""
        sim = self._make_sim(a=2.1, b=0.6)
        assert np.isclose(sim.compute_divergence(), -2.7)

    def test_divergence_always_negative(self):
        """Divergence should be negative for positive a, b (dissipative)."""
        for a, b in [(1.0, 0.5), (5.0, 2.0), (0.1, 0.01)]:
            sim = self._make_sim(a=a, b=b)
            assert sim.compute_divergence() < 0

    def test_divergence_custom_params(self):
        """Divergence with a=5, b=2 should be -7."""
        sim = self._make_sim(a=5.0, b=2.0)
        assert np.isclose(sim.compute_divergence(), -7.0)


class TestTiganTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> TiganSimulation:
        defaults = {
            "a": 2.1, "b": 0.6, "c": 30.0,
            "x_0": 0.1, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return TiganSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Tigan trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 500, (
                f"Trajectory diverged: {state}"
            )

    def test_attractor_statistics(self):
        """Trajectory statistics should be reasonable for chaotic attractor."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # z_mean should be near (c-a)/a ~ 13.3 on the attractor
        assert stats["z_mean"] > 0, f"z_mean={stats['z_mean']:.1f}"
        # x and y should have nonzero spread (chaotic)
        assert stats["x_std"] > 0.5, "x_std too small for chaotic regime"
        assert stats["y_std"] > 0.5, "y_std too small for chaotic regime"

    def test_different_c_gives_different_trajectory(self):
        """Changing c should change the trajectory behavior."""
        sim1 = self._make_sim(c=30.0)
        sim2 = self._make_sim(c=10.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestTiganChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Tigan at standard parameters should have positive Lyapunov."""
        config = SimulationConfig(
            domain=Domain.TIGAN,
            dt=0.005,
            n_steps=20000,
            parameters={"a": 2.1, "b": 0.6, "c": 30.0},
        )
        sim = TiganSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert lam > 0.1, f"Lyapunov {lam:.3f} too small for chaos"
        assert lam < 20.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_c(self):
        """Lyapunov exponent should change as c varies."""
        lyap_c30 = _compute_lyapunov_at_c(30.0)
        lyap_c10 = _compute_lyapunov_at_c(10.0)
        assert lyap_c30 != lyap_c10, "Lyapunov should differ for different c"


class TestTiganRediscovery:
    """Tests for Tigan data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.tigan import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 2.1
        assert data["b"] == 0.6
        assert data["c"] == 30.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.tigan import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.tigan import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(
            n_c=5, n_steps=2000, dt=0.005
        )
        assert len(data["c"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.tigan import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.005)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_c_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.tigan import (
            generate_lyapunov_vs_c_data,
        )

        data = generate_lyapunov_vs_c_data(n_c=5, n_steps=3000, dt=0.005)
        assert len(data["c"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_c(c: float) -> float:
    """Helper to compute Lyapunov exponent at a given c."""
    config = SimulationConfig(
        domain=Domain.TIGAN,
        dt=0.005,
        n_steps=20000,
        parameters={
            "a": 2.1, "b": 0.6, "c": c,
            "x_0": 0.1, "y_0": 1.0, "z_0": 1.0,
        },
    )
    sim = TiganSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.005)
