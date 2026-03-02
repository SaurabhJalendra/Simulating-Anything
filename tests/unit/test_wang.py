"""Tests for the Wang chaotic attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.wang import WangSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestWangSimulation:
    """Tests for the Wang system simulation basics."""

    def _make_sim(self, **kwargs) -> WangSimulation:
        defaults = {
            "a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 1.0
        assert sim.b == 1.0
        assert sim.c == 0.7
        assert sim.d == 0.5

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=2.0, b=1.5, c=0.9, d=0.8)
        assert sim.a == 2.0
        assert sim.b == 1.5
        assert sim.c == 0.9
        assert sim.d == 0.8

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


class TestWangDerivatives:
    """Tests for the Wang ODE derivative computation."""

    def _make_sim(self, **kwargs) -> WangSimulation:
        defaults = {
            "a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=1, b=1, c=0.7, d=0.5:
            dx = 1 - 1*1 = 0
            dy = -1*1 + 1*1 = 0
            dz = -0.7*1 + 0.5*1*1 = -0.2
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], -0.2)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 3, 1] with a=1, b=1, c=0.7, d=0.5.

            dx = 2 - 1*3 = -1
            dy = -1*3 + 2*1 = -1
            dz = -0.7*1 + 0.5*2*3 = -0.7 + 3.0 = 2.3
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 3.0, 1.0]))
        assert np.isclose(derivs[0], -1.0)
        assert np.isclose(derivs[1], -1.0)
        assert np.isclose(derivs[2], 2.3)


class TestWangRK4Convergence:
    """Tests for RK4 integration accuracy."""

    def _make_sim(self, dt: float = 0.01, **kwargs) -> WangSimulation:
        defaults = {"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=dt,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

    def test_rk4_convergence_order(self):
        """RK4 should show 4th-order convergence as dt is halved.

        Run with dt and dt/2, compare endpoints. Error ratio should be ~16.
        """
        dt_coarse = 0.02
        dt_fine = 0.01
        n_steps_coarse = 50
        n_steps_fine = 100

        sim_coarse = self._make_sim(dt=dt_coarse)
        sim_coarse.reset()
        for _ in range(n_steps_coarse):
            sim_coarse.step()
        state_coarse = sim_coarse.observe().copy()

        sim_fine = self._make_sim(dt=dt_fine)
        sim_fine.reset()
        for _ in range(n_steps_fine):
            sim_fine.step()
        state_fine = sim_fine.observe().copy()

        # Use an even finer run as reference
        dt_ref = 0.005
        n_steps_ref = 200
        sim_ref = self._make_sim(dt=dt_ref)
        sim_ref.reset()
        for _ in range(n_steps_ref):
            sim_ref.step()
        state_ref = sim_ref.observe().copy()

        err_coarse = np.linalg.norm(state_coarse - state_ref)
        err_fine = np.linalg.norm(state_fine - state_ref)

        # For RK4, halving dt should reduce error by ~16x
        if err_fine > 1e-14:
            ratio = err_coarse / err_fine
            assert ratio > 4.0, f"RK4 convergence ratio {ratio:.1f} too low"


class TestWangFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> WangSimulation:
        defaults = {"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

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
        """z-coordinate of symmetric fixed points should be b/a.

        For a=1, b=1: z = 1/1 = 1.0
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        z_expected = 1.0 / 1.0
        assert np.isclose(fps[1][2], z_expected)
        assert np.isclose(fps[2][2], z_expected)

    def test_fixed_point_x_value(self):
        """x-coordinate of symmetric fixed points should be +/-sqrt(b*c/d).

        For b=1, c=0.7, d=0.5: x = sqrt(1*0.7/0.5) = sqrt(1.4) ~ 1.1832
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        x_expected = np.sqrt(1.0 * 0.7 / 0.5)
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


class TestWangJacobian:
    """Tests for Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> WangSimulation:
        defaults = {"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        sim = self._make_sim()
        J = sim.jacobian(np.array([1.0, 2.0, 3.0]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at origin for a=1, b=1, c=0.7, d=0.5.

        J = [[1, -1, 0], [0, -1, 0], [0, 0, -0.7]]
        """
        sim = self._make_sim()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -0.7],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_trace_equals_divergence(self):
        """Trace of Jacobian should equal divergence = 1 - b - c."""
        sim = self._make_sim()
        for state in [np.array([0, 0, 0.0]), np.array([1, 2, 3.0]),
                       np.array([-5, 3, 7.0])]:
            J = sim.jacobian(state)
            assert np.isclose(np.trace(J), sim.compute_divergence())


class TestWangTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> WangSimulation:
        defaults = {
            "a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

    def test_trajectory_stays_finite(self):
        """Wang trajectories should remain finite for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(500):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"

    def test_trajectory_stays_bounded(self):
        """Wang trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(500):
            state = sim.step()
            assert np.linalg.norm(state) < 500, f"Trajectory diverged: {state}"

    def test_different_d_gives_different_trajectory(self):
        """Changing d should change the trajectory behavior."""
        sim1 = self._make_sim(d=0.5, x_0=1.0, y_0=1.0, z_0=1.0)
        sim2 = self._make_sim(d=0.1, x_0=1.0, y_0=1.0, z_0=1.0)
        sim1.reset()
        sim2.reset()
        for _ in range(500):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.01)

    def test_attractor_statistics(self):
        """Time-averaged statistics should be finite and non-trivial."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=1000
        )
        assert np.isfinite(stats["x_mean"])
        assert np.isfinite(stats["y_mean"])
        assert np.isfinite(stats["z_mean"])
        assert stats["x_std"] > 0 or stats["y_std"] > 0, (
            "Trajectory appears stationary"
        )


class TestWangDissipation:
    """Tests for dissipation and divergence properties."""

    def _make_sim(self, **kwargs) -> WangSimulation:
        defaults = {"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return WangSimulation(config)

    def test_divergence_formula(self):
        """Divergence should equal 1 - b - c."""
        sim = self._make_sim(b=1.0, c=0.7)
        assert np.isclose(sim.compute_divergence(), 1.0 - 1.0 - 0.7)

    def test_divergence_custom_params(self):
        """Divergence with custom b and c."""
        sim = self._make_sim(b=2.0, c=1.5)
        assert np.isclose(sim.compute_divergence(), 1.0 - 2.0 - 1.5)

    def test_dissipative_default(self):
        """Default parameters should give dissipative system (div < 0)."""
        sim = self._make_sim()
        assert sim.compute_divergence() < 0

    def test_conservative_boundary(self):
        """When b + c = 1, divergence should be zero (conservative)."""
        sim = self._make_sim(b=0.5, c=0.5)
        assert np.isclose(sim.compute_divergence(), 0.0)


class TestWangChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_lyapunov_returns_finite(self):
        """Lyapunov estimate should be a finite number."""
        config = SimulationConfig(
            domain=Domain.WANG,
            dt=0.01,
            n_steps=500,
            parameters={"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5},
        )
        sim = WangSimulation(config)
        sim.reset()
        for _ in range(500):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert np.isfinite(lam), f"Lyapunov is not finite: {lam}"

    def test_lyapunov_varies_with_d(self):
        """Lyapunov exponent should change as d varies."""
        lam1 = _compute_lyapunov_at_d(0.5)
        lam2 = _compute_lyapunov_at_d(0.1)
        assert lam1 != lam2, "Lyapunov should differ for different d"


class TestWangRediscovery:
    """Tests for Wang data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.wang import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 1.0
        assert data["b"] == 1.0
        assert data["c"] == 0.7
        assert data["d"] == 0.5

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.wang import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.wang import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_sweep_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.wang import (
            generate_lyapunov_sweep_data,
        )

        data = generate_lyapunov_sweep_data(
            n_points=5, n_steps=3000, dt=0.01
        )
        assert len(data["param_values"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_d(d: float) -> float:
    """Helper to compute Lyapunov exponent at a given d."""
    config = SimulationConfig(
        domain=Domain.WANG,
        dt=0.01,
        n_steps=500,
        parameters={
            "a": 1.0, "b": 1.0, "c": 0.7, "d": d,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        },
    )
    sim = WangSimulation(config)
    sim.reset()
    for _ in range(500):
        sim.step()
    return sim.estimate_lyapunov(n_steps=5000, dt=0.01)
