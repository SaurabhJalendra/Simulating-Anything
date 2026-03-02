"""Tests for the Burke-Shaw chaotic system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.burke_shaw import BurkeShawSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestBurkeShawSimulation:
    """Tests for the Burke-Shaw system simulation basics."""

    def _make_sim(self, **kwargs) -> BurkeShawSimulation:
        defaults = {
            "s": 10.0, "v": 4.272,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return BurkeShawSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.s == 10.0
        assert sim.v == 4.272

    def test_creation_custom_params(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(s=5.0, v=2.0)
        assert sim.s == 5.0
        assert sim.v == 2.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=2.0, y_0=-1.0, z_0=0.5)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -1.0)
        assert np.isclose(state[2], 0.5)

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
        sim1 = self._make_sim(s=10.0, v=4.272, x_0=1.0, y_0=0.5, z_0=0.0)
        sim2 = self._make_sim(s=10.0, v=4.272, x_0=1.0, y_0=0.5, z_0=0.0)
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_step_count_increments(self):
        """Step count should increment after each step."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


class TestBurkeShawDerivatives:
    """Tests for the Burke-Shaw ODE derivatives."""

    def _make_sim(self, **kwargs) -> BurkeShawSimulation:
        defaults = {"s": 10.0, "v": 4.272}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return BurkeShawSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin: dx=-s*(0+0)=0, dy=-0-s*0*0=0, dz=s*0*0+v=v."""
        sim = self._make_sim(s=10.0, v=4.272)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 4.272)

    def test_derivatives_known_point(self):
        """Test derivatives at state [1, 1, 1] with s=10, v=4.272."""
        sim = self._make_sim(s=10.0, v=4.272)
        sim.reset()
        # At (1, 1, 1):
        # dx = -10*(1 + 1) = -20
        # dy = -1 - 10*1*1 = -11
        # dz = 10*1*1 + 4.272 = 14.272
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], -20.0)
        assert np.isclose(derivs[1], -11.0)
        assert np.isclose(derivs[2], 14.272)

    def test_derivatives_negative_point(self):
        """Test derivatives at state [-1, 2, -0.5] with s=10, v=4.272."""
        sim = self._make_sim(s=10.0, v=4.272)
        sim.reset()
        # dx = -10*(-1 + 2) = -10
        # dy = -2 - 10*(-1)*(-0.5) = -2 - 5 = -7
        # dz = 10*(-1)*2 + 4.272 = -20 + 4.272 = -15.728
        derivs = sim._derivatives(np.array([-1.0, 2.0, -0.5]))
        assert np.isclose(derivs[0], -10.0)
        assert np.isclose(derivs[1], -7.0)
        assert np.isclose(derivs[2], -15.728)


class TestBurkeShawFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> BurkeShawSimulation:
        defaults = {"s": 10.0, "v": 4.272}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return BurkeShawSimulation(config)

    def test_two_fixed_points(self):
        """Standard parameters should give exactly two fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 2

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

    def test_fixed_point_analytical_values(self):
        """Verify fixed points against analytical formulas.

        x = +/- sqrt(v/s), y = -x, z = 1/s
        """
        s, v = 10.0, 4.272
        sim = self._make_sim(s=s, v=v)
        sim.reset()
        fps = sim.fixed_points

        x_expected = np.sqrt(v / s)
        z_expected = 1.0 / s

        # FP1: positive x
        np.testing.assert_almost_equal(fps[0][0], x_expected, decimal=10)
        np.testing.assert_almost_equal(fps[0][1], -x_expected, decimal=10)
        np.testing.assert_almost_equal(fps[0][2], z_expected, decimal=10)

        # FP2: negative x
        np.testing.assert_almost_equal(fps[1][0], -x_expected, decimal=10)
        np.testing.assert_almost_equal(fps[1][1], x_expected, decimal=10)
        np.testing.assert_almost_equal(fps[1][2], z_expected, decimal=10)

    def test_fixed_points_symmetric(self):
        """The two fixed points should be related by (x,y,z) -> (-x,-y,z)."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 2
        np.testing.assert_almost_equal(fps[0][0], -fps[1][0], decimal=10)
        np.testing.assert_almost_equal(fps[0][1], -fps[1][1], decimal=10)
        np.testing.assert_almost_equal(fps[0][2], fps[1][2], decimal=10)


class TestBurkeShawTrajectory:
    """Tests for trajectory behavior."""

    def _make_sim(self, **kwargs) -> BurkeShawSimulation:
        defaults = {"s": 10.0, "v": 4.272}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return BurkeShawSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Burke-Shaw trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 100, f"Trajectory diverged: {state}"

    def test_trajectory_shape_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_stability_no_nan(self):
        """No NaN or Inf after many steps."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_attractor_is_nontrivial(self):
        """After transient, trajectory should not converge to a fixed point."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        # Collect states
        states = []
        for _ in range(1000):
            states.append(sim.step().copy())
        states = np.array(states)
        # Standard deviation of x should be nonzero (not stuck at fixed point)
        assert np.std(states[:, 0]) > 0.01, "Trajectory collapsed to a point"


class TestBurkeShawLyapunov:
    """Tests for Lyapunov exponent and chaos detection."""

    def test_chaotic_regime_positive_lyapunov(self):
        """At classic chaotic parameters, largest Lyapunov exponent should be positive."""
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=30000,
            parameters={"s": 10.0, "v": 4.272, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = BurkeShawSimulation(config)
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.005)
        assert lam > 0.01, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_is_chaotic_property(self):
        """is_chaotic property should be True at classic parameters."""
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=1000,
            parameters={"s": 10.0, "v": 4.272},
        )
        sim = BurkeShawSimulation(config)
        assert sim.is_chaotic is True

    def test_not_chaotic_small_s(self):
        """For small s, the system should not be classified as chaotic."""
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=1000,
            parameters={"s": 1.0, "v": 0.5},
        )
        sim = BurkeShawSimulation(config)
        assert sim.is_chaotic is False

    def test_lyapunov_returns_float(self):
        """Lyapunov estimation should return a finite float."""
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=5000,
            parameters={"s": 10.0, "v": 4.272},
        )
        sim = BurkeShawSimulation(config)
        sim.reset()
        for _ in range(1000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.005)
        assert isinstance(lam, float)
        assert np.isfinite(lam)


class TestBurkeShawNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_accuracy(self):
        """Smaller dt should give more accurate results (convergence test)."""
        # Run with dt=0.005
        config1 = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=200,
            parameters={"s": 10.0, "v": 4.272, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim1 = BurkeShawSimulation(config1)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.001 (5x finer, same total time = 200*0.005 = 1.0)
        config2 = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.001,
            n_steps=1000,
            parameters={"s": 10.0, "v": 4.272, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim2 = BurkeShawSimulation(config2)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state_fine = sim2.observe().copy()

        # For RK4, the error should scale as dt^4; states should be close
        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.1, (
            f"RK4 convergence error {error:.6f} too large between dt=0.005 and dt=0.001"
        )

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computable and finite."""
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=10000,
            parameters={"s": 10.0, "v": 4.272, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = BurkeShawSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Std should be positive in chaotic regime
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0
        assert stats["z_std"] > 0

    def test_measure_period_returns_finite(self):
        """Period measurement should return a positive value or inf."""
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=30000,
            parameters={"s": 10.0, "v": 4.272, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = BurkeShawSimulation(config)
        sim.reset()
        T = sim.measure_period(n_transient=3000, n_measure=10000)
        assert T > 0, f"Period should be positive, got {T}"


class TestBurkeShawRediscovery:
    """Tests for Burke-Shaw rediscovery data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.burke_shaw import generate_trajectory_data

        data = generate_trajectory_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["s"] == 10.0
        assert data["v"] == 4.272

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.burke_shaw import generate_trajectory_data

        data = generate_trajectory_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_s_data_shape(self):
        from simulating_anything.rediscovery.burke_shaw import (
            generate_lyapunov_vs_s_data,
        )

        data = generate_lyapunov_vs_s_data(n_s=5, n_steps=3000, dt=0.005)
        assert len(data["s"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_fixed_point_data(self):
        from simulating_anything.rediscovery.burke_shaw import (
            generate_fixed_point_data,
        )

        data = generate_fixed_point_data(s_values=np.array([5.0, 10.0]))
        assert len(data["data"]) == 2
        # Standard params should always have 2 fixed points
        assert data["data"][0]["n_fixed_points"] == 2
        assert data["data"][1]["n_fixed_points"] == 2

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.burke_shaw import generate_trajectory_data

        data = generate_trajectory_data(n_steps=200, dt=0.005)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
