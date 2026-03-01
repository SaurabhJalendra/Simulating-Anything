"""Tests for the Rossler system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.rossler import RosslerSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestRosslerSimulation:
    """Tests for the Rossler system simulation."""

    def _make_sim(self, **kwargs) -> RosslerSimulation:
        defaults = {
            "a": 0.2, "b": 0.2, "c": 5.7,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RosslerSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.a == 0.2
        assert sim.b == 0.2
        assert sim.c == 5.7

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

    def test_derivatives_at_origin(self):
        """Derivatives at origin: dx=-0-0=0, dy=0+0=0, dz=b+0*(0-c)=b."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 0.2)  # b = 0.2

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point."""
        sim = self._make_sim(a=0.2, b=0.2, c=5.7)
        sim.reset()
        # At state [1, 1, 1]:
        # dx = -1 - 1 = -2
        # dy = 1 + 0.2*1 = 1.2
        # dz = 0.2 + 1*(1-5.7) = 0.2 - 4.7 = -4.5
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], -2.0)
        assert np.isclose(derivs[1], 1.2)
        assert np.isclose(derivs[2], -4.5)


class TestRosslerFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> RosslerSimulation:
        defaults = {"a": 0.2, "b": 0.2, "c": 5.7}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return RosslerSimulation(config)

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
        """Verify fixed points against analytical formulas."""
        a, b, c = 0.2, 0.2, 5.7
        sim = self._make_sim(a=a, b=b, c=c)
        sim.reset()
        fps = sim.fixed_points

        disc = c**2 - 4 * a * b
        sqrt_disc = np.sqrt(disc)

        # FP1: y = (-c + sqrt(disc)) / (2a)
        y1_expected = (-c + sqrt_disc) / (2 * a)
        x1_expected = -a * y1_expected
        z1_expected = -y1_expected

        np.testing.assert_almost_equal(fps[0][0], x1_expected, decimal=10)
        np.testing.assert_almost_equal(fps[0][1], y1_expected, decimal=10)
        np.testing.assert_almost_equal(fps[0][2], z1_expected, decimal=10)

    def test_no_fixed_points_when_discriminant_negative(self):
        """When c^2 < 4ab, no real fixed points exist."""
        # a=1, b=1, c=1: discriminant = 1 - 4 = -3 < 0
        sim = self._make_sim(a=1.0, b=1.0, c=1.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 0


class TestRosslerTrajectory:
    """Tests for trajectory behavior."""

    def _make_sim(self, **kwargs) -> RosslerSimulation:
        defaults = {"a": 0.2, "b": 0.2, "c": 5.7}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RosslerSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Rossler trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_trajectory_shape_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_chaotic_regime_positive_lyapunov(self):
        """At c=5.7 (chaotic), largest Lyapunov exponent should be positive."""
        sim = self._make_sim(c=5.7)
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.005)
        assert lam > 0.01, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_periodic_regime_nonpositive_lyapunov(self):
        """At c=3.0 (periodic), Lyapunov exponent should be near zero or negative."""
        sim = self._make_sim(c=3.0)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.005)
        assert lam < 0.05, f"Lyapunov {lam:.4f} too large for periodic regime"

    def test_period_at_c3_less_than_c4(self):
        """Period-doubling: period at c=3 < period at c=4."""
        sim3 = self._make_sim(c=3.0)
        sim3.reset()
        T3 = sim3.measure_period(n_transient=5000, n_measure=30000)

        sim4 = self._make_sim(c=4.0)
        sim4.reset()
        T4 = sim4.measure_period(n_transient=5000, n_measure=30000)

        assert T3 < T4, f"Period at c=3 ({T3:.2f}) >= period at c=4 ({T4:.2f})"

    def test_is_chaotic_property(self):
        """is_chaotic property should be True at c=5.7 and False at c=3."""
        sim_chaotic = self._make_sim(c=5.7)
        assert sim_chaotic.is_chaotic is True

        sim_periodic = self._make_sim(c=3.0)
        assert sim_periodic.is_chaotic is False


class TestRosslerRediscovery:
    """Tests for Rossler data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.rossler import generate_trajectory_data

        data = generate_trajectory_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.2
        assert data["b"] == 0.2
        assert data["c"] == 5.7

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.rossler import generate_trajectory_data

        data = generate_trajectory_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_period_data_shape(self):
        from simulating_anything.rediscovery.rossler import generate_period_data

        data = generate_period_data(n_c=5, dt=0.005)
        assert len(data["c"]) == 5
        assert len(data["period"]) == 5
        assert len(data["lyapunov_exponent"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.rossler import generate_trajectory_data

        data = generate_trajectory_data(n_steps=200, dt=0.005)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
