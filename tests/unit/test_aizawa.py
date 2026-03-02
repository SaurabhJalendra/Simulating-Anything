"""Tests for the Aizawa attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.aizawa import AizawaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestAizawaSimulation:
    """Tests for the Aizawa attractor simulation basics."""

    def _make_sim(self, **kwargs) -> AizawaSimulation:
        defaults = {
            "a": 0.95, "b": 0.7, "c": 0.6,
            "d": 3.5, "e": 0.25, "f": 0.1,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return AizawaSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.a == 0.95
        assert sim.b == 0.7
        assert sim.c == 0.6
        assert sim.d == 3.5
        assert sim.e == 0.25
        assert sim.f == 0.1

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=-0.5, z_0=0.3)
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], -0.5)
        assert np.isclose(state[2], 0.3)

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
        for _ in range(200):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_step_count_increments(self):
        """Step count increments correctly."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        for _ in range(9):
            sim.step()
        assert sim._step_count == 10


class TestAizawaDerivatives:
    """Tests for the Aizawa ODE derivatives."""

    def _make_sim(self, **kwargs) -> AizawaSimulation:
        defaults = {
            "a": 0.95, "b": 0.7, "c": 0.6,
            "d": 3.5, "e": 0.25, "f": 0.1,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=100,
            parameters=defaults,
        )
        return AizawaSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin (0,0,0):
        dx = (0-b)*0 - d*0 = 0
        dy = d*0 + (0-b)*0 = 0
        dz = c + a*0 - 0 - 0 + 0 = c
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 0.6)  # c = 0.6

    def test_derivatives_known_point(self):
        """Test derivatives at state [1, 0, 0] with classic parameters."""
        sim = self._make_sim()
        sim.reset()
        # At (1, 0, 0):
        # dx = (0-0.7)*1 - 3.5*0 = -0.7
        # dy = 3.5*1 + (0-0.7)*0 = 3.5
        # dz = 0.6 + 0.95*0 - 0 - 1*(1+0) + 0.1*0*1 = 0.6 - 1.0 = -0.4
        derivs = sim._derivatives(np.array([1.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], -0.7)
        assert np.isclose(derivs[1], 3.5)
        assert np.isclose(derivs[2], -0.4)

    def test_derivatives_xy_rotation(self):
        """The d parameter controls xy-plane rotation rate.

        At (0, 1, b): dx = 0 - d*1 = -d, dy = d*0 + 0 = 0.
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 1.0, 0.7]))
        assert np.isclose(derivs[0], -3.5)  # -d
        assert np.isclose(derivs[1], 0.0)   # d*0 + (z-b)*y = 0

    def test_derivatives_z_cubic(self):
        """The z-equation includes -z^3/3 cubic damping."""
        sim = self._make_sim()
        sim.reset()
        # At (0, 0, 3): dz = 0.6 + 0.95*3 - 27/3 - 0 + 0
        #             = 0.6 + 2.85 - 9.0 = -5.55
        derivs = sim._derivatives(np.array([0.0, 0.0, 3.0]))
        expected_dz = 0.6 + 0.95 * 3.0 - 27.0 / 3.0
        assert np.isclose(derivs[2], expected_dz)


class TestAizawaFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> AizawaSimulation:
        defaults = {
            "a": 0.95, "b": 0.7, "c": 0.6,
            "d": 3.5, "e": 0.25, "f": 0.1,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=100,
            parameters=defaults,
        )
        return AizawaSimulation(config)

    def test_fixed_points_exist(self):
        """At least one fixed point should be found for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) >= 1, "No fixed points found"

    def test_fixed_points_have_x_y_zero(self):
        """All fixed points should have x=0, y=0 (for d != 0)."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            assert np.isclose(fp[0], 0.0), f"x != 0 at fixed point {fp}"
            assert np.isclose(fp[1], 0.0), f"y != 0 at fixed point {fp}"

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be near zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=8,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_cubic_gives_correct_count(self):
        """The cubic z^3 - 3*a*z - 3*c = 0 determines fixed point count.

        Discriminant D = -4*(-3a)^3 - 27*(-3c)^2 = 108*a^3 - 243*c^2
        For a=0.95, c=0.6: D = 108*(0.857375) - 243*(0.36) = 92.60 - 87.48 > 0
        Three real roots expected.
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        # With classic params, discriminant is positive, so 3 real roots
        assert len(fps) == 3, f"Expected 3 fixed points, got {len(fps)}"


class TestAizawaTrajectory:
    """Tests for trajectory behavior."""

    def _make_sim(self, **kwargs) -> AizawaSimulation:
        defaults = {
            "a": 0.95, "b": 0.7, "c": 0.6,
            "d": 3.5, "e": 0.25, "f": 0.1,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return AizawaSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Aizawa trajectory should remain bounded for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 50, f"Trajectory diverged: {state}"

    def test_trajectory_shape_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_attractor_nontrivial(self):
        """After transient, trajectory should explore a nontrivial region."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        # Collect some points
        states = []
        for _ in range(5000):
            states.append(sim.step().copy())
        states = np.array(states)
        # Should have significant spread in all coordinates
        assert np.std(states[:, 0]) > 0.1, "x not spread on attractor"
        assert np.std(states[:, 1]) > 0.1, "y not spread on attractor"
        assert np.std(states[:, 2]) > 0.1, "z not spread on attractor"

    def test_attractor_radial_extent(self):
        """The mushroom attractor should have a finite radial extent in xy."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(3000):
            sim.step()
        r_values = []
        for _ in range(5000):
            state = sim.step()
            r_values.append(np.sqrt(state[0]**2 + state[1]**2))
        r_max = max(r_values)
        r_mean = np.mean(r_values)
        # Radial extent should be finite and positive
        assert r_max > 0.1, "Radial extent too small"
        assert r_max < 20.0, "Radial extent too large"
        assert r_mean > 0.01, "Mean radius too small"


class TestAizawaLyapunov:
    """Tests for Lyapunov exponent and chaos detection."""

    def test_classic_params_positive_lyapunov(self):
        """At classic parameters, the largest Lyapunov exponent should be positive."""
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=30000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, f"Lyapunov {lam:.4f} not positive for classic parameters"

    def test_lyapunov_returns_float(self):
        """Lyapunov estimate should return a finite float."""
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        sim.reset()
        for _ in range(1000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=3000, dt=0.01)
        assert isinstance(lam, float)
        assert np.isfinite(lam)

    def test_lyapunov_uses_default_dt(self):
        """When dt=None, should use config dt."""
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.005,
            n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        sim.reset()
        for _ in range(1000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=2000)
        assert isinstance(lam, float)
        assert np.isfinite(lam)


class TestAizawaNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_convergence(self):
        """Smaller dt should give more accurate results."""
        # Run with dt=0.01
        config1 = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=100,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim1 = AizawaSimulation(config1)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.002 (5x finer, same total time)
        config2 = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.002,
            n_steps=500,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim2 = AizawaSimulation(config2)
        sim2.reset()
        for _ in range(500):
            sim2.step()
        state_fine = sim2.observe().copy()

        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.01, (
            f"RK4 convergence error {error:.6f} too large"
        )

    def test_no_nan_long_integration(self):
        """No NaN or Inf after many steps."""
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=20000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computable and finite."""
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=10000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Std should be positive on chaotic attractor
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0
        assert stats["z_std"] > 0
        # r_max should be positive
        assert stats["r_max"] > 0
        assert stats["r_mean"] > 0

    def test_measure_period(self):
        """measure_period should return a finite positive value or inf."""
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=0.01,
            n_steps=30000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        sim.reset()
        T = sim.measure_period(n_transient=3000, n_measure=15000)
        # Should be either a finite positive period or inf
        assert T > 0, f"Period should be positive, got {T}"


class TestAizawaRediscovery:
    """Tests for Aizawa rediscovery data generation."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.aizawa import generate_trajectory_data

        data = generate_trajectory_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.95
        assert data["b"] == 0.7
        assert data["c"] == 0.6
        assert data["d"] == 3.5
        assert data["e"] == 0.25
        assert data["f"] == 0.1

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.aizawa import generate_trajectory_data

        data = generate_trajectory_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.aizawa import generate_trajectory_data

        data = generate_trajectory_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_sweep_shape(self):
        from simulating_anything.rediscovery.aizawa import (
            generate_lyapunov_sweep_data,
        )

        data = generate_lyapunov_sweep_data(n_a=5, n_steps=3000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_attractor_statistics(self):
        from simulating_anything.rediscovery.aizawa import (
            generate_attractor_statistics,
        )

        stats = generate_attractor_statistics(dt=0.01)
        assert "x_mean" in stats
        assert "r_max" in stats
        assert np.isfinite(stats["r_max"])
        assert stats["r_max"] > 0
