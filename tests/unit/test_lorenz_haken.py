"""Tests for the Lorenz-Haken (Maxwell-Bloch) laser simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.lorenz_haken import LorenzHakenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_DOMAIN = Domain.LORENZ_HAKEN


class TestLorenzHakenConstruction:
    """Tests for simulation construction and initialization."""

    def _make_sim(self, **kwargs) -> LorenzHakenSimulation:
        defaults = {
            "sigma": 3.0, "r": 25.0, "b": 1.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzHakenSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.sigma == 3.0
        assert sim.r == 25.0
        assert sim.b == 1.0

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

    def test_default_laser_parameters(self):
        """Default parameters are the classic laser values, not Lorenz."""
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=1000, parameters={},
        )
        sim = LorenzHakenSimulation(config)
        assert sim.sigma == 3.0
        assert sim.r == 25.0
        assert sim.b == 1.0

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

    def test_state_shape_preserved(self):
        """State shape should remain (3,) throughout simulation."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(100):
            state = sim.step()
            assert state.shape == (3,)

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))


class TestLorenzHakenDerivatives:
    """Tests for the Maxwell-Bloch ODEs at known points."""

    def _make_sim(self, **kwargs) -> LorenzHakenSimulation:
        defaults = {"sigma": 3.0, "r": 25.0, "b": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=1000, parameters=defaults,
        )
        return LorenzHakenSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (non-lasing fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at state [1, 1, 1] with laser parameters."""
        sim = self._make_sim(sigma=3.0, r=25.0, b=1.0)
        sim.reset()
        # At state [1, 1, 1]:
        # dx = 3*(1-1) = 0
        # dy = (25-1)*1 - 1 = 23
        # dz = 1*1 - 1*1 = 0
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 23.0)
        assert np.isclose(derivs[2], 0.0)

    def test_derivatives_general_point(self):
        """Test derivatives at a general state."""
        sim = self._make_sim(sigma=3.0, r=25.0, b=1.0)
        sim.reset()
        # At state [2, 3, 5]:
        # dx = 3*(3-2) = 3
        # dy = (25-5)*2 - 3 = 40 - 3 = 37
        # dz = 2*3 - 1*5 = 1
        derivs = sim._derivatives(np.array([2.0, 3.0, 5.0]))
        assert np.isclose(derivs[0], 3.0)
        assert np.isclose(derivs[1], 37.0)
        assert np.isclose(derivs[2], 1.0)


class TestLorenzHakenFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, r=25.0, sigma=3.0, b=1.0) -> LorenzHakenSimulation:
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=1000,
            parameters={"sigma": sigma, "r": r, "b": b},
        )
        return LorenzHakenSimulation(config)

    def test_three_fixed_points_above_threshold(self):
        """For r > 1, there should be three fixed points."""
        sim = self._make_sim(r=25.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_one_fixed_point_below_threshold(self):
        """For r < 1, only the origin (non-lasing state) exists."""
        sim = self._make_sim(r=0.5)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_fixed_point_symmetry(self):
        """The two lasing fixed points should be symmetric."""
        sim = self._make_sim(r=25.0)
        sim.reset()
        fps = sim.fixed_points
        # C+ = (c, c, r-1), C- = (-c, -c, r-1)
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], -fps[2][1])
        assert np.isclose(fps[1][2], fps[2][2])
        assert np.isclose(fps[1][2], 24.0)  # r - 1

    def test_fixed_point_values(self):
        """Check exact fixed point values for sigma=3, r=25, b=1."""
        sim = self._make_sim(r=25.0, b=1.0)
        sim.reset()
        fps = sim.fixed_points
        c = np.sqrt(1.0 * 24.0)  # sqrt(b*(r-1))
        np.testing.assert_array_almost_equal(fps[1], [c, c, 24.0])

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim(r=25.0)
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_fixed_points_at_threshold(self):
        """At r=1 exactly, lasing fixed points collapse to origin."""
        sim = self._make_sim(r=1.0 + 1e-12)
        sim.reset()
        fps = sim.fixed_points
        # Three points exist but lasing ones are very close to origin
        assert len(fps) == 3
        assert np.linalg.norm(fps[1]) < 1e-5


class TestLorenzHakenThresholds:
    """Tests for lasing and second (Hopf) thresholds."""

    def _make_sim(self, sigma=3.0, r=25.0, b=1.0) -> LorenzHakenSimulation:
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=1000,
            parameters={"sigma": sigma, "r": r, "b": b},
        )
        return LorenzHakenSimulation(config)

    def test_lasing_threshold_is_one(self):
        """Lasing threshold should always be r=1."""
        sim = self._make_sim(sigma=3.0, b=1.0)
        assert sim.lasing_threshold == 1.0

    def test_lasing_threshold_independent_of_parameters(self):
        """Threshold r=1 does not depend on sigma or b."""
        sim1 = self._make_sim(sigma=10.0, b=0.5)
        sim2 = self._make_sim(sigma=1.0, b=5.0)
        assert sim1.lasing_threshold == 1.0
        assert sim2.lasing_threshold == 1.0

    def test_second_threshold_formula(self):
        """r_H = sigma*(sigma+b+3)/(sigma-b-1) for sigma > b+1."""
        sim = self._make_sim(sigma=3.0, b=1.0)
        r_H = sim.second_threshold
        expected = 3.0 * (3.0 + 1.0 + 3.0) / (3.0 - 1.0 - 1.0)  # 21.0
        assert np.isclose(r_H, expected)
        assert np.isclose(r_H, 21.0)

    def test_second_threshold_inf_when_bad_class(self):
        """When sigma <= b+1, no second instability exists."""
        sim = self._make_sim(sigma=1.5, b=1.0)
        assert sim.second_threshold == np.inf

    def test_second_threshold_different_params(self):
        """Test second threshold with sigma=10, b=8/3 (Lorenz-like)."""
        sim = self._make_sim(sigma=10.0, b=8.0 / 3.0)
        r_H = sim.second_threshold
        expected = 10.0 * (10.0 + 8.0 / 3.0 + 3.0) / (10.0 - 8.0 / 3.0 - 1.0)
        assert np.isclose(r_H, expected)


class TestLorenzHakenTrajectory:
    """Tests for trajectory behavior of the laser model."""

    def _make_sim(self, **kwargs) -> LorenzHakenSimulation:
        defaults = {"sigma": 3.0, "r": 25.0, "b": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=10000, parameters=defaults,
        )
        return LorenzHakenSimulation(config)

    def test_trajectory_bounded(self):
        """Trajectories should remain bounded for classic laser parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_deterministic(self):
        """Two runs with same initial conditions should be identical."""
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        traj1 = sim1.run(n_steps=500)
        traj2 = sim2.run(n_steps=500)
        np.testing.assert_array_almost_equal(
            traj1.states, traj2.states, decimal=12,
        )

    def test_different_parameters_diverge(self):
        """Different r values should yield different trajectories."""
        sim1 = self._make_sim(r=25.0)
        sim2 = self._make_sim(r=5.0)
        traj1 = sim1.run(n_steps=500)
        traj2 = sim2.run(n_steps=500)
        assert not np.allclose(traj1.states, traj2.states)

    def test_below_threshold_decays_to_origin(self):
        """For r < 1, the field should decay to zero (non-lasing)."""
        sim = self._make_sim(r=0.5, x_0=0.1, y_0=0.1, z_0=0.0)
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.linalg.norm(state) < 0.01, (
            f"State should decay to origin for r<1, got {state}"
        )

    def test_above_threshold_nonzero_field(self):
        """For r > 1 (steady lasing), the field should be nonzero."""
        sim = self._make_sim(r=5.0, x_0=0.1, y_0=0.1, z_0=0.0)
        sim.reset()
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        # Intensity should be nonzero for steady lasing
        intensity = state[0] ** 2
        assert intensity > 0.01, (
            f"Field should be nonzero above threshold, intensity={intensity}"
        )


class TestLorenzHakenLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """At classic laser parameters (r=25), Lyapunov should be positive."""
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=20000,
            parameters={"sigma": 3.0, "r": 25.0, "b": 1.0},
        )
        sim = LorenzHakenSimulation(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.1, (
            f"Lyapunov {lam:.3f} should be positive for chaotic regime"
        )

    def test_negative_lyapunov_stable_lasing(self):
        """Below second threshold but above first, Lyapunov should be negative."""
        # For sigma=3, b=1: r_H = 21, so r=5 is stable lasing
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=20000,
            parameters={"sigma": 3.0, "r": 5.0, "b": 1.0},
        )
        sim = LorenzHakenSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam < 0.1, (
            f"Lyapunov {lam:.3f} should be small/negative for stable lasing"
        )


class TestLorenzHakenIntensity:
    """Tests for laser intensity measurement."""

    def _make_sim(self, **kwargs) -> LorenzHakenSimulation:
        defaults = {"sigma": 3.0, "r": 25.0, "b": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=30000, parameters=defaults,
        )
        return LorenzHakenSimulation(config)

    def test_intensity_positive(self):
        """Intensity (x^2) should always be non-negative."""
        sim = self._make_sim()
        sim.reset()
        result = sim.measure_intensity(n_transient=5000, n_measure=5000)
        assert result["mean_intensity"] >= 0
        assert np.all(result["intensities"] >= 0)

    def test_intensity_nonzero_above_threshold(self):
        """Above lasing threshold, mean intensity should be nonzero."""
        sim = self._make_sim(r=5.0)
        sim.reset()
        result = sim.measure_intensity(n_transient=10000, n_measure=5000)
        assert result["mean_intensity"] > 0.01


class TestLorenzHakenRediscovery:
    """Tests for Lorenz-Haken data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.lorenz_haken import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["sigma"] == 3.0
        assert data["r"] == 25.0
        assert data["b"] == 1.0

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.lorenz_haken import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_pump_sweep_data(self):
        from simulating_anything.rediscovery.lorenz_haken import (
            generate_pump_sweep_data,
        )

        data = generate_pump_sweep_data(n_r=5, n_steps=2000, dt=0.01)
        assert len(data["r"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5
        assert len(data["mean_intensity"]) == 5

    def test_pump_sweep_contains_types(self):
        """Sweep from r=0.5 to r=35 should find both non-lasing and chaotic."""
        from simulating_anything.rediscovery.lorenz_haken import (
            generate_pump_sweep_data,
        )

        data = generate_pump_sweep_data(n_r=10, n_steps=5000, dt=0.01)
        types = set(data["attractor_type"])
        has_nonlasing = "non_lasing" in types
        has_active = (
            "chaotic" in types
            or "steady_lasing" in types
            or "periodic_or_transient" in types
        )
        assert has_nonlasing, f"No non-lasing regime found in types: {types}"
        assert has_active, f"No active regime found in types: {types}"

    def test_sindy_ready_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.lorenz_haken import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
