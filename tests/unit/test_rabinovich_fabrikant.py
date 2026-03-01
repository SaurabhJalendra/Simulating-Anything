"""Tests for the Rabinovich-Fabrikant system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.rabinovich_fabrikant import (
    RabinovichFabrikantSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestRabinovichFabrikantCreation:
    """Tests for simulation creation and parameter handling."""

    def _make_sim(self, **kwargs) -> RabinovichFabrikantSimulation:
        defaults = {
            "alpha": 1.1, "gamma": 0.87,
            "x_0": -1.0, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RabinovichFabrikantSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.alpha == 1.1
        assert sim.gamma == 0.87

    def test_creation_custom_parameters(self):
        """Simulation accepts custom alpha and gamma."""
        sim = self._make_sim(alpha=0.5, gamma=1.0)
        assert sim.alpha == 0.5
        assert sim.gamma == 1.0

    def test_domain_enum_exists(self):
        """RABINOVICH_FABRIKANT should exist in the Domain enum."""
        assert Domain.RABINOVICH_FABRIKANT == "rabinovich_fabrikant"

    def test_config_stored(self):
        """Config is accessible on the simulation."""
        sim = self._make_sim()
        assert sim.config.dt == 0.005
        assert sim.config.n_steps == 10000


class TestRabinovichFabrikantState:
    """Tests for state initialization and shape."""

    def _make_sim(self, **kwargs) -> RabinovichFabrikantSimulation:
        defaults = {
            "alpha": 1.1, "gamma": 0.87,
            "x_0": -1.0, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RabinovichFabrikantSimulation(config)

    def test_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)

    def test_state_dtype(self):
        """State should be float64 for numerical precision."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.dtype == np.float64

    def test_initial_state_default_values(self):
        """Default initial state is [-1, 0, 0.5]."""
        sim = self._make_sim()
        state = sim.reset()
        assert np.isclose(state[0], -1.0)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], 0.5)

    def test_initial_state_custom_values(self):
        """Custom initial conditions are respected."""
        sim = self._make_sim(x_0=2.0, y_0=-1.5, z_0=0.3)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -1.5)
        assert np.isclose(state[2], 0.3)

    def test_observe_returns_current_state(self):
        """observe() returns the same state as current internal state."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs, sim._state)


class TestRabinovichFabrikantIntegration:
    """Tests for RK4 integration stability and correctness."""

    def _make_sim(self, **kwargs) -> RabinovichFabrikantSimulation:
        defaults = {
            "alpha": 1.1, "gamma": 0.87,
            "x_0": -1.0, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RabinovichFabrikantSimulation(config)

    def test_step_advances_state(self):
        """A single step changes the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_step_count_increments(self):
        """Step count increments with each step."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_rk4_stability_short_trajectory(self):
        """RK4 integration stays finite for 1000 steps at dt=0.005."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(1000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_derivatives_at_origin(self):
        """At the origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with alpha=1.1, gamma=0.87:
            dx = 1*(1 - 1 + 1) + 0.87*1 = 1.87
            dy = 1*(3 + 1 - 1) + 0.87*1 = 3.87
            dz = -2*1*(1.1 + 1*1) = -2*2.1 = -4.2
        """
        sim = self._make_sim(alpha=1.1, gamma=0.87)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.87)
        assert np.isclose(derivs[1], 3.87)
        assert np.isclose(derivs[2], -4.2)

    def test_reset_restores_initial_state(self):
        """Reset restores the simulation to initial conditions."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(100):
            sim.step()
        state_after_steps = sim.observe().copy()

        sim.reset()
        state_after_reset = sim.observe()
        assert not np.allclose(state_after_steps, state_after_reset)
        assert np.isclose(state_after_reset[0], -1.0)


class TestRabinovichFabrikantTrajectory:
    """Tests for trajectory collection and attractor properties."""

    def _make_sim(self, **kwargs) -> RabinovichFabrikantSimulation:
        defaults = {"alpha": 1.1, "gamma": 0.87}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RabinovichFabrikantSimulation(config)

    def test_trajectory_collection(self):
        """run() returns TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_trajectory_timestamps(self):
        """Trajectory timestamps are correctly spaced."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        expected_times = np.arange(101) * 0.005
        np.testing.assert_array_almost_equal(traj.timestamps, expected_times)

    def test_attractor_bounded(self):
        """Trajectory stays bounded for classic chaotic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"
            norm = np.linalg.norm(state)
            assert norm < 100, f"Trajectory diverged: norm={norm:.2f}, state={state}"

    def test_z_stays_nonnegative(self):
        """z coordinate should stay non-negative for the attractor.

        The dz/dt = -2z(alpha + xy) equation with alpha > 0 ensures z cannot
        cross zero when starting positive, provided xy does not overwhelm alpha.
        For standard parameters, z remains non-negative.
        """
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            # Allow small numerical error
            assert state[2] > -0.01, f"z went below -0.01: {state[2]}"


class TestRabinovichFabrikantChaos:
    """Tests for chaotic dynamics and Lyapunov exponents."""

    def _make_sim(self, **kwargs) -> RabinovichFabrikantSimulation:
        defaults = {"alpha": 1.1, "gamma": 0.87}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=50000,
            parameters=defaults,
        )
        return RabinovichFabrikantSimulation(config)

    def test_positive_lyapunov_chaotic_regime(self):
        """At alpha=1.1, gamma=0.87, Lyapunov exponent should be positive."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.005)
        assert lam > 0.01, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_sensitivity_to_initial_conditions(self):
        """Two nearby trajectories should diverge (chaos hallmark)."""
        config1 = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=20000,
            parameters={
                "alpha": 1.1, "gamma": 0.87,
                "x_0": -1.0, "y_0": 0.0, "z_0": 0.5,
            },
        )
        config2 = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=20000,
            parameters={
                "alpha": 1.1, "gamma": 0.87,
                "x_0": -1.0 + 1e-6, "y_0": 0.0, "z_0": 0.5,
            },
        )
        sim1 = RabinovichFabrikantSimulation(config1)
        sim2 = RabinovichFabrikantSimulation(config2)
        sim1.reset()
        sim2.reset()

        for _ in range(20000):
            sim1.step()
            sim2.step()

        dist = np.linalg.norm(sim1.observe() - sim2.observe())
        # After 20000 steps (100 time units), perturbation should grow
        assert dist > 1e-4, f"Trajectories did not diverge enough: dist={dist:.2e}"

    def test_different_gamma_changes_dynamics(self):
        """Different gamma values produce different dynamics."""
        sim_low = self._make_sim(gamma=0.1)
        sim_low.reset()
        for _ in range(2000):
            sim_low.step()
        state_low = sim_low.observe().copy()

        sim_high = self._make_sim(gamma=0.87)
        sim_high.reset()
        for _ in range(2000):
            sim_high.step()
        state_high = sim_high.observe().copy()

        # Different gamma should lead to different attractor structure
        assert not np.allclose(state_low, state_high, atol=0.1)


class TestRabinovichFabrikantFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> RabinovichFabrikantSimulation:
        defaults = {"alpha": 1.1, "gamma": 0.87}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return RabinovichFabrikantSimulation(config)

    def test_origin_is_fixed_point(self):
        """The origin should always be a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) >= 1
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_derivatives_at_origin(self):
        """Derivatives should be zero at the origin."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0], decimal=12)


class TestRabinovichFabrikantRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.rabinovich_fabrikant import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["alpha"] == 1.1
        assert data["gamma"] == 0.87

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.rabinovich_fabrikant import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_gamma_sweep_data_shape(self):
        from simulating_anything.rediscovery.rabinovich_fabrikant import (
            generate_gamma_sweep_data,
        )

        data = generate_gamma_sweep_data(n_gamma=5, n_steps=2000, dt=0.005)
        assert len(data["gamma"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.rabinovich_fabrikant import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=200, dt=0.005)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
