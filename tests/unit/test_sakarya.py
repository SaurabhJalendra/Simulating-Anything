"""Tests for the Sakarya attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.sakarya import SakaryaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestSakaryaCreation:
    """Tests for simulation creation and parameter handling."""

    def _make_sim(self, **kwargs) -> SakaryaSimulation:
        defaults = {
            "a": 0.4, "b": 0.3,
            "x_0": 1.0, "y_0": -1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return SakaryaSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 0.4
        assert sim.b == 0.3

    def test_creation_custom_parameters(self):
        """Simulation accepts custom a and b values."""
        sim = self._make_sim(a=0.8, b=1.0)
        assert sim.a == 0.8
        assert sim.b == 1.0

    def test_domain_enum_exists(self):
        """SAKARYA should exist in the Domain enum."""
        assert Domain.SAKARYA == "sakarya"

    def test_config_stored(self):
        """Config is accessible on the simulation."""
        sim = self._make_sim()
        assert sim.config.dt == 0.01
        assert sim.config.n_steps == 10000


class TestSakaryaState:
    """Tests for state initialization and shape."""

    def _make_sim(self, **kwargs) -> SakaryaSimulation:
        defaults = {
            "a": 0.4, "b": 0.3,
            "x_0": 1.0, "y_0": -1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return SakaryaSimulation(config)

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
        """Default initial state is [1, -1, 1]."""
        sim = self._make_sim()
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], -1.0)
        assert np.isclose(state[2], 1.0)

    def test_initial_state_custom_values(self):
        """Custom initial conditions are respected."""
        sim = self._make_sim(x_0=2.0, y_0=-0.5, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -0.5)
        assert np.isclose(state[2], 3.0)

    def test_observe_returns_current_state(self):
        """observe() returns the same state as current internal state."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_step_advances_state(self):
        """A single step changes the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)


class TestSakaryaDerivatives:
    """Tests for the Sakarya ODE equations."""

    def _make_sim(self, **kwargs) -> SakaryaSimulation:
        defaults = {"a": 0.4, "b": 0.3}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return SakaryaSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_at_known_point(self):
        """Test derivatives at (1, 1, 1) with a=0.4, b=0.3.

        dx = -1 + 1 + 1*1 = 1.0
        dy = -1 - 1 + 0.4*1*1 = -1.6
        dz = 1 - 0.3*1*1 = 0.7
        """
        sim = self._make_sim(a=0.4, b=0.3)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], -1.6)
        assert np.isclose(derivs[2], 0.7)

    def test_derivatives_at_negative_point(self):
        """Test derivatives at (-1, -1, -1) with a=0.4, b=0.3.

        dx = -(-1) + (-1) + (-1)*(-1) = 1 - 1 + 1 = 1.0
        dy = -(-1) - (-1) + 0.4*(-1)*(-1) = 1 + 1 + 0.4 = 2.4
        dz = -1 - 0.3*(-1)*(-1) = -1 - 0.3 = -1.3
        """
        sim = self._make_sim(a=0.4, b=0.3)
        sim.reset()
        derivs = sim._derivatives(np.array([-1.0, -1.0, -1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 2.4)
        assert np.isclose(derivs[2], -1.3)


class TestSakaryaFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> SakaryaSimulation:
        defaults = {"a": 0.4, "b": 0.3}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return SakaryaSimulation(config)

    def test_origin_is_fixed_point(self):
        """Origin (0,0,0) is always a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        origin_found = False
        for fp in fps:
            if np.linalg.norm(fp) < 1e-6:
                origin_found = True
                break
        assert origin_found, "Origin not found among fixed points"

    def test_fixed_points_exist(self):
        """At least one fixed point should be found (the origin)."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) >= 1

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be near zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        for fp in fps:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=6,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestSakaryaTrajectory:
    """Tests for trajectory behavior and boundedness."""

    def _make_sim(self, **kwargs) -> SakaryaSimulation:
        defaults = {
            "a": 0.4, "b": 0.3,
            "x_0": 1.0, "y_0": -1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return SakaryaSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Trajectory should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"
            assert np.linalg.norm(state) < 100, f"Trajectory diverged: {state}"

    def test_no_nan_or_inf(self):
        """No NaN or Inf after many steps."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

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

    def test_attractor_bounded_long_run(self):
        """Trajectory should stay bounded over a longer integration."""
        sim = self._make_sim()
        sim.reset()
        max_norm = 0.0
        for _ in range(50000):
            state = sim.step()
            norm = np.linalg.norm(state)
            if norm > max_norm:
                max_norm = norm
        # Based on numerical experiments, max norm stays below 50
        assert max_norm < 100, f"Max norm {max_norm:.2f} exceeded bound"
        assert max_norm > 1.0, "Trajectory collapsed to near origin"


class TestSakaryaLyapunov:
    """Tests for Lyapunov exponent and chaos detection."""

    def test_chaotic_regime_positive_lyapunov(self):
        """Standard parameters (a=0.4, b=0.3) should give positive Lyapunov."""
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=50000,
            parameters={
                "a": 0.4, "b": 0.3,
                "x_0": 1.0, "y_0": -1.0, "z_0": 1.0,
            },
        )
        sim = SakaryaSimulation(config)
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.01)
        assert lam > 0.05, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_near_stable_regime(self):
        """At a=0.8, b=0.3 the Lyapunov should be lower than chaotic regime."""
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=50000,
            parameters={
                "a": 0.8, "b": 0.3,
                "x_0": 1.0, "y_0": -1.0, "z_0": 1.0,
            },
        )
        sim = SakaryaSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.01)
        # At a=0.8, b=0.3 we get near-zero Lyapunov
        assert lam < 0.1, f"Lyapunov {lam:.4f} too large for near-stable regime"

    def test_lyapunov_transition(self):
        """Lyapunov at a=0.4 should exceed Lyapunov at a=0.8."""
        lam_chaotic = _compute_lyapunov_at_params(0.4, 0.3)
        lam_stable = _compute_lyapunov_at_params(0.8, 0.3)
        assert lam_chaotic > lam_stable, (
            f"Lyapunov at a=0.4 ({lam_chaotic:.4f}) should exceed "
            f"a=0.8 ({lam_stable:.4f})"
        )

    def test_is_chaotic_property(self):
        """is_chaotic property should return True for standard parameters."""
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=1000,
            parameters={"a": 0.4, "b": 0.3},
        )
        sim = SakaryaSimulation(config)
        assert sim.is_chaotic is True


class TestSakaryaNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_convergence(self):
        """Smaller dt should give more accurate results."""
        # Run with dt=0.01
        config1 = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=100,
            parameters={"a": 0.4, "b": 0.3, "x_0": 1.0, "y_0": -1.0, "z_0": 1.0},
        )
        sim1 = SakaryaSimulation(config1)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.002 (5x finer, same total time = 1.0)
        config2 = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.002,
            n_steps=500,
            parameters={"a": 0.4, "b": 0.3, "x_0": 1.0, "y_0": -1.0, "z_0": 1.0},
        )
        sim2 = SakaryaSimulation(config2)
        sim2.reset()
        for _ in range(500):
            sim2.step()
        state_fine = sim2.observe().copy()

        # For RK4, error scales as dt^4; states should be close
        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.01, (
            f"RK4 convergence error {error:.6f} too large between dt=0.01 and dt=0.002"
        )

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computable and finite."""
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=10000,
            parameters={"a": 0.4, "b": 0.3, "x_0": 1.0, "y_0": -1.0, "z_0": 1.0},
        )
        sim = SakaryaSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Std should be positive for chaotic regime
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0
        assert stats["z_std"] > 0

    def test_bifurcation_sweep(self):
        """Sweep should produce valid data for all a values."""
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=0.01,
            n_steps=5000,
            parameters={"a": 0.4, "b": 0.3},
        )
        sim = SakaryaSimulation(config)
        sim.reset()
        a_values = np.array([0.3, 0.4, 0.5])
        data = sim.bifurcation_sweep(
            a_values, n_transient=1000, n_measure=5000
        )
        assert len(data["a"]) == 3
        assert len(data["lyapunov_exponent"]) == 3
        assert len(data["attractor_type"]) == 3
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


class TestSakaryaRediscovery:
    """Tests for the Sakarya rediscovery data generation."""

    def test_trajectory_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.sakarya import generate_trajectory_data

        data = generate_trajectory_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert np.isclose(data["a"], 0.4)
        assert np.isclose(data["b"], 0.3)
        assert np.all(np.isfinite(data["states"]))

    def test_trajectory_data_stays_finite(self):
        """Trajectory data should remain finite over many steps."""
        from simulating_anything.rediscovery.sakarya import generate_trajectory_data

        data = generate_trajectory_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.sakarya import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.sakarya import generate_trajectory_data

        data = generate_trajectory_data(n_steps=200, dt=0.01)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data


def _compute_lyapunov_at_params(a: float, b: float) -> float:
    """Helper to compute Lyapunov exponent at given parameters."""
    config = SimulationConfig(
        domain=Domain.SAKARYA,
        dt=0.01,
        n_steps=30000,
        parameters={
            "a": a, "b": b,
            "x_0": 1.0, "y_0": -1.0, "z_0": 1.0,
        },
    )
    sim = SakaryaSimulation(config)
    sim.reset()
    for _ in range(5000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=20000, dt=0.01)
