"""Tests for the Dadras chaotic attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.dadras import DadrasSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_DADRAS_DOMAIN = Domain.DADRAS


class TestDadrasSimulation:
    """Tests for the Dadras system simulation basics."""

    def _make_sim(self, **kwargs) -> DadrasSimulation:
        defaults = {
            "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return DadrasSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 3.0
        assert sim.b == 2.7
        assert sim.c == 1.7
        assert sim.d == 2.0
        assert sim.e == 9.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=4.0, b=3.0, c=2.0, d=1.5, e=8.0)
        assert sim.a == 4.0
        assert sim.b == 3.0
        assert sim.c == 2.0
        assert sim.d == 1.5
        assert sim.e == 8.0

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
        sim1 = self._make_sim(a=3.0, b=2.7, c=1.7, d=2.0, e=9.0)
        sim2 = self._make_sim(a=3.0, b=2.7, c=1.7, d=2.0, e=9.0)
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


class TestDadrasDerivatives:
    """Tests for the Dadras ODE derivative computation."""

    def _make_sim(self, **kwargs) -> DadrasSimulation:
        defaults = {
            "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return DadrasSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=3, b=2.7, c=1.7, d=2, e=9:
            dx = 1 - 3*1 + 2.7*1*1 = 1 - 3 + 2.7 = 0.7
            dy = 1.7*1 - 1*1 + 1 = 1.7 - 1 + 1 = 1.7
            dz = 2*1*1 - 9*1 = 2 - 9 = -7
        """
        sim = self._make_sim(a=3.0, b=2.7, c=1.7, d=2.0, e=9.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 0.7)
        assert np.isclose(derivs[1], 1.7)
        assert np.isclose(derivs[2], -7.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 3, 1] with a=3, b=2.7, c=1.7, d=2, e=9.

            dx = 3 - 3*2 + 2.7*3*1 = 3 - 6 + 8.1 = 5.1
            dy = 1.7*3 - 2*1 + 1 = 5.1 - 2 + 1 = 4.1
            dz = 2*2*3 - 9*1 = 12 - 9 = 3
        """
        sim = self._make_sim(a=3.0, b=2.7, c=1.7, d=2.0, e=9.0)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 3.0, 1.0]))
        assert np.isclose(derivs[0], 5.1)
        assert np.isclose(derivs[1], 4.1)
        assert np.isclose(derivs[2], 3.0)

    def test_derivatives_z_only(self):
        """Test derivatives at [0, 0, 1] -- only z-damping term active.

            dx = 0 - 0 + 0 = 0
            dy = 0 - 0 + 1 = 1
            dz = 0 - 9*1 = -9
        """
        sim = self._make_sim(a=3.0, b=2.7, c=1.7, d=2.0, e=9.0)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], -9.0)


class TestDadrasFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> DadrasSimulation:
        defaults = {"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return DadrasSimulation(config)

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

    def test_at_least_one_fixed_point(self):
        """At least the origin should be found."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) >= 1

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=8,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestDadrasTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> DadrasSimulation:
        defaults = {
            "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return DadrasSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Dadras trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_trajectory_stays_finite(self):
        """No NaN or Inf after many steps."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_attractor_statistics(self):
        """Trajectory statistics should have positive std for chaotic regime."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=10000, n_transient=3000
        )
        # x and y should have significant spread in the chaotic regime
        assert stats["x_std"] > 0.01, "x_std too small for chaotic regime"
        assert stats["y_std"] > 0.01, "y_std too small for chaotic regime"
        assert stats["z_std"] > 0.01, "z_std too small for chaotic regime"

    def test_different_a_gives_different_trajectory(self):
        """Changing a should change the trajectory behavior."""
        sim1 = self._make_sim(a=3.0)
        sim2 = self._make_sim(a=5.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestDadrasDissipation:
    """Tests for the dissipation property."""

    def _make_sim(self, **kwargs) -> DadrasSimulation:
        defaults = {"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return DadrasSimulation(config)

    def test_divergence_value(self):
        """Divergence should be -a + c - e = -3 + 1.7 - 9 = -10.3."""
        sim = self._make_sim()
        assert np.isclose(sim.divergence, -10.3)

    def test_divergence_negative(self):
        """System should be dissipative (negative divergence) at classic params."""
        sim = self._make_sim()
        assert sim.divergence < 0, f"Divergence {sim.divergence} not negative"

    def test_is_chaotic_property(self):
        """is_chaotic should be True at standard params."""
        sim = self._make_sim()
        assert sim.is_chaotic is True

    def test_is_chaotic_false_for_weak_coupling(self):
        """is_chaotic should be False when nonlinear coupling is too weak."""
        sim = self._make_sim(b=0.1, d=0.1)
        assert sim.is_chaotic is False


class TestDadrasChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Dadras at standard parameters should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={
                "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
                "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim = DadrasSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert lam > 0.0, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_lyapunov_bounded(self):
        """Lyapunov exponent should be bounded (not diverging)."""
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={
                "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
            },
        )
        sim = DadrasSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert abs(lam) < 20.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_a(self):
        """Lyapunov exponent should change as a varies."""
        lam_a3 = _compute_lyapunov_at_a(3.0)
        lam_a5 = _compute_lyapunov_at_a(5.0)
        # Different a values should give different Lyapunov exponents
        assert lam_a3 != lam_a5, "Lyapunov should differ for different a"


class TestDadrasNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_convergence(self):
        """Smaller dt should give more accurate results (convergence test)."""
        # Run with dt=0.005
        config1 = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=200,
            parameters={
                "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
                "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim1 = DadrasSimulation(config1)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.001 (5x finer, same total time = 1.0)
        config2 = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters={
                "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
                "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim2 = DadrasSimulation(config2)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state_fine = sim2.observe().copy()

        # For RK4, the error should scale as dt^4; states should be close
        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.01, (
            f"RK4 convergence error {error:.6f} too large between dt=0.005 and dt=0.001"
        )

    def test_trajectory_statistics_all_finite(self):
        """Trajectory statistics should all be finite numbers."""
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters={"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0},
        )
        sim = DadrasSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"


class TestDadrasRediscovery:
    """Tests for Dadras data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.dadras import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 3.0
        assert data["b"] == 2.7
        assert data["c"] == 1.7
        assert data["d"] == 2.0
        assert data["e"] == 9.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.dadras import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep over a should produce valid data."""
        from simulating_anything.rediscovery.dadras import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.005)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_lyapunov_vs_e_data(self):
        """Lyapunov sweep over e should produce valid data."""
        from simulating_anything.rediscovery.dadras import (
            generate_lyapunov_vs_e_data,
        )

        data = generate_lyapunov_vs_e_data(n_e=5, n_steps=3000, dt=0.005)
        assert len(data["e"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.dadras import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.005)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data


def _compute_lyapunov_at_a(a: float) -> float:
    """Helper to compute Lyapunov exponent at a given a."""
    config = SimulationConfig(
        domain=_DADRAS_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={
            "a": a, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        },
    )
    sim = DadrasSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.005)
