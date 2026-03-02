"""Tests for the Halvorsen attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.halvorsen import HalvorsenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_HALVORSEN_DOMAIN = Domain.HALVORSEN


class TestHalvorsenSimulation:
    """Tests for the Halvorsen system simulation basics."""

    def _make_sim(self, **kwargs) -> HalvorsenSimulation:
        defaults = {
            "a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return HalvorsenSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 1.89
        assert sim.x_0 == -5.0
        assert sim.y_0 == 0.0
        assert sim.z_0 == 0.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=2.0, x_0=1.0, y_0=-1.0, z_0=0.5)
        assert sim.a == 2.0
        assert sim.x_0 == 1.0
        assert sim.y_0 == -1.0
        assert sim.z_0 == 0.5

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=2.0, y_0=-3.0, z_0=1.5)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -3.0)
        assert np.isclose(state[2], 1.5)

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
        sim1 = self._make_sim(a=1.89, x_0=-5.0, y_0=0.0, z_0=0.0)
        sim2 = self._make_sim(a=1.89, x_0=-5.0, y_0=0.0, z_0=0.0)
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


class TestHalvorsenDerivatives:
    """Tests for the Halvorsen ODE derivative computation."""

    def _make_sim(self, **kwargs) -> HalvorsenSimulation:
        defaults = {"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return HalvorsenSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=1.89:
            dx = -1.89*1 - 4*1 - 4*1 - 1^2 = -1.89 - 4 - 4 - 1 = -10.89
            dy = -1.89*1 - 4*1 - 4*1 - 1^2 = -1.89 - 4 - 4 - 1 = -10.89
            dz = -1.89*1 - 4*1 - 4*1 - 1^2 = -1.89 - 4 - 4 - 1 = -10.89
        """
        sim = self._make_sim(a=1.89)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        expected = -1.89 - 4.0 - 4.0 - 1.0  # = -10.89
        np.testing.assert_array_almost_equal(
            derivs, [expected, expected, expected]
        )

    def test_derivatives_asymmetric_point(self):
        """Test derivatives at an asymmetric point.

        At state [2, 0, -1] with a=2.0:
            dx = -2*2 - 4*0 - 4*(-1) - 0^2 = -4 + 0 + 4 - 0 = 0
            dy = -2*0 - 4*(-1) - 4*2 - (-1)^2 = 0 + 4 - 8 - 1 = -5
            dz = -2*(-1) - 4*2 - 4*0 - 2^2 = 2 - 8 + 0 - 4 = -10
        """
        sim = self._make_sim(a=2.0)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 0.0, -1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], -5.0)
        assert np.isclose(derivs[2], -10.0)

    def test_derivatives_cyclic_structure(self):
        """The derivative function should have cyclic symmetry.

        If we permute the input (x,y,z) -> (y,z,x), the output should
        also be permuted: (dx,dy,dz) -> (dy,dz,dx).
        """
        sim = self._make_sim(a=1.89)
        sim.reset()
        state1 = np.array([2.0, -1.0, 3.0])
        state2 = np.array([-1.0, 3.0, 2.0])  # cyclic permutation

        d1 = sim._derivatives(state1)
        d2 = sim._derivatives(state2)

        # d2 should be cyclic permutation of d1
        np.testing.assert_array_almost_equal(
            d2, np.array([d1[1], d1[2], d1[0]])
        )


class TestHalvorsenCyclicSymmetry:
    """Tests for cyclic symmetry verification."""

    def _make_sim(self, **kwargs) -> HalvorsenSimulation:
        defaults = {"a": 1.89, "x_0": -5.0, "y_0": 1.0, "z_0": -1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=5000,
            parameters=defaults,
        )
        return HalvorsenSimulation(config)

    def test_cyclic_symmetry(self):
        """(x, y, z) -> (y, z, x) should give same dynamics."""
        sim = self._make_sim()
        sim.reset()
        result = sim.cyclic_symmetry_check(n_steps=500)
        assert result["max_deviation"] < 1e-10, (
            f"Symmetry deviation {result['max_deviation']:.2e} too large"
        )

    def test_cyclic_symmetry_different_ic(self):
        """Symmetry should hold for any initial conditions."""
        sim = self._make_sim(x_0=3.0, y_0=-2.0, z_0=1.0)
        sim.reset()
        result = sim.cyclic_symmetry_check(n_steps=500)
        assert result["max_deviation"] < 1e-10

    def test_mean_symmetry_on_attractor(self):
        """Due to cyclic symmetry, standard deviations should be similar on the attractor."""
        sim = self._make_sim(a=1.89, x_0=-5.0, y_0=0.0, z_0=0.0)
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # Standard deviations should be roughly equal (cyclic symmetry)
        stds = [stats["x_std"], stats["y_std"], stats["z_std"]]
        for s in stds:
            assert s > 0.1, f"Std too small: {s}"
        max_std = max(stds)
        min_std = min(stds)
        assert min_std > 0.3 * max_std, (
            f"Stds not symmetric: {stds}"
        )


class TestHalvorsenTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> HalvorsenSimulation:
        defaults = {
            "a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return HalvorsenSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Halvorsen trajectory should remain bounded for classic parameters."""
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
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_attractor_statistics_finite(self):
        """Trajectory statistics should be computable and finite."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Std should be positive in chaotic regime
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0
        assert stats["z_std"] > 0

    def test_different_a_gives_different_trajectory(self):
        """Changing a should change the trajectory behavior."""
        sim1 = self._make_sim(a=1.89)
        sim2 = self._make_sim(a=1.5)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestHalvorsenLyapunov:
    """Tests for Lyapunov exponent estimation and chaos detection."""

    def test_positive_lyapunov_chaotic(self):
        """Classic a=1.89 should have positive Lyapunov exponent (chaos)."""
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=30000,
            parameters={"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = HalvorsenSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, f"Lyapunov {lam:.4f} not positive for chaotic a=1.89"

    def test_high_dissipation_non_chaotic(self):
        """Large a should suppress chaos (high dissipation)."""
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=30000,
            parameters={"a": 5.0, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = HalvorsenSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam < 0.5, f"Lyapunov {lam:.4f} too large for dissipative a=5.0"

    def test_lyapunov_varies_with_a(self):
        """Lyapunov exponent should change as a varies."""
        lyap_189 = _compute_lyapunov_at_a(1.89)
        lyap_300 = _compute_lyapunov_at_a(3.0)
        # a=1.89 is chaotic; a=3.0 is more dissipative
        assert lyap_189 > lyap_300, (
            f"Lyapunov at a=1.89 ({lyap_189:.4f}) should exceed "
            f"a=3.0 ({lyap_300:.4f})"
        )

    def test_lyapunov_is_finite(self):
        """Lyapunov estimate should be finite."""
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=10000,
            parameters={"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = HalvorsenSimulation(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert np.isfinite(lam), f"Non-finite Lyapunov: {lam}"


class TestHalvorsenNumerics:
    """Tests for numerical accuracy and RK4 integration."""

    def test_rk4_convergence(self):
        """Smaller dt should give more accurate results (convergence test)."""
        # Run with dt=0.01
        config1 = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=100,
            parameters={"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim1 = HalvorsenSimulation(config1)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.002 (5x finer, same total time = 1.0)
        config2 = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.002,
            n_steps=500,
            parameters={"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim2 = HalvorsenSimulation(config2)
        sim2.reset()
        for _ in range(500):
            sim2.step()
        state_fine = sim2.observe().copy()

        # For RK4, error scales as dt^4; states should be very close at T=1
        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.1, (
            f"RK4 convergence error {error:.6f} too large between dt=0.01 and dt=0.002"
        )

    def test_bifurcation_sweep(self):
        """Sweep should produce valid data for all a values."""
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=0.01,
            n_steps=5000,
            parameters={"a": 1.89},
        )
        sim = HalvorsenSimulation(config)
        sim.reset()
        a_values = np.linspace(1.5, 2.5, 5)
        data = sim.bifurcation_sweep(
            a_values, n_transient=1000, n_measure=5000
        )
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


class TestHalvorsenRediscovery:
    """Tests for the Halvorsen rediscovery data generation."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.halvorsen import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert np.isclose(data["a"], 1.89)
        assert np.all(np.isfinite(data["states"]))

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.halvorsen import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.halvorsen import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(n_a=5, n_steps=2000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.halvorsen import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.halvorsen import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data


def _compute_lyapunov_at_a(a: float) -> float:
    """Helper to compute Lyapunov exponent at a given a."""
    config = SimulationConfig(
        domain=_HALVORSEN_DOMAIN,
        dt=0.01,
        n_steps=20000,
        parameters={"a": a, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim = HalvorsenSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.01)
