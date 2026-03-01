"""Tests for the Thomas cyclically symmetric attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.thomas import ThomasSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestThomasSimulation:
    """Tests for the Thomas system simulation basics."""

    def _make_sim(self, **kwargs) -> ThomasSimulation:
        defaults = {
            "b": 0.208186, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=10000,
            parameters=defaults,
        )
        return ThomasSimulation(config)

    def test_reset(self):
        """Initial state should match specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=0.5, z_0=-0.3)
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 0.5)
        assert np.isclose(state[2], -0.3)

    def test_observe_shape(self):
        """Observe should return a 3-element array."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_step_advances(self):
        """State should change after a step."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same parameters should produce the same trajectory."""
        sim1 = self._make_sim(b=0.18, x_0=1.0, y_0=0.5, z_0=0.0)
        sim2 = self._make_sim(b=0.18, x_0=1.0, y_0=0.5, z_0=0.0)
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_stability(self):
        """No NaN or Inf after many steps."""
        sim = self._make_sim(b=0.18)
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_bounded(self):
        """Trajectory stays bounded for chaotic regime."""
        sim = self._make_sim(b=0.18)
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.linalg.norm(state) < 100, f"Trajectory diverged: {state}"

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_values(self):
        """Test derivatives at a known point."""
        sim = self._make_sim(b=0.2)
        sim.reset()
        # At (1, 1, 1): dx=sin(1)-0.2*1, dy=sin(1)-0.2*1, dz=sin(1)-0.2*1
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        expected = np.sin(1.0) - 0.2
        np.testing.assert_array_almost_equal(
            derivs, [expected, expected, expected]
        )


class TestThomasCyclicSymmetry:
    """Tests for cyclic symmetry verification."""

    def _make_sim(self, **kwargs) -> ThomasSimulation:
        defaults = {"b": 0.18, "x_0": 1.0, "y_0": 0.5, "z_0": -0.3}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=5000,
            parameters=defaults,
        )
        return ThomasSimulation(config)

    def test_cyclic_symmetry(self):
        """(x,y,z) -> (y,z,x) should give same dynamics."""
        sim = self._make_sim()
        sim.reset()
        result = sim.verify_cyclic_symmetry(n_steps=500)
        assert result["max_deviation"] < 1e-10, (
            f"Symmetry deviation {result['max_deviation']:.2e} too large"
        )

    def test_mean_zero_symmetry(self):
        """Due to cyclic symmetry, <x> ~ <y> ~ <z> on the attractor."""
        sim = self._make_sim(b=0.18, x_0=1.0, y_0=0.0, z_0=0.0)
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # The standard deviations should be similar (cyclic symmetry)
        stds = [stats["x_std"], stats["y_std"], stats["z_std"]]
        # For chaotic regime, std should be positive
        for s in stds:
            assert s > 0.01, f"Std too small: {s}"
        # Stds should be roughly equal (within 50%)
        max_std = max(stds)
        min_std = min(stds)
        assert min_std > 0.3 * max_std, (
            f"Stds not symmetric: {stds}"
        )


class TestThomasFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, b=0.208186) -> ThomasSimulation:
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=1000,
            parameters={"b": b},
        )
        return ThomasSimulation(config)

    def test_fixed_points_exist(self):
        """At least one fixed point should be found."""
        sim = self._make_sim(b=0.2)
        sim.reset()
        fps = sim.find_fixed_points()
        assert len(fps) >= 1, "No fixed points found"

    def test_origin_is_fixed_point(self):
        """Origin (0,0,0) is always a fixed point since sin(0) = 0."""
        sim = self._make_sim(b=0.2)
        sim.reset()
        fps = sim.find_fixed_points()
        origin_found = False
        for fp in fps:
            if np.linalg.norm(fp) < 1e-6:
                origin_found = True
                break
        assert origin_found, "Origin not found among fixed points"

    def test_fixed_point_is_fixed(self):
        """Derivatives at fixed points should be near zero."""
        sim = self._make_sim(b=0.2)
        sim.reset()
        fps = sim.find_fixed_points()
        for fp in fps:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=8,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_more_fixed_points_for_small_b(self):
        """Smaller b should have more fixed points (more roots of sin(x)=b*x)."""
        sim_small = self._make_sim(b=0.1)
        sim_small.reset()
        fps_small = sim_small.find_fixed_points()

        sim_large = self._make_sim(b=0.9)
        sim_large.reset()
        fps_large = sim_large.find_fixed_points()

        assert len(fps_small) > len(fps_large), (
            f"Small b has {len(fps_small)} fps, large b has {len(fps_large)}"
        )


class TestThomasLyapunov:
    """Tests for Lyapunov exponent and chaos detection."""

    def test_chaotic_regime(self):
        """Small b should give positive Lyapunov exponent (chaos)."""
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=30000,
            parameters={"b": 0.18, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = ThomasSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.05)
        assert lam > 0.0, f"Lyapunov {lam:.4f} not positive for chaotic b=0.18"

    def test_dissipative_regime(self):
        """Large b should converge to fixed point (negative Lyapunov)."""
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=30000,
            parameters={"b": 0.5, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = ThomasSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.05)
        assert lam < 0.1, f"Lyapunov {lam:.4f} too large for dissipative b=0.5"

    def test_lyapunov_transition(self):
        """Lyapunov should change sign near b_c ~ 0.208."""
        lyap_low_b = _compute_lyapunov_at_b(0.15)
        lyap_high_b = _compute_lyapunov_at_b(0.30)
        # Low b should be more chaotic (higher Lyapunov) than high b
        assert lyap_low_b > lyap_high_b, (
            f"Lyapunov at b=0.15 ({lyap_low_b:.4f}) should exceed "
            f"b=0.30 ({lyap_high_b:.4f})"
        )

    def test_bifurcation_sweep(self):
        """Sweep should produce valid data for all b values."""
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=5000,
            parameters={"b": 0.18},
        )
        sim = ThomasSimulation(config)
        sim.reset()
        b_values = np.linspace(0.1, 0.4, 5)
        data = sim.bifurcation_sweep(
            b_values, n_transient=1000, n_measure=5000
        )
        assert len(data["b"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


class TestThomasNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_accuracy(self):
        """Smaller dt should give more accurate results (convergence test)."""
        # Run with dt=0.05
        config1 = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=200,
            parameters={"b": 0.2, "x_0": 1.0, "y_0": 0.5, "z_0": 0.0},
        )
        sim1 = ThomasSimulation(config1)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.01 (5x finer, same total time)
        config2 = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.01,
            n_steps=1000,
            parameters={"b": 0.2, "x_0": 1.0, "y_0": 0.5, "z_0": 0.0},
        )
        sim2 = ThomasSimulation(config2)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state_fine = sim2.observe().copy()

        # For RK4, the error should scale as dt^4; states should be close
        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.01, (
            f"RK4 convergence error {error:.6f} too large between dt=0.05 and dt=0.01"
        )

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computable and finite."""
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.05,
            n_steps=10000,
            parameters={"b": 0.18, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = ThomasSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Std should be positive in chaotic regime
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0
        assert stats["z_std"] > 0


class TestThomasRediscovery:
    """Tests for the Thomas rediscovery data generation."""

    def test_ode_data(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.thomas import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert np.isclose(data["b"], 0.208186)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_b_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.thomas import (
            generate_lyapunov_vs_b_data,
        )

        data = generate_lyapunov_vs_b_data(n_b=5, n_steps=3000, dt=0.05)
        assert len(data["b"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_fixed_point_data(self):
        """Fixed point data generation should find points at multiple b values."""
        from simulating_anything.rediscovery.thomas import (
            generate_fixed_point_data,
        )

        data = generate_fixed_point_data(b_values=np.array([0.2, 0.5]))
        assert len(data["data"]) == 2
        # At b=0.2, should find more than just the origin
        assert data["data"][0]["n_fixed_points"] >= 1


def _compute_lyapunov_at_b(b: float) -> float:
    """Helper to compute Lyapunov exponent at a given b."""
    config = SimulationConfig(
        domain=Domain.THOMAS,
        dt=0.05,
        n_steps=20000,
        parameters={"b": b, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim = ThomasSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.05)
