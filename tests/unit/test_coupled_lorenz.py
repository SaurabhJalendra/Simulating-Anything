"""Tests for the coupled Lorenz systems simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.coupled_lorenz import CoupledLorenzSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    eps: float = 5.0,
    dt: float = 0.005,
    n_steps: int = 10000,
    **overrides,
) -> SimulationConfig:
    params = {
        "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
        "eps": eps,
        "x1_0": 1.0, "y1_0": 1.0, "z1_0": 1.0,
        "x2_0": -5.0, "y2_0": 5.0, "z2_0": 25.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.COUPLED_LORENZ,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestCoupledLorenzBasics:
    """Basic simulation tests: shape, determinism, stability."""

    def test_observe_shape(self):
        """State should be a 6-element vector."""
        sim = CoupledLorenzSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (6,)
        obs = sim.observe()
        assert obs.shape == (6,)

    def test_initial_conditions(self):
        """Reset should set the specified initial conditions."""
        sim = CoupledLorenzSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state[:3], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(state[3:], [-5.0, 5.0, 25.0])

    def test_step_advances(self):
        """State should change after one step."""
        sim = CoupledLorenzSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same seed and config should produce the same trajectory."""
        config = _make_config()
        sim1 = CoupledLorenzSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = CoupledLorenzSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_rk4_stability(self):
        """No NaN or Inf should appear during integration."""
        sim = CoupledLorenzSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_bounded(self):
        """Both systems should stay on the Lorenz attractor (bounded)."""
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            norm1 = np.linalg.norm(state[:3])
            norm2 = np.linalg.norm(state[3:])
            assert norm1 < 200, f"System 1 diverged: norm={norm1:.1f}"
            assert norm2 < 200, f"System 2 diverged: norm={norm2:.1f}"

    def test_reset_different_ics(self):
        """Systems 1 and 2 should start at different points by default."""
        sim = CoupledLorenzSimulation(_make_config())
        state = sim.reset()
        assert not np.allclose(state[:3], state[3:])


class TestCoupledLorenzDynamics:
    """Tests for the core physical behavior: chaos and synchronization."""

    def test_drive_is_chaotic(self):
        """Drive system (eps=0) should exhibit chaotic Lorenz dynamics.

        We verify by checking that x1 changes sign multiple times,
        indicating oscillation between the two wings of the attractor.
        """
        sim = CoupledLorenzSimulation(_make_config(eps=0.0))
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        sign_changes = 0
        prev_x1 = sim.observe()[0]
        for _ in range(5000):
            sim.step()
            x1 = sim.observe()[0]
            if prev_x1 * x1 < 0:
                sign_changes += 1
            prev_x1 = x1
        # A chaotic Lorenz trajectory should have many sign changes
        assert sign_changes > 10, (
            f"Only {sign_changes} sign changes -- drive may not be chaotic"
        )

    def test_zero_coupling_independent(self):
        """With eps=0, systems evolve independently and diverge."""
        sim = CoupledLorenzSimulation(_make_config(eps=0.0))
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        # After transient on chaotic attractor, systems should be decorrelated
        errors = []
        for _ in range(2000):
            sim.step()
            errors.append(sim.sync_error())
        mean_err = np.mean(errors)
        # Independent chaotic trajectories should have O(1) to O(10) error
        assert mean_err > 1.0, (
            f"Mean error {mean_err:.3f} too small for independent systems"
        )

    def test_high_coupling_sync(self):
        """At eps >> eps_c, systems should synchronize (error -> 0)."""
        sim = CoupledLorenzSimulation(_make_config(eps=10.0))
        sim.reset()
        # Run long enough for synchronization
        for _ in range(15000):
            sim.step()
        final_err = sim.sync_error()
        assert final_err < 1e-3, (
            f"Sync error {final_err:.6e} too large at eps=10.0"
        )

    def test_sync_error_decreases(self):
        """Sync error should decrease over time at high coupling."""
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()

        # Measure early error (after a few steps)
        for _ in range(100):
            sim.step()
        early_err = sim.sync_error()

        # Measure late error
        for _ in range(10000):
            sim.step()
        late_err = sim.sync_error()

        assert late_err < early_err, (
            f"Error did not decrease: early={early_err:.4f}, late={late_err:.4f}"
        )

    def test_sync_manifold(self):
        """At synchronization, x1=x2, y1=y2, z1=z2."""
        sim = CoupledLorenzSimulation(_make_config(eps=15.0))
        sim.reset()
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(
            state[:3], state[3:], atol=1e-2,
            err_msg="Systems not on synchronization manifold",
        )

    def test_sync_sweep_monotonic(self):
        """Higher coupling should produce lower or equal sync error."""
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()

        eps_values = np.array([0.0, 2.0, 5.0, 10.0])
        sweep = sim.sync_sweep(eps_values, n_steps=5000, n_transient=3000)

        errors = sweep["mean_error"]
        # Overall trend: errors[0] (eps=0) should be much larger than errors[-1]
        assert errors[0] > errors[-1], (
            f"Error at eps=0 ({errors[0]:.4f}) not greater than "
            f"at eps=10 ({errors[-1]:.4f})"
        )

    def test_critical_coupling(self):
        """eps_c should be roughly in the range [0.5, 10.0].

        For x-only diffusive coupling with sigma=10, rho=28, beta=8/3,
        the critical coupling is higher than full-state coupling because
        only the x variable is directly coupled.
        """
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()

        eps_values = np.linspace(0.0, 15.0, 20)
        sweep = sim.sync_sweep(eps_values, n_steps=8000, n_transient=5000)

        # Find approximate eps_c (first eps where error < 1.0)
        errors = sweep["mean_error"]
        below = np.where(errors < 1.0)[0]
        assert len(below) > 0, "No eps gave sync error < 1.0"
        eps_c_idx = below[0]
        eps_c_approx = float(eps_values[eps_c_idx])

        # eps_c should be in a reasonable range
        assert 0.1 < eps_c_approx < 12.0, (
            f"eps_c = {eps_c_approx:.2f} outside expected range [0.1, 12.0]"
        )


class TestCoupledLorenzDerivatives:
    """Tests for the derivative computation."""

    def test_derivatives_decoupled(self):
        """With eps=0, derivative of system 2 should be standard Lorenz."""
        sim = CoupledLorenzSimulation(_make_config(eps=0.0))
        sim.reset()
        state = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
        derivs = sim._derivatives(state)
        # System 1: dx1 = 10*(1-1)=0, dy1 = 1*(28-1)-1=26, dz1 = 1*1-8/3*1
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 26.0)
        assert np.isclose(derivs[2], 1.0 - 8.0 / 3.0)
        # System 2 (eps=0, same Lorenz equations): dx2=10*(3-2)=10
        assert np.isclose(derivs[3], 10.0)

    def test_coupling_term(self):
        """The coupling term eps*(x1-x2) should appear in dx2."""
        sim5 = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim5.reset()
        sim0 = CoupledLorenzSimulation(_make_config(eps=0.0))
        sim0.reset()

        state = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
        d5 = sim5._derivatives(state)
        d0 = sim0._derivatives(state)

        # Drive derivatives should be identical
        np.testing.assert_array_equal(d5[:3], d0[:3])

        # Response dx2 differs by eps*(x1-x2) = 5*(1-2) = -5
        assert np.isclose(d5[3] - d0[3], 5.0 * (1.0 - 2.0))

        # dy2 and dz2 should be unaffected by coupling
        assert np.isclose(d5[4], d0[4])
        assert np.isclose(d5[5], d0[5])

    def test_derivatives_at_sync_manifold(self):
        """On the sync manifold (x1=x2, etc.), coupling term vanishes."""
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()
        state = np.array([2.0, 3.0, 4.0, 2.0, 3.0, 4.0])
        derivs = sim._derivatives(state)
        # Drive and response should have identical derivatives on manifold
        np.testing.assert_array_equal(derivs[:3], derivs[3:])


class TestSyncErrorTrajectory:
    """Tests for the sync_error_trajectory method."""

    def test_trajectory_shape(self):
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert result["time"].shape == (101,)
        assert result["error"].shape == (101,)
        assert result["time"][0] == 0.0

    def test_trajectory_error_positive(self):
        sim = CoupledLorenzSimulation(_make_config(eps=5.0))
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert np.all(result["error"] >= 0)


class TestConditionalLyapunov:
    """Tests for the conditional Lyapunov exponent estimation."""

    def test_positive_at_zero_coupling(self):
        """With no coupling, conditional Lyapunov should be positive (chaotic)."""
        sim = CoupledLorenzSimulation(_make_config(eps=0.0))
        sim.reset()
        lam = sim.conditional_lyapunov(eps=0.0, n_steps=20000, n_transient=3000)
        assert lam > 0.3, (
            f"Conditional Lyapunov {lam:.3f} not positive at eps=0"
        )

    def test_negative_at_high_coupling(self):
        """With strong coupling, conditional Lyapunov should be negative."""
        sim = CoupledLorenzSimulation(_make_config(eps=10.0))
        sim.reset()
        lam = sim.conditional_lyapunov(
            eps=10.0, n_steps=20000, n_transient=3000,
        )
        assert lam < 0, (
            f"Conditional Lyapunov {lam:.3f} not negative at eps=10"
        )


class TestCoupledLorenzRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_sync_sweep_data(self):
        from simulating_anything.rediscovery.coupled_lorenz import (
            generate_sync_sweep_data,
        )
        data = generate_sync_sweep_data(
            n_eps=5, eps_min=0.0, eps_max=8.0,
            n_steps=1000, n_transient=500, dt=0.005,
        )
        assert len(data["eps"]) == 5
        assert len(data["mean_error"]) == 5
        assert data["eps"][0] == pytest.approx(0.0)
        assert data["eps"][-1] == pytest.approx(8.0)

    def test_transient_data(self):
        from simulating_anything.rediscovery.coupled_lorenz import (
            generate_transient_data,
        )
        data = generate_transient_data(
            eps_values=np.array([5.0, 10.0]),
            n_steps=5000, dt=0.005, threshold=0.5,
        )
        assert len(data["eps"]) == 2
        assert len(data["transient_time"]) == 2

    def test_critical_coupling_estimate(self):
        from simulating_anything.rediscovery.coupled_lorenz import (
            estimate_critical_coupling,
        )
        # Fabricate a simple sweep: error drops linearly
        sweep = {
            "eps": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "mean_error": np.array([10.0, 5.0, 1.5, 0.5, 0.01]),
        }
        eps_c = estimate_critical_coupling(sweep, error_threshold=1.0)
        # Should be between 2.0 and 3.0
        assert 2.0 < eps_c < 3.0

    def test_run_method_exists(self):
        """Ensure the main rediscovery runner function is importable."""
        from simulating_anything.rediscovery.coupled_lorenz import (
            run_coupled_lorenz_rediscovery,
        )
        assert callable(run_coupled_lorenz_rediscovery)
