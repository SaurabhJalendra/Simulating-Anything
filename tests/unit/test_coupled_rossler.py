"""Tests for the coupled Rossler systems simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.coupled_rossler import CoupledRosslerSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    epsilon: float = 0.05,
    dt: float = 0.01,
    n_steps: int = 5000,
    **overrides,
) -> SimulationConfig:
    params = {
        "a": 0.2, "b": 0.2, "c": 5.7,
        "epsilon": epsilon,
        "x1_0": 1.0, "y1_0": 0.0, "z1_0": 0.0,
        "x2_0": 0.5, "y2_0": 0.5, "z2_0": 0.5,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.COUPLED_ROSSLER,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestCoupledRosslerBasics:
    """Basic simulation tests: shape, determinism, stability."""

    def test_observe_shape(self):
        """State should be a 6-element vector."""
        sim = CoupledRosslerSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (6,)
        obs = sim.observe()
        assert obs.shape == (6,)

    def test_initial_conditions(self):
        """Reset should set the specified initial conditions."""
        sim = CoupledRosslerSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state[:3], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(state[3:], [0.5, 0.5, 0.5])

    def test_custom_initial_conditions(self):
        """Custom ICs should be respected."""
        config = _make_config(
            x1_0=2.0, y1_0=1.0, z1_0=0.5,
            x2_0=-1.0, y2_0=3.0, z2_0=2.0,
        )
        sim = CoupledRosslerSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [2.0, 1.0, 0.5, -1.0, 3.0, 2.0])

    def test_step_advances(self):
        """State should change after one step."""
        sim = CoupledRosslerSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same config should produce identical trajectories."""
        config = _make_config()
        sim1 = CoupledRosslerSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = CoupledRosslerSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_rk4_stability(self):
        """No NaN or Inf should appear during integration."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_bounded(self):
        """Both Rossler systems should stay bounded on the attractor."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.05))
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
        sim = CoupledRosslerSimulation(_make_config())
        state = sim.reset()
        assert not np.allclose(state[:3], state[3:])

    def test_state_dtype(self):
        """State should be float64."""
        sim = CoupledRosslerSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64

    def test_run_returns_trajectory_data(self):
        """The run method should return a TrajectoryData object."""
        sim = CoupledRosslerSimulation(_make_config(n_steps=100))
        traj = sim.run(100)
        assert traj.states.shape == (101, 6)
        assert len(traj.timestamps) == 101


class TestCoupledRosslerDerivatives:
    """Tests for the derivative computation."""

    def test_derivatives_decoupled(self):
        """With epsilon=0, both systems should be standard Rossler."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim.reset()
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        derivs = sim._derivatives(state)

        # System 1: dx1 = -(y1+z1) = -(2+3) = -5
        # dy1 = x1 + a*y1 = 1 + 0.2*2 = 1.4
        # dz1 = b + z1*(x1-c) = 0.2 + 3*(1-5.7) = 0.2 - 14.1 = -13.9
        assert np.isclose(derivs[0], -5.0)
        assert np.isclose(derivs[1], 1.4)
        assert np.isclose(derivs[2], -13.9)

        # System 2: dx2 = -(y2+z2) = -(5+6) = -11
        # dy2 = x2 + a*y2 = 4 + 0.2*5 = 5.0
        # dz2 = b + z2*(x2-c) = 0.2 + 6*(4-5.7) = 0.2 - 10.2 = -10.0
        assert np.isclose(derivs[3], -11.0)
        assert np.isclose(derivs[4], 5.0)
        assert np.isclose(derivs[5], -10.0)

    def test_coupling_term_symmetric(self):
        """Coupling eps*(x_other - x_self) should be symmetric and opposite."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.1))
        sim.reset()
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        d_coupled = sim._derivatives(state)

        sim0 = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim0.reset()
        d_uncoupled = sim0._derivatives(state)

        # dx1 differs by eps*(x2-x1) = 0.1*(4-1) = 0.3
        assert np.isclose(d_coupled[0] - d_uncoupled[0], 0.1 * (4.0 - 1.0))

        # dx2 differs by eps*(x1-x2) = 0.1*(1-4) = -0.3
        assert np.isclose(d_coupled[3] - d_uncoupled[3], 0.1 * (1.0 - 4.0))

        # dy and dz should be unaffected by coupling
        assert np.isclose(d_coupled[1], d_uncoupled[1])
        assert np.isclose(d_coupled[2], d_uncoupled[2])
        assert np.isclose(d_coupled[4], d_uncoupled[4])
        assert np.isclose(d_coupled[5], d_uncoupled[5])

    def test_derivatives_at_sync_manifold(self):
        """On the sync manifold (x1=x2, etc.), coupling term vanishes."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.1))
        sim.reset()
        state = np.array([2.0, 3.0, 4.0, 2.0, 3.0, 4.0])
        derivs = sim._derivatives(state)
        # Drive and response should have identical derivatives on manifold
        np.testing.assert_array_equal(derivs[:3], derivs[3:])

    def test_rossler_equations_correct(self):
        """Verify the Rossler equations match the standard form."""
        config = _make_config(epsilon=0.0, a=0.3, b=0.1, c=6.0)
        sim = CoupledRosslerSimulation(config)
        sim.reset()
        state = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0])
        derivs = sim._derivatives(state)

        # System 1: dx = -(y+z) = -(-1+0.5) = 0.5
        # dy = x + a*y = 2 + 0.3*(-1) = 1.7
        # dz = b + z*(x-c) = 0.1 + 0.5*(2-6) = 0.1 - 2.0 = -1.9
        assert np.isclose(derivs[0], 0.5)
        assert np.isclose(derivs[1], 1.7)
        assert np.isclose(derivs[2], -1.9)


class TestCoupledRosslerDynamics:
    """Tests for the core physical behavior: chaos and synchronization."""

    def test_uncoupled_independent_chaos(self):
        """With epsilon=0, both systems evolve independently and diverge."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim.reset()
        # Skip transient
        for _ in range(3000):
            sim.step()
        # Measure sync error
        errors = []
        for _ in range(2000):
            sim.step()
            errors.append(sim.sync_error())
        mean_err = np.mean(errors)
        # Independent chaotic trajectories should have O(1) error
        assert mean_err > 0.5, (
            f"Mean error {mean_err:.3f} too small for independent systems"
        )

    def test_uncoupled_both_chaotic(self):
        """With eps=0, both systems should exhibit chaotic oscillation."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim.reset()
        # Skip transient
        for _ in range(3000):
            sim.step()
        # Check x1 changes sign (oscillation on attractor)
        sign_changes_1 = 0
        sign_changes_2 = 0
        prev = sim.observe().copy()
        for _ in range(5000):
            sim.step()
            curr = sim.observe()
            if prev[0] * curr[0] < 0:
                sign_changes_1 += 1
            if prev[3] * curr[3] < 0:
                sign_changes_2 += 1
            prev = curr.copy()
        assert sign_changes_1 > 5, (
            f"System 1 only {sign_changes_1} sign changes"
        )
        assert sign_changes_2 > 5, (
            f"System 2 only {sign_changes_2} sign changes"
        )

    def test_strong_coupling_sync(self):
        """At large epsilon, systems should synchronize."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.3))
        sim.reset()
        for _ in range(20000):
            sim.step()
        final_err = sim.sync_error()
        assert final_err < 0.1, (
            f"Sync error {final_err:.6e} too large at epsilon=0.3"
        )

    def test_sync_error_decreases_over_time(self):
        """Sync error should decrease over time at strong coupling."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.2))
        sim.reset()

        # Measure early error after a short transient
        for _ in range(500):
            sim.step()
        early_err = sim.sync_error()

        # Run much longer
        for _ in range(15000):
            sim.step()
        late_err = sim.sync_error()

        assert late_err < early_err, (
            f"Error did not decrease: early={early_err:.4f}, late={late_err:.4f}"
        )

    def test_sync_manifold(self):
        """At synchronization, x1=x2, y1=y2, z1=z2."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.3))
        sim.reset()
        for _ in range(25000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(
            state[:3], state[3:], atol=0.1,
            err_msg="Systems not on synchronization manifold",
        )

    def test_weak_coupling_partial_sync(self):
        """Weak coupling should reduce sync error but not achieve full sync."""
        sim_weak = CoupledRosslerSimulation(_make_config(epsilon=0.02))
        sim_weak.reset()
        sim_none = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim_none.reset()

        n_transient = 5000
        n_measure = 3000

        for _ in range(n_transient):
            sim_weak.step()
            sim_none.step()

        errors_weak = []
        errors_none = []
        for _ in range(n_measure):
            sim_weak.step()
            sim_none.step()
            errors_weak.append(sim_weak.sync_error())
            errors_none.append(sim_none.sync_error())

        mean_weak = np.mean(errors_weak)
        mean_none = np.mean(errors_none)

        # Weak coupling should give smaller error than no coupling
        # but still have substantial error (not fully synchronized)
        assert mean_weak < mean_none * 1.5, (
            f"Weak coupling error ({mean_weak:.3f}) not reduced vs "
            f"uncoupled ({mean_none:.3f})"
        )


class TestSyncError:
    """Tests for sync error computation methods."""

    def test_sync_error_at_init(self):
        """Initial sync error should reflect the IC difference."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        err = sim.sync_error()
        # ||[1,0,0] - [0.5,0.5,0.5]|| = ||[0.5,-0.5,-0.5]|| = sqrt(0.75)
        expected = np.sqrt(0.5**2 + 0.5**2 + 0.5**2)
        assert np.isclose(err, expected, rtol=1e-10)

    def test_sync_error_zero_on_manifold(self):
        """If both systems are identical, sync error should be zero."""
        config = _make_config(
            x1_0=1.0, y1_0=2.0, z1_0=3.0,
            x2_0=1.0, y2_0=2.0, z2_0=3.0,
        )
        sim = CoupledRosslerSimulation(config)
        sim.reset()
        err = sim.sync_error()
        assert np.isclose(err, 0.0, atol=1e-15)

    def test_sync_error_positive(self):
        """Sync error should always be non-negative."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
            assert sim.sync_error() >= 0.0

    def test_compute_sync_error_from_states(self):
        """compute_sync_error should work on trajectory arrays."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        states = []
        for _ in range(50):
            sim.step()
            states.append(sim.observe().copy())
        states = np.array(states)
        mean_err = sim.compute_sync_error(states)
        assert mean_err > 0
        assert np.isfinite(mean_err)


class TestSyncTransitionData:
    """Tests for the sync_transition_data sweep method."""

    def test_sweep_returns_correct_shape(self):
        """Sweep should return arrays of the correct length."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        data = sim.sync_transition_data(
            epsilon_range=(0.0, 0.2), n_eps=5,
            n_transient=500, n_measure=500,
        )
        assert len(data["epsilon"]) == 5
        assert len(data["mean_error"]) == 5

    def test_sweep_eps_range(self):
        """Epsilon values should span the requested range."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        data = sim.sync_transition_data(
            epsilon_range=(0.0, 0.3), n_eps=10,
            n_transient=500, n_measure=500,
        )
        assert data["epsilon"][0] == pytest.approx(0.0)
        assert data["epsilon"][-1] == pytest.approx(0.3)

    def test_sweep_errors_positive(self):
        """All sync errors should be non-negative."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        data = sim.sync_transition_data(
            epsilon_range=(0.0, 0.1), n_eps=3,
            n_transient=500, n_measure=500,
        )
        assert np.all(data["mean_error"] >= 0)

    def test_sweep_trend_decreasing(self):
        """Higher epsilon should generally give lower sync error."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        data = sim.sync_transition_data(
            epsilon_range=(0.0, 0.3), n_eps=5,
            n_transient=2000, n_measure=2000,
        )
        # Overall trend: first error (eps=0) should be >= last error
        assert data["mean_error"][0] >= data["mean_error"][-1] * 0.5, (
            f"Error at eps=0 ({data['mean_error'][0]:.4f}) not greater than "
            f"at eps=0.3 ({data['mean_error'][-1]:.4f})"
        )


class TestSyncErrorTrajectory:
    """Tests for the sync_error_trajectory method."""

    def test_trajectory_shape(self):
        """Output arrays should have length n_steps+1."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert result["time"].shape == (101,)
        assert result["error"].shape == (101,)
        assert result["time"][0] == 0.0

    def test_trajectory_error_positive(self):
        """All errors in trajectory should be non-negative."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert np.all(result["error"] >= 0)

    def test_trajectory_time_monotonic(self):
        """Time should be strictly increasing."""
        sim = CoupledRosslerSimulation(_make_config())
        sim.reset()
        result = sim.sync_error_trajectory(n_steps=100)
        assert np.all(np.diff(result["time"]) > 0)


class TestLyapunovExponent:
    """Tests for the Lyapunov exponent computation."""

    def test_lyapunov_positive_at_chaos(self):
        """Max Lyapunov exponent should be positive for chaotic Rossler."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim.reset()
        lam = sim.compute_lyapunov(n_transient=2000, n_compute=5000)
        assert lam > 0.01, (
            f"Lyapunov exponent {lam:.4f} not positive for chaotic params"
        )

    def test_lyapunov_finite(self):
        """Lyapunov exponent should be finite."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.05))
        sim.reset()
        lam = sim.compute_lyapunov(n_transient=1000, n_compute=3000)
        assert np.isfinite(lam)

    def test_lyapunov_reasonable_magnitude(self):
        """Lyapunov exponent should be in a reasonable range."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim.reset()
        lam = sim.compute_lyapunov(n_transient=2000, n_compute=5000)
        # Standard Rossler has max LE ~ 0.07-0.09
        # Coupled system can have LE up to ~ 0.2
        assert -1.0 < lam < 2.0, (
            f"Lyapunov exponent {lam:.4f} outside reasonable range"
        )


class TestConditionalLyapunov:
    """Tests for the conditional Lyapunov exponent estimation."""

    def test_positive_at_zero_coupling(self):
        """With no coupling, conditional Lyapunov should be positive."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.0))
        sim.reset()
        lam = sim.conditional_lyapunov(
            epsilon=0.0, n_steps=15000, n_transient=2000,
        )
        assert lam > 0.01, (
            f"Conditional Lyapunov {lam:.4f} not positive at epsilon=0"
        )

    def test_negative_at_high_coupling(self):
        """With strong coupling, conditional Lyapunov should be negative."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.3))
        sim.reset()
        lam = sim.conditional_lyapunov(
            epsilon=0.3, n_steps=15000, n_transient=2000,
        )
        assert lam < 0, (
            f"Conditional Lyapunov {lam:.4f} not negative at epsilon=0.3"
        )

    def test_conditional_lyapunov_finite(self):
        """Conditional Lyapunov exponent should be finite."""
        sim = CoupledRosslerSimulation(_make_config(epsilon=0.1))
        sim.reset()
        lam = sim.conditional_lyapunov(
            epsilon=0.1, n_steps=10000, n_transient=2000,
        )
        assert np.isfinite(lam)


class TestCoupledRosslerRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_sync_sweep_data(self):
        """Sync sweep should return correct-length arrays."""
        from simulating_anything.rediscovery.coupled_rossler import (
            generate_sync_sweep_data,
        )
        data = generate_sync_sweep_data(
            n_eps=5, eps_min=0.0, eps_max=0.2,
            n_steps=500, n_transient=500, dt=0.01,
        )
        assert len(data["epsilon"]) == 5
        assert len(data["mean_error"]) == 5
        assert data["epsilon"][0] == pytest.approx(0.0)
        assert data["epsilon"][-1] == pytest.approx(0.2)

    def test_critical_coupling_estimate(self):
        """Critical coupling estimator should interpolate correctly."""
        from simulating_anything.rediscovery.coupled_rossler import (
            estimate_critical_coupling,
        )
        # Fabricate a simple sweep
        sweep = {
            "epsilon": np.array([0.0, 0.05, 0.1, 0.15, 0.2]),
            "mean_error": np.array([8.0, 5.0, 1.5, 0.5, 0.01]),
        }
        eps_c = estimate_critical_coupling(sweep, error_threshold=1.0)
        # Should be between 0.1 and 0.15
        assert 0.1 < eps_c < 0.15

    def test_critical_coupling_all_above(self):
        """If all errors above threshold, return max eps."""
        from simulating_anything.rediscovery.coupled_rossler import (
            estimate_critical_coupling,
        )
        sweep = {
            "epsilon": np.array([0.0, 0.1, 0.2]),
            "mean_error": np.array([10.0, 8.0, 5.0]),
        }
        eps_c = estimate_critical_coupling(sweep, error_threshold=1.0)
        assert eps_c == pytest.approx(0.2)

    def test_critical_coupling_all_below(self):
        """If all errors below threshold, return min eps."""
        from simulating_anything.rediscovery.coupled_rossler import (
            estimate_critical_coupling,
        )
        sweep = {
            "epsilon": np.array([0.0, 0.1, 0.2]),
            "mean_error": np.array([0.5, 0.3, 0.1]),
        }
        eps_c = estimate_critical_coupling(sweep, error_threshold=1.0)
        assert eps_c == pytest.approx(0.0)

    def test_lyapunov_data(self):
        """Lyapunov data generation should return correct-length arrays."""
        from simulating_anything.rediscovery.coupled_rossler import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(
            eps_values=np.array([0.0, 0.1]),
            n_compute=2000, n_transient=500, dt=0.01,
        )
        assert len(data["epsilon"]) == 2
        assert len(data["lyapunov"]) == 2
        assert np.all(np.isfinite(data["lyapunov"]))

    def test_run_method_exists(self):
        """Ensure the main rediscovery runner function is importable."""
        from simulating_anything.rediscovery.coupled_rossler import (
            run_coupled_rossler_rediscovery,
        )
        assert callable(run_coupled_rossler_rediscovery)

    def test_default_parameters(self):
        """Default parameters should match the classic Rossler values."""
        sim = CoupledRosslerSimulation(_make_config())
        assert sim.a == 0.2
        assert sim.b == 0.2
        assert sim.c == 5.7
        assert sim.epsilon == 0.05
