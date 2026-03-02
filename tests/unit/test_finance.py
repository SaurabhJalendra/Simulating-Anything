"""Tests for the chaotic financial system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.finance import FinanceSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_FINANCE_DOMAIN = Domain.FINANCE


class TestFinanceSimulation:
    """Tests for the financial system simulation basics."""

    def _make_sim(self, **kwargs) -> FinanceSimulation:
        defaults = {
            "a": 1.0, "b": 0.1, "c": 1.0,
            "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return FinanceSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 1.0
        assert sim.b == 0.1
        assert sim.c == 1.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=0.9, b=0.2, c=1.2)
        assert sim.a == 0.9
        assert sim.b == 0.2
        assert sim.c == 1.2

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=1.5, y_0=-0.5, z_0=0.3)
        state = sim.reset()
        assert np.isclose(state[0], 1.5)
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
        """Same parameters should produce the same trajectory."""
        sim1 = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim2 = self._make_sim(a=1.0, b=0.1, c=1.0)
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


class TestFinanceDerivatives:
    """Tests for the financial system ODE derivative computation."""

    def _make_sim(self, **kwargs) -> FinanceSimulation:
        defaults = {
            "a": 1.0, "b": 0.1, "c": 1.0,
            "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return FinanceSimulation(config)

    def test_derivatives_at_trivial_fixed_point(self):
        """At (0, 1/b, 0) with a=1, b=0.1, c=1: derivatives should be zero.

        FP = (0, 10, 0):
            dx = 0 + (10 - 1)*0 = 0
            dy = 1 - 0.1*10 - 0 = 0
            dz = 0 - 0 = 0
        """
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 10.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at state [1, 1, 1] with a=1, b=0.1, c=1.

            dx = 1 + (1 - 1)*1 = 1
            dy = 1 - 0.1*1 - 1 = -0.1
            dz = -1 - 1*1 = -2
        """
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], -0.1)
        assert np.isclose(derivs[2], -2.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 3, 1] with a=1, b=0.1, c=1.

            dx = 1 + (3 - 1)*2 = 1 + 4 = 5
            dy = 1 - 0.1*3 - 4 = 1 - 0.3 - 4 = -3.3
            dz = -2 - 1*1 = -3
        """
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 3.0, 1.0]))
        assert np.isclose(derivs[0], 5.0)
        assert np.isclose(derivs[1], -3.3)
        assert np.isclose(derivs[2], -3.0)

    def test_derivatives_z_only(self):
        """Test derivatives at [0, 0, 1] -- only z terms active.

            dx = 1 + (0 - 1)*0 = 1
            dy = 1 - 0 - 0 = 1
            dz = 0 - 1*1 = -1
        """
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], -1.0)


class TestFinanceFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> FinanceSimulation:
        defaults = {"a": 1.0, "b": 0.1, "c": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return FinanceSimulation(config)

    def test_trivial_fixed_point(self):
        """The fixed point (0, 1/b, 0) should always exist."""
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        fps = sim.fixed_points
        # First fixed point should be (0, 1/b, 0) = (0, 10, 0)
        assert len(fps) >= 1
        fp0 = fps[0]
        assert np.isclose(fp0[0], 0.0)
        assert np.isclose(fp0[1], 10.0)
        assert np.isclose(fp0[2], 0.0)

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_nontrivial_fixed_points_exist(self):
        """For classic parameters, non-trivial (non-zero x) fixed points exist.

        y = a + 1/c = 1 + 1 = 2, x^2 = 1 - b*y = 1 - 0.2 = 0.8 > 0
        So there should be 3 fixed points total.
        """
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_nontrivial_fixed_points_symmetric(self):
        """Non-trivial fixed points should be symmetric: +x and -x."""
        sim = self._make_sim(a=1.0, b=0.1, c=1.0)
        sim.reset()
        fps = sim.fixed_points
        if len(fps) == 3:
            # fps[1] and fps[2] should have opposite x and z, same y
            assert np.isclose(fps[1][0], -fps[2][0])
            assert np.isclose(fps[1][1], fps[2][1])
            assert np.isclose(fps[1][2], -fps[2][2])

    def test_no_nontrivial_when_discriminant_negative(self):
        """When 1 - b*(a + 1/c) < 0, only trivial fixed point exists.

        For a=5, b=0.5, c=1: 1 - 0.5*(5 + 1) = 1 - 3 = -2 < 0.
        """
        sim = self._make_sim(a=5.0, b=0.5, c=1.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1


class TestFinanceTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> FinanceSimulation:
        defaults = {
            "a": 1.0, "b": 0.1, "c": 1.0,
            "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return FinanceSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Financial system should remain bounded for standard parameters."""
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
        # All variables should have significant spread in chaotic regime
        assert stats["x_std"] > 0.01, "x_std too small for chaotic regime"
        assert stats["y_std"] > 0.01, "y_std too small for chaotic regime"
        assert stats["z_std"] > 0.01, "z_std too small for chaotic regime"

    def test_different_a_gives_different_trajectory(self):
        """Changing a should change the trajectory behavior."""
        sim1 = self._make_sim(a=1.0)
        sim2 = self._make_sim(a=2.5)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)

    def test_alt_params_bounded(self):
        """Alternative chaotic parameters (a=0.9, b=0.2, c=1.2) should be bounded."""
        sim = self._make_sim(a=0.9, b=0.2, c=1.2)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"


class TestFinanceDissipation:
    """Tests for the dissipation and divergence properties."""

    def _make_sim(self, **kwargs) -> FinanceSimulation:
        defaults = {"a": 1.0, "b": 0.1, "c": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return FinanceSimulation(config)

    def test_divergence_value(self):
        """Divergence should be -a - b - c = -1 - 0.1 - 1 = -2.1."""
        sim = self._make_sim()
        assert np.isclose(sim.divergence, -2.1)

    def test_divergence_negative(self):
        """System should be dissipative (negative divergence) for positive params."""
        sim = self._make_sim()
        assert sim.divergence < 0, f"Divergence {sim.divergence} not negative"

    def test_is_chaotic_property(self):
        """is_chaotic should be True at standard params."""
        sim = self._make_sim()
        assert sim.is_chaotic is True

    def test_is_chaotic_false_for_large_b(self):
        """is_chaotic should be False when b >= 1."""
        sim = self._make_sim(b=1.5)
        assert sim.is_chaotic is False


class TestFinanceJacobian:
    """Tests for the Jacobian computation."""

    def _make_sim(self, **kwargs) -> FinanceSimulation:
        defaults = {"a": 1.0, "b": 0.1, "c": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return FinanceSimulation(config)

    def test_jacobian_at_origin(self):
        """Jacobian at (0, 0, 0) should have known structure.

        J = [[0 - 1, 0, 1],    = [[-1, 0, 1],
             [0, -0.1, 0],         [0, -0.1, 0],
             [-1, 0, -1]]          [-1, 0, -1]]
        """
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-1.0, 0.0, 1.0],
            [0.0, -0.1, 0.0],
            [-1.0, 0.0, -1.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_trace_equals_divergence_at_origin(self):
        """Trace of Jacobian at origin should match divergence formula."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        # trace at (0,0,0) = (0 - a) + (-b) + (-c) = -a - b - c
        assert np.isclose(np.trace(J), -sim.a - sim.b - sim.c)

    def test_eigenvalues_at_fixed_points(self):
        """Eigenvalue analysis should return valid results for each fixed point."""
        sim = self._make_sim()
        sim.reset()
        eig_results = sim.eigenvalues_at_fixed_points()
        assert len(eig_results) == len(sim.fixed_points)
        for r in eig_results:
            assert "stability" in r
            assert r["stability"] in ("stable", "unstable", "saddle")
            assert np.isfinite(r["max_real_part"])


class TestFinanceChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Financial system at standard parameters should have positive Lyapunov."""
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={
                "a": 1.0, "b": 0.1, "c": 1.0,
                "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
            },
        )
        sim = FinanceSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert lam > 0.0, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_lyapunov_bounded(self):
        """Lyapunov exponent should be bounded (not diverging)."""
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={"a": 1.0, "b": 0.1, "c": 1.0},
        )
        sim = FinanceSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert abs(lam) < 20.0, f"Lyapunov {lam:.3f} unreasonably large"


class TestFinanceNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_convergence(self):
        """Smaller dt should give more accurate results (convergence test)."""
        # Run with dt=0.005
        config1 = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=200,
            parameters={
                "a": 1.0, "b": 0.1, "c": 1.0,
                "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
            },
        )
        sim1 = FinanceSimulation(config1)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.001 (5x finer, same total time = 1.0)
        config2 = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters={
                "a": 1.0, "b": 0.1, "c": 1.0,
                "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
            },
        )
        sim2 = FinanceSimulation(config2)
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
            domain=_FINANCE_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters={"a": 1.0, "b": 0.1, "c": 1.0},
        )
        sim = FinanceSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"


class TestFinanceRediscovery:
    """Tests for financial system data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.finance import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 1.0
        assert data["b"] == 0.1
        assert data["c"] == 1.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.finance import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep over a should produce valid data."""
        from simulating_anything.rediscovery.finance import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.005)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_lyapunov_vs_b_data(self):
        """Lyapunov sweep over b should produce valid data."""
        from simulating_anything.rediscovery.finance import (
            generate_lyapunov_vs_b_data,
        )

        data = generate_lyapunov_vs_b_data(n_b=5, n_steps=3000, dt=0.005)
        assert len(data["b"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.finance import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.005)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
