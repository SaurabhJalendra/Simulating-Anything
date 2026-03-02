"""Tests for the Qi 4D chaotic system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.qi import QiSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_QI_DOMAIN = Domain.QI


class TestQiCreation:
    """Tests for simulation creation and parameter handling."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 10.0
        assert np.isclose(sim.b, 8.0 / 3.0)
        assert sim.c == 28.0
        assert sim.d == 1.0

    def test_creation_custom_parameters(self):
        """Simulation accepts custom parameter values."""
        sim = self._make_sim(a=5.0, b=2.0, c=20.0, d=3.0)
        assert sim.a == 5.0
        assert sim.b == 2.0
        assert sim.c == 20.0
        assert sim.d == 3.0

    def test_config_stored(self):
        """Config is accessible on the simulation."""
        sim = self._make_sim()
        assert sim.config.dt == 0.001
        assert sim.config.n_steps == 10000


class TestQiState:
    """Tests for state initialization and shape."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_state_shape(self):
        """State vector has shape (4,) for the 4D system."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)

    def test_state_dtype(self):
        """State should be float64 for numerical precision."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.dtype == np.float64

    def test_initial_state_default_values(self):
        """Default initial state is [1, 0, 0, 0]."""
        sim = self._make_sim()
        state = sim.reset()
        np.testing.assert_allclose(state, [1.0, 0.0, 0.0, 0.0])

    def test_initial_state_custom_values(self):
        """Custom initial conditions are respected."""
        sim = self._make_sim(x_0=2.0, y_0=-1.5, z_0=0.3, w_0=0.5)
        state = sim.reset()
        np.testing.assert_allclose(state, [2.0, -1.5, 0.3, 0.5])

    def test_observe_returns_current_state(self):
        """observe() returns the same state with correct shape."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances_state(self):
        """A single step changes the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_step_count_increments(self):
        """Step counter increments with each step."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


class TestQiDerivatives:
    """Tests for the derivative computations."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a known point [1, 1, 1, 1].

        At [1, 1, 1, 1] with a=10, b=8/3, c=28, d=1:
            dx = 10*(1-1) + 1*1 = 1
            dy = 28*1 - 1 - 1*1 = 26
            dz = 1*1 - (8/3)*1 = 1 - 8/3 = -5/3
            dw = -1*1 + 1*1 = 0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 26.0)
        assert np.isclose(derivs[2], 1.0 - 8.0 / 3.0)
        assert np.isclose(derivs[3], 0.0)

    def test_derivatives_x_equation(self):
        """x equation: dx/dt = a*(y-x) + y*z, independent of w."""
        sim = self._make_sim()
        sim.reset()
        d1 = sim._derivatives(np.array([2.0, 3.0, 4.0, 0.0]))
        d2 = sim._derivatives(np.array([2.0, 3.0, 4.0, 100.0]))
        assert np.isclose(d1[0], d2[0])
        # dx = 10*(3-2) + 3*4 = 10 + 12 = 22
        assert np.isclose(d1[0], 22.0)

    def test_derivatives_w_equation(self):
        """w equation: dw/dt = -d*w + x*z, independent of y."""
        sim = self._make_sim(d=2.0)
        sim.reset()
        derivs = sim._derivatives(np.array([3.0, 999.0, 4.0, 5.0]))
        # dw = -2*5 + 3*4 = -10 + 12 = 2
        assert np.isclose(derivs[3], 2.0)

    def test_derivatives_z_equation(self):
        """z equation: dz/dt = x*y - b*z, independent of w."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([3.0, 4.0, 6.0, 999.0]))
        # dz = 3*4 - (8/3)*6 = 12 - 16 = -4
        assert np.isclose(derivs[2], -4.0)


class TestQiJacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be 4x4."""
        sim = self._make_sim()
        sim.reset()
        J = sim._jacobian(np.array([1.0, 2.0, 3.0, 4.0]))
        assert J.shape == (4, 4)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should have known structure."""
        sim = self._make_sim(a=10.0, b=8.0 / 3.0, c=28.0, d=1.0)
        sim.reset()
        J = sim._jacobian(np.array([0.0, 0.0, 0.0, 0.0]))
        expected = np.array([
            [-10.0, 10.0, 0.0, 0.0],
            [28.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -8.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ])
        np.testing.assert_allclose(J, expected)

    def test_jacobian_state_dependent_entries(self):
        """Jacobian entries that depend on state variables."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([5.0, 2.0, 7.0, 1.0])
        J = sim._jacobian(state)
        # J[0,1] = a + z = 10 + 7 = 17
        assert np.isclose(J[0, 1], 17.0)
        # J[0,2] = y = 2
        assert np.isclose(J[0, 2], 2.0)
        # J[1,0] = c - z = 28 - 7 = 21
        assert np.isclose(J[1, 0], 21.0)
        # J[1,2] = -x = -5
        assert np.isclose(J[1, 2], -5.0)
        # J[2,0] = y = 2
        assert np.isclose(J[2, 0], 2.0)
        # J[2,1] = x = 5
        assert np.isclose(J[2, 1], 5.0)
        # J[3,0] = z = 7
        assert np.isclose(J[3, 0], 7.0)
        # J[3,2] = x = 5
        assert np.isclose(J[3, 2], 5.0)

    def test_jacobian_numerical_consistency(self):
        """Jacobian matches numerical finite differences of derivatives."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([2.0, -3.0, 1.5, 0.5])
        J = sim._jacobian(state)
        eps = 1e-7

        for j in range(4):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[j] += eps
            state_minus[j] -= eps
            J_col_num = (
                sim._derivatives(state_plus) - sim._derivatives(state_minus)
            ) / (2 * eps)
            np.testing.assert_allclose(J[:, j], J_col_num, atol=1e-5)


class TestQiTrajectory:
    """Tests for trajectory behavior and boundedness."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 1000, f"Trajectory diverged: {state}"

    def test_trajectory_shape_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert np.all(np.isfinite(traj.states))

    def test_deterministic(self):
        """Two runs with same parameters should produce identical trajectories."""
        sim1 = self._make_sim()
        traj1 = sim1.run(n_steps=200)

        sim2 = self._make_sim()
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_sensitivity_to_initial_conditions(self):
        """Chaotic system should show sensitivity to initial conditions."""
        sim1 = self._make_sim(x_0=1.0)
        sim1.reset()
        for _ in range(20000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = self._make_sim(x_0=1.0 + 1e-6)
        sim2.reset()
        for _ in range(20000):
            sim2.step()
        state2 = sim2.observe().copy()

        # After 20000 steps with dt=0.001 (t=20), states should diverge
        assert np.linalg.norm(state1 - state2) > 0.01, (
            "No divergence detected -- expected chaotic sensitivity"
        )

    def test_w_component_active(self):
        """The w component should be nontrivial (not stuck at zero)."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(3000):
            sim.step()
        w_vals = []
        for _ in range(5000):
            state = sim.step()
            w_vals.append(abs(state[3]))
        max_w = np.max(w_vals)
        assert max_w > 0.1, f"max |w| = {max_w:.4f} too small; w should be active"


class TestQiFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_origin_always_fixed_point(self):
        """The origin [0,0,0,0] should always be a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(
            fps[0], [0.0, 0.0, 0.0, 0.0]
        )

    def test_derivatives_at_origin_zero(self):
        """Derivatives at origin should be exactly zero."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(
            derivs, [0.0, 0.0, 0.0, 0.0], decimal=15
        )

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be (approximately) zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0, 0.0], decimal=8,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestQiLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_largest_lyapunov_positive(self):
        """At classic parameters, largest LE should be positive (chaos)."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.001)
        assert lam > 0.01, f"Largest Lyapunov {lam:.4f} not positive"

    def test_lyapunov_spectrum_returns_four(self):
        """Spectrum should contain exactly 4 exponents for 4D system."""
        sim = self._make_sim()
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(n_steps=20000, dt=0.001)
        assert len(spectrum) == 4

    def test_lyapunov_spectrum_sorted_descending(self):
        """Lyapunov spectrum should be sorted from largest to smallest."""
        sim = self._make_sim()
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(n_steps=20000, dt=0.001)
        for i in range(len(spectrum) - 1):
            assert spectrum[i] >= spectrum[i + 1], (
                f"Spectrum not sorted: {spectrum}"
            )

    def test_dissipation_sum_negative(self):
        """Sum of Lyapunov exponents should be negative (dissipative system).

        The divergence of the Qi system is -(a + 1 + b + d), which is negative
        for standard parameters: -(10 + 1 + 8/3 + 1) = -14.667.
        """
        sim = self._make_sim()
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(
            n_steps=40000, dt=0.001, n_transient=10000
        )
        assert np.sum(spectrum) < 0, (
            f"Sum of Lyapunov exponents should be negative, got {np.sum(spectrum):.4f}"
        )


class TestQiKaplanYorke:
    """Tests for Kaplan-Yorke dimension computation."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_kaplan_yorke_known_spectrum(self):
        """Verify D_KY for a hand-calculated spectrum.

        Spectrum: [0.1, 0.05, -0.01, -0.3]
        Cumsum:   [0.1, 0.15, 0.14, -0.16]
        j=3 (first 3 sums are non-negative)
        D_KY = 3 + 0.14 / 0.3 = 3.4667
        """
        sim = self._make_sim()
        sim.reset()
        spectrum = np.array([0.1, 0.05, -0.01, -0.3])
        d_ky = sim.kaplan_yorke_dimension(spectrum=spectrum)
        np.testing.assert_allclose(d_ky, 3.0 + 0.14 / 0.3, rtol=1e-10)

    def test_kaplan_yorke_all_negative(self):
        """If all exponents are negative, D_KY = 0."""
        sim = self._make_sim()
        sim.reset()
        spectrum = np.array([-0.1, -0.2, -0.3, -0.5])
        d_ky = sim.kaplan_yorke_dimension(spectrum=spectrum)
        assert d_ky == 0.0

    def test_kaplan_yorke_positive(self):
        """D_KY should be positive for a chaotic attractor spectrum."""
        sim = self._make_sim()
        sim.reset()
        spectrum = np.array([1.0, -0.5, -2.0, -5.0])
        d_ky = sim.kaplan_yorke_dimension(spectrum=spectrum)
        assert d_ky > 0, f"D_KY should be positive, got {d_ky}"
        # D_KY = 1 + 0.5/0.5 = 2... wait
        # Cumsum: [1.0, 0.5, -1.5, ...]
        # j=2, D_KY = 2 + 0.5/2.0 = 2.25
        np.testing.assert_allclose(d_ky, 2.25, rtol=1e-10)


class TestQiStatistics:
    """Tests for trajectory statistics computation."""

    def _make_sim(self, **kwargs) -> QiSimulation:
        defaults = {
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=0.001,
            n_steps=20000,
            parameters=defaults,
        )
        return QiSimulation(config)

    def test_statistics_keys(self):
        """compute_trajectory_statistics returns expected keys."""
        sim = self._make_sim()
        sim.reset()
        stats = sim.compute_trajectory_statistics(n_steps=2000, n_transient=500)
        expected_keys = {
            "x_mean", "y_mean", "z_mean", "w_mean",
            "x_std", "y_std", "z_std", "w_std",
            "x_range", "y_range", "z_range", "w_range",
        }
        assert set(stats.keys()) == expected_keys

    def test_statistics_values_finite(self):
        """All trajectory statistics should be finite numbers."""
        sim = self._make_sim()
        sim.reset()
        stats = sim.compute_trajectory_statistics(n_steps=2000, n_transient=500)
        for key, val in stats.items():
            assert np.isfinite(val), f"Statistic {key} is not finite: {val}"

    def test_statistics_positive_ranges(self):
        """State ranges should be positive (attractor is not a fixed point)."""
        sim = self._make_sim()
        sim.reset()
        stats = sim.compute_trajectory_statistics(n_steps=5000, n_transient=2000)
        assert stats["x_range"] > 0.1, "x range too small"
        assert stats["y_range"] > 0.1, "y range too small"
        assert stats["z_range"] > 0.1, "z range too small"


class TestQiRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.qi import generate_trajectory_data

        data = generate_trajectory_data(n_steps=100, dt=0.001)
        assert data["states"].shape == (101, 4)
        assert data["a"] == 10.0
        assert np.isclose(data["b"], 8.0 / 3.0)
        assert data["c"] == 28.0
        assert data["d"] == 1.0

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.qi import generate_trajectory_data

        data = generate_trajectory_data(n_steps=1000, dt=0.001)
        assert np.all(np.isfinite(data["states"]))

    def test_d_sweep_data_shape(self):
        from simulating_anything.rediscovery.qi import generate_d_sweep_data

        data = generate_d_sweep_data(n_d_values=3, dt=0.001)
        assert len(data["d"]) == 3
        assert len(data["lyapunov_exponent"]) == 3
        assert len(data["max_amplitude"]) == 3
        assert len(data["w_amplitude"]) == 3

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.qi import generate_trajectory_data

        data = generate_trajectory_data(n_steps=200, dt=0.001)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 4
        assert states.dtype == np.float64
        assert "dt" in data
