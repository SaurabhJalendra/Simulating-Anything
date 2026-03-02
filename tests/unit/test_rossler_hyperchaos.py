"""Tests for the Rossler hyperchaotic system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.rossler_hyperchaos import (
    RosslerHyperchaosSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestRosslerHyperchaosSimulation:
    """Tests for the Rossler hyperchaotic system simulation."""

    def _make_sim(self, **kwargs) -> RosslerHyperchaosSimulation:
        defaults = {
            "a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05,
            "x_0": -10.0, "y_0": -6.0, "z_0": 0.0, "w_0": 10.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RosslerHyperchaosSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.a == 0.25
        assert sim.b == 3.0
        assert sim.c == 0.5
        assert sim.d == 0.05

    def test_initial_state_shape(self):
        """State vector has shape (4,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=-2.0, z_0=0.5, w_0=3.0)
        state = sim.reset()
        np.testing.assert_allclose(state, [1.0, -2.0, 0.5, 3.0])

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
        assert obs.shape == (4,)

    def test_step_count_increments(self):
        """Step counter increments with each step."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


class TestRosslerHyperchaosDerivatives:
    """Tests for the derivative computations."""

    def _make_sim(self, **kwargs) -> RosslerHyperchaosSimulation:
        defaults = {"a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return RosslerHyperchaosSimulation(config)

    def test_derivatives_at_origin(self):
        """Derivatives at origin: dx=0, dy=0, dz=b, dw=0."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0, 0.0]))
        # dx = -(0+0) = 0
        # dy = 0 + 0.25*0 + 0 = 0
        # dz = 3.0 + 0*0 = 3.0
        # dw = -0.5*0 + 0.05*0 = 0
        np.testing.assert_allclose(derivs, [0.0, 0.0, 3.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point [1, 1, 1, 1]."""
        sim = self._make_sim(a=0.25, b=3.0, c=0.5, d=0.05)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0, 1.0]))
        # dx = -(1 + 1) = -2
        # dy = 1 + 0.25*1 + 1 = 2.25
        # dz = 3.0 + 1*1 = 4.0
        # dw = -0.5*1 + 0.05*1 = -0.45
        np.testing.assert_allclose(derivs, [-2.0, 2.25, 4.0, -0.45])

    def test_derivatives_x_equation(self):
        """x equation: dx/dt = -(y + z) does not depend on x or w."""
        sim = self._make_sim()
        sim.reset()
        d1 = sim._derivatives(np.array([0.0, 2.0, 3.0, 0.0]))
        d2 = sim._derivatives(np.array([100.0, 2.0, 3.0, 50.0]))
        assert np.isclose(d1[0], d2[0])
        assert np.isclose(d1[0], -5.0)

    def test_derivatives_w_equation(self):
        """w equation: dw/dt = -c*z + d*w, independent of x, y."""
        sim = self._make_sim(c=0.5, d=0.05)
        sim.reset()
        derivs = sim._derivatives(np.array([999.0, 999.0, 4.0, 10.0]))
        # dw = -0.5*4 + 0.05*10 = -2.0 + 0.5 = -1.5
        assert np.isclose(derivs[3], -1.5)


class TestRosslerHyperchaosJacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> RosslerHyperchaosSimulation:
        defaults = {"a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return RosslerHyperchaosSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be 4x4."""
        sim = self._make_sim()
        sim.reset()
        J = sim._jacobian(np.array([1.0, 2.0, 3.0, 4.0]))
        assert J.shape == (4, 4)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should have known structure."""
        sim = self._make_sim(a=0.25, c=0.5, d=0.05)
        sim.reset()
        J = sim._jacobian(np.array([0.0, 0.0, 0.0, 0.0]))
        expected = np.array([
            [0.0, -1.0, -1.0, 0.0],
            [1.0, 0.25, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.05],
        ])
        np.testing.assert_allclose(J, expected)

    def test_jacobian_state_dependent_entries(self):
        """Jacobian entries J[2,0]=z and J[2,2]=x depend on state."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([5.0, 2.0, 7.0, 1.0])
        J = sim._jacobian(state)
        assert np.isclose(J[2, 0], 7.0)  # z
        assert np.isclose(J[2, 2], 5.0)  # x

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
            J_col_num = (sim._derivatives(state_plus) - sim._derivatives(state_minus)) / (2 * eps)
            np.testing.assert_allclose(J[:, j], J_col_num, atol=1e-5)


class TestRosslerHyperchaosTrajectory:
    """Tests for trajectory behavior."""

    def _make_sim(self, **kwargs) -> RosslerHyperchaosSimulation:
        defaults = {
            "a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05,
            "x_0": -10.0, "y_0": -6.0, "z_0": 0.0, "w_0": 10.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return RosslerHyperchaosSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 500, f"Trajectory diverged: {state}"

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
        sim1 = self._make_sim(x_0=-10.0)
        sim1.reset()
        for _ in range(20000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = self._make_sim(x_0=-10.0 + 1e-6)
        sim2.reset()
        for _ in range(20000):
            sim2.step()
        state2 = sim2.observe().copy()

        # After 20000 steps with dt=0.005 (t=100), states should diverge
        assert np.linalg.norm(state1 - state2) > 0.01, (
            "No divergence detected -- expected chaotic sensitivity"
        )


class TestRosslerHyperchaosLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def _make_sim(self, **kwargs) -> RosslerHyperchaosSimulation:
        defaults = {
            "a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05,
            "x_0": -10.0, "y_0": -6.0, "z_0": 0.0, "w_0": 10.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return RosslerHyperchaosSimulation(config)

    def test_largest_lyapunov_positive(self):
        """At classic hyperchaotic parameters, largest LE should be positive."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.005)
        assert lam > 0.01, f"Largest Lyapunov {lam:.4f} not positive"

    def test_lyapunov_spectrum_returns_four(self):
        """Spectrum should contain exactly 4 exponents for 4D system."""
        sim = self._make_sim()
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(n_steps=20000, dt=0.005)
        assert len(spectrum) == 4

    def test_lyapunov_spectrum_sorted_descending(self):
        """Lyapunov spectrum should be sorted from largest to smallest."""
        sim = self._make_sim()
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(n_steps=20000, dt=0.005)
        for i in range(len(spectrum) - 1):
            assert spectrum[i] >= spectrum[i + 1], (
                f"Spectrum not sorted: {spectrum}"
            )

    def test_is_hyperchaotic_classic_params(self):
        """Classic parameters should yield hyperchaos (2 positive LEs)."""
        sim = self._make_sim()
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(
            n_steps=40000, dt=0.005, n_transient=10000
        )
        n_positive = int(np.sum(spectrum > 0.001))
        # Hyperchaos requires at least 2 positive exponents
        assert n_positive >= 2, (
            f"Expected 2 positive exponents, got {n_positive}. "
            f"Spectrum: {spectrum}"
        )

    def test_d_zero_not_hyperchaotic(self):
        """With d=0, the w variable decouples and system is 3D -- not hyperchaotic."""
        sim = self._make_sim(d=0.0, a=0.2, b=0.2, c=5.7)
        sim.reset()
        spectrum = sim.estimate_lyapunov_spectrum(
            n_steps=30000, dt=0.005, n_transient=5000
        )
        n_positive = int(np.sum(spectrum > 0.01))
        # With d=0, should have at most 1 positive LE
        assert n_positive <= 1, (
            f"d=0 should not be hyperchaotic, got {n_positive} positive. "
            f"Spectrum: {spectrum}"
        )


class TestRosslerHyperchaosKaplanYorke:
    """Tests for Kaplan-Yorke dimension."""

    def _make_sim(self, **kwargs) -> RosslerHyperchaosSimulation:
        defaults = {
            "a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return RosslerHyperchaosSimulation(config)

    def test_kaplan_yorke_dimension_positive(self):
        """D_KY should be positive for a bounded chaotic attractor."""
        sim = self._make_sim()
        sim.reset()
        spectrum = np.array([0.1, 0.05, -0.01, -0.3])
        d_ky = sim.kaplan_yorke_dimension(spectrum=spectrum)
        assert d_ky > 0, f"D_KY should be positive, got {d_ky}"

    def test_kaplan_yorke_known_spectrum(self):
        """Verify D_KY for a hand-calculated spectrum."""
        sim = self._make_sim()
        sim.reset()
        # Spectrum: [0.1, 0.05, -0.01, -0.3]
        # Cumsum: [0.1, 0.15, 0.14, -0.16]
        # j=3 (first 3 sums are non-negative)
        # D_KY = 3 + 0.14 / 0.3 = 3.4667
        spectrum = np.array([0.1, 0.05, -0.01, -0.3])
        d_ky = sim.kaplan_yorke_dimension(spectrum=spectrum)
        np.testing.assert_allclose(d_ky, 3.0 + 0.14 / 0.3, rtol=1e-10)

    def test_kaplan_yorke_all_negative(self):
        """If all exponents are negative, D_KY = 0 (fixed point attractor)."""
        sim = self._make_sim()
        sim.reset()
        spectrum = np.array([-0.1, -0.2, -0.3, -0.5])
        d_ky = sim.kaplan_yorke_dimension(spectrum=spectrum)
        assert d_ky == 0.0


class TestRosslerHyperchaosRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.rossler_hyperchaos import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 4)
        assert data["a"] == 0.25
        assert data["b"] == 3.0
        assert data["c"] == 0.5
        assert data["d"] == 0.05

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.rossler_hyperchaos import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_data_shape(self):
        from simulating_anything.rediscovery.rossler_hyperchaos import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_d_values=3, dt=0.005)
        assert len(data["d"]) == 3
        assert len(data["max_lyapunov"]) == 3
        assert len(data["second_lyapunov"]) == 3
        assert len(data["n_positive"]) == 3

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.rossler_hyperchaos import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=200, dt=0.005)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 4
        assert states.dtype == np.float64
        assert "dt" in data
