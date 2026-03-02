"""Tests for the Vallis ENSO model simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.vallis import VallisSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_VALLIS_DOMAIN = Domain.VALLIS


class TestVallisSimulation:
    """Tests for the Vallis system simulation basics."""

    def _make_sim(self, **kwargs) -> VallisSimulation:
        defaults = {
            "B": 102.0, "C": 3.0, "p": 0.0,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return VallisSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.B == 102.0
        assert sim.C == 3.0
        assert sim.p == 0.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(B=50.0, C=5.0, p=1.0)
        assert sim.B == 50.0
        assert sim.C == 5.0
        assert sim.p == 1.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=5.0, y_0=-3.0, z_0=20.0)
        state = sim.reset()
        assert np.isclose(state[0], 5.0)
        assert np.isclose(state[1], -3.0)
        assert np.isclose(state[2], 20.0)

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
        sim1 = self._make_sim(B=102.0, C=3.0)
        sim2 = self._make_sim(B=102.0, C=3.0)
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


class TestVallisDerivatives:
    """Tests for the Vallis ODE derivative computation."""

    def _make_sim(self, **kwargs) -> VallisSimulation:
        defaults = {
            "B": 102.0, "C": 3.0, "p": 0.0,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return VallisSimulation(config)

    def test_derivatives_known_point_p0(self):
        """Test derivatives at [1, 0, 1] with B=102, C=3, p=0.

            dx = 102*0 - 3*(1 - 0) = -3
            dy = -0 + 1*1 = 1
            dz = -1 - 1*0 + 1 = 0
        """
        sim = self._make_sim(B=102.0, C=3.0, p=0.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 0.0, 1.0]))
        assert np.isclose(derivs[0], -3.0)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], 0.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 1, 3] with B=102, C=3, p=0.

            dx = 102*1 - 3*(2 - 0) = 102 - 6 = 96
            dy = -1 + 2*3 = 5
            dz = -3 - 2*1 + 1 = -4
        """
        sim = self._make_sim(B=102.0, C=3.0, p=0.0)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 1.0, 3.0]))
        assert np.isclose(derivs[0], 96.0)
        assert np.isclose(derivs[1], 5.0)
        assert np.isclose(derivs[2], -4.0)

    def test_derivatives_with_nonzero_p(self):
        """Test derivatives at [1, 0, 1] with B=102, C=3, p=1.

            dx = 102*0 - 3*(1 - 1) = 0
            dy = -0 + 1*1 = 1
            dz = -1 - 1*0 + 1 = 0
        """
        sim = self._make_sim(B=102.0, C=3.0, p=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 0.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], 0.0)


class TestVallisFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> VallisSimulation:
        defaults = {"B": 102.0, "C": 3.0, "p": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return VallisSimulation(config)

    def test_three_fixed_points(self):
        """Standard parameters (B > C) should give three fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_first_fixed_point_is_known(self):
        """First fixed point should be (0, 0, 1) for p=0."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 1.0])

    def test_fixed_point_symmetry(self):
        """Non-origin fixed points should be symmetric in x, y (opposite sign)."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], -fps[2][1])

    def test_fixed_point_x_value(self):
        """Non-origin x should satisfy x^2 = B/C - 1 for p=0.

        For B=102, C=3: x^2 = 102/3 - 1 = 33, so x ~ 5.745
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        x_expected = np.sqrt(102.0 / 3.0 - 1.0)
        assert np.isclose(abs(fps[1][0]), x_expected, rtol=1e-6)
        assert np.isclose(abs(fps[2][0]), x_expected, rtol=1e-6)

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_one_fixed_point_when_B_lt_C(self):
        """When B < C, only one fixed point should exist (the origin-like one)."""
        sim = self._make_sim(B=1.0, C=3.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 1.0])


class TestVallisJacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> VallisSimulation:
        defaults = {"B": 102.0, "C": 3.0, "p": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return VallisSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([1.0, 2.0, 3.0]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at (0,0,0) should have known structure.

        J = [[-C,  B,  0],
             [ 0, -1,  0],
             [ 0,  0, -1]]
        """
        sim = self._make_sim(B=102.0, C=3.0)
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-3.0, 102.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_trace_is_constant(self):
        """Trace of Jacobian should be -(C+2) at any state."""
        sim = self._make_sim(C=3.0)
        sim.reset()
        for state in [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-5.0, 10.0, -7.0]),
        ]:
            J = sim.jacobian(state)
            assert np.isclose(np.trace(J), -(3.0 + 2.0))


class TestVallisDivergence:
    """Tests for divergence computation."""

    def _make_sim(self, **kwargs) -> VallisSimulation:
        defaults = {"B": 102.0, "C": 3.0, "p": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return VallisSimulation(config)

    def test_divergence_default_C(self):
        """Divergence should be -(C+2) = -5 for C=3."""
        sim = self._make_sim(C=3.0)
        div = sim.compute_divergence(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(div, -5.0)

    def test_divergence_custom_C(self):
        """Divergence should be -(C+2) for any C."""
        for C_val in [1.0, 5.0, 10.0]:
            sim = self._make_sim(C=C_val)
            div = sim.compute_divergence(np.array([0.0, 0.0, 0.0]))
            assert np.isclose(div, -(C_val + 2.0))

    def test_divergence_state_independent(self):
        """Divergence should be the same at any state."""
        sim = self._make_sim(C=3.0)
        d1 = sim.compute_divergence(np.array([0.0, 0.0, 0.0]))
        d2 = sim.compute_divergence(np.array([10.0, -5.0, 3.0]))
        d3 = sim.compute_divergence(np.array([-100.0, 50.0, 25.0]))
        assert np.isclose(d1, d2)
        assert np.isclose(d2, d3)

    def test_divergence_matches_jacobian_trace(self):
        """Divergence should equal the trace of the Jacobian."""
        sim = self._make_sim(C=3.0)
        state = np.array([2.0, -1.0, 4.0])
        div = sim.compute_divergence(state)
        J = sim.jacobian(state)
        assert np.isclose(div, np.trace(J))


class TestVallisTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> VallisSimulation:
        defaults = {
            "B": 102.0, "C": 3.0, "p": 0.0,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return VallisSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Vallis trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 500, (
                f"Trajectory diverged: {state}"
            )

    def test_attractor_statistics(self):
        """Trajectory statistics should reflect bounded attractor."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # x should have nonzero std on the attractor
        assert stats["x_std"] > 0.1, "x_std too small for attractor"
        assert stats["y_std"] > 0.001, "y_std too small for attractor"

    def test_different_B_gives_different_trajectory(self):
        """Changing B should change the trajectory behavior."""
        sim1 = self._make_sim(B=102.0)
        sim2 = self._make_sim(B=20.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestVallisChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Vallis at standard parameters should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={"B": 102.0, "C": 3.0, "p": 0.0},
        )
        sim = VallisSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert lam > 0.1, (
            f"Lyapunov {lam:.3f} too small for chaotic regime"
        )
        assert lam < 50.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_B(self):
        """Lyapunov exponent should change as B varies."""
        lyap_B102 = _compute_lyapunov_at_B(102.0)
        lyap_B10 = _compute_lyapunov_at_B(10.0)
        assert lyap_B102 != lyap_B10, (
            "Lyapunov should differ for different B"
        )

    def test_bifurcation_sweep_returns_data(self):
        """Bifurcation sweep should return valid data."""
        config = SimulationConfig(
            domain=_VALLIS_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters={"B": 102.0, "C": 3.0, "p": 0.0},
        )
        sim = VallisSimulation(config)
        sim.reset()
        B_values = np.linspace(20.0, 120.0, 5)
        result = sim.bifurcation_sweep(
            B_values, n_transient=1000, n_measure=3000
        )
        assert len(result["B"]) == 5
        assert len(result["lyapunov_exponent"]) == 5
        assert len(result["attractor_type"]) == 5


class TestVallisRediscovery:
    """Tests for Vallis data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.vallis import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["B"] == 102.0
        assert data["C"] == 3.0
        assert data["p"] == 0.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.vallis import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.vallis import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=200, dt=0.005)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_B_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.vallis import (
            generate_lyapunov_vs_B_data,
        )

        data = generate_lyapunov_vs_B_data(
            n_B=5, n_steps=3000, dt=0.005
        )
        assert len(data["B"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_B(B: float) -> float:
    """Helper to compute Lyapunov exponent at a given B."""
    config = SimulationConfig(
        domain=_VALLIS_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={
            "B": B, "C": 3.0, "p": 0.0,
            "x_0": 0.1, "y_0": 0.2, "z_0": 0.3,
        },
    )
    sim = VallisSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.005)
