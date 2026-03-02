"""Tests for the Liu chaotic attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.liu import LiuSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_LIU_DOMAIN = Domain.LIU


class TestLiuSimulation:
    """Tests for the Liu system simulation basics."""

    def _make_sim(self, **kwargs) -> LiuSimulation:
        defaults = {
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
            "x_0": 0.2, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LiuSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 1.0
        assert sim.b == 2.5
        assert sim.c == 5.0
        assert sim.e == 1.0
        assert sim.k == 4.0
        assert sim.m == 4.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=2.0, b=3.0, c=6.0, e=0.5, k=3.0, m=5.0)
        assert sim.a == 2.0
        assert sim.b == 3.0
        assert sim.c == 6.0
        assert sim.e == 0.5
        assert sim.k == 3.0
        assert sim.m == 5.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=-2.0, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], -2.0)
        assert np.isclose(state[2], 3.0)

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
        sim1 = self._make_sim()
        sim2 = self._make_sim()
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


class TestLiuDerivatives:
    """Tests for the Liu ODE derivative computation."""

    def _make_sim(self, **kwargs) -> LiuSimulation:
        defaults = {
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return LiuSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 2, 3] with a=1, b=2.5, c=5, e=1, k=4, m=4:
            dx = -1*1 - 1*4 = -5
            dy = 2.5*2 - 4*1*3 = 5 - 12 = -7
            dz = -5*3 + 4*1*2 = -15 + 8 = -7
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(derivs[0], -5.0)
        assert np.isclose(derivs[1], -7.0)
        assert np.isclose(derivs[2], -7.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [0.5, 1.0, 0.0] with default parameters.

            dx = -1*0.5 - 1*1 = -1.5
            dy = 2.5*1 - 4*0.5*0 = 2.5
            dz = -5*0 + 4*0.5*1 = 2.0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.5, 1.0, 0.0]))
        assert np.isclose(derivs[0], -1.5)
        assert np.isclose(derivs[1], 2.5)
        assert np.isclose(derivs[2], 2.0)


class TestLiuJacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> LiuSimulation:
        defaults = {
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return LiuSimulation(config)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should be the linear part."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, -5.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([1.0, 2.0, 3.0]))
        assert J.shape == (3, 3)

    def test_jacobian_trace_equals_divergence(self):
        """Trace of Jacobian should equal divergence (constant for Liu)."""
        sim = self._make_sim()
        sim.reset()
        # Divergence is constant, independent of state
        for state in [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, 0.5, 2.0]),
        ]:
            J = sim.jacobian(state)
            trace = np.trace(J)
            assert np.isclose(trace, sim.compute_divergence())


class TestLiuFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> LiuSimulation:
        defaults = {
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return LiuSimulation(config)

    def test_origin_only_fixed_point(self):
        """Standard parameters should give only the origin as fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1

    def test_origin_is_fixed_point(self):
        """First fixed point should be the origin."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

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


class TestLiuDivergence:
    """Tests for the divergence / dissipation computation."""

    def _make_sim(self, **kwargs) -> LiuSimulation:
        defaults = {
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return LiuSimulation(config)

    def test_divergence_default_params(self):
        """Divergence should be -3.5 for default parameters.

        div = -a + b - c = -1 + 2.5 - 5 = -3.5
        """
        sim = self._make_sim()
        div = sim.compute_divergence()
        assert np.isclose(div, -3.5)

    def test_divergence_is_negative_dissipative(self):
        """Default parameters should be dissipative (negative divergence)."""
        sim = self._make_sim()
        assert sim.compute_divergence() < 0

    def test_divergence_formula(self):
        """Test divergence formula with custom parameters."""
        sim = self._make_sim(a=2.0, b=3.0, c=6.0)
        # div = -2 + 3 - 6 = -5
        assert np.isclose(sim.compute_divergence(), -5.0)

    def test_divergence_conservative_case(self):
        """When a - b + c = 0, the system is volume-preserving."""
        # a - b + c = 0 => b = a + c
        sim = self._make_sim(a=1.0, b=6.0, c=5.0)
        assert np.isclose(sim.compute_divergence(), 0.0)


class TestLiuTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> LiuSimulation:
        defaults = {
            "a": 1.0, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
            "x_0": 0.2, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LiuSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Liu trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, (
                f"Trajectory diverged: {state}"
            )

    def test_trajectory_leaves_origin(self):
        """Starting near origin, trajectory should eventually move away."""
        sim = self._make_sim(x_0=0.01, y_0=0.01, z_0=0.01)
        sim.reset()
        max_dist = 0.0
        for _ in range(5000):
            state = sim.step()
            max_dist = max(max_dist, np.linalg.norm(state))
        assert max_dist > 0.1, "Trajectory stuck near origin"

    def test_attractor_statistics(self):
        """Trajectory should have non-trivial statistics after transient."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # At least one component should have meaningful spread
        total_std = stats["x_std"] + stats["y_std"] + stats["z_std"]
        assert total_std > 0.01, (
            f"Total std too small: {total_std:.6f}"
        )

    def test_different_a_gives_different_trajectory(self):
        """Changing a should change the trajectory behavior."""
        sim1 = self._make_sim(a=1.0)
        sim2 = self._make_sim(a=2.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestLiuChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_lyapunov_exponent_computed(self):
        """Lyapunov exponent should be a finite number."""
        config = SimulationConfig(
            domain=_LIU_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={
                "a": 1.0, "b": 2.5, "c": 5.0,
                "e": 1.0, "k": 4.0, "m": 4.0,
            },
        )
        sim = LiuSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert np.isfinite(lam), f"Lyapunov {lam} is not finite"

    def test_lyapunov_varies_with_a(self):
        """Lyapunov exponent should change as a varies."""
        lyap_a1 = _compute_lyapunov_at_a(1.0)
        lyap_a2 = _compute_lyapunov_at_a(2.0)
        assert lyap_a1 != lyap_a2, (
            "Lyapunov should differ for different a values"
        )


class TestLiuRediscovery:
    """Tests for Liu data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.liu import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 1.0
        assert data["b"] == 2.5
        assert data["c"] == 5.0
        assert data["e"] == 1.0
        assert data["k"] == 4.0
        assert data["m"] == 4.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.liu import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.liu import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(n_a=5, n_steps=2000, dt=0.005)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.liu import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.005)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.liu import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.005)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_a(a: float) -> float:
    """Helper to compute Lyapunov exponent at a given a."""
    config = SimulationConfig(
        domain=_LIU_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={
            "a": a, "b": 2.5, "c": 5.0,
            "e": 1.0, "k": 4.0, "m": 4.0,
            "x_0": 0.2, "y_0": 0.0, "z_0": 0.5,
        },
    )
    sim = LiuSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.005)
