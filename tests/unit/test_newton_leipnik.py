"""Tests for the Newton-Leipnik attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.newton_leipnik import NewtonLeipnikSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_NL_DOMAIN = Domain.NEWTON_LEIPNIK


class TestNewtonLeipnikSimulation:
    """Tests for Newton-Leipnik system simulation basics."""

    def _make_sim(self, **kwargs) -> NewtonLeipnikSimulation:
        defaults = {
            "a": 0.4, "b": 0.175,
            "x_0": 0.349, "y_0": 0.0, "z_0": -0.16,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters=defaults,
        )
        return NewtonLeipnikSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 0.4
        assert sim.b == 0.175

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=0.5, b=0.2)
        assert sim.a == 0.5
        assert sim.b == 0.2

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=0.5, y_0=-0.3, z_0=0.1)
        state = sim.reset()
        assert np.isclose(state[0], 0.5)
        assert np.isclose(state[1], -0.3)
        assert np.isclose(state[2], 0.1)

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


class TestNewtonLeipnikDerivatives:
    """Tests for the Newton-Leipnik ODE derivative computation."""

    def _make_sim(self, **kwargs) -> NewtonLeipnikSimulation:
        defaults = {
            "a": 0.4, "b": 0.175,
            "x_0": 0.349, "y_0": 0.0, "z_0": -0.16,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters=defaults,
        )
        return NewtonLeipnikSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=0.4, b=0.175:
            dx = -0.4*1 + 1 + 10*1*1 = -0.4 + 1 + 10 = 10.6
            dy = -1 - 0.4*1 + 5*1*1 = -1 - 0.4 + 5 = 3.6
            dz = 0.175*1 - 5*1*1 = 0.175 - 5 = -4.825
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 10.6)
        assert np.isclose(derivs[1], 3.6)
        assert np.isclose(derivs[2], -4.825)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 0.5, -0.5] with a=0.4, b=0.175.

            dx = -0.4*2 + 0.5 + 10*0.5*(-0.5) = -0.8 + 0.5 - 2.5 = -2.8
            dy = -2 - 0.4*0.5 + 5*2*(-0.5) = -2 - 0.2 - 5 = -7.2
            dz = 0.175*(-0.5) - 5*2*0.5 = -0.0875 - 5 = -5.0875
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 0.5, -0.5]))
        assert np.isclose(derivs[0], -2.8)
        assert np.isclose(derivs[1], -7.2)
        assert np.isclose(derivs[2], -5.0875)


class TestNewtonLeipnikRK4Convergence:
    """Tests for RK4 integration accuracy."""

    def test_rk4_convergence_order(self):
        """RK4 should show 4th-order convergence as dt decreases."""
        errors = []
        dts = [0.01, 0.005, 0.0025]
        reference_dt = 0.001
        n_time = 0.5  # Integrate for 0.5 time units

        # Get reference solution with very small dt
        n_ref = int(n_time / reference_dt)
        config_ref = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=reference_dt,
            n_steps=n_ref,
            parameters={"a": 0.4, "b": 0.175, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
        )
        sim_ref = NewtonLeipnikSimulation(config_ref)
        sim_ref.reset()
        for _ in range(n_ref):
            sim_ref.step()
        ref_state = sim_ref.observe().copy()

        for dt in dts:
            n = int(n_time / dt)
            config = SimulationConfig(
                domain=_NL_DOMAIN,
                dt=dt,
                n_steps=n,
                parameters={
                    "a": 0.4, "b": 0.175,
                    "x_0": 0.349, "y_0": 0.0, "z_0": -0.16,
                },
            )
            sim = NewtonLeipnikSimulation(config)
            sim.reset()
            for _ in range(n):
                sim.step()
            errors.append(np.linalg.norm(sim.observe() - ref_state))

        # Check roughly 4th-order: halving dt should reduce error by ~16x
        # Allow some margin because reference solution also has error
        ratio = errors[0] / errors[1] if errors[1] > 0 else 0
        assert ratio > 8.0, f"RK4 convergence ratio {ratio:.1f}, expected >8"


class TestNewtonLeipnikFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> NewtonLeipnikSimulation:
        defaults = {"a": 0.4, "b": 0.175}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters=defaults,
        )
        return NewtonLeipnikSimulation(config)

    def test_origin_is_fixed_point(self):
        """Origin should always be a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        origin_found = any(np.linalg.norm(fp) < 1e-10 for fp in fps)
        assert origin_found, "Origin not found among fixed points"

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

    def test_at_least_one_fixed_point(self):
        """There should be at least one fixed point (the origin)."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) >= 1


class TestNewtonLeipnikJacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> NewtonLeipnikSimulation:
        defaults = {"a": 0.4, "b": 0.175}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters=defaults,
        )
        return NewtonLeipnikSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be a 3x3 matrix."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should have known structure.

        J(0,0,0) = [[-a,  1,    0  ],
                     [-1, -0.4,  0  ],
                     [0,   0,    b  ]]

        With a=0.4, b=0.175:
        J = [[-0.4, 1.0, 0.0],
             [-1.0, -0.4, 0.0],
             [0.0, 0.0, 0.175]]
        """
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-0.4, 1.0, 0.0],
            [-1.0, -0.4, 0.0],
            [0.0, 0.0, 0.175],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_at_known_point(self):
        """Jacobian at (1, 1, 1) with a=0.4, b=0.175.

        J = [[-0.4,  1+10*1,  10*1 ],   = [[-0.4, 11,  10  ],
             [-1+5*1, -0.4,    5*1 ],       [4,   -0.4,  5  ],
             [-5*1,   -5*1,    0.175]]       [-5,  -5,    0.175]]
        """
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([1.0, 1.0, 1.0]))
        expected = np.array([
            [-0.4, 11.0, 10.0],
            [4.0, -0.4, 5.0],
            [-5.0, -5.0, 0.175],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_trace_equals_divergence(self):
        """Trace of Jacobian should equal the divergence at any state."""
        sim = self._make_sim()
        sim.reset()
        for state in [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-0.5, 0.3, -0.1]),
        ]:
            J = sim.jacobian(state)
            assert np.isclose(np.trace(J), sim.divergence)


class TestNewtonLeipnikDissipation:
    """Tests for dissipation and divergence properties."""

    def _make_sim(self, **kwargs) -> NewtonLeipnikSimulation:
        defaults = {"a": 0.4, "b": 0.175}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters=defaults,
        )
        return NewtonLeipnikSimulation(config)

    def test_divergence_value(self):
        """Divergence should be -(a + 0.4 - b) = -(0.4 + 0.4 - 0.175) = -0.625."""
        sim = self._make_sim()
        assert np.isclose(sim.divergence, -0.625)

    def test_divergence_negative(self):
        """System should be dissipative (negative divergence) for standard params."""
        sim = self._make_sim()
        assert sim.divergence < 0, f"Divergence {sim.divergence} not negative"

    def test_compute_divergence_matches_property(self):
        """compute_divergence() should match divergence property."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([0.5, 0.3, -0.2])
        assert np.isclose(sim.compute_divergence(state), sim.divergence)

    def test_divergence_state_independent(self):
        """Divergence should be the same at any state."""
        sim = self._make_sim()
        sim.reset()
        div1 = sim.compute_divergence(np.array([0.0, 0.0, 0.0]))
        div2 = sim.compute_divergence(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(div1, div2)


class TestNewtonLeipnikTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> NewtonLeipnikSimulation:
        defaults = {
            "a": 0.4, "b": 0.175,
            "x_0": 0.349, "y_0": 0.0, "z_0": -0.16,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters=defaults,
        )
        return NewtonLeipnikSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(500):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 10, f"Trajectory diverged: {state}"

    def test_trajectory_stays_finite_long(self):
        """Long trajectory should remain finite."""
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=5000,
            parameters={"a": 0.4, "b": 0.175, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
        )
        sim = NewtonLeipnikSimulation(config)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"

    def test_different_a_gives_different_trajectory(self):
        """Changing a should change the trajectory behavior."""
        sim1 = self._make_sim(a=0.4)
        sim2 = self._make_sim(a=0.8)
        sim1.reset()
        sim2.reset()
        for _ in range(500):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.01)


class TestNewtonLeipnikChaos:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Standard parameters should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=20000,
            parameters={"a": 0.4, "b": 0.175, "x_0": 0.349, "y_0": 0.0, "z_0": -0.16},
        )
        sim = NewtonLeipnikSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert lam > 0.0, f"Lyapunov {lam:.3f} not positive for chaotic regime"

    def test_is_chaotic_property(self):
        """is_chaotic should be True at standard parameters."""
        config = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters={"a": 0.4, "b": 0.175},
        )
        sim = NewtonLeipnikSimulation(config)
        assert sim.is_chaotic is True

    def test_lyapunov_varies_with_a(self):
        """Lyapunov exponent should change as a varies."""
        lyap1 = _compute_lyapunov_at_a(0.4)
        lyap2 = _compute_lyapunov_at_a(0.8)
        assert lyap1 != lyap2, "Lyapunov should differ for different a"


class TestNewtonLeipnikMultistability:
    """Tests for multistability -- two coexisting attractors."""

    def test_different_ic_different_trajectory(self):
        """Different initial z should lead to different trajectories."""
        config1 = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters={
                "a": 0.4, "b": 0.175,
                "x_0": 0.349, "y_0": 0.0, "z_0": -0.16,
            },
        )
        config2 = SimulationConfig(
            domain=_NL_DOMAIN,
            dt=0.005,
            n_steps=500,
            parameters={
                "a": 0.4, "b": 0.175,
                "x_0": 0.349, "y_0": 0.0, "z_0": 0.16,
            },
        )
        sim1 = NewtonLeipnikSimulation(config1)
        sim2 = NewtonLeipnikSimulation(config2)
        sim1.reset()
        sim2.reset()
        for _ in range(500):
            s1 = sim1.step()
            s2 = sim2.step()
        # The two attractors should give different z values
        assert not np.allclose(s1, s2, atol=0.01), (
            "Trajectories from different IC should diverge"
        )

    def test_both_attractors_bounded(self):
        """Both attractors should produce bounded trajectories."""
        for z0 in [-0.16, 0.16]:
            config = SimulationConfig(
                domain=_NL_DOMAIN,
                dt=0.005,
                n_steps=500,
                parameters={
                    "a": 0.4, "b": 0.175,
                    "x_0": 0.349, "y_0": 0.0, "z_0": z0,
                },
            )
            sim = NewtonLeipnikSimulation(config)
            sim.reset()
            for _ in range(500):
                state = sim.step()
                assert np.all(np.isfinite(state)), (
                    f"State NaN/Inf for z_0={z0}: {state}"
                )


class TestNewtonLeipnikRediscovery:
    """Tests for Newton-Leipnik data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.newton_leipnik import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.4
        assert data["b"] == 0.175

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.newton_leipnik import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.newton_leipnik import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(n_a=5, n_steps=2000, dt=0.005)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.newton_leipnik import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.005)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.newton_leipnik import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.005)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_multistability_data(self):
        """Multistability data should have two different trajectories."""
        from simulating_anything.rediscovery.newton_leipnik import (
            generate_multistability_data,
        )

        data = generate_multistability_data(dt=0.005, n_steps=500)
        assert data["states_attractor1"].shape == (501, 3)
        assert data["states_attractor2"].shape == (501, 3)
        assert np.all(np.isfinite(data["states_attractor1"]))
        assert np.all(np.isfinite(data["states_attractor2"]))


def _compute_lyapunov_at_a(a: float) -> float:
    """Helper to compute Lyapunov exponent at a given a."""
    config = SimulationConfig(
        domain=_NL_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={
            "a": a, "b": 0.175,
            "x_0": 0.349, "y_0": 0.0, "z_0": -0.16,
        },
    )
    sim = NewtonLeipnikSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.005)
