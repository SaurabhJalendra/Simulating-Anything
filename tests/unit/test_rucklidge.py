"""Tests for the Rucklidge attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.rucklidge import RucklidgeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestRucklidgeSimulation:
    """Tests for the Rucklidge system simulation basics."""

    def _make_sim(self, **kwargs) -> RucklidgeSimulation:
        defaults = {
            "kappa": 2.0, "lambda_param": 6.7,
            "x_0": 1.0, "y_0": 0.0, "z_0": 4.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return RucklidgeSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.kappa == 2.0
        assert sim.lambda_param == 6.7

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(kappa=3.0, lambda_param=8.0)
        assert sim.kappa == 3.0
        assert sim.lambda_param == 8.0

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
        sim1 = self._make_sim(kappa=2.0, lambda_param=6.7)
        sim2 = self._make_sim(kappa=2.0, lambda_param=6.7)
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


class TestRucklidgeDerivatives:
    """Tests for the Rucklidge ODE derivative computation."""

    def _make_sim(self, **kwargs) -> RucklidgeSimulation:
        defaults = {
            "kappa": 2.0, "lambda_param": 6.7,
            "x_0": 1.0, "y_0": 0.0, "z_0": 4.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return RucklidgeSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 2, 3] with kappa=2, lambda=6.7:
            dx = -2*1 + 6.7*2 - 2*3 = -2 + 13.4 - 6 = 5.4
            dy = 1
            dz = -3 + 2^2 = -3 + 4 = 1
        """
        sim = self._make_sim(kappa=2.0, lambda_param=6.7)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(derivs[0], 5.4)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], 1.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [0, 3, 5] with kappa=2, lambda=6.7.

            dx = -2*0 + 6.7*3 - 3*5 = 0 + 20.1 - 15 = 5.1
            dy = 0
            dz = -5 + 3^2 = -5 + 9 = 4
        """
        sim = self._make_sim(kappa=2.0, lambda_param=6.7)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 3.0, 5.0]))
        assert np.isclose(derivs[0], 5.1)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 4.0)


class TestRucklidgeFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> RucklidgeSimulation:
        defaults = {"kappa": 2.0, "lambda_param": 6.7}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return RucklidgeSimulation(config)

    def test_three_fixed_points(self):
        """Standard parameters should give three fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_origin_is_fixed_point(self):
        """First fixed point should be the origin."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_fixed_point_symmetry(self):
        """The two non-origin fixed points should be symmetric in y."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        # x=0 for both
        assert np.isclose(fps[1][0], 0.0)
        assert np.isclose(fps[2][0], 0.0)
        # y values are opposite
        assert np.isclose(fps[1][1], -fps[2][1])
        # z values are the same
        assert np.isclose(fps[1][2], fps[2][2])

    def test_fixed_point_z_value(self):
        """z-coordinate of symmetric fixed points should be lambda.

        For lambda=6.7: z = 6.7
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][2], 6.7)
        assert np.isclose(fps[2][2], 6.7)

    def test_fixed_point_y_value(self):
        """y-coordinate of symmetric fixed points should be +/-sqrt(lambda).

        For lambda=6.7: y = +/-sqrt(6.7) ~ 2.588
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        y_expected = np.sqrt(6.7)
        assert np.isclose(abs(fps[1][1]), y_expected)
        assert np.isclose(abs(fps[2][1]), y_expected)

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


class TestRucklidgeJacobian:
    """Tests for Jacobian computation."""

    def _make_sim(self, **kwargs) -> RucklidgeSimulation:
        defaults = {"kappa": 2.0, "lambda_param": 6.7}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return RucklidgeSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([1.0, 2.0, 3.0]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at origin with kappa=2, lambda=6.7.

        J = [[-2,  6.7,  0],
             [ 1,  0,    0],
             [ 0,  0,   -1]]
        """
        sim = self._make_sim(kappa=2.0, lambda_param=6.7)
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-2.0, 6.7, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_at_known_point(self):
        """Jacobian at [1, 2, 3] with kappa=2, lambda=6.7.

        J = [[-2,  6.7-3,  -2],
             [ 1,  0,       0],
             [ 0,  2*2,    -1]]
          = [[-2,  3.7,  -2],
             [ 1,  0,     0],
             [ 0,  4,    -1]]
        """
        sim = self._make_sim(kappa=2.0, lambda_param=6.7)
        sim.reset()
        J = sim.jacobian(np.array([1.0, 2.0, 3.0]))
        expected = np.array([
            [-2.0, 3.7, -2.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, -1.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)


class TestRucklidgeDivergence:
    """Tests for divergence computation."""

    def _make_sim(self, **kwargs) -> RucklidgeSimulation:
        defaults = {"kappa": 2.0, "lambda_param": 6.7}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return RucklidgeSimulation(config)

    def test_divergence_standard(self):
        """Divergence should be -(kappa + 1) = -3.0 for kappa=2."""
        sim = self._make_sim(kappa=2.0)
        div = sim.compute_divergence()
        assert np.isclose(div, -3.0)

    def test_divergence_custom_kappa(self):
        """Divergence should be -(kappa + 1) for arbitrary kappa."""
        sim = self._make_sim(kappa=5.0)
        div = sim.compute_divergence()
        assert np.isclose(div, -6.0)

    def test_divergence_always_negative(self):
        """Divergence is negative for any positive kappa (dissipative)."""
        for kappa in [0.5, 1.0, 2.0, 5.0, 10.0]:
            sim = self._make_sim(kappa=kappa)
            div = sim.compute_divergence()
            assert div < 0, f"Divergence not negative for kappa={kappa}"


class TestRucklidgeTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> RucklidgeSimulation:
        defaults = {
            "kappa": 2.0, "lambda_param": 6.7,
            "x_0": 1.0, "y_0": 0.0, "z_0": 4.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return RucklidgeSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Rucklidge trajectories should remain bounded for standard params."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, (
                f"Trajectory diverged: {state}"
            )

    def test_z_stays_nonnegative_on_attractor(self):
        """z = y^2 at equilibrium so z should stay nonneg after transient."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        for _ in range(5000):
            state = sim.step()
            assert state[2] > -1.0, f"z went too negative: {state[2]}"

    def test_attractor_statistics(self):
        """Time-averaged z should be in a reasonable range."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # z should be positive on the attractor
        assert stats["z_mean"] > 0.0, (
            f"z_mean={stats['z_mean']:.1f}, expected positive"
        )
        # x should oscillate around 0
        assert abs(stats["x_mean"]) < 10.0, (
            f"x_mean={stats['x_mean']:.1f}, expected near 0"
        )

    def test_different_lambda_gives_different_trajectory(self):
        """Changing lambda should change the trajectory behavior."""
        sim1 = self._make_sim(lambda_param=6.7)
        sim2 = self._make_sim(lambda_param=3.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestRucklidgeChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Rucklidge at standard params should have positive Lyapunov."""
        config = SimulationConfig(
            domain=Domain.RUCKLIDGE,
            dt=0.01,
            n_steps=20000,
            parameters={"kappa": 2.0, "lambda_param": 6.7},
        )
        sim = RucklidgeSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, (
            f"Lyapunov {lam:.3f} not positive for chaotic regime"
        )
        assert lam < 10.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_lambda(self):
        """Lyapunov exponent should change as lambda varies."""
        lyap1 = _compute_lyapunov_at_lambda(6.7)
        lyap2 = _compute_lyapunov_at_lambda(3.0)
        assert lyap1 != lyap2, (
            "Lyapunov should differ for different lambda"
        )


class TestRucklidgeRediscovery:
    """Tests for Rucklidge data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.rucklidge import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["kappa"] == 2.0
        assert data["lambda_param"] == 6.7

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.rucklidge import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.rucklidge import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(
            n_lam=5, n_steps=2000, dt=0.01
        )
        assert len(data["lambda_param"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.rucklidge import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_lambda_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.rucklidge import (
            generate_lyapunov_vs_lambda_data,
        )

        data = generate_lyapunov_vs_lambda_data(
            n_lam=5, n_steps=3000, dt=0.01
        )
        assert len(data["lambda_param"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_lambda(lam: float) -> float:
    """Helper to compute Lyapunov exponent at a given lambda."""
    config = SimulationConfig(
        domain=Domain.RUCKLIDGE,
        dt=0.01,
        n_steps=20000,
        parameters={
            "kappa": 2.0, "lambda_param": lam,
            "x_0": 1.0, "y_0": 0.0, "z_0": 4.5,
        },
    )
    sim = RucklidgeSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.01)
