"""Tests for the Hadley circulation simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.hadley import HadleySimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestHadleySimulation:
    """Tests for the Hadley system simulation basics."""

    def _make_sim(self, **kwargs) -> HadleySimulation:
        defaults = {
            "a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return HadleySimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 0.2
        assert sim.b == 4.0
        assert sim.F == 8.0
        assert sim.G == 1.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=0.3, b=5.0, F=6.0, G=0.5)
        assert sim.a == 0.3
        assert sim.b == 5.0
        assert sim.F == 6.0
        assert sim.G == 0.5

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=2.0, y_0=-1.0, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -1.0)
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
        """Same parameters produce the same trajectory."""
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


class TestHadleyDerivatives:
    """Tests for the Hadley ODE derivative computation."""

    def _make_sim(self, **kwargs) -> HadleySimulation:
        defaults = {
            "a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return HadleySimulation(config)

    def test_derivatives_at_hadley_fp_g0(self):
        """At (F, 0, 0) with G=0, all derivatives should be zero."""
        sim = self._make_sim(G=0.0)
        sim.reset()
        derivs = sim._derivatives(np.array([sim.F, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at state [1, 1, 0] with a=0.2, b=4, F=8, G=1.

        dx = -1^2 - 0^2 - 0.2*1 + 0.2*8 = -1 - 0.2 + 1.6 = 0.4
        dy = 1*1 - 4*1*0 - 1 + 1 = 1 - 0 - 1 + 1 = 1.0
        dz = 4*1*1 + 1*0 - 0 = 4.0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 0.0]))
        assert np.isclose(derivs[0], 0.4)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], 4.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 1, 1] with a=0.2, b=4, F=8, G=1.

        dx = -1^2 - 1^2 - 0.2*2 + 0.2*8 = -1 - 1 - 0.4 + 1.6 = -0.8
        dy = 2*1 - 4*2*1 - 1 + 1 = 2 - 8 - 1 + 1 = -6.0
        dz = 4*2*1 + 2*1 - 1 = 8 + 2 - 1 = 9.0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], -0.8)
        assert np.isclose(derivs[1], -6.0)
        assert np.isclose(derivs[2], 9.0)

    def test_derivatives_origin(self):
        """At origin with a=0.2, F=8, G=1:

        dx = 0 - 0 - 0 + 0.2*8 = 1.6
        dy = 0 - 0 - 0 + 1 = 1.0
        dz = 0 + 0 - 0 = 0.0
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 1.6)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], 0.0)


class TestHadleyJacobian:
    """Tests for the Jacobian computation."""

    def _make_sim(self, **kwargs) -> HadleySimulation:
        defaults = {"a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return HadleySimulation(config)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should match analytical form.

        At (0,0,0): J = [[-a, 0, 0], [0, -1, 0], [0, 0, -1]]
        """
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [-0.2, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_at_hadley_fp(self):
        """Jacobian at (F, 0, 0) with F=8.

        J = [[-a, 0, 0], [0, F-1, -b*F], [0, b*F, F-1]]
          = [[-0.2, 0, 0], [0, 7, -32], [0, 32, 7]]
        """
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([8.0, 0.0, 0.0]))
        expected = np.array([
            [-0.2, 0.0, 0.0],
            [0.0, 7.0, -32.0],
            [0.0, 32.0, 7.0],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_numerical_consistency(self):
        """Jacobian should be consistent with finite-difference approximation."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([2.0, 1.0, -0.5])
        J_analytic = sim.jacobian(state)

        eps = 1e-7
        J_numeric = np.zeros((3, 3))
        f0 = sim._derivatives(state)
        for j in range(3):
            state_pert = state.copy()
            state_pert[j] += eps
            f1 = sim._derivatives(state_pert)
            J_numeric[:, j] = (f1 - f0) / eps

        np.testing.assert_array_almost_equal(
            J_analytic, J_numeric, decimal=5
        )


class TestHadleyDivergence:
    """Tests for divergence computation."""

    def _make_sim(self, **kwargs) -> HadleySimulation:
        defaults = {"a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return HadleySimulation(config)

    def test_divergence_default_params(self):
        """Divergence should be -(a + 2) = -2.2 for a=0.2."""
        sim = self._make_sim()
        sim.reset()
        div = sim.compute_divergence()
        assert np.isclose(div, -2.2)

    def test_divergence_custom_a(self):
        """Divergence should be -(a + 2) for any a value."""
        sim = self._make_sim(a=0.5)
        sim.reset()
        div = sim.compute_divergence()
        assert np.isclose(div, -2.5)

    def test_divergence_is_negative(self):
        """Flow is dissipative (negative divergence) for positive a."""
        sim = self._make_sim(a=0.1)
        sim.reset()
        assert sim.compute_divergence() < 0


class TestHadleyFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> HadleySimulation:
        defaults = {"a": 0.2, "b": 4.0, "F": 8.0, "G": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return HadleySimulation(config)

    def test_hadley_fp_exists_g0(self):
        """The Hadley fixed point (F, 0, 0) should exist for G=0."""
        sim = self._make_sim(G=0.0, F=2.0)
        sim.reset()
        fps = sim.fixed_points()
        # Check that (F, 0, 0) is among the fixed points
        found_hadley = False
        for fp in fps:
            if np.allclose(fp, [2.0, 0.0, 0.0], atol=1e-6):
                found_hadley = True
                break
        assert found_hadley, f"Hadley FP not found among {fps}"

    def test_hadley_fp_property(self):
        """The hadley_fixed_point property returns (F, 0, 0)."""
        sim = self._make_sim(F=5.0)
        assert np.allclose(sim.hadley_fixed_point, [5.0, 0.0, 0.0])

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim(G=0.0, F=0.5)
        sim.reset()
        for fp in sim.fixed_points():
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=6,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_fixed_points_with_G_nonzero(self):
        """Fixed points should still be found for G != 0."""
        sim = self._make_sim(G=1.0, F=8.0)
        sim.reset()
        fps = sim.fixed_points()
        # Should find at least one fixed point
        assert len(fps) >= 1
        for fp in fps:
            derivs = sim._derivatives(fp)
            assert np.linalg.norm(derivs) < 1e-6, (
                f"FP {fp} has |deriv|={np.linalg.norm(derivs):.2e}"
            )


class TestHadleyTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> HadleySimulation:
        defaults = {
            "a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return HadleySimulation(config)

    def test_trajectory_stays_bounded(self):
        """Hadley trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, (
                f"Trajectory diverged: {state}"
            )

    def test_trajectory_stays_finite_long_run(self):
        """Long trajectory should remain finite."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_different_F_gives_different_trajectory(self):
        """Changing F should change the trajectory behavior."""
        sim1 = self._make_sim(F=8.0)
        sim2 = self._make_sim(F=2.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)

    def test_attractor_statistics(self):
        """Trajectory statistics should have nonzero variance for chaotic regime."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=10000, n_transient=3000
        )
        # For chaotic motion, there should be variance
        assert stats["x_std"] > 0.1, "x_std too small for chaotic regime"
        assert stats["y_std"] > 0.1, "y_std too small for chaotic regime"


class TestHadleyLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """Hadley at F=8, G=1 should show positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=20000,
            parameters={
                "a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0,
                "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim = HadleySimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.lyapunov_exponent(n_steps=20000, dt=0.01)
        assert lam > 0.0, f"Lyapunov {lam:.3f} not positive for chaotic regime"
        assert lam < 20.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_F(self):
        """Lyapunov exponent should change as F varies."""
        lam1 = _compute_lyapunov_at_F(8.0)
        lam2 = _compute_lyapunov_at_F(2.0)
        assert lam1 != lam2, "Lyapunov should differ for different F"

    def test_seed_perturbation(self):
        """Seed-based perturbation should give a slightly different IC."""
        sim = HadleySimulation(SimulationConfig(
            domain=Domain.HADLEY,
            dt=0.01,
            n_steps=100,
            parameters={"a": 0.2, "b": 4.0, "F": 8.0, "G": 1.0},
        ))
        s1 = sim.reset(seed=42)
        s2 = sim.reset(seed=99)
        assert not np.allclose(s1, s2), "Different seeds should differ"


class TestHadleyRediscovery:
    """Tests for Hadley data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.hadley import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.2
        assert data["b"] == 4.0
        assert data["F"] == 8.0
        assert data["G"] == 1.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.hadley import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_sweep_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.hadley import (
            generate_lyapunov_vs_F_data,
        )

        data = generate_lyapunov_vs_F_data(n_F=5, n_steps=3000, dt=0.01)
        assert len(data["F"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_hadley_fp_verification_data(self):
        """Hadley FP verification should produce valid data."""
        from simulating_anything.rediscovery.hadley import (
            generate_hadley_fp_verification_data,
        )

        data = generate_hadley_fp_verification_data(
            n_F=5, n_steps=2000, dt=0.01
        )
        assert len(data["F"]) == 5
        assert len(data["x_final"]) == 5
        assert np.all(np.isfinite(data["x_final"]))

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.hadley import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data


def _compute_lyapunov_at_F(F: float) -> float:
    """Helper to compute Lyapunov exponent at a given F."""
    config = SimulationConfig(
        domain=Domain.HADLEY,
        dt=0.01,
        n_steps=20000,
        parameters={
            "a": 0.2, "b": 4.0, "F": F, "G": 1.0,
            "x_0": 0.0, "y_0": 1.0, "z_0": 0.0,
        },
    )
    sim = HadleySimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.lyapunov_exponent(n_steps=15000, dt=0.01)
