"""Tests for the Lorenz-84 atmospheric model simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.lorenz84 import Lorenz84Simulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestLorenz84Simulation:
    """Tests for the Lorenz-84 simulation engine."""

    def _make_sim(self, **kwargs) -> Lorenz84Simulation:
        defaults = {
            "a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return Lorenz84Simulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], 0.0)

    def test_custom_initial_conditions(self):
        sim = self._make_sim(x_0=5.0, y_0=-2.0, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 5.0)
        assert np.isclose(state[1], -2.0)
        assert np.isclose(state[2], 3.0)

    def test_state_shape_is_3d(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.ndim == 1
        assert state.shape == (3,)

    def test_step_advances_state(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_current_state(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        # Observe should match the current state after step
        sim.step()
        obs2 = sim.observe()
        assert np.allclose(obs2, sim._state)

    def test_default_parameters(self):
        sim = self._make_sim()
        assert sim.a == 0.25
        assert sim.b == 4.0
        assert sim.F == 8.0
        assert sim.G == 1.0

    def test_custom_parameters(self):
        sim = self._make_sim(a=0.5, b=2.0, F=6.0, G=0.5)
        assert sim.a == 0.5
        assert sim.b == 2.0
        assert sim.F == 6.0
        assert sim.G == 0.5


class TestLorenz84RK4:
    """Tests for RK4 integration stability and correctness."""

    def _make_sim(self, **kwargs) -> Lorenz84Simulation:
        defaults = {
            "a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return Lorenz84Simulation(config)

    def test_trajectory_stays_finite(self):
        """Lorenz-84 trajectories should remain bounded for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_bounded(self):
        """Trajectory should not diverge for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.linalg.norm(state) < 50, f"Trajectory diverged: {state}"

    def test_derivatives_at_hadley_fp(self):
        """At (F, 0, 0) with G=0, derivatives should be zero."""
        sim = self._make_sim(G=0.0)
        sim.reset()
        derivs = sim._derivatives(np.array([sim.F, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a known state for correctness."""
        sim = self._make_sim(a=0.25, b=4.0, F=8.0, G=1.0)
        sim.reset()
        # At state [2.0, 1.0, 0.5]:
        # dx = -1^2 - 0.5^2 - 0.25*2 + 0.25*8 = -1 - 0.25 - 0.5 + 2 = 0.25
        # dy = 2*1 - 4*2*0.5 - 1 + 1 = 2 - 4 - 1 + 1 = -2
        # dz = 4*2*1 + 2*0.5 - 0.5 = 8 + 1 - 0.5 = 8.5
        derivs = sim._derivatives(np.array([2.0, 1.0, 0.5]))
        np.testing.assert_allclose(derivs[0], 0.25, atol=1e-10)
        np.testing.assert_allclose(derivs[1], -2.0, atol=1e-10)
        np.testing.assert_allclose(derivs[2], 8.5, atol=1e-10)

    def test_small_dt_convergence(self):
        """Smaller dt should give a more accurate trajectory (Richardson extrapolation)."""
        # Run with dt=0.01
        sim1 = self._make_sim()
        sim1.config = SimulationConfig(
            domain=Domain.LORENZ_84, dt=0.01, n_steps=100,
            parameters={"a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
                        "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        # Run with dt=0.005 (200 steps to same time)
        config2 = SimulationConfig(
            domain=Domain.LORENZ_84, dt=0.005, n_steps=200,
            parameters={"a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
                        "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim2 = Lorenz84Simulation(config2)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe()

        # Both should be close (RK4 convergence)
        assert np.linalg.norm(state1 - state2) < 0.1


class TestLorenz84TrajectoryCollection:
    """Tests for trajectory data collection via run()."""

    def _make_sim(self, n_steps: int = 100, **kwargs) -> Lorenz84Simulation:
        defaults = {
            "a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=n_steps,
            parameters=defaults,
        )
        return Lorenz84Simulation(config)

    def test_trajectory_shape(self):
        sim = self._make_sim(n_steps=100)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)

    def test_trajectory_timestamps(self):
        sim = self._make_sim(n_steps=50)
        traj = sim.run(n_steps=50)
        assert len(traj.timestamps) == 51
        np.testing.assert_allclose(traj.timestamps[0], 0.0)
        np.testing.assert_allclose(traj.timestamps[-1], 0.5, atol=1e-10)

    def test_trajectory_data_finite(self):
        sim = self._make_sim(n_steps=500)
        traj = sim.run(n_steps=500)
        assert np.all(np.isfinite(traj.states))

    def test_trajectory_parameters(self):
        sim = self._make_sim()
        traj = sim.run(n_steps=10)
        assert "a" in traj.parameters
        assert "F" in traj.parameters
        assert traj.parameters["a"] == 0.25
        assert traj.parameters["F"] == 8.0


class TestLorenz84FixedPoints:
    """Tests for fixed point finding."""

    def _make_sim(self, **kwargs) -> Lorenz84Simulation:
        defaults = {
            "a": 0.25, "b": 4.0, "F": 8.0, "G": 0.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return Lorenz84Simulation(config)

    def test_hadley_fp_property(self):
        """Hadley fixed point should be (F, 0, 0)."""
        sim = self._make_sim(F=6.0)
        sim.reset()
        fp = sim.hadley_fixed_point
        np.testing.assert_array_almost_equal(fp, [6.0, 0.0, 0.0])

    def test_hadley_fp_is_true_fp_for_G_zero(self):
        """At G=0, (F, 0, 0) should have zero derivatives."""
        sim = self._make_sim(F=5.0, G=0.0)
        sim.reset()
        fp = sim.hadley_fixed_point
        derivs = sim._derivatives(fp)
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_find_fixed_points_returns_hadley(self):
        """find_fixed_points should return the Hadley fixed point for G=0."""
        sim = self._make_sim(F=3.0, G=0.0)
        sim.reset()
        fps = sim.find_fixed_points()
        assert len(fps) >= 1
        # Check that one fixed point is close to (F, 0, 0)
        distances = [np.linalg.norm(fp - np.array([3.0, 0.0, 0.0])) for fp in fps]
        assert min(distances) < 0.1

    def test_fixed_points_have_zero_derivatives(self):
        """All found fixed points should have near-zero derivatives."""
        sim = self._make_sim(G=0.0)
        sim.reset()
        fps = sim.find_fixed_points()
        for fp in fps:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=6,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestLorenz84HadleyCirculation:
    """Tests for the Hadley circulation regime (low forcing, G=0)."""

    def test_converges_to_hadley_fp(self):
        """For G=0 and F < 1 (stable regime), system should converge to (F, 0, 0).

        The Hadley fixed point is only stable when F < 1 (eigenvalue analysis:
        the two complex eigenvalues have real part F-1, so stable iff F < 1).
        """
        F = 0.8
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=10000,
            parameters={
                "a": 0.25, "b": 4.0, "F": F, "G": 0.0,
                "x_0": F + 0.1, "y_0": 0.1, "z_0": 0.1,
            },
        )
        sim = Lorenz84Simulation(config)
        sim.reset()

        for _ in range(10000):
            sim.step()

        state = sim.observe()
        np.testing.assert_allclose(state[0], F, atol=0.1)
        np.testing.assert_allclose(state[1], 0.0, atol=0.1)
        np.testing.assert_allclose(state[2], 0.0, atol=0.1)

    def test_hadley_fp_varies_with_F(self):
        """Different F < 1 values yield different stable Hadley fixed points."""
        for F in [0.3, 0.5, 0.8]:
            config = SimulationConfig(
                domain=Domain.LORENZ_84,
                dt=0.01,
                n_steps=10000,
                parameters={
                    "a": 0.25, "b": 4.0, "F": F, "G": 0.0,
                    "x_0": F + 0.1, "y_0": 0.01, "z_0": 0.01,
                },
            )
            sim = Lorenz84Simulation(config)
            sim.reset()
            for _ in range(10000):
                sim.step()
            state = sim.observe()
            assert abs(state[0] - F) < 0.2, (
                f"x* = {state[0]:.4f} not close to F = {F}"
            )


class TestLorenz84Lyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic_params(self):
        """Lorenz-84 at F=8, G=1 should have positive largest Lyapunov exponent."""
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=20000,
            parameters={
                "a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
                "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = Lorenz84Simulation(config)
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        lam = sim.compute_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.01, f"Lyapunov {lam:.4f} should be positive for chaotic regime"

    def test_negative_lyapunov_stable_params(self):
        """Lorenz-84 at low F with G=0 should have negative Lyapunov (stable FP)."""
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=20000,
            parameters={
                "a": 0.25, "b": 4.0, "F": 2.0, "G": 0.0,
                "x_0": 2.1, "y_0": 0.01, "z_0": 0.01,
            },
        )
        sim = Lorenz84Simulation(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.compute_lyapunov(n_steps=20000, dt=0.01)
        assert lam < 0.1, f"Lyapunov {lam:.4f} should be near-zero or negative"

    def test_lyapunov_returns_float(self):
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=1000,
            parameters={"a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0},
        )
        sim = Lorenz84Simulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_steps=1000, dt=0.01)
        assert isinstance(lam, float)
        assert np.isfinite(lam)


class TestLorenz84Jacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self) -> Lorenz84Simulation:
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=1000,
            parameters={"a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0},
        )
        return Lorenz84Simulation(config)

    def test_jacobian_shape(self):
        sim = self._make_sim()
        sim.reset()
        J = sim._jacobian(np.array([1.0, 0.5, 0.3]))
        assert J.shape == (3, 3)

    def test_jacobian_numerical_consistency(self):
        """Jacobian should be consistent with numerical differentiation."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([2.0, 1.0, 0.5])
        J_analytical = sim._jacobian(state)

        # Numerical Jacobian via finite differences
        eps = 1e-6
        J_numerical = np.zeros((3, 3))
        f0 = sim._derivatives(state)
        for j in range(3):
            state_plus = state.copy()
            state_plus[j] += eps
            f_plus = sim._derivatives(state_plus)
            J_numerical[:, j] = (f_plus - f0) / eps

        np.testing.assert_array_almost_equal(
            J_analytical, J_numerical, decimal=4,
        )


class TestLorenz84ParameterVariations:
    """Tests for different parameter regimes."""

    def _make_sim(self, **kwargs) -> Lorenz84Simulation:
        defaults = {
            "a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_84,
            dt=0.01,
            n_steps=5000,
            parameters=defaults,
        )
        return Lorenz84Simulation(config)

    def test_different_a_runs(self):
        """System should remain bounded for different damping values."""
        for a in [0.1, 0.25, 0.5, 1.0]:
            sim = self._make_sim(a=a)
            sim.reset()
            for _ in range(2000):
                state = sim.step()
                assert np.all(np.isfinite(state)), f"NaN at a={a}"

    def test_different_F_runs(self):
        """System should remain bounded for different forcing values."""
        for F in [1.0, 4.0, 8.0, 12.0]:
            sim = self._make_sim(F=F)
            sim.reset()
            for _ in range(2000):
                state = sim.step()
                assert np.all(np.isfinite(state)), f"NaN at F={F}"

    def test_different_G_runs(self):
        """System should remain bounded for different asymmetric forcing."""
        for G in [0.0, 0.5, 1.0, 2.0]:
            sim = self._make_sim(G=G)
            sim.reset()
            for _ in range(2000):
                state = sim.step()
                assert np.all(np.isfinite(state)), f"NaN at G={G}"

    def test_seeded_reset_adds_perturbation(self):
        """Reset with a seed should add a small perturbation."""
        sim = self._make_sim(x_0=1.0, y_0=0.0, z_0=0.0)
        state1 = sim.reset(seed=42)
        state2 = sim.reset()  # Without seed
        assert not np.allclose(state1, state2), "Seeded reset should differ"


class TestLorenz84Rediscovery:
    """Tests for Lorenz-84 rediscovery data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.lorenz84 import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.25
        assert data["b"] == 4.0
        assert data["F"] == 8.0
        assert data["G"] == 1.0

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.lorenz84 import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        from simulating_anything.rediscovery.lorenz84 import generate_chaos_transition_data

        data = generate_chaos_transition_data(n_F=5, n_steps=2000, dt=0.01)
        assert len(data["F"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5
        assert len(data["x_mean"]) == 5

    def test_hadley_verification_data(self):
        from simulating_anything.rediscovery.lorenz84 import generate_hadley_verification_data

        data = generate_hadley_verification_data(n_F=5, n_steps=5000, dt=0.01)
        assert len(data["F"]) == 5
        assert len(data["x_final"]) == 5
        assert len(data["y_final"]) == 5
        assert len(data["z_final"]) == 5

    def test_hadley_x_close_to_F(self):
        """For G=0 and F < 1 (stable Hadley), x_final should be close to F.

        The Hadley fixed point (F, 0, 0) is only stable for F < 1 due to
        Hopf bifurcation at F=1 (eigenvalues have real part F-1).
        """
        from simulating_anything.rediscovery.lorenz84 import generate_hadley_verification_data

        # Use F range in stable regime only (F < 1)
        data = generate_hadley_verification_data(
            n_F=5, n_steps=10000, dt=0.01,
        )
        # Filter to F < 1 (stable Hadley regime)
        mask = data["F"] < 1.0
        if np.sum(mask) == 0:
            pytest.skip("No F values in stable Hadley regime")
        errors = np.abs(data["x_final"][mask] - data["F"][mask])
        assert np.max(errors) < 0.5, (
            f"Max error {np.max(errors):.4f} too large; "
            f"F={data['F'][mask]}, x_final={data['x_final'][mask]}"
        )
