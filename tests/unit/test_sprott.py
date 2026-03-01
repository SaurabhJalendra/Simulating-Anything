"""Tests for the Sprott minimal chaotic flow simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sprott import SprottSimulation


def _make_sim(system: str = "B", dt: float = 0.05, n_steps: int = 10000,
              x_0: float = 0.1, y_0: float = 0.1, z_0: float = 0.1
              ) -> SprottSimulation:
    """Helper to create a SprottSimulation via the factory method."""
    return SprottSimulation.create(
        system=system, dt=dt, n_steps=n_steps,
        x_0=x_0, y_0=y_0, z_0=z_0,
    )


class TestSprottSimulation:
    """Tests for the Sprott simulation engine."""

    def test_creation_default(self):
        """Default creation should produce a Sprott-B system."""
        sim = _make_sim()
        assert sim.system == "B"

    def test_initial_state_shape(self):
        """State should be a 3D vector."""
        sim = _make_sim()
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        """Initial state should match provided initial conditions."""
        sim = _make_sim(x_0=0.5, y_0=-0.3, z_0=1.0)
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, -0.3, 1.0])

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = _make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_matches_step(self):
        """observe() should return the same state as the last step()."""
        sim = _make_sim()
        sim.reset()
        state1 = sim.step()
        obs = sim.observe()
        np.testing.assert_array_equal(state1, obs)

    def test_rk4_integration_stability(self):
        """Sprott-B should remain finite for 10000 steps at dt=0.05."""
        sim = _make_sim(dt=0.05, n_steps=10000)
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_collection(self):
        """run() should return a TrajectoryData with correct shape."""
        sim = _make_sim(n_steps=100)
        traj = sim.run(n_steps=100)
        # n_steps + 1 (includes initial state)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_attractor_bounded(self):
        """Sprott-B attractor should remain bounded for chaotic trajectory."""
        sim = _make_sim(dt=0.01, n_steps=20000)
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            norm = np.linalg.norm(state)
            assert norm < 100, f"Trajectory diverged: norm={norm:.1f}, state={state}"


class TestSprottDerivatives:
    """Tests for Sprott system derivative computations."""

    def test_sprott_b_derivatives_at_fixed_point(self):
        """Derivatives should be zero at fixed points of Sprott-B."""
        sim = _make_sim(system="B")
        sim.reset()

        fp1 = np.array([1.0, 1.0, 0.0])
        derivs = sim._derivatives(fp1)
        np.testing.assert_array_almost_equal(
            derivs, [0.0, 0.0, 0.0], decimal=10,
            err_msg=f"Non-zero derivatives at fixed point {fp1}",
        )

        fp2 = np.array([-1.0, -1.0, 0.0])
        derivs2 = sim._derivatives(fp2)
        np.testing.assert_array_almost_equal(
            derivs2, [0.0, 0.0, 0.0], decimal=10,
            err_msg=f"Non-zero derivatives at fixed point {fp2}",
        )

    def test_sprott_b_derivatives_known_point(self):
        """Test derivatives at a known point for Sprott-B.

        At state [1, 0, 1]: dx=0*1=0, dy=1-0=1, dz=1-1*0=1
        """
        sim = _make_sim(system="B")
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 0.0, 1.0]))
        np.testing.assert_allclose(derivs, [0.0, 1.0, 1.0])

    def test_sprott_a_derivatives(self):
        """Test Sprott-A derivatives at a known point.

        At [1, 0, 0]: dx=0, dy=-1+0==-1, dz=1-0=1
        """
        sim = _make_sim(system="A")
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(derivs, [0.0, -1.0, 1.0])

    def test_sprott_c_derivatives(self):
        """Test Sprott-C derivatives at a known point.

        At [0, 0, 1]: dx=0, dy=0-0=0, dz=1-0=1
        """
        sim = _make_sim(system="C")
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(derivs, [0.0, 0.0, 1.0])

    def test_unknown_system_raises(self):
        """Using an unknown system letter should raise ValueError."""
        sim = _make_sim(system="Z")
        sim.reset()
        with pytest.raises(ValueError, match="Unknown Sprott system"):
            sim._derivatives(np.array([0.0, 0.0, 0.0]))


class TestSprottVariants:
    """Tests for different Sprott system variants."""

    def test_sprott_a_runs(self):
        """Sprott-A should run without error."""
        sim = _make_sim(system="A", dt=0.05, n_steps=1000)
        sim.reset()
        for _ in range(1000):
            state = sim.step()
            assert np.all(np.isfinite(state))

    def test_sprott_c_runs(self):
        """Sprott-C should run without error."""
        sim = _make_sim(system="C", dt=0.05, n_steps=1000)
        sim.reset()
        for _ in range(1000):
            state = sim.step()
            assert np.all(np.isfinite(state))

    def test_sprott_e_runs(self):
        """Sprott-E should run without error."""
        sim = _make_sim(system="E", dt=0.01, n_steps=1000)
        sim.reset()
        for _ in range(1000):
            state = sim.step()
            assert np.all(np.isfinite(state))

    def test_different_systems_produce_different_trajectories(self):
        """Different Sprott systems should produce different trajectories."""
        sim_a = _make_sim(system="A", dt=0.01)
        sim_b = _make_sim(system="B", dt=0.01)

        sim_a.reset()
        sim_b.reset()

        for _ in range(100):
            sim_a.step()
            sim_b.step()

        state_a = sim_a.observe()
        state_b = sim_b.observe()
        assert not np.allclose(state_a, state_b)


class TestSprottLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_sprott_b(self):
        """Sprott-B should have a positive largest Lyapunov exponent."""
        sim = _make_sim(system="B", dt=0.01)
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0, f"Lyapunov {lam:.4f} is not positive for Sprott-B"
        assert lam < 5.0, f"Lyapunov {lam:.4f} unreasonably large"

    def test_lyapunov_returns_float(self):
        """estimate_lyapunov should return a finite float."""
        sim = _make_sim(system="B", dt=0.01)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert isinstance(lam, float)
        assert np.isfinite(lam)


class TestSprottSensitivity:
    """Tests for sensitivity to initial conditions (hallmark of chaos)."""

    def test_nearby_trajectories_diverge(self):
        """Two nearby initial conditions should diverge for Sprott-B."""
        eps = 1e-6
        sim1 = _make_sim(system="B", dt=0.01, x_0=0.1, y_0=0.1, z_0=0.1)
        sim2 = _make_sim(system="B", dt=0.01, x_0=0.1 + eps, y_0=0.1, z_0=0.1)

        sim1.reset()
        sim2.reset()

        # Initial distance
        d0 = np.linalg.norm(sim1.observe() - sim2.observe())
        assert d0 == pytest.approx(eps, abs=1e-10)

        # Evolve for many steps
        for _ in range(5000):
            sim1.step()
            sim2.step()

        d_final = np.linalg.norm(sim1.observe() - sim2.observe())
        # Distance should have grown significantly (exponential divergence)
        assert d_final > 1e-3, (
            f"Trajectories did not diverge: d0={d0:.2e}, d_final={d_final:.2e}"
        )


class TestSprottFixedPoints:
    """Tests for fixed point computation."""

    def test_sprott_b_has_two_fixed_points(self):
        """Sprott-B should have exactly two fixed points."""
        sim = _make_sim(system="B")
        sim.reset()
        fps = sim.sprott_b_fixed_points
        assert len(fps) == 2

    def test_sprott_b_fixed_point_values(self):
        """Sprott-B fixed points should be (1,1,0) and (-1,-1,0)."""
        sim = _make_sim(system="B")
        sim.reset()
        fps = sim.sprott_b_fixed_points
        np.testing.assert_allclose(fps[0], [1.0, 1.0, 0.0])
        np.testing.assert_allclose(fps[1], [-1.0, -1.0, 0.0])

    def test_non_b_system_returns_empty(self):
        """fixed_points should return empty for non-B systems."""
        sim = _make_sim(system="A")
        sim.reset()
        fps = sim.sprott_b_fixed_points
        assert len(fps) == 0


class TestSprottRediscovery:
    """Tests for Sprott rediscovery data generation functions."""

    def test_ode_data_generation(self):
        """generate_ode_data should produce finite trajectory data."""
        from simulating_anything.rediscovery.sprott import generate_ode_data

        data = generate_ode_data(system="B", n_steps=200, dt=0.01)
        assert data["states"].shape == (201, 3)
        assert np.all(np.isfinite(data["states"]))
        assert data["system"] == "B"
        assert data["dt"] == 0.01

    def test_attractor_characterization(self):
        """characterize_attractor should return valid statistics."""
        from simulating_anything.rediscovery.sprott import characterize_attractor

        result = characterize_attractor(system="B", n_steps=5000, dt=0.01)
        assert result["is_bounded"]
        assert result["max_norm"] < 100
        assert len(result["x_range"]) == 2
        assert result["x_range"][0] < result["x_range"][1]

    def test_lyapunov_comparison(self):
        """generate_lyapunov_comparison should return exponents for each system."""
        from simulating_anything.rediscovery.sprott import generate_lyapunov_comparison

        result = generate_lyapunov_comparison(
            systems=["A", "B"], n_steps=5000, dt=0.01
        )
        assert "A" in result["lyapunov_exponents"]
        assert "B" in result["lyapunov_exponents"]
        assert len(result["systems"]) == 2
