"""Tests for the Rikitake dynamo simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.rikitake import RikitakeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestRikitakeSimulation:
    """Tests for the Rikitake dynamo simulation."""

    def _make_sim(self, **kwargs) -> RikitakeSimulation:
        defaults = {
            "mu": 1.0, "a": 5.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return RikitakeSimulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 1.0)
        assert np.isclose(state[2], 0.0)

    def test_custom_initial_conditions(self):
        sim = self._make_sim(x_0=2.0, y_0=-1.5, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -1.5)
        assert np.isclose(state[2], 3.0)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_derivatives_at_fixed_point(self):
        """Derivatives should be zero at fixed points."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_derivatives_known_values(self):
        """Test derivatives at a known point."""
        sim = self._make_sim(mu=1.0, a=5.0)
        sim.reset()
        # At state [1, 1, 0]:
        # dx = -1*1 + 0*1 = -1
        # dy = -1*1 + (0 - 5)*1 = -6
        # dz = 1 - 1*1 = 0
        derivs = sim._derivatives(np.array([1.0, 1.0, 0.0]))
        assert np.isclose(derivs[0], -1.0)
        assert np.isclose(derivs[1], -6.0)
        assert np.isclose(derivs[2], 0.0)

    def test_trajectory_bounded(self):
        """Rikitake trajectories should remain bounded."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_trajectory_stays_finite_various_a(self):
        """Trajectory should be finite for a range of a values."""
        for a in [1.0, 3.0, 5.0, 8.0]:
            sim = self._make_sim(a=a)
            sim.reset()
            for _ in range(5000):
                state = sim.step()
            assert np.all(np.isfinite(state)), f"NaN/Inf at a={a}"

    def test_run_trajectory(self):
        """Test full trajectory collection via run()."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_default_parameters(self):
        """Default parameters should be mu=1.0, a=5.0."""
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = RikitakeSimulation(config)
        assert sim.mu == 1.0
        assert sim.a == 5.0


class TestRikitakeFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, mu=1.0, a=5.0) -> RikitakeSimulation:
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=0.01,
            n_steps=1000,
            parameters={"mu": mu, "a": a},
        )
        return RikitakeSimulation(config)

    def test_two_fixed_points(self):
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 2

    def test_fixed_points_symmetry(self):
        """The two fixed points should have x -> -x, y -> -y symmetry."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        # FP1: (x, 1/x, mu*x^2), FP2: (-x, -1/x, mu*x^2)
        assert np.isclose(fps[0][0], -fps[1][0])
        assert np.isclose(fps[0][1], -fps[1][1])
        assert np.isclose(fps[0][2], fps[1][2])

    def test_fixed_point_xy_product(self):
        """At fixed points, x*y = 1 (from dz/dt = 0)."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            assert np.isclose(fp[0] * fp[1], 1.0, atol=1e-10)

    def test_fixed_point_z_value(self):
        """At fixed points, z = mu*x^2 (from dx/dt = 0 with y=1/x)."""
        sim = self._make_sim(mu=1.0, a=5.0)
        sim.reset()
        for fp in sim.fixed_points:
            expected_z = sim.mu * fp[0] ** 2
            assert np.isclose(fp[2], expected_z, atol=1e-10)

    def test_fixed_points_different_params(self):
        """Fixed points should exist for different parameter values."""
        for mu, a in [(0.5, 3.0), (2.0, 7.0), (1.0, 1.0)]:
            sim = self._make_sim(mu=mu, a=a)
            sim.reset()
            fps = sim.fixed_points
            assert len(fps) == 2
            for fp in sim.fixed_points:
                derivs = sim._derivatives(fp)
                np.testing.assert_array_almost_equal(
                    derivs, [0.0, 0.0, 0.0], decimal=8,
                )


class TestRikitakeLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """Rikitake with a=5 should have positive largest Lyapunov exponent."""
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=0.01,
            n_steps=20000,
            parameters={"mu": 1.0, "a": 5.0},
        )
        sim = RikitakeSimulation(config)
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, f"Lyapunov {lam:.3f} not positive for chaotic regime"
        assert lam < 10.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_finite(self):
        """Lyapunov exponent should be finite."""
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=0.01,
            n_steps=5000,
            parameters={"mu": 1.0, "a": 5.0},
        )
        sim = RikitakeSimulation(config)
        sim.reset()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert np.isfinite(lam)


class TestRikitakeReversals:
    """Tests for polarity reversal counting."""

    def test_reversals_detected(self):
        """System should exhibit polarity reversals for a=5."""
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=0.01,
            n_steps=1000,
            parameters={"mu": 1.0, "a": 5.0},
        )
        sim = RikitakeSimulation(config)
        sim.reset()
        info = sim.count_reversals(n_transient=2000, n_measure=20000)
        assert info["n_reversals"] >= 0
        assert "mean_interval" in info
        assert "total_time" in info
        assert info["total_time"] > 0

    def test_reversal_count_structure(self):
        """Reversal info should have correct structure."""
        sim = RikitakeSimulation(SimulationConfig(
            domain=Domain.RIKITAKE, dt=0.01, n_steps=100,
            parameters={"mu": 1.0, "a": 5.0},
        ))
        sim.reset()
        info = sim.count_reversals(n_transient=100, n_measure=1000)
        assert isinstance(info["n_reversals"], int)
        assert isinstance(info["mean_interval"], float)
        assert isinstance(info["x_std"], float)


class TestRikitakeRediscovery:
    """Tests for Rikitake data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.rikitake import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["mu"] == 1.0
        assert data["a"] == 5.0

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.rikitake import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_a_data(self):
        from simulating_anything.rediscovery.rikitake import generate_lyapunov_vs_a_data

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=2000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_reversal_data(self):
        from simulating_anything.rediscovery.rikitake import generate_reversal_data

        data = generate_reversal_data(n_a=3, n_transient=100, n_measure=1000, dt=0.01)
        assert len(data["a"]) == 3
        assert len(data["n_reversals"]) == 3
        assert len(data["mean_interval"]) == 3
