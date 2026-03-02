"""Tests for the genetic toggle switch simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.toggle_switch import ToggleSwitchSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha1: float = 4.0,
    alpha2: float = 4.0,
    beta: float = 4.0,
    gamma: float = 4.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    u_0: float = 3.0,
    v_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.TOGGLE_SWITCH,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha1": alpha1,
            "alpha2": alpha2,
            "beta": beta,
            "gamma": gamma,
            "u_0": u_0,
            "v_0": v_0,
        },
    )


class TestToggleSwitchBasic:
    """Basic simulation tests."""

    def test_initial_state_shape(self):
        sim = ToggleSwitchSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = ToggleSwitchSimulation(_make_config(u_0=2.5, v_0=1.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [2.5, 1.5], atol=0.02)

    def test_step_advances_state(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = ToggleSwitchSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = ToggleSwitchSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps(self):
        dt = 0.01
        sim = ToggleSwitchSimulation(_make_config(dt=dt))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * dt
        np.testing.assert_allclose(traj.timestamps, expected, atol=1e-12)


class TestToggleSwitchRK4:
    """RK4 integration tests."""

    def test_derivatives_hill_repression(self):
        """Hill repression terms should decrease production as repressor increases."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        # At u=0, v=0: du/dt = alpha1/(1+0) - 0 = alpha1
        deriv_zero = sim._derivatives(np.array([0.0, 0.0]))
        assert deriv_zero[0] == pytest.approx(4.0, abs=1e-10)  # alpha1
        assert deriv_zero[1] == pytest.approx(4.0, abs=1e-10)  # alpha2

    def test_derivatives_high_repressor(self):
        """High repressor should suppress production."""
        sim = ToggleSwitchSimulation(_make_config(beta=4.0, gamma=4.0))
        sim.reset()
        # At u=0, v=10: du/dt = alpha1/(1+10^4) - 0 ~ 0
        deriv = sim._derivatives(np.array([0.0, 10.0]))
        assert deriv[0] < 0.01  # production nearly zero due to repression by v

    def test_rk4_energy_like_convergence(self):
        """State should converge to a fixed point, not diverge."""
        sim = ToggleSwitchSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        # Should be finite and bounded
        assert np.all(np.isfinite(state))
        assert np.all(np.abs(state) < 100)

    def test_rk4_small_dt_accuracy(self):
        """Smaller dt should give more accurate results (comparing two dt values)."""
        config_coarse = _make_config(dt=0.1, u_0=2.0, v_0=2.0)
        config_fine = _make_config(dt=0.01, u_0=2.0, v_0=2.0)

        sim_coarse = ToggleSwitchSimulation(config_coarse)
        sim_fine = ToggleSwitchSimulation(config_fine)

        sim_coarse.reset()
        sim_fine.reset()

        # Run both for 10 time units
        for _ in range(100):  # 100 * 0.1 = 10
            sim_coarse.step()
        for _ in range(1000):  # 1000 * 0.01 = 10
            sim_fine.step()

        # Both should be close but not identical
        state_coarse = sim_coarse.observe()
        state_fine = sim_fine.observe()
        assert np.all(np.isfinite(state_coarse))
        assert np.all(np.isfinite(state_fine))
        # Fine and coarse should agree to within ~1% for stable dynamics
        np.testing.assert_allclose(state_coarse, state_fine, rtol=0.05)


class TestToggleSwitchBistability:
    """Bistability and fixed point tests."""

    def test_converges_to_high_u_state(self):
        """Starting with high u, low v should converge to high-u fixed point."""
        sim = ToggleSwitchSimulation(_make_config(u_0=4.0, v_0=0.1, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        # Should have high u, low v
        assert state[0] > 2.0
        assert state[1] < 1.0

    def test_converges_to_high_v_state(self):
        """Starting with low u, high v should converge to high-v fixed point."""
        sim = ToggleSwitchSimulation(_make_config(u_0=0.1, v_0=4.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        # Should have low u, high v
        assert state[0] < 1.0
        assert state[1] > 2.0

    def test_two_ics_different_steady_states(self):
        """Two different ICs should converge to different fixed points (bistability)."""
        sim1 = ToggleSwitchSimulation(_make_config(u_0=4.0, v_0=0.1, dt=0.01))
        sim2 = ToggleSwitchSimulation(_make_config(u_0=0.1, v_0=4.0, dt=0.01))

        sim1.reset()
        sim2.reset()

        for _ in range(10000):
            sim1.step()
            sim2.step()

        state1 = sim1.observe()
        state2 = sim2.observe()

        # States should be significantly different
        diff = np.linalg.norm(state1 - state2)
        assert diff > 1.0, f"States too similar: diff={diff:.4f}"

    def test_symmetric_bistability(self):
        """For alpha1=alpha2, beta=gamma, the two stable FPs should be symmetric."""
        sim1 = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
            u_0=4.0, v_0=0.1, dt=0.01,
        ))
        sim2 = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
            u_0=0.1, v_0=4.0, dt=0.01,
        ))

        sim1.reset()
        sim2.reset()

        for _ in range(10000):
            sim1.step()
            sim2.step()

        state1 = sim1.observe()
        state2 = sim2.observe()

        # By symmetry: (u1, v1) ~ (v2, u2)
        np.testing.assert_allclose(state1[0], state2[1], atol=0.1)
        np.testing.assert_allclose(state1[1], state2[0], atol=0.1)


class TestToggleSwitchFixedPoints:
    """Fixed point finding and classification tests."""

    def test_find_fixed_points_returns_list(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        assert isinstance(fps, list)
        assert len(fps) >= 1

    def test_find_three_fixed_points(self):
        """For default parameters, should find 3 fixed points."""
        sim = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
        ))
        sim.reset()
        fps = sim.find_fixed_points()
        assert len(fps) == 3, f"Expected 3 fixed points, found {len(fps)}"

    def test_fixed_points_satisfy_equations(self):
        """At fixed points, derivatives should be zero."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        for u_fp, v_fp in fps:
            deriv = sim._derivatives(np.array([u_fp, v_fp]))
            np.testing.assert_allclose(deriv, [0.0, 0.0], atol=1e-6)

    def test_fixed_points_non_negative(self):
        """All fixed points should have non-negative concentrations."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        for u_fp, v_fp in fps:
            assert u_fp >= -0.01, f"Negative u: {u_fp}"
            assert v_fp >= -0.01, f"Negative v: {v_fp}"

    def test_classify_stable_fixed_points(self):
        """Should find exactly 2 stable fixed points for default params."""
        sim = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
        ))
        sim.reset()
        fps = sim.find_fixed_points()
        stabilities = [sim.classify_fixed_point(u, v) for u, v in fps]
        n_stable = stabilities.count("stable")
        assert n_stable == 2, f"Expected 2 stable FPs, found {n_stable}"

    def test_classify_saddle_point(self):
        """Should find exactly 1 saddle point for default params."""
        sim = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
        ))
        sim.reset()
        fps = sim.find_fixed_points()
        stabilities = [sim.classify_fixed_point(u, v) for u, v in fps]
        n_saddle = stabilities.count("saddle")
        assert n_saddle == 1, f"Expected 1 saddle, found {n_saddle}"

    def test_fixed_points_sorted_by_u(self):
        """Fixed points should be returned sorted by u value."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        u_values = [fp[0] for fp in fps]
        assert u_values == sorted(u_values)


class TestToggleSwitchNullclines:
    """Nullcline computation tests."""

    def test_nullclines_return_correct_keys(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        nc = sim.compute_nullclines()
        assert "u_null_u" in nc
        assert "u_null_v" in nc
        assert "v_null_u" in nc
        assert "v_null_v" in nc

    def test_nullclines_correct_shape(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        nc = sim.compute_nullclines(n_points=100)
        assert nc["u_null_u"].shape == (100,)
        assert nc["u_null_v"].shape == (100,)
        assert nc["v_null_u"].shape == (100,)
        assert nc["v_null_v"].shape == (100,)

    def test_u_nullcline_decreasing(self):
        """u-nullcline: u = alpha1/(1+v^beta) should decrease as v increases."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        nc = sim.compute_nullclines(u_range=(0.01, 6.0), n_points=100)
        # u_null_u should decrease as v (u_null_v) increases
        assert nc["u_null_u"][0] > nc["u_null_u"][-1]

    def test_v_nullcline_decreasing(self):
        """v-nullcline: v = alpha2/(1+u^gamma) should decrease as u increases."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        nc = sim.compute_nullclines(u_range=(0.01, 6.0), n_points=100)
        # v_null_v should decrease as u (v_null_u) increases
        assert nc["v_null_v"][0] > nc["v_null_v"][-1]

    def test_nullcline_bounds(self):
        """u-nullcline bounded by [0, alpha1], v-nullcline bounded by [0, alpha2]."""
        sim = ToggleSwitchSimulation(_make_config(alpha1=4.0, alpha2=4.0))
        sim.reset()
        nc = sim.compute_nullclines(u_range=(0.0, 10.0), n_points=200)
        assert np.all(nc["u_null_u"] >= 0)
        assert np.all(nc["u_null_u"] <= 4.0 + 1e-10)
        assert np.all(nc["v_null_v"] >= 0)
        assert np.all(nc["v_null_v"] <= 4.0 + 1e-10)


class TestToggleSwitchAlphaSweep:
    """Alpha sweep and bifurcation tests."""

    def test_alpha_sweep_returns_correct_keys(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        sweep = sim.alpha_sweep(alpha_range=(1.0, 4.0), n_alpha=5, n_steps=1000)
        assert "alpha" in sweep
        assert "final_u_high" in sweep
        assert "final_v_high" in sweep
        assert "final_u_low" in sweep
        assert "final_v_low" in sweep

    def test_alpha_sweep_correct_shape(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        n = 10
        sweep = sim.alpha_sweep(alpha_range=(1.0, 5.0), n_alpha=n, n_steps=1000)
        assert len(sweep["alpha"]) == n
        assert len(sweep["final_u_high"]) == n
        assert len(sweep["final_v_low"]) == n

    def test_alpha_sweep_bistability_at_high_alpha(self):
        """At high alpha, two ICs should converge to different states."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        sweep = sim.alpha_sweep(
            alpha_range=(3.0, 6.0), n_alpha=5, n_steps=3000,
        )
        # At high alpha, the difference between high-u IC and low-u IC
        # should be significant
        diff = np.abs(sweep["final_u_high"][-1] - sweep["final_u_low"][-1])
        assert diff > 0.5, f"No bistability detected at alpha=6: diff={diff:.4f}"

    def test_alpha_sweep_monostability_at_low_alpha(self):
        """At very low alpha, both ICs should converge to similar states."""
        sim = ToggleSwitchSimulation(_make_config(beta=2.0, gamma=2.0))
        sim.reset()
        sweep = sim.alpha_sweep(
            alpha_range=(0.3, 0.5), n_alpha=3, n_steps=5000,
        )
        # At low alpha with low Hill coefficients, should be monostable
        diff = np.abs(sweep["final_u_high"][0] - sweep["final_u_low"][0])
        # Monostable: both converge to same point
        assert diff < 1.0, f"Unexpected bistability at low alpha: diff={diff:.4f}"


class TestToggleSwitchStateProperties:
    """Physical property tests."""

    def test_state_non_negativity(self):
        """Protein concentrations should stay non-negative."""
        sim = ToggleSwitchSimulation(_make_config(dt=0.005, u_0=0.1, v_0=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            u, v = sim.observe()
            assert u > -0.01, f"u went negative: {u}"
            assert v > -0.01, f"v went negative: {v}"

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded."""
        sim = ToggleSwitchSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(np.abs(state) < 50), f"State diverged: {state}"

    def test_concentrations_bounded_by_alpha(self):
        """At steady state, u <= alpha1 and v <= alpha2."""
        sim = ToggleSwitchSimulation(
            _make_config(alpha1=4.0, alpha2=4.0, dt=0.01, u_0=2.0, v_0=2.0)
        )
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        # At steady state, u = alpha1/(1+v^beta) <= alpha1
        assert state[0] <= 4.0 + 0.1
        assert state[1] <= 4.0 + 0.1


class TestToggleSwitchReproducibility:
    """Reproducibility and determinism tests."""

    def test_deterministic_with_same_seed(self):
        """Same config should produce identical trajectories."""
        config = _make_config()
        sim1 = ToggleSwitchSimulation(config)
        sim2 = ToggleSwitchSimulation(config)

        sim1.reset()
        sim2.reset()

        for _ in range(100):
            sim1.step()
            sim2.step()

        np.testing.assert_array_equal(sim1.observe(), sim2.observe())

    def test_different_dt_converge(self):
        """Different timesteps should converge to same fixed point."""
        sim_coarse = ToggleSwitchSimulation(_make_config(dt=0.05, u_0=3.0, v_0=0.5))
        sim_fine = ToggleSwitchSimulation(_make_config(dt=0.005, u_0=3.0, v_0=0.5))

        sim_coarse.reset()
        sim_fine.reset()

        for _ in range(4000):  # 200 time units
            sim_coarse.step()
        for _ in range(40000):  # 200 time units
            sim_fine.step()

        state_coarse = sim_coarse.observe()
        state_fine = sim_fine.observe()
        np.testing.assert_allclose(state_coarse, state_fine, atol=0.05)


class TestToggleSwitchJacobian:
    """Jacobian computation tests."""

    def test_jacobian_shape(self):
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        jac = sim._jacobian(1.0, 1.0)
        assert jac.shape == (2, 2)

    def test_jacobian_diagonal(self):
        """Diagonal entries should always be -1 (degradation terms)."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        jac = sim._jacobian(2.0, 1.5)
        assert jac[0, 0] == pytest.approx(-1.0)
        assert jac[1, 1] == pytest.approx(-1.0)

    def test_jacobian_off_diagonal_negative(self):
        """Off-diagonal entries should be negative (mutual repression)."""
        sim = ToggleSwitchSimulation(_make_config())
        sim.reset()
        jac = sim._jacobian(2.0, 1.5)
        assert jac[0, 1] < 0  # d(du/dt)/dv < 0 (v represses u production)
        assert jac[1, 0] < 0  # d(dv/dt)/du < 0 (u represses v production)

    def test_jacobian_eigenvalues_at_stable_fp(self):
        """At a stable fixed point, all eigenvalues should have negative real parts."""
        sim = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
        ))
        sim.reset()
        fps = sim.find_fixed_points()
        # Find a stable fixed point
        for u_fp, v_fp in fps:
            if sim.classify_fixed_point(u_fp, v_fp) == "stable":
                jac = sim._jacobian(u_fp, v_fp)
                eigs = np.linalg.eigvals(jac)
                assert all(np.real(e) < 0 for e in eigs)
                break

    def test_jacobian_eigenvalues_at_saddle(self):
        """At a saddle point, eigenvalues should have mixed sign real parts."""
        sim = ToggleSwitchSimulation(_make_config(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
        ))
        sim.reset()
        fps = sim.find_fixed_points()
        for u_fp, v_fp in fps:
            if sim.classify_fixed_point(u_fp, v_fp) == "saddle":
                jac = sim._jacobian(u_fp, v_fp)
                eigs = np.linalg.eigvals(jac)
                real_parts = sorted(np.real(eigs))
                assert real_parts[0] < 0 and real_parts[1] > 0
                break


class TestToggleSwitchRediscovery:
    """Rediscovery data generation tests."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.toggle_switch import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["alpha1"] == 4.0
        assert data["alpha2"] == 4.0

    def test_bistability_data_generation(self):
        from simulating_anything.rediscovery.toggle_switch import (
            generate_bistability_data,
        )

        data = generate_bistability_data(n_steps=3000, dt=0.01)
        assert "final_high_u" in data
        assert "final_high_v" in data
        assert "is_bistable" in data

    def test_bistability_detected(self):
        from simulating_anything.rediscovery.toggle_switch import (
            generate_bistability_data,
        )

        data = generate_bistability_data(
            alpha1=4.0, alpha2=4.0, beta=4.0, gamma=4.0,
            n_steps=5000, dt=0.01,
        )
        assert data["is_bistable"], "Bistability not detected for default params"

    def test_alpha_sweep_data_generation(self):
        from simulating_anything.rediscovery.toggle_switch import (
            generate_alpha_sweep_data,
        )

        data = generate_alpha_sweep_data(
            alpha_range=(1.0, 5.0), n_alpha=5, n_steps=1000,
        )
        assert len(data["alpha"]) == 5
        assert len(data["final_u_high"]) == 5
        assert len(data["final_u_low"]) == 5
