"""Tests for the stochastic genetic toggle switch simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.toggle_switch_stochastic import (
    ToggleSwitchStochasticSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha1: float = 5.0,
    alpha2: float = 5.0,
    beta_hill: float = 2.0,
    gamma_hill: float = 2.0,
    sigma: float = 0.3,
    u_0: float = 0.5,
    v_0: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 5000,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "alpha1": alpha1,
            "alpha2": alpha2,
            "beta_hill": beta_hill,
            "gamma_hill": gamma_hill,
            "sigma": sigma,
            "u_0": u_0,
            "v_0": v_0,
        },
    )


class TestCreation:
    """Tests for simulation object creation and parameter handling."""

    def test_creation_default_parameters(self):
        config = SimulationConfig(
            domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = ToggleSwitchStochasticSimulation(config)
        assert sim.alpha1 == 5.0
        assert sim.alpha2 == 5.0
        assert sim.beta_hill == 2.0
        assert sim.gamma_hill == 2.0
        assert sim.sigma == 0.3
        assert sim.u_0 == 0.5
        assert sim.v_0 == 0.5

    def test_creation_custom_parameters(self):
        config = _make_config(alpha1=8.0, alpha2=6.0, beta_hill=3.0, gamma_hill=4.0, sigma=0.5)
        sim = ToggleSwitchStochasticSimulation(config)
        assert sim.alpha1 == 8.0
        assert sim.alpha2 == 6.0
        assert sim.beta_hill == 3.0
        assert sim.gamma_hill == 4.0
        assert sim.sigma == 0.5

    def test_creation_stores_config(self):
        config = _make_config()
        sim = ToggleSwitchStochasticSimulation(config)
        assert sim.config is config


class TestBasics:
    """Tests for reset, step, observe, run."""

    def test_reset_returns_initial_state(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [0.5, 0.5])

    def test_reset_with_custom_initial(self):
        sim = ToggleSwitchStochasticSimulation(_make_config(u_0=4.0, v_0=0.1))
        state = sim.reset(seed=42)
        np.testing.assert_allclose(state, [4.0, 0.1])

    def test_reset_resets_step_count(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(10):
            sim.step()
        assert sim._step_count == 10
        sim.reset(seed=42)
        assert sim._step_count == 0

    def test_step_advances_state(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        state = sim.reset(seed=42)
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)

    def test_state_shape_is_2(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            state = sim.step()
            assert state.shape == (2,), f"State shape {state.shape} != (2,)"

    def test_run_returns_trajectory_data(self):
        sim = ToggleSwitchStochasticSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states is not None
        assert traj.states.shape == (101, 2)
        assert traj.timestamps is not None
        assert len(traj.timestamps) == 101

    def test_run_trajectory_starts_at_initial_state(self):
        sim = ToggleSwitchStochasticSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        np.testing.assert_allclose(traj.states[0], [0.5, 0.5])


class TestNonNegativity:
    """Tests that concentrations stay non-negative."""

    def test_concentrations_non_negative(self):
        """Protein concentrations should never go below zero."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=1.0, n_steps=10000, seed=42)
        )
        traj = sim.run(n_steps=10000)
        assert np.all(traj.states >= 0.0), (
            f"Negative concentrations found: min_u={np.min(traj.states[:, 0])}, "
            f"min_v={np.min(traj.states[:, 1])}"
        )

    def test_non_negative_with_large_noise(self):
        """Even with very large noise, clamp should prevent negatives."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=3.0, n_steps=5000, u_0=0.1, v_0=0.1, seed=123)
        )
        traj = sim.run(n_steps=5000)
        assert np.all(traj.states >= 0.0)

    def test_non_negative_near_zero_initial(self):
        """Starting near zero should still keep non-negative."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.5, u_0=0.01, v_0=0.01, n_steps=2000, seed=99)
        )
        traj = sim.run(n_steps=2000)
        assert np.all(traj.states >= 0.0)


class TestStochasticity:
    """Tests for stochastic behavior: reproducibility and randomness."""

    def test_different_seeds_different_trajectories(self):
        """Different seeds should produce different stochastic trajectories."""
        config1 = _make_config(sigma=0.3, n_steps=1000, seed=42)
        sim1 = ToggleSwitchStochasticSimulation(config1)
        traj1 = sim1.run(n_steps=1000)

        config2 = _make_config(sigma=0.3, n_steps=1000, seed=99)
        sim2 = ToggleSwitchStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=1000)

        assert not np.allclose(traj1.states, traj2.states), (
            "Different seeds should produce different trajectories"
        )

    def test_same_seed_same_trajectory(self):
        """Same seed should produce identical trajectories (reproducible)."""
        config = _make_config(sigma=0.3, n_steps=500, seed=42)

        sim1 = ToggleSwitchStochasticSimulation(config)
        traj1 = sim1.run(n_steps=500)

        sim2 = ToggleSwitchStochasticSimulation(config)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_zero_noise_is_deterministic(self):
        """With sigma=0, trajectories should be deterministic regardless of seed."""
        config1 = _make_config(sigma=0.0, n_steps=500, seed=42)
        sim1 = ToggleSwitchStochasticSimulation(config1)
        traj1 = sim1.run(n_steps=500)

        config2 = _make_config(sigma=0.0, n_steps=500, seed=99)
        sim2 = ToggleSwitchStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)

    def test_zero_noise_no_randomness(self):
        """Zero noise should produce no stochastic variation."""
        config = _make_config(sigma=0.0, n_steps=1000, seed=42, u_0=4.0, v_0=0.5)
        sim = ToggleSwitchStochasticSimulation(config)
        traj = sim.run(n_steps=1000)

        # Run again from different seed
        config2 = _make_config(sigma=0.0, n_steps=1000, seed=12345, u_0=4.0, v_0=0.5)
        sim2 = ToggleSwitchStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=1000)

        np.testing.assert_allclose(traj.states, traj2.states, atol=1e-12)


class TestBistability:
    """Tests for bistability in the deterministic system."""

    def test_deterministic_has_multiple_fixed_points(self):
        """The deterministic system should have 3 fixed points (2 stable + 1 saddle)."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        fps = sim.find_fixed_points()
        assert len(fps) >= 2, f"Expected at least 2 fixed points, got {len(fps)}"

    def test_two_stable_fixed_points(self):
        """There should be exactly 2 stable fixed points."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        fps = sim.find_fixed_points()
        stable_fps = [fp for fp in fps if sim.classify_fixed_point(*fp) == "stable"]
        assert len(stable_fps) == 2, (
            f"Expected 2 stable fixed points, got {len(stable_fps)}: {stable_fps}"
        )

    def test_stable_fixed_points_are_asymmetric(self):
        """With equal parameters, stable FPs should be (high, low) and (low, high)."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        fps = sim.find_fixed_points()
        stable_fps = [fp for fp in fps if sim.classify_fixed_point(*fp) == "stable"]
        assert len(stable_fps) == 2

        fp1, fp2 = stable_fps
        # One should have u > v, the other v > u (symmetric system)
        assert (fp1[0] > fp1[1]) != (fp2[0] > fp2[1]), (
            f"Stable FPs should be asymmetric: {fp1}, {fp2}"
        )

    def test_saddle_point_exists(self):
        """With equal parameters and high Hill coefficients, a saddle should exist."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        fps = sim.find_fixed_points()
        saddle_fps = [fp for fp in fps if sim.classify_fixed_point(*fp) == "saddle"]
        assert len(saddle_fps) >= 1, (
            f"Expected at least 1 saddle point, got {len(saddle_fps)}"
        )

    def test_deterministic_converges_to_high_u(self):
        """Starting near high-u stable FP should converge there deterministically."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.0, u_0=4.5, v_0=0.1, n_steps=5000)
        )
        traj = sim.run(n_steps=5000)
        final = traj.states[-1]
        assert final[0] > final[1], (
            f"Expected high-u state, got u={final[0]:.3f}, v={final[1]:.3f}"
        )

    def test_deterministic_converges_to_high_v(self):
        """Starting near high-v stable FP should converge there deterministically."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.0, u_0=0.1, v_0=4.5, n_steps=5000)
        )
        traj = sim.run(n_steps=5000)
        final = traj.states[-1]
        assert final[1] > final[0], (
            f"Expected high-v state, got u={final[0]:.3f}, v={final[1]:.3f}"
        )


class TestSwitching:
    """Tests for noise-induced switching between states."""

    def test_high_noise_induces_switching(self):
        """With high noise, the system should switch between states."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=1.5, n_steps=100000, u_0=4.0, v_0=0.5, seed=42)
        )
        traj = sim.run(n_steps=100000)
        n_switches = sim.count_switches(traj.states[:, 0])
        assert n_switches > 0, (
            "Expected switching events with sigma=1.5"
        )

    def test_low_noise_few_switches(self):
        """With very low noise, switching should be rare."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.05, n_steps=20000, u_0=4.0, v_0=0.5, seed=42)
        )
        traj = sim.run(n_steps=20000)
        n_switches = sim.count_switches(traj.states[:, 0])
        # Low noise: expect very few or no switches in short time
        assert n_switches < 5, (
            f"Expected few switches with sigma=0.05, got {n_switches}"
        )

    def test_zero_noise_no_switching(self):
        """With zero noise, no switching should occur from a stable state."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.0, n_steps=10000, u_0=4.0, v_0=0.5)
        )
        traj = sim.run(n_steps=10000)
        n_switches = sim.count_switches(traj.states[:, 0])
        assert n_switches == 0, (
            f"Expected 0 switches with sigma=0, got {n_switches}"
        )

    def test_count_switches_method(self):
        """count_switches should detect threshold crossings correctly."""
        sim = ToggleSwitchStochasticSimulation(_make_config())
        # Synthetic trace crossing threshold 2.5 at known points
        u_trace = np.array([1.0, 1.5, 3.0, 3.5, 2.0, 1.0, 3.5, 4.0])
        n = sim.count_switches(u_trace, threshold=2.5)
        # Crossings: 1.5->3.0 (up), 3.5->2.0 (down), 1.0->3.5 (up) = 3
        assert n == 3, f"Expected 3 switches, got {n}"


class TestBoundedness:
    """Tests that trajectories remain bounded."""

    def test_trajectory_bounded(self):
        """State should not diverge for reasonable parameters."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.3, n_steps=10000, seed=42)
        )
        traj = sim.run(n_steps=10000)
        # Concentrations should stay within reasonable bounds
        assert np.all(traj.states[:, 0] < 20), "u diverged"
        assert np.all(traj.states[:, 1] < 20), "v diverged"

    def test_no_nans(self):
        """State should not contain NaN values."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.5, n_steps=5000, seed=42)
        )
        traj = sim.run(n_steps=5000)
        assert not np.any(np.isnan(traj.states)), "NaN in trajectory"

    def test_all_finite(self):
        """All values should be finite."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=1.0, n_steps=5000, seed=42)
        )
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states)), "Non-finite values in trajectory"


class TestDeterministicDynamics:
    """Tests for the deterministic part of the dynamics."""

    def test_derivatives_at_alpha_high_u(self):
        """At u=alpha1, v=0, du/dt should be alpha1-alpha1 = 0."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        y = np.array([5.0, 0.0])
        dy = sim._derivatives(y)
        # du = alpha1/(1+0^beta) - u = 5 - 5 = 0
        # dv = alpha2/(1+5^gamma) - 0 = 5/(1+25) = 5/26
        np.testing.assert_allclose(dy[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dy[1], 5.0 / 26.0, atol=1e-10)

    def test_derivatives_at_origin(self):
        """At u=0, v=0, both derivatives should be positive (alpha push)."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        y = np.array([0.0, 0.0])
        dy = sim._derivatives(y)
        # du = alpha1/(1+0) - 0 = 5.0
        # dv = alpha2/(1+0) - 0 = 5.0
        np.testing.assert_allclose(dy, [5.0, 5.0], atol=1e-10)

    def test_derivatives_hill_repression(self):
        """High v should strongly suppress du/dt."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        y_low_v = np.array([2.0, 0.1])
        y_high_v = np.array([2.0, 4.0])
        dy_low = sim._derivatives(y_low_v)
        dy_high = sim._derivatives(y_high_v)
        # Higher v => lower production rate for u => lower du/dt
        assert dy_low[0] > dy_high[0], (
            "Higher v should reduce du/dt via Hill repression"
        )

    def test_jacobian_diagonal(self):
        """Jacobian diagonal entries should be -1 (degradation)."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        jac = sim._jacobian(2.0, 2.0)
        assert jac[0, 0] == -1.0
        assert jac[1, 1] == -1.0

    def test_jacobian_off_diagonal_negative(self):
        """Off-diagonal entries should be negative (mutual repression)."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        jac = sim._jacobian(2.0, 2.0)
        assert jac[0, 1] < 0, "J[0,1] should be negative (v represses u)"
        assert jac[1, 0] < 0, "J[1,0] should be negative (u represses v)"


class TestNullclines:
    """Tests for nullcline computation."""

    def test_nullclines_shape(self):
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        nc = sim.compute_nullclines(n_points=100)
        assert len(nc["u_null_u"]) == 100
        assert len(nc["u_null_v"]) == 100
        assert len(nc["v_null_u"]) == 100
        assert len(nc["v_null_v"]) == 100

    def test_u_nullcline_at_v_zero(self):
        """At v=0, u-nullcline gives u = alpha1."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        nc = sim.compute_nullclines(u_range=(0.0, 6.0), n_points=200)
        # First entry corresponds to v=0
        np.testing.assert_allclose(nc["u_null_u"][0], 5.0, atol=0.1)

    def test_v_nullcline_at_u_zero(self):
        """At u=0, v-nullcline gives v = alpha2."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        nc = sim.compute_nullclines(u_range=(0.0, 6.0), n_points=200)
        # First entry corresponds to u=0
        np.testing.assert_allclose(nc["v_null_v"][0], 5.0, atol=0.1)

    def test_nullclines_non_negative(self):
        """Nullcline values should be non-negative."""
        sim = ToggleSwitchStochasticSimulation(_make_config(sigma=0.0))
        nc = sim.compute_nullclines()
        assert np.all(nc["u_null_u"] >= 0)
        assert np.all(nc["v_null_v"] >= 0)


class TestParameterSweep:
    """Tests for parameter variation effects."""

    def test_higher_alpha_larger_fixed_points(self):
        """Increasing alpha should move stable FPs to higher concentrations."""
        sim_low = ToggleSwitchStochasticSimulation(
            _make_config(alpha1=3.0, alpha2=3.0, sigma=0.0)
        )
        fps_low = sim_low.find_fixed_points()
        stable_low = [fp for fp in fps_low if sim_low.classify_fixed_point(*fp) == "stable"]

        sim_high = ToggleSwitchStochasticSimulation(
            _make_config(alpha1=8.0, alpha2=8.0, sigma=0.0)
        )
        fps_high = sim_high.find_fixed_points()
        stable_high = [fp for fp in fps_high if sim_high.classify_fixed_point(*fp) == "stable"]

        if len(stable_low) >= 2 and len(stable_high) >= 2:
            max_u_low = max(fp[0] for fp in stable_low)
            max_u_high = max(fp[0] for fp in stable_high)
            assert max_u_high > max_u_low, (
                "Higher alpha should produce larger high-u fixed point"
            )

    def test_sigma_zero_vs_nonzero_variance(self):
        """Nonzero sigma should produce more variance in the ensemble."""
        n_runs = 10
        finals_det = []
        finals_stoch = []

        for i in range(n_runs):
            config_det = _make_config(sigma=0.0, n_steps=3000, seed=i, u_0=4.0, v_0=0.5)
            sim_det = ToggleSwitchStochasticSimulation(config_det)
            traj_det = sim_det.run(n_steps=3000)
            finals_det.append(traj_det.states[-1, 0])

            config_stoch = _make_config(sigma=0.5, n_steps=3000, seed=i, u_0=4.0, v_0=0.5)
            sim_stoch = ToggleSwitchStochasticSimulation(config_stoch)
            traj_stoch = sim_stoch.run(n_steps=3000)
            finals_stoch.append(traj_stoch.states[-1, 0])

        var_det = np.var(finals_det)
        var_stoch = np.var(finals_stoch)
        # Deterministic ensemble should have near-zero variance
        assert var_det < 1e-6, f"Deterministic variance too high: {var_det}"
        # Stochastic ensemble should have some variance
        assert var_stoch > var_det, (
            f"Stochastic variance ({var_stoch}) should exceed deterministic ({var_det})"
        )


class TestDwellTimes:
    """Tests for dwell time analysis."""

    def test_dwell_times_returns_dict(self):
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=1.0, n_steps=10000, seed=42)
        )
        traj = sim.run(n_steps=10000)
        dwell = sim.dwell_times(traj.states[:, 0])
        assert "high_state_times" in dwell
        assert "low_state_times" in dwell

    def test_dwell_times_positive(self):
        """All dwell times should be positive."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=1.0, n_steps=50000, seed=42)
        )
        traj = sim.run(n_steps=50000)
        dwell = sim.dwell_times(traj.states[:, 0])
        if len(dwell["high_state_times"]) > 0:
            assert np.all(dwell["high_state_times"] > 0)
        if len(dwell["low_state_times"]) > 0:
            assert np.all(dwell["low_state_times"] > 0)

    def test_dwell_times_empty_for_zero_noise(self):
        """Deterministic trajectory from stable state should have no switches => no dwell pairs."""
        sim = ToggleSwitchStochasticSimulation(
            _make_config(sigma=0.0, n_steps=5000, u_0=4.0, v_0=0.5)
        )
        traj = sim.run(n_steps=5000)
        dwell = sim.dwell_times(traj.states[:, 0])
        # Starting in high-u basin with no noise: no switches
        n_total = len(dwell["high_state_times"]) + len(dwell["low_state_times"])
        assert n_total == 0, (
            f"Expected 0 dwell time entries with sigma=0, got {n_total}"
        )


class TestWhichState:
    """Tests for state identification."""

    def test_which_state_high_u(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        sim.reset(seed=42)
        sim._state = np.array([4.0, 0.5])
        assert sim.which_state() == "high_u"

    def test_which_state_high_v(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        sim.reset(seed=42)
        sim._state = np.array([0.5, 4.0])
        assert sim.which_state() == "high_v"

    def test_which_state_boundary(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        sim.reset(seed=42)
        sim._state = np.array([2.5, 2.5])
        assert sim.which_state() == "boundary"

    def test_which_state_with_explicit_state(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        assert sim.which_state(np.array([4.0, 1.0])) == "high_u"
        assert sim.which_state(np.array([1.0, 4.0])) == "high_v"


class TestEnsembleStatistics:
    """Tests for ensemble-level properties."""

    def test_ensemble_explores_both_basins(self):
        """With moderate noise from the boundary, ensemble should visit both basins."""
        n_runs = 20
        final_high_u = 0
        final_high_v = 0

        for i in range(n_runs):
            config = _make_config(
                sigma=0.8, n_steps=10000, seed=i * 7 + 100,
                u_0=0.5, v_0=0.5,
            )
            sim = ToggleSwitchStochasticSimulation(config)
            traj = sim.run(n_steps=10000)
            final = traj.states[-1]
            if final[0] > final[1]:
                final_high_u += 1
            else:
                final_high_v += 1

        # With symmetric parameters and noise, both basins should be populated
        assert final_high_u > 0, "No trajectories ended in high-u basin"
        assert final_high_v > 0, "No trajectories ended in high-v basin"

    def test_ensemble_mean_near_symmetric(self):
        """For symmetric parameters, ensemble mean should be roughly symmetric."""
        n_runs = 30
        finals = []

        for i in range(n_runs):
            config = _make_config(
                sigma=0.8, n_steps=10000, seed=i * 11 + 200,
                u_0=2.5, v_0=2.5,
            )
            sim = ToggleSwitchStochasticSimulation(config)
            traj = sim.run(n_steps=10000)
            finals.append(traj.states[-1].copy())

        finals_arr = np.array(finals)
        mean_u = np.mean(finals_arr[:, 0])
        mean_v = np.mean(finals_arr[:, 1])
        # Symmetric parameters should give similar mean u and v
        assert abs(mean_u - mean_v) < 2.0, (
            f"Ensemble means should be roughly symmetric: u={mean_u:.2f}, v={mean_v:.2f}"
        )


class TestTrajectoryData:
    """Tests for trajectory data structure."""

    def test_trajectory_parameters(self):
        sim = ToggleSwitchStochasticSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert "alpha1" in traj.parameters
        assert "alpha2" in traj.parameters
        assert "sigma" in traj.parameters

    def test_trajectory_timestamps_monotonic(self):
        sim = ToggleSwitchStochasticSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        diffs = np.diff(traj.timestamps)
        assert np.all(diffs > 0), "Timestamps should be strictly increasing"

    def test_trajectory_timestamps_spacing(self):
        """Timestamps should be spaced by dt."""
        dt = 0.01
        sim = ToggleSwitchStochasticSimulation(_make_config(dt=dt, n_steps=100))
        traj = sim.run(n_steps=100)
        diffs = np.diff(traj.timestamps)
        np.testing.assert_allclose(diffs, dt, atol=1e-12)


class TestRediscovery:
    """Tests for rediscovery data generation and runner."""

    def test_switching_rate_data_generation(self):
        from simulating_anything.rediscovery.toggle_switch_stochastic import (
            generate_switching_rate_data,
        )
        data = generate_switching_rate_data(
            n_sigma=3,
            sigma_min=0.1,
            sigma_max=1.0,
            n_steps=2000,
            n_trials=1,
            seed=42,
        )
        assert len(data["sigma"]) == 3
        assert len(data["switch_count"]) == 3
        assert np.all(data["switch_count"] >= 0)

    def test_switching_rate_sigma_range(self):
        from simulating_anything.rediscovery.toggle_switch_stochastic import (
            generate_switching_rate_data,
        )
        data = generate_switching_rate_data(
            n_sigma=3,
            sigma_min=0.2,
            sigma_max=1.5,
            n_steps=1000,
            n_trials=1,
            seed=42,
        )
        np.testing.assert_allclose(data["sigma"][0], 0.2)
        np.testing.assert_allclose(data["sigma"][-1], 1.5)

    def test_ensemble_comparison(self):
        from simulating_anything.rediscovery.toggle_switch_stochastic import (
            generate_ensemble_comparison,
        )
        results = generate_ensemble_comparison(
            n_ensemble=5,
            n_steps=2000,
            dt=0.01,
            sigma=0.5,
            seed=42,
        )
        assert "fixed_points" in results
        assert "n_high_u" in results
        assert "n_high_v" in results
        assert results["n_high_u"] + results["n_high_v"] == 5

    def test_basin_map(self):
        from simulating_anything.rediscovery.toggle_switch_stochastic import (
            generate_basin_map,
        )
        data = generate_basin_map(
            n_grid=5,
            n_steps=1000,
            dt=0.01,
        )
        assert data["basin"].shape == (5, 5)
        assert "nullclines" in data

    def test_rediscovery_runner_returns_dict(self):
        from simulating_anything.rediscovery.toggle_switch_stochastic import (
            run_toggle_switch_stochastic_rediscovery,
        )
        results = run_toggle_switch_stochastic_rediscovery(
            output_dir="output/rediscovery/toggle_switch_stochastic_test",
            n_sigma=3,
            n_trials=1,
        )
        assert isinstance(results, dict)
        assert results["domain"] == "toggle_switch_stochastic"
        assert "targets" in results
        assert "switching_rate" in results
        assert "ensemble" in results
        assert "basins" in results
