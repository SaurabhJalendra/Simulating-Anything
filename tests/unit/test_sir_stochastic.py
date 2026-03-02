"""Tests for the stochastic SIR epidemic model with demographic noise."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sir_stochastic import SIRStochasticSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    beta: float = 0.5,
    gamma: float = 0.1,
    N0: float = 1000.0,
    sigma: float = 0.1,
    I_0: float = 10.0,
    dt: float = 0.1,
    n_steps: int = 2000,
    seed: int = 42,
) -> SimulationConfig:
    """Create a SimulationConfig for SIR stochastic with optional overrides."""
    return SimulationConfig(
        domain=Domain.SIR_STOCHASTIC,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "beta": beta,
            "gamma": gamma,
            "N0": N0,
            "sigma": sigma,
            "I_0": I_0,
        },
    )


class TestSIRStochasticCreation:
    """Tests for simulation object creation and parameter handling."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SIRStochasticSimulation(_make_config())
        assert sim.beta == 0.5
        assert sim.gamma == 0.1
        assert sim.N0 == 1000.0
        assert sim.sigma == 0.1
        assert sim.I_0 == 10.0

    def test_default_parameters(self):
        """Default parameters should be set when not provided."""
        config = SimulationConfig(
            domain=Domain.SIR_STOCHASTIC,
            dt=0.1,
            n_steps=100,
            parameters={},
        )
        sim = SIRStochasticSimulation(config)
        assert sim.beta == 0.5
        assert sim.gamma == 0.1
        assert sim.N0 == 1000.0
        assert sim.sigma == 0.1
        assert sim.I_0 == 10.0

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SIRStochasticSimulation(_make_config(
            beta=0.8, gamma=0.2, N0=5000.0, sigma=0.5, I_0=50.0,
        ))
        assert sim.beta == 0.8
        assert sim.gamma == 0.2
        assert sim.N0 == 5000.0
        assert sim.sigma == 0.5
        assert sim.I_0 == 50.0

    def test_initial_state_shape(self):
        """State vector should be [S, I, R] with 3 components."""
        sim = SIRStochasticSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state.shape == (3,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SIRStochasticSimulation(_make_config(N0=1000.0, I_0=10.0))
        state = sim.reset(seed=42)
        np.testing.assert_allclose(state, [990.0, 10.0, 0.0])


class TestSIRStochasticBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = SIRStochasticSimulation(_make_config())
        state0 = sim.reset(seed=42).copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SIRStochasticSimulation(_make_config())
        sim.reset(seed=42)
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_run_returns_trajectory_data(self):
        """run() should produce a TrajectoryData object."""
        sim = SIRStochasticSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_trajectory_starts_at_initial(self):
        """First state in trajectory should match initial conditions."""
        sim = SIRStochasticSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        np.testing.assert_allclose(traj.states[0], [990.0, 10.0, 0.0])

    def test_no_nan_in_trajectory(self):
        """No NaN values should appear in trajectory."""
        sim = SIRStochasticSimulation(_make_config(n_steps=2000))
        traj = sim.run(n_steps=2000)
        assert not np.any(np.isnan(traj.states)), "NaN in trajectory"

    def test_long_run_stability(self):
        """No NaN or Inf after many steps."""
        sim = SIRStochasticSimulation(_make_config(dt=0.1, n_steps=500))
        sim.reset(seed=42)
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"Non-finite values: {state}"


class TestSIRStochasticConservation:
    """Approximate conservation S+I+R ~ N."""

    def test_initial_total_population(self):
        """S + I + R should equal N0 at initialization."""
        sim = SIRStochasticSimulation(_make_config(N0=1000.0))
        state = sim.reset(seed=42)
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_approximate_conservation_small_sigma(self):
        """With small noise, S+I+R should stay close to N."""
        sim = SIRStochasticSimulation(_make_config(sigma=0.01, n_steps=500))
        sim.reset(seed=42)
        for _ in range(500):
            state = sim.step()
            total = state.sum()
            # With very small noise, conservation should be approximate
            assert abs(total - 1000.0) < 100.0, (
                f"Total population drifted too far: {total}"
            )

    def test_conservation_zero_noise(self):
        """With zero noise, S+I+R should be well conserved."""
        sim = SIRStochasticSimulation(_make_config(sigma=0.0, n_steps=500))
        sim.reset(seed=42)
        for _ in range(500):
            state = sim.step()
            # Euler-Maruyama with sigma=0 is just forward Euler, not exact
            # but should be close for small dt=0.1
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=5.0,
                err_msg="Conservation violated with zero noise",
            )


class TestSIRStochasticNonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_all_compartments(self):
        """S, I, R should all remain >= 0."""
        sim = SIRStochasticSimulation(_make_config(dt=0.1, n_steps=2000))
        sim.reset(seed=42)
        for _ in range(2000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"
            assert state[2] >= 0, f"R became negative: {state[2]}"

    def test_non_negative_large_noise(self):
        """Non-negativity should hold even with large noise."""
        sim = SIRStochasticSimulation(_make_config(sigma=1.0, n_steps=1000))
        sim.reset(seed=42)
        for _ in range(1000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"

    def test_non_negative_high_beta(self):
        """Non-negativity with high transmission rate."""
        sim = SIRStochasticSimulation(_make_config(beta=2.0, sigma=0.2, n_steps=1000))
        sim.reset(seed=42)
        for _ in range(1000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


class TestSIRStochasticStochasticity:
    """Tests for stochastic behavior."""

    def test_different_seeds_different_trajectories(self):
        """Different seeds should produce different trajectories."""
        config1 = _make_config(sigma=0.1, n_steps=500, seed=42)
        sim1 = SIRStochasticSimulation(config1)
        traj1 = sim1.run(n_steps=500)

        config2 = _make_config(sigma=0.1, n_steps=500, seed=99)
        sim2 = SIRStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=500)

        assert not np.allclose(traj1.states, traj2.states), (
            "Different seeds should produce different trajectories"
        )

    def test_same_seed_same_trajectory(self):
        """Same seed should produce identical trajectories (reproducible)."""
        config = _make_config(sigma=0.1, n_steps=500, seed=42)

        sim1 = SIRStochasticSimulation(config)
        traj1 = sim1.run(n_steps=500)

        sim2 = SIRStochasticSimulation(config)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_zero_noise_deterministic_across_seeds(self):
        """With sigma=0, trajectories should be the same regardless of seed."""
        config1 = _make_config(sigma=0.0, n_steps=500, seed=42)
        sim1 = SIRStochasticSimulation(config1)
        traj1 = sim1.run(n_steps=500)

        config2 = _make_config(sigma=0.0, n_steps=500, seed=99)
        sim2 = SIRStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)

    def test_larger_sigma_more_variance(self):
        """Larger sigma should produce more variance across runs."""
        n_runs = 20
        n_steps = 300

        # Small sigma
        config_small = _make_config(sigma=0.01, n_steps=n_steps)
        sim_small = SIRStochasticSimulation(config_small)
        ens_small = sim_small.run_ensemble(n_runs=n_runs, n_steps=n_steps, base_seed=0)
        var_small = np.var(ens_small[:, n_steps // 2, 1])

        # Large sigma
        config_large = _make_config(sigma=0.5, n_steps=n_steps)
        sim_large = SIRStochasticSimulation(config_large)
        ens_large = sim_large.run_ensemble(n_runs=n_runs, n_steps=n_steps, base_seed=0)
        var_large = np.var(ens_large[:, n_steps // 2, 1])

        assert var_large > var_small, (
            f"Larger sigma should give more variance: {var_large} vs {var_small}"
        )


class TestSIRStochasticResetDeterminism:
    """Tests for reset and reproducibility."""

    def test_reset_with_same_seed(self):
        """Resetting with the same seed should produce the same trajectory."""
        sim = SIRStochasticSimulation(_make_config(sigma=0.1))
        sim.reset(seed=42)
        states1 = []
        for _ in range(100):
            states1.append(sim.step().copy())

        sim.reset(seed=42)
        states2 = []
        for _ in range(100):
            states2.append(sim.step().copy())

        np.testing.assert_array_equal(states1, states2)

    def test_reset_clears_step_count(self):
        """Reset should clear the step count."""
        sim = SIRStochasticSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        assert sim._step_count == 50
        sim.reset(seed=42)
        assert sim._step_count == 0

    def test_reset_returns_initial_state(self):
        """Reset should return the initial state."""
        sim = SIRStochasticSimulation(_make_config(N0=1000.0, I_0=10.0))
        state = sim.reset(seed=42)
        np.testing.assert_allclose(state, [990.0, 10.0, 0.0])


class TestSIRStochasticEpidemicDynamics:
    """Tests for correct epidemic behavior."""

    def test_epidemic_occurs_high_r0(self):
        """With high R0, an epidemic should occur (peak I rises)."""
        sim = SIRStochasticSimulation(_make_config(
            beta=0.5, gamma=0.1, sigma=0.05, n_steps=5000,
        ))
        sim.reset(seed=42)
        max_I = 0.0
        for _ in range(5000):
            state = sim.step()
            max_I = max(max_I, state[1])
        # R0 = 5.0, should have significant epidemic
        assert max_I > 50.0, f"Peak I too low for R0=5: {max_I}"

    def test_no_epidemic_low_r0(self):
        """With R0 < 1, infected should generally decline."""
        sim = SIRStochasticSimulation(_make_config(
            beta=0.05, gamma=0.1, sigma=0.01, n_steps=2000,
        ))
        sim.reset(seed=42)
        initial_I = sim.observe()[1]
        for _ in range(2000):
            sim.step()
        final_I = sim.observe()[1]
        assert final_I < initial_I, (
            f"I should decrease when R0 < 1: final={final_I}, initial={initial_I}"
        )

    def test_r0_property(self):
        """R0 should equal beta / gamma."""
        sim = SIRStochasticSimulation(_make_config(beta=0.5, gamma=0.1))
        assert sim.R0 == pytest.approx(5.0)

    def test_r0_varies_correctly(self):
        """R0 should change with beta and gamma."""
        sim = SIRStochasticSimulation(_make_config(beta=0.3, gamma=0.15))
        assert sim.R0 == pytest.approx(2.0)


class TestSIRStochasticDeterministicComparison:
    """Tests comparing stochastic and deterministic trajectories."""

    def test_run_deterministic(self):
        """Deterministic trajectory should be produced without noise."""
        sim = SIRStochasticSimulation(_make_config())
        det_traj = sim.run_deterministic(n_steps=100)
        assert det_traj.shape == (101, 3)
        # Should be deterministic (no noise)
        det_traj2 = sim.run_deterministic(n_steps=100)
        np.testing.assert_allclose(det_traj, det_traj2)

    def test_deterministic_conservation(self):
        """Deterministic trajectory should conserve population well (RK4)."""
        sim = SIRStochasticSimulation(_make_config())
        det_traj = sim.run_deterministic(n_steps=2000)
        totals = det_traj.sum(axis=1)
        np.testing.assert_allclose(totals, 1000.0, atol=1e-4)

    def test_deterministic_non_negative(self):
        """Deterministic trajectory should remain non-negative."""
        sim = SIRStochasticSimulation(_make_config())
        det_traj = sim.run_deterministic(n_steps=2000)
        assert np.all(det_traj >= 0), "Negative values in deterministic trajectory"

    def test_ensemble_mean_near_deterministic(self):
        """Ensemble mean should approximate deterministic trajectory."""
        config = _make_config(sigma=0.05, n_steps=500)
        sim = SIRStochasticSimulation(config)

        det_traj = sim.run_deterministic(n_steps=500)
        ensemble = sim.run_ensemble(n_runs=50, n_steps=500, base_seed=0)
        ens_mean = np.mean(ensemble, axis=0)

        # Relative error at midpoint for each compartment
        mid = 250
        for j in range(3):
            if det_traj[mid, j] > 1.0:
                rel_error = abs(ens_mean[mid, j] - det_traj[mid, j]) / det_traj[mid, j]
                assert rel_error < 0.3, (
                    f"Ensemble mean too far from deterministic at midpoint: "
                    f"compartment {j}, rel_error={rel_error:.4f}"
                )


class TestSIRStochasticEnsemble:
    """Tests for the ensemble runner."""

    def test_ensemble_shape(self):
        """Ensemble should have correct shape."""
        sim = SIRStochasticSimulation(_make_config(n_steps=100))
        ensemble = sim.run_ensemble(n_runs=5, n_steps=100, base_seed=0)
        assert ensemble.shape == (5, 101, 3)

    def test_ensemble_different_runs(self):
        """Different runs in ensemble should differ (different seeds)."""
        sim = SIRStochasticSimulation(_make_config(sigma=0.1, n_steps=200))
        ensemble = sim.run_ensemble(n_runs=5, n_steps=200, base_seed=0)
        # Check that at least two runs are different at the end
        assert not np.allclose(ensemble[0, -1], ensemble[1, -1]), (
            "Ensemble runs should differ"
        )

    def test_ensemble_initial_conditions(self):
        """All ensemble runs should start from the same initial condition."""
        sim = SIRStochasticSimulation(_make_config(n_steps=100))
        ensemble = sim.run_ensemble(n_runs=10, n_steps=100, base_seed=0)
        for i in range(10):
            np.testing.assert_allclose(ensemble[i, 0], [990.0, 10.0, 0.0])

    def test_ensemble_non_negative(self):
        """All ensemble values should be non-negative."""
        sim = SIRStochasticSimulation(_make_config(sigma=0.3, n_steps=500))
        ensemble = sim.run_ensemble(n_runs=10, n_steps=500, base_seed=0)
        assert np.all(ensemble >= 0), "Negative values in ensemble"


class TestSIRStochasticDerivatives:
    """Tests for the deterministic derivatives function."""

    def test_derivatives_shape(self):
        """Derivatives should return array of shape (3,)."""
        sim = SIRStochasticSimulation(_make_config())
        y = np.array([990.0, 10.0, 0.0])
        dy = sim._derivatives(y)
        assert dy.shape == (3,)

    def test_derivatives_zero_infected(self):
        """With I=0, all derivatives should be 0 (no force of infection)."""
        sim = SIRStochasticSimulation(_make_config())
        y = np.array([1000.0, 0.0, 0.0])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-12)

    def test_derivatives_conservation(self):
        """Sum of derivatives should be 0 (population conservation)."""
        sim = SIRStochasticSimulation(_make_config())
        y = np.array([500.0, 200.0, 300.0])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy.sum(), 0.0, atol=1e-10)

    def test_derivatives_signs(self):
        """During an epidemic: dS < 0, dR > 0."""
        sim = SIRStochasticSimulation(_make_config(beta=0.5, gamma=0.1))
        y = np.array([800.0, 150.0, 50.0])
        dy = sim._derivatives(y)
        assert dy[0] < 0, "dS should be negative during epidemic"
        assert dy[2] > 0, "dR should be positive during epidemic"

    def test_derivatives_zero_population(self):
        """With N=0, derivatives should be zero (no division by zero)."""
        sim = SIRStochasticSimulation(_make_config())
        y = np.array([0.0, 0.0, 0.0])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0])


class TestSIRStochasticParameterSweeps:
    """Tests for behavior under parameter variations."""

    def test_higher_beta_faster_epidemic(self):
        """Higher beta should lead to earlier epidemic peak."""
        peaks = []
        for beta in [0.3, 0.5, 0.8]:
            sim = SIRStochasticSimulation(
                _make_config(beta=beta, sigma=0.01, n_steps=3000)
            )
            sim.reset(seed=42)
            max_I = 0
            peak_step = 0
            for step in range(3000):
                state = sim.step()
                if state[1] > max_I:
                    max_I = state[1]
                    peak_step = step
            peaks.append(peak_step)
        # Higher beta should mean earlier peak
        assert peaks[2] < peaks[0], (
            f"Higher beta should give earlier peak: {peaks}"
        )

    def test_higher_gamma_faster_recovery(self):
        """Higher gamma should lead to faster epidemic decline."""
        final_Is = []
        for gamma in [0.05, 0.1, 0.3]:
            sim = SIRStochasticSimulation(
                _make_config(beta=0.5, gamma=gamma, sigma=0.01, n_steps=2000)
            )
            sim.reset(seed=42)
            for _ in range(2000):
                sim.step()
            final_Is.append(sim.observe()[1])
        # Higher gamma should mean lower final I (faster recovery)
        assert final_Is[2] <= final_Is[0] + 1.0, (
            f"Higher gamma should give faster recovery: {final_Is}"
        )

    def test_different_population_sizes(self):
        """Simulation should work with different population sizes."""
        for N0 in [100.0, 1000.0, 10000.0]:
            I_0 = max(1.0, N0 * 0.01)
            sim = SIRStochasticSimulation(
                _make_config(N0=N0, I_0=I_0, sigma=0.1, n_steps=500)
            )
            traj = sim.run(n_steps=500)
            assert not np.any(np.isnan(traj.states)), f"NaN with N0={N0}"
            assert np.all(traj.states >= 0), f"Negative values with N0={N0}"

    def test_sigma_zero_matches_euler(self):
        """With sigma=0, Euler-Maruyama reduces to forward Euler."""
        sim = SIRStochasticSimulation(_make_config(sigma=0.0, n_steps=100))
        traj = sim.run(n_steps=100)
        # Forward Euler: population should be approximately conserved
        totals = traj.states.sum(axis=1)
        # Not as precise as RK4 but should be reasonable
        np.testing.assert_allclose(totals, 1000.0, atol=10.0)


class TestSIRStochasticNoiseScaling:
    """Tests verifying noise scales with sqrt of transition rates."""

    def test_noise_proportional_to_sigma(self):
        """Trajectory variance should increase with sigma."""
        variances = []
        for sigma in [0.01, 0.1, 0.5]:
            config = _make_config(sigma=sigma, n_steps=300)
            sim = SIRStochasticSimulation(config)
            ensemble = sim.run_ensemble(n_runs=20, n_steps=300, base_seed=0)
            var_I = np.var(ensemble[:, 150, 1])
            variances.append(var_I)
        # Variance should increase monotonically with sigma
        assert variances[1] > variances[0], (
            f"Variance should increase with sigma: {variances}"
        )
        assert variances[2] > variances[1], (
            f"Variance should increase with sigma: {variances}"
        )

    def test_zero_sigma_zero_variance(self):
        """With sigma=0, ensemble should have zero variance."""
        config = _make_config(sigma=0.0, n_steps=200)
        sim = SIRStochasticSimulation(config)
        ensemble = sim.run_ensemble(n_runs=5, n_steps=200, base_seed=0)
        # All runs identical when sigma=0
        var = np.var(ensemble[:, -1, 1])
        np.testing.assert_allclose(var, 0.0, atol=1e-20)


class TestSIRStochasticRediscoveryData:
    """Tests for rediscovery data generation functions."""

    def test_ensemble_comparison_data(self):
        """Ensemble comparison data should produce valid results."""
        from simulating_anything.rediscovery.sir_stochastic import (
            generate_ensemble_comparison_data,
        )
        data = generate_ensemble_comparison_data(
            n_runs=5, n_steps=200, dt=0.1,
        )
        assert "deterministic" in data
        assert "ensemble_mean" in data
        assert "ensemble_std" in data
        assert "mae" in data
        assert "peak_rel_error" in data
        assert data["deterministic"].shape == (201, 3)
        assert data["ensemble_mean"].shape == (201, 3)
        assert data["mae"] >= 0

    def test_variance_scaling_data(self):
        """Variance scaling data should produce valid arrays."""
        from simulating_anything.rediscovery.sir_stochastic import (
            generate_variance_scaling_data,
        )
        data = generate_variance_scaling_data(
            sigma_values=np.array([0.01, 0.1, 0.5]),
            N_values=np.array([100.0, 1000.0]),
            n_runs=5,
            n_steps=100,
            dt=0.1,
        )
        assert len(data["sigma_values"]) == 3
        assert len(data["sigma_variances"]) == 3
        assert len(data["N_values"]) == 2
        assert len(data["n_variances"]) == 2
        assert np.all(data["sigma_variances"] >= 0)
        assert np.all(data["n_variances"] >= 0)

    def test_extinction_data(self):
        """Extinction data should produce valid arrays."""
        from simulating_anything.rediscovery.sir_stochastic import (
            generate_extinction_data,
        )
        data = generate_extinction_data(
            R0_values=np.array([0.5, 1.0, 3.0]),
            n_runs=5,
            n_steps=500,
            dt=0.1,
        )
        assert len(data["R0_values"]) == 3
        assert len(data["extinction_probs"]) == 3
        assert len(data["mean_peak_I"]) == 3
        # Extinction probs should be between 0 and 1
        assert np.all(data["extinction_probs"] >= 0)
        assert np.all(data["extinction_probs"] <= 1)

    def test_extinction_higher_at_low_r0(self):
        """Extinction probability should be higher for low R0."""
        from simulating_anything.rediscovery.sir_stochastic import (
            generate_extinction_data,
        )
        data = generate_extinction_data(
            R0_values=np.array([0.5, 5.0]),
            n_runs=20,
            n_steps=1000,
            dt=0.1,
            sigma=0.2,
        )
        # R0=0.5 should have higher extinction prob than R0=5.0
        assert data["extinction_probs"][0] >= data["extinction_probs"][1], (
            f"Low R0 should have higher extinction: "
            f"P_ext(R0=0.5)={data['extinction_probs'][0]}, "
            f"P_ext(R0=5.0)={data['extinction_probs'][1]}"
        )

    def test_rediscovery_runner_returns_dict(self):
        """The full rediscovery runner should return a valid results dict."""
        from simulating_anything.rediscovery.sir_stochastic import (
            run_sir_stochastic_rediscovery,
        )
        results = run_sir_stochastic_rediscovery(
            output_dir="output/rediscovery/sir_stochastic_test",
            n_iterations=1,
        )
        assert isinstance(results, dict)
        assert results["domain"] == "sir_stochastic"
        assert "targets" in results
        assert "ensemble_comparison" in results

    def test_ensemble_comparison_mae_finite(self):
        """MAE from ensemble comparison should be finite and non-negative."""
        from simulating_anything.rediscovery.sir_stochastic import (
            generate_ensemble_comparison_data,
        )
        data = generate_ensemble_comparison_data(
            n_runs=5, n_steps=100, dt=0.1,
        )
        assert np.isfinite(data["mae"])
        assert data["mae"] >= 0

    def test_variance_increases_with_sigma(self):
        """Measured variance should increase with noise intensity."""
        from simulating_anything.rediscovery.sir_stochastic import (
            generate_variance_scaling_data,
        )
        data = generate_variance_scaling_data(
            sigma_values=np.array([0.01, 0.5]),
            N_values=np.array([1000.0]),
            n_runs=10,
            n_steps=200,
            dt=0.1,
        )
        # Variance at sigma=0.5 should be larger than at sigma=0.01
        assert data["sigma_variances"][1] > data["sigma_variances"][0], (
            f"Variance should increase with sigma: "
            f"{data['sigma_variances']}"
        )
