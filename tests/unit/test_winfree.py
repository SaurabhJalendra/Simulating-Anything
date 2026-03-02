"""Tests for the Winfree pulse-coupled oscillators simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.winfree import WinfreeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    kappa: float = 1.0,
    N: int = 50,
    dt: float = 0.01,
    omega0: float = 1.0,
    delta: float = 0.1,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.WINFREE,  # Placeholder domain
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "kappa": kappa,
            "omega0": omega0,
            "delta": delta,
        },
    )


class TestWinfreeBasics:
    """Basic simulation tests: creation, shape, determinism, stability."""

    def test_initial_state_shape(self):
        """State should be an N-element vector of phases."""
        sim = WinfreeSimulation(_make_config(N=50))
        state = sim.reset(seed=42)
        assert state.shape == (50,)

    def test_initial_state_range(self):
        """All initial phases should be in [0, 2*pi)."""
        sim = WinfreeSimulation(_make_config(N=100))
        state = sim.reset(seed=42)
        assert np.all(state >= 0)
        assert np.all(state < 2 * np.pi)

    def test_step_advances(self):
        """State should change after one step."""
        sim = WinfreeSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same seed should produce identical trajectories."""
        config = _make_config()
        sim1 = WinfreeSimulation(config)
        sim1.reset(seed=42)
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = WinfreeSimulation(config)
        sim2.reset(seed=42)
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_phases_stay_finite(self):
        """No NaN or Inf should appear during integration."""
        sim = WinfreeSimulation(_make_config(kappa=3.0))
        sim.reset(seed=42)
        for _ in range(1000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        config = _make_config(N=30, n_steps=100)
        sim = WinfreeSimulation(config)
        traj = sim.run(100)
        assert traj.states.shape == (101, 30)
        assert traj.timestamps.shape == (101,)

    def test_observe_matches_state(self):
        """observe() should return the same as the internal state."""
        sim = WinfreeSimulation(_make_config())
        sim.reset(seed=42)
        sim.step()
        np.testing.assert_array_equal(sim.observe(), sim._state)

    def test_default_parameters(self):
        """Default parameters should match specification."""
        config = SimulationConfig(
            domain=Domain.WINFREE,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = WinfreeSimulation(config)
        assert sim.N == 50
        assert sim.kappa == 1.0
        assert sim.omega0 == 1.0
        assert sim.delta == 0.1

    def test_different_N_sizes(self):
        """Should work for different system sizes."""
        for N in [10, 50, 100, 200]:
            sim = WinfreeSimulation(_make_config(N=N))
            state = sim.reset(seed=42)
            assert state.shape == (N,)
            sim.step()
            assert sim.observe().shape == (N,)


class TestPhaseWrapping:
    """Tests for phase wrapping to [0, 2*pi)."""

    def test_phases_wrapped_after_step(self):
        """All phases should be in [0, 2*pi) after each step."""
        sim = WinfreeSimulation(_make_config(kappa=2.0))
        sim.reset(seed=42)
        for _ in range(500):
            state = sim.step()
            assert np.all(state >= 0), f"Negative phase: {state.min()}"
            assert np.all(state < 2 * np.pi), (
                f"Phase >= 2*pi: {state.max()}"
            )

    def test_phases_wrapped_long_run(self):
        """Phases remain wrapped even during long simulations."""
        sim = WinfreeSimulation(_make_config(kappa=1.0))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        assert np.all(state >= 0)
        assert np.all(state < 2 * np.pi)

    def test_wrap_with_high_frequency(self):
        """High-frequency oscillators should still wrap correctly."""
        sim = WinfreeSimulation(_make_config(omega0=10.0, dt=0.01))
        sim.reset(seed=42)
        for _ in range(1000):
            state = sim.step()
            assert np.all(state >= 0)
            assert np.all(state < 2 * np.pi)


class TestOrderParameter:
    """Tests for the order parameter computation."""

    def test_order_parameter_range(self):
        """Order parameter r should be in [0, 1]."""
        sim = WinfreeSimulation(_make_config(kappa=1.0))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        r, psi = sim.order_parameter
        assert 0 <= r <= 1.0 + 1e-10
        assert -np.pi <= psi <= np.pi + 1e-10

    def test_order_parameter_r_property(self):
        """order_parameter_r should return just the magnitude."""
        sim = WinfreeSimulation(_make_config())
        sim.reset(seed=42)
        r_prop = sim.order_parameter_r
        r_full, _ = sim.order_parameter
        assert r_prop == r_full

    def test_compute_order_parameter_method(self):
        """compute_order_parameter() should match property."""
        sim = WinfreeSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        r_method = sim.compute_order_parameter()
        r_property = sim.order_parameter_r
        np.testing.assert_allclose(r_method, r_property)

    def test_compute_order_parameter_with_state(self):
        """compute_order_parameter() should accept explicit state."""
        sim = WinfreeSimulation(_make_config(N=4))
        sim.reset(seed=42)
        # All phases identical => r = 1
        state = np.array([0.5, 0.5, 0.5, 0.5])
        r = sim.compute_order_parameter(state=state)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_uniform_phases_low_r(self):
        """Uniformly distributed phases should give r ~ 0."""
        sim = WinfreeSimulation(_make_config(N=1000))
        sim.reset(seed=42)
        # Manually set uniformly spaced phases
        sim._state = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        r = sim.order_parameter_r
        assert r < 0.01, f"r={r} for uniform phases"

    def test_identical_phases_r_one(self):
        """Identical phases should give r = 1."""
        sim = WinfreeSimulation(_make_config(N=50))
        sim.reset(seed=42)
        sim._state = np.full(50, 1.5)
        r = sim.order_parameter_r
        np.testing.assert_allclose(r, 1.0, atol=1e-10)


class TestZeroCoupling:
    """Tests for uncoupled (kappa=0) behavior."""

    def test_zero_coupling_desynchronized(self):
        """With kappa=0, oscillators drift independently and r stays low."""
        sim = WinfreeSimulation(_make_config(kappa=0.0, N=200))
        sim.reset(seed=42)
        for _ in range(2000):
            sim.step()
        r = sim.order_parameter_r
        # For random phases, r ~ 1/sqrt(N) ~ 0.07, should be small
        assert r < 0.3, f"r={r} too high for kappa=0"

    def test_zero_coupling_frequencies_unchanged(self):
        """With kappa=0, effective freq should equal natural freq."""
        sim = WinfreeSimulation(_make_config(kappa=0.0))
        sim.reset(seed=42)
        eff_freq = sim.compute_effective_frequency()
        np.testing.assert_allclose(eff_freq, sim.omega)

    def test_zero_coupling_independent_evolution(self):
        """With kappa=0, each oscillator evolves at its own natural freq."""
        config = _make_config(kappa=0.0, N=10, dt=0.001)
        sim = WinfreeSimulation(config)
        sim.reset(seed=42)
        theta0 = sim._state.copy()
        omega = sim.omega.copy()

        # One step with RK4 at kappa=0 should advance by omega*dt
        sim.step()
        expected = (theta0 + omega * 0.001) % (2 * np.pi)
        np.testing.assert_allclose(sim._state, expected, atol=1e-8)


class TestStrongCoupling:
    """Tests for strong coupling behavior."""

    def test_strong_coupling_synchronized(self):
        """With strong kappa and small delta, oscillators should partially sync.

        The Winfree model's multiplicative coupling P(theta)*R(theta) creates
        richer dynamics than Kuramoto. The Lorentzian frequency distribution
        has heavy tails making full sync harder, but strong coupling should
        still produce r > 0.4.
        """
        sim = WinfreeSimulation(_make_config(kappa=5.0, N=100, delta=0.05))
        r = sim.measure_steady_state_r(
            n_transient_steps=5000, n_measure_steps=2000, seed=42,
        )
        assert r > 0.4, f"r={r} too low for strong Winfree coupling"

    def test_coupling_increases_r_vs_uncoupled(self):
        """Coupled Winfree should have higher r than uncoupled baseline."""
        sim_zero = WinfreeSimulation(_make_config(kappa=0.0, N=100, delta=0.05))
        r_zero = sim_zero.measure_steady_state_r(
            n_transient_steps=3000, n_measure_steps=1000, seed=42,
        )

        sim_coupled = WinfreeSimulation(_make_config(kappa=0.5, N=100, delta=0.05))
        r_coupled = sim_coupled.measure_steady_state_r(
            n_transient_steps=3000, n_measure_steps=1000, seed=42,
        )

        assert r_coupled > r_zero, (
            f"Coupled r={r_coupled:.4f} not higher than uncoupled r={r_zero:.4f}"
        )


class TestSyncTransition:
    """Tests for the synchronization transition as kappa increases."""

    def test_r_increases_with_kappa(self):
        """Order parameter should generally increase with coupling."""
        kappa_values = [0.0, 0.3, 0.8]
        r_values = []
        for kappa in kappa_values:
            sim = WinfreeSimulation(_make_config(kappa=kappa, N=100, delta=0.1))
            r = sim.measure_steady_state_r(
                n_transient_steps=3000, n_measure_steps=1000, seed=42,
            )
            r_values.append(r)

        # r at highest kappa should exceed r at lowest
        assert r_values[-1] > r_values[0], (
            f"No transition detected: r_low={r_values[0]:.4f}, "
            f"r_high={r_values[-1]:.4f}"
        )

    def test_transition_exists_in_sweep(self):
        """A kappa sweep should show a transition from low to high r."""
        r_low_sum = 0.0
        r_high_sum = 0.0
        n_low = 3
        n_high = 3

        # Low kappa: should have small r
        for i in range(n_low):
            sim = WinfreeSimulation(_make_config(kappa=0.01, N=80, delta=0.1))
            r_low_sum += sim.measure_steady_state_r(
                n_transient_steps=2000, n_measure_steps=500, seed=42 + i,
            )

        # High kappa: should have larger r
        for i in range(n_high):
            sim = WinfreeSimulation(_make_config(kappa=1.0, N=80, delta=0.1))
            r_high_sum += sim.measure_steady_state_r(
                n_transient_steps=2000, n_measure_steps=500, seed=42 + i,
            )

        r_low_mean = r_low_sum / n_low
        r_high_mean = r_high_sum / n_high
        assert r_high_mean > r_low_mean + 0.1, (
            f"Transition too weak: r_low={r_low_mean:.4f}, "
            f"r_high={r_high_mean:.4f}"
        )


class TestSensitivityAndPulse:
    """Tests for the P(theta) and R(theta) functions."""

    def test_sensitivity_at_zero(self):
        """P(0) = 1 - cos(0) = 0."""
        assert WinfreeSimulation.sensitivity(0.0) == pytest.approx(0.0)

    def test_sensitivity_at_pi(self):
        """P(pi) = 1 - cos(pi) = 2."""
        assert WinfreeSimulation.sensitivity(np.pi) == pytest.approx(2.0)

    def test_sensitivity_range(self):
        """P(theta) in [0, 2] for all theta."""
        theta = np.linspace(0, 2 * np.pi, 1000)
        P = WinfreeSimulation.sensitivity(theta)
        assert np.all(P >= -1e-10)
        assert np.all(P <= 2.0 + 1e-10)

    def test_pulse_at_zero(self):
        """R(0) = 1 + cos(0) = 2."""
        assert WinfreeSimulation.pulse(0.0) == pytest.approx(2.0)

    def test_pulse_at_pi(self):
        """R(pi) = 1 + cos(pi) = 0."""
        assert WinfreeSimulation.pulse(np.pi) == pytest.approx(0.0)

    def test_pulse_range(self):
        """R(theta) in [0, 2] for all theta."""
        theta = np.linspace(0, 2 * np.pi, 1000)
        R = WinfreeSimulation.pulse(theta)
        assert np.all(R >= -1e-10)
        assert np.all(R <= 2.0 + 1e-10)

    def test_pulse_sensitivity_product_bounded(self):
        """P(theta)*R(theta) should be bounded in [0, 4]."""
        theta = np.linspace(0, 2 * np.pi, 1000)
        product = WinfreeSimulation.sensitivity(theta) * WinfreeSimulation.pulse(theta)
        assert np.all(product >= -1e-10)
        assert np.all(product <= 4.0 + 1e-10)


class TestMeanField:
    """Tests for mean-field computations."""

    def test_mean_field_ranges(self):
        """Mean pulse and sensitivity should be in [0, 2]."""
        sim = WinfreeSimulation(_make_config())
        sim.reset(seed=42)
        R_mean, P_mean = sim.compute_mean_field()
        assert 0 <= R_mean <= 2.0 + 1e-10
        assert 0 <= P_mean <= 2.0 + 1e-10

    def test_mean_field_for_uniform_phases(self):
        """For uniformly spaced phases, mean P and R should be ~1."""
        sim = WinfreeSimulation(_make_config(N=1000))
        sim.reset(seed=42)
        sim._state = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        R_mean, P_mean = sim.compute_mean_field()
        # <1 + cos(theta)> = 1 for uniform, <1 - cos(theta)> = 1
        np.testing.assert_allclose(R_mean, 1.0, atol=0.01)
        np.testing.assert_allclose(P_mean, 1.0, atol=0.01)


class TestFrequencyDistribution:
    """Tests for the natural frequency distribution."""

    def test_lorentzian_frequencies_generated(self):
        """Default distribution should generate Lorentzian frequencies."""
        sim = WinfreeSimulation(_make_config(N=100))
        sim.reset(seed=42)
        assert hasattr(sim, "omega")
        assert len(sim.omega) == 100
        # Lorentzian median should be near omega0
        median = np.median(sim.omega)
        assert abs(median - sim.omega0) < 0.5

    def test_gaussian_distribution(self):
        """Gaussian distribution should produce normally distributed freqs."""
        sim = WinfreeSimulation(_make_config(N=1000, delta=0.5))
        sim.distribution = "gaussian"
        sim.reset(seed=42)
        # Mean should be near omega0
        assert abs(np.mean(sim.omega) - sim.omega0) < 0.1
        # Std should be near delta
        assert abs(np.std(sim.omega) - 0.5) < 0.1

    def test_different_seeds_different_freqs(self):
        """Different seeds should give different natural frequencies."""
        sim = WinfreeSimulation(_make_config(N=50))
        sim.reset(seed=42)
        omega1 = sim.omega.copy()
        sim.reset(seed=99)
        omega2 = sim.omega.copy()
        assert not np.allclose(omega1, omega2)


class TestNScaling:
    """Tests for N-independence and finite-size effects."""

    def test_works_with_small_N(self):
        """Should work with very small N."""
        sim = WinfreeSimulation(_make_config(N=3))
        state = sim.reset(seed=42)
        assert state.shape == (3,)
        sim.step()
        assert sim.observe().shape == (3,)

    def test_works_with_large_N(self):
        """Should work with large N."""
        sim = WinfreeSimulation(_make_config(N=500))
        state = sim.reset(seed=42)
        assert state.shape == (500,)
        sim.step()
        assert np.all(np.isfinite(sim.observe()))


class TestMeasureSteadyState:
    """Tests for steady-state measurement methods."""

    def test_measure_steady_state_r_range(self):
        """Steady-state r should be in [0, 1]."""
        sim = WinfreeSimulation(_make_config(kappa=1.0, N=50))
        r = sim.measure_steady_state_r(
            n_transient_steps=1000, n_measure_steps=500, seed=42,
        )
        assert 0 <= r <= 1.0 + 1e-10

    def test_measure_steady_state_freq_spread_positive(self):
        """Frequency spread should be non-negative."""
        sim = WinfreeSimulation(_make_config(kappa=1.0, N=50))
        fs = sim.measure_steady_state_freq_spread(
            n_transient_steps=1000, n_measure_steps=500, seed=42,
        )
        assert fs >= 0

    def test_detect_amplitude_death_returns_bool(self):
        """detect_amplitude_death should return a boolean."""
        sim = WinfreeSimulation(_make_config(kappa=1.0, N=30))
        result = sim.detect_amplitude_death(
            n_transient_steps=1000, n_measure_steps=500, seed=42,
        )
        assert isinstance(result, bool)


class TestDerivatives:
    """Tests for the derivative computation."""

    def test_derivatives_shape(self):
        """Derivatives should have the same shape as state."""
        sim = WinfreeSimulation(_make_config(N=20))
        sim.reset(seed=42)
        derivs = sim._derivatives(sim._state)
        assert derivs.shape == (20,)

    def test_derivatives_finite(self):
        """Derivatives should always be finite."""
        sim = WinfreeSimulation(_make_config(kappa=5.0, N=50))
        sim.reset(seed=42)
        derivs = sim._derivatives(sim._state)
        assert np.all(np.isfinite(derivs))

    def test_derivatives_zero_coupling(self):
        """At kappa=0, derivatives should equal natural frequencies."""
        sim = WinfreeSimulation(_make_config(kappa=0.0, N=20))
        sim.reset(seed=42)
        derivs = sim._derivatives(sim._state)
        np.testing.assert_allclose(derivs, sim.omega)

    def test_derivatives_coupling_term(self):
        """Coupling term should be (kappa/N)*P(theta_i)*sum(R(theta_j))."""
        sim = WinfreeSimulation(_make_config(kappa=1.0, N=5))
        sim.reset(seed=42)
        theta = sim._state
        derivs = sim._derivatives(theta)

        # Manually compute expected derivatives
        P = 1.0 - np.cos(theta)
        R = 1.0 + np.cos(theta)
        R_sum = np.sum(R)
        coupling = (1.0 / 5) * P * R_sum
        expected = sim.omega + coupling
        np.testing.assert_allclose(derivs, expected)


class TestRediscoveryData:
    """Tests for rediscovery data generation functions."""

    def test_sync_transition_data(self):
        """generate_sync_transition_data should return correct arrays."""
        from simulating_anything.rediscovery.winfree import (
            generate_sync_transition_data,
        )
        data = generate_sync_transition_data(
            n_kappa=5, N=30, n_trials=2,
            n_transient=500, n_measure=200, dt=0.05,
        )
        assert len(data["kappa"]) == 5
        assert len(data["r_mean"]) == 5
        assert len(data["r_std"]) == 5
        assert np.all(data["r_mean"] >= 0)
        assert np.all(data["r_mean"] <= 1.0 + 0.1)

    def test_transition_occurs_in_data(self):
        """r should increase with kappa."""
        from simulating_anything.rediscovery.winfree import (
            generate_sync_transition_data,
        )
        data = generate_sync_transition_data(
            n_kappa=10, N=50, n_trials=3,
            n_transient=2000, n_measure=1000, dt=0.02,
        )
        # r at highest kappa should be larger than at lowest kappa
        assert data["r_mean"][-1] > data["r_mean"][0], (
            f"No transition: r_low={data['r_mean'][0]}, "
            f"r_high={data['r_mean'][-1]}"
        )

    def test_finite_size_data(self):
        """generate_finite_size_data should return correct arrays."""
        from simulating_anything.rediscovery.winfree import (
            generate_finite_size_data,
        )
        data = generate_finite_size_data(
            N_values=[10, 30], kappa=0.5, n_trials=2, dt=0.02,
        )
        assert len(data["N"]) == 2
        assert len(data["r"]) == 2

    def test_run_function_exists(self):
        """Ensure the main rediscovery runner function is importable."""
        from simulating_anything.rediscovery.winfree import (
            run_winfree_rediscovery,
        )
        assert callable(run_winfree_rediscovery)

    def test_kuramoto_comparison_data(self):
        """generate_kuramoto_comparison_data should return correct arrays."""
        from simulating_anything.rediscovery.winfree import (
            generate_kuramoto_comparison_data,
        )
        data = generate_kuramoto_comparison_data(
            n_kappa=5, N=30, n_trials=2,
            n_transient=500, n_measure=200, dt=0.05,
        )
        assert len(data["kappa"]) == 5
        assert len(data["r_mean"]) == 5
        assert len(data["freq_spread_mean"]) == 5
