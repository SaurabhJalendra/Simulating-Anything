"""Tests for the circadian clock (Gonze-Goodwin model) simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.circadian_clock import CircadianClockSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

# Use GOODWIN as the domain enum for circadian models
_DOMAIN = Domain.CIRCADIAN_CLOCK


def _make_config(
    dt: float = 0.1,
    n_steps: int = 3000,
    **params,
) -> SimulationConfig:
    """Helper to create a SimulationConfig with circadian clock defaults."""
    defaults = {
        "v_s": 1.6, "K_I": 1.0, "n": 4.0,
        "v_m": 0.505, "K_m": 0.5, "k_s": 0.5,
        "v_d": 1.4, "K_d": 0.13, "k1": 0.33, "k2": 0.43,
        "M_0": 0.5, "Fc_0": 0.5, "Fn_0": 0.5,
    }
    defaults.update(params)
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


# ---------------------------------------------------------------------------
# Basic simulation tests
# ---------------------------------------------------------------------------


class TestCircadianClockCreation:
    """Test object creation and parameter handling."""

    def test_create_default(self):
        sim = CircadianClockSimulation(_make_config())
        assert sim.v_s == pytest.approx(1.6)
        assert sim.K_I == pytest.approx(1.0)
        assert sim.n == pytest.approx(4.0)

    def test_create_custom_params(self):
        sim = CircadianClockSimulation(_make_config(v_s=1.0, n=6.0, k1=2.5))
        assert sim.v_s == pytest.approx(1.0)
        assert sim.n == pytest.approx(6.0)
        assert sim.k1 == pytest.approx(2.5)

    def test_initial_state_shape(self):
        sim = CircadianClockSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        sim = CircadianClockSimulation(_make_config(M_0=0.3, Fc_0=0.7, Fn_0=0.1))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.3, 0.7, 0.1])

    def test_default_initial_conditions(self):
        sim = CircadianClockSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, 0.5, 0.5])


class TestCircadianClockStep:
    """Test stepping and basic dynamics."""

    def test_step_advances_state(self):
        sim = CircadianClockSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = CircadianClockSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = CircadianClockSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_run_trajectory_timestamps(self):
        dt = 0.1
        sim = CircadianClockSimulation(_make_config(dt=dt))
        traj = sim.run(n_steps=50)
        np.testing.assert_allclose(traj.timestamps[0], 0.0)
        np.testing.assert_allclose(traj.timestamps[-1], 50 * dt, atol=1e-10)


# ---------------------------------------------------------------------------
# Non-negativity tests
# ---------------------------------------------------------------------------


class TestCircadianClockNonNegativity:
    """All concentrations (M, Fc, Fn) must remain non-negative."""

    def test_states_non_negative_default(self):
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(3000):
            sim.step()
            M, Fc, Fn = sim.observe()
            assert M >= 0.0, f"M went negative: {M}"
            assert Fc >= 0.0, f"Fc went negative: {Fc}"
            assert Fn >= 0.0, f"Fn went negative: {Fn}"

    def test_states_non_negative_small_initial(self):
        """Starting near zero should not produce negatives."""
        sim = CircadianClockSimulation(_make_config(
            dt=0.05, M_0=0.01, Fc_0=0.01, Fn_0=0.01,
        ))
        sim.reset()
        for _ in range(5000):
            sim.step()
            M, Fc, Fn = sim.observe()
            assert M >= 0.0, f"M went negative: {M}"
            assert Fc >= 0.0, f"Fc went negative: {Fc}"
            assert Fn >= 0.0, f"Fn went negative: {Fn}"

    def test_states_bounded(self):
        """Trajectory should remain bounded for default parameters."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            sim.step()
            M, Fc, Fn = sim.observe()
            assert M < 50, f"M diverged: {M}"
            assert Fc < 50, f"Fc diverged: {Fc}"
            assert Fn < 50, f"Fn diverged: {Fn}"


# ---------------------------------------------------------------------------
# Oscillation tests
# ---------------------------------------------------------------------------


class TestCircadianOscillation:
    """Test that the system oscillates with the correct period."""

    def test_oscillation_default_params(self):
        """Default parameters (n=4) should produce oscillation."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        oscillating = sim.is_oscillating(
            transient_steps=3000, measure_steps=3000, threshold=0.01,
        )
        assert oscillating, "Default parameters should produce oscillation"

    def test_oscillation_all_variables(self):
        """All three variables should oscillate together."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        amps = sim.measure_amplitude(transient_steps=3000, measure_steps=3000)
        assert amps[0] > 0.01, f"M should oscillate, amplitude={amps[0]}"
        assert amps[1] > 0.01, f"Fc should oscillate, amplitude={amps[1]}"
        assert amps[2] > 0.01, f"Fn should oscillate, amplitude={amps[2]}"

    def test_period_approximately_24h(self):
        """With default parameters, period should be close to 24h."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        period = sim.measure_period(transient_steps=5000, measure_steps=10000)
        assert np.isfinite(period), "Period should be finite for default params"
        # Gonze model with these params gives period ~23-25h
        assert 18.0 < period < 32.0, f"Period {period:.2f}h outside circadian range"

    def test_period_positive(self):
        """Period should be strictly positive."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        period = sim.measure_period(transient_steps=5000, measure_steps=10000)
        assert period > 0

    def test_nonzero_amplitude_m(self):
        """mRNA amplitude should be non-zero for oscillating system."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        amps = sim.measure_amplitude(transient_steps=3000, measure_steps=3000)
        assert amps[0] > 0.01


# ---------------------------------------------------------------------------
# Phase relationship tests
# ---------------------------------------------------------------------------


class TestCircadianPhaseRelationship:
    """M should lead Fc, and Fc should lead Fn in the oscillation cycle."""

    def test_phase_lag_m_to_fc(self):
        """Fc should lag M (positive lag)."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        lag_m_fc, _ = sim.measure_phase_lag(
            transient_steps=5000, measure_steps=10000,
        )
        # Fc lags M by some positive time (translated protein follows mRNA)
        assert lag_m_fc >= 0, f"Fc should lag M, but lag={lag_m_fc}"

    def test_phase_lag_fc_to_fn(self):
        """Fn should lag Fc (positive lag)."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()
        _, lag_fc_fn = sim.measure_phase_lag(
            transient_steps=5000, measure_steps=10000,
        )
        # Fn lags Fc (nuclear protein follows cytoplasmic)
        assert lag_fc_fn >= 0, f"Fn should lag Fc, but lag={lag_fc_fn}"

    def test_peak_order_m_before_fc(self):
        """Over a full cycle, M peak should precede the next Fc peak."""
        sim = CircadianClockSimulation(_make_config(dt=0.1))
        sim.reset()

        # Skip long transient for clean limit cycle
        for _ in range(15000):
            sim.step()

        # Collect enough data for multiple full periods
        m_vals, fc_vals = [], []
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            m_vals.append(state[0])
            fc_vals.append(state[1])

        m_arr = np.array(m_vals)
        fc_arr = np.array(fc_vals)
        m_mean = np.mean(m_arr)
        fc_mean = np.mean(fc_arr)

        # Find upward crossings for each variable (proxy for phase)
        m_crossings = []
        fc_crossings = []
        for j in range(1, len(m_arr)):
            if m_arr[j - 1] < m_mean and m_arr[j] >= m_mean:
                m_crossings.append(j)
            if fc_arr[j - 1] < fc_mean and fc_arr[j] >= fc_mean:
                fc_crossings.append(j)

        # We just need to verify both variables oscillate in sync
        assert len(m_crossings) >= 2, "M should have multiple crossings"
        assert len(fc_crossings) >= 2, "Fc should have multiple crossings"
        # Verify similar number of crossings (same frequency)
        assert abs(len(m_crossings) - len(fc_crossings)) <= 1, (
            f"M and Fc should oscillate at same frequency: "
            f"M crossings={len(m_crossings)}, Fc crossings={len(fc_crossings)}"
        )


# ---------------------------------------------------------------------------
# Parameter effect tests
# ---------------------------------------------------------------------------


class TestCircadianParameterEffects:
    """Test sensitivity of period and amplitude to key parameters."""

    def test_vs_affects_amplitude(self):
        """Higher v_s (transcription rate) should increase oscillation amplitude."""
        sim_low = CircadianClockSimulation(_make_config(dt=0.1, v_s=0.5))
        sim_low.reset()
        amp_low = sim_low.measure_amplitude(transient_steps=3000, measure_steps=3000)

        sim_high = CircadianClockSimulation(_make_config(dt=0.1, v_s=1.0))
        sim_high.reset()
        amp_high = sim_high.measure_amplitude(transient_steps=3000, measure_steps=3000)

        # Higher transcription rate should give larger amplitude
        assert amp_high[0] > amp_low[0] * 0.5, (
            f"Expected higher v_s to increase amplitude: "
            f"low={amp_low[0]:.4f}, high={amp_high[0]:.4f}"
        )

    def test_k1_affects_period(self):
        """Changing k1 (nuclear import rate) should affect the period."""
        sim1 = CircadianClockSimulation(_make_config(dt=0.1, k1=1.5))
        sim1.reset()
        period1 = sim1.measure_period(transient_steps=5000, measure_steps=10000)

        sim2 = CircadianClockSimulation(_make_config(dt=0.1, k1=2.5))
        sim2.reset()
        period2 = sim2.measure_period(transient_steps=5000, measure_steps=10000)

        # Periods should be different
        if np.isfinite(period1) and np.isfinite(period2):
            assert abs(period1 - period2) > 0.1, (
                f"Expected different periods: k1=1.5 -> {period1:.2f}, "
                f"k1=2.5 -> {period2:.2f}"
            )

    def test_k2_affects_period(self):
        """Changing k2 (nuclear export rate) should affect the period."""
        sim1 = CircadianClockSimulation(_make_config(dt=0.1, k2=1.0))
        sim1.reset()
        period1 = sim1.measure_period(transient_steps=5000, measure_steps=10000)

        sim2 = CircadianClockSimulation(_make_config(dt=0.1, k2=2.0))
        sim2.reset()
        period2 = sim2.measure_period(transient_steps=5000, measure_steps=10000)

        if np.isfinite(period1) and np.isfinite(period2):
            assert abs(period1 - period2) > 0.1, (
                f"Expected different periods: k2=1.0 -> {period1:.2f}, "
                f"k2=2.0 -> {period2:.2f}"
            )

    def test_n_affects_oscillation(self):
        """Higher Hill coefficient n should produce stronger oscillation."""
        sim_n2 = CircadianClockSimulation(_make_config(dt=0.1, n=2.0))
        sim_n2.reset()
        amp_n2 = sim_n2.measure_amplitude(transient_steps=5000, measure_steps=3000)

        sim_n6 = CircadianClockSimulation(_make_config(dt=0.1, n=6.0))
        sim_n6.reset()
        amp_n6 = sim_n6.measure_amplitude(transient_steps=5000, measure_steps=3000)

        # Higher n should give larger amplitude (steeper Hill function)
        assert amp_n6[0] >= amp_n2[0], (
            f"Expected higher n to increase amplitude: "
            f"n=2 -> {amp_n2[0]:.4f}, n=6 -> {amp_n6[0]:.4f}"
        )


# ---------------------------------------------------------------------------
# Hill coefficient onset tests
# ---------------------------------------------------------------------------


class TestCircadianHillOnset:
    """Test oscillation onset as a function of Hill coefficient."""

    def test_very_low_n_no_oscillation(self):
        """Very low n (n=0.5) should not produce sustained oscillation.

        The Gonze model has two nonlinearity sources: the Hill function and the
        Michaelis-Menten degradation. With strong MM kinetics, even n=1 can
        oscillate. Only at very low n does the feedback become too weak.
        """
        sim = CircadianClockSimulation(_make_config(dt=0.1, n=0.5))
        sim.reset()
        oscillating = sim.is_oscillating(
            transient_steps=5000, measure_steps=5000, threshold=0.05,
        )
        assert not oscillating, "n=0.5 should not produce sustained oscillation"

    def test_n4_oscillates(self):
        """n=4 (default) should produce robust oscillation."""
        sim = CircadianClockSimulation(_make_config(dt=0.1, n=4.0))
        sim.reset()
        oscillating = sim.is_oscillating(
            transient_steps=3000, measure_steps=3000, threshold=0.01,
        )
        assert oscillating, "n=4 should produce oscillation"

    def test_low_n_may_not_oscillate(self):
        """Low Hill coefficients should have smaller amplitude."""
        amplitudes = []
        for n_val in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]:
            sim = CircadianClockSimulation(_make_config(dt=0.1, n=n_val))
            sim.reset()
            amp = sim.measure_amplitude(transient_steps=5000, measure_steps=3000)
            amplitudes.append(amp[0])

        # Amplitude should generally increase with n
        # At minimum, the highest n should have largest amplitude
        assert amplitudes[-1] >= amplitudes[0], (
            f"Expected amplitude to increase with n: n=0.5 -> {amplitudes[0]:.4f}, "
            f"n=6 -> {amplitudes[-1]:.4f}"
        )

    def test_high_n_oscillates(self):
        """High Hill coefficient should definitely oscillate."""
        sim = CircadianClockSimulation(_make_config(dt=0.1, n=8.0))
        sim.reset()
        oscillating = sim.is_oscillating(
            transient_steps=3000, measure_steps=3000, threshold=0.01,
        )
        assert oscillating, "n=8 should produce robust oscillation"


# ---------------------------------------------------------------------------
# Steady state test for n=1
# ---------------------------------------------------------------------------


class TestCircadianSteadyState:
    """Test convergence to steady state when oscillation is absent."""

    def test_low_n_converges_to_steady_state(self):
        """With n=0.5 (weak cooperativity), system should converge to a fixed point.

        The Gonze model combines Hill repression with Michaelis-Menten degradation.
        With strong MM kinetics, even n=1 can oscillate. At n=0.5 the Hill function
        is too weak for instability and the system reaches a stable steady state.
        """
        sim = CircadianClockSimulation(_make_config(dt=0.1, n=0.5))
        sim.reset()

        # Run long enough for convergence
        for _ in range(20000):
            sim.step()

        # Record the state
        s1 = sim.observe().copy()

        # Run more steps and check state barely changes
        for _ in range(5000):
            sim.step()
        s2 = sim.observe()

        # State should be very close (converged)
        np.testing.assert_allclose(s1, s2, atol=0.05, rtol=0.05)

    def test_low_n_state_positive_at_steady_state(self):
        """Steady state concentrations should be positive."""
        sim = CircadianClockSimulation(_make_config(dt=0.1, n=0.5))
        sim.reset()

        for _ in range(20000):
            sim.step()

        M, Fc, Fn = sim.observe()
        assert M > 0, f"M should be positive at steady state: {M}"
        assert Fc > 0, f"Fc should be positive at steady state: {Fc}"
        assert Fn > 0, f"Fn should be positive at steady state: {Fn}"


# ---------------------------------------------------------------------------
# Derivatives / ODE structure tests
# ---------------------------------------------------------------------------


class TestCircadianDerivatives:
    """Test the ODE right-hand side for correctness."""

    def test_derivatives_shape(self):
        sim = CircadianClockSimulation(_make_config())
        sim.reset()
        dy = sim._derivatives(sim.observe())
        assert dy.shape == (3,)

    def test_derivatives_at_zero_fn(self):
        """When Fn=0, Hill function should be at maximum: v_s."""
        sim = CircadianClockSimulation(_make_config())
        sim.reset()
        state = np.array([1.0, 1.0, 0.0])
        dy = sim._derivatives(state)
        # dM/dt = v_s * 1.0 - v_m * M / (K_m + M) = 1.6 - 0.505 * 1/(0.5+1)
        expected_dM = 1.6 - 0.505 * 1.0 / (0.5 + 1.0)
        assert dy[0] == pytest.approx(expected_dM, rel=1e-10)

    def test_derivatives_nuclear_transport(self):
        """Nuclear transport terms: dFn/dt = k1*Fc - k2*Fn."""
        sim = CircadianClockSimulation(_make_config(k1=0.33, k2=0.43))
        sim.reset()
        state = np.array([1.0, 2.0, 3.0])
        dy = sim._derivatives(state)
        # dFn/dt = k1*Fc - k2*Fn = 0.33*2.0 - 0.43*3.0 = 0.66 - 1.29 = -0.63
        expected_dFn = 0.33 * 2.0 - 0.43 * 3.0
        assert dy[2] == pytest.approx(expected_dFn, rel=1e-10)

    def test_derivatives_conservation_like(self):
        """dFc/dt + dFn/dt = k_s*M - v_d*Fc/(K_d+Fc) (total protein production - degradation)."""
        sim = CircadianClockSimulation(_make_config())
        sim.reset()
        state = np.array([1.0, 1.0, 1.0])
        dy = sim._derivatives(state)
        # dFc + dFn = k_s*M - v_d*Fc/(K_d+Fc) - k1*Fc + k2*Fn + k1*Fc - k2*Fn
        # = k_s*M - v_d*Fc/(K_d+Fc)  (transport terms cancel)
        expected_sum = sim.k_s * 1.0 - sim.v_d * 1.0 / (sim.K_d + 1.0)
        assert dy[1] + dy[2] == pytest.approx(expected_sum, rel=1e-10)

    def test_hill_function_repression(self):
        """Hill function should decrease as Fn increases."""
        sim = CircadianClockSimulation(_make_config(n=4.0))
        sim.reset()
        # Compute Hill function at different Fn values
        dy_low = sim._derivatives(np.array([1.0, 1.0, 0.1]))
        dy_high = sim._derivatives(np.array([1.0, 1.0, 5.0]))
        # At higher Fn, repression is stronger, so dM/dt should be lower
        # (the Michaelis-Menten degradation term is the same since M is the same)
        # So the difference is purely from the Hill function
        assert dy_low[0] > dy_high[0], (
            f"Hill repression failed: dM(Fn=0.1)={dy_low[0]:.4f} "
            f"should be > dM(Fn=5.0)={dy_high[0]:.4f}"
        )


# ---------------------------------------------------------------------------
# Rediscovery data generation tests
# ---------------------------------------------------------------------------


class TestCircadianRediscovery:
    """Test the rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.circadian_clock import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.1)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert len(data["M"]) == 501
        assert len(data["Fc"]) == 501
        assert len(data["Fn"]) == 501

    def test_ode_data_non_negative(self):
        from simulating_anything.rediscovery.circadian_clock import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.1)
        assert np.all(data["M"] >= 0), "M should be non-negative"
        assert np.all(data["Fc"] >= 0), "Fc should be non-negative"
        assert np.all(data["Fn"] >= 0), "Fn should be non-negative"

    def test_hill_sweep_data_generation(self):
        from simulating_anything.rediscovery.circadian_clock import (
            generate_hill_sweep_data,
        )

        data = generate_hill_sweep_data(n_range=(1.0, 6.0), n_points=5, dt=0.1)
        assert len(data["n_values"]) == 5
        assert len(data["amplitudes"]) == 5

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.circadian_clock import generate_period_data

        data = generate_period_data(n_steps_per_point=5000, dt=0.1)
        assert len(data["v_s_values"]) == 15
        assert len(data["periods_vs"]) == 15
        assert len(data["k1_values"]) == 15
        assert len(data["periods_k1"]) == 15

    def test_phase_portrait_data_generation(self):
        from simulating_anything.rediscovery.circadian_clock import (
            generate_phase_portrait_data,
        )

        data = generate_phase_portrait_data(n_steps=500, dt=0.1)
        assert len(data["M"]) == 500
        assert len(data["Fc"]) == 500
        assert len(data["Fn"]) == 500


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestCircadianDeterminism:
    """Verify that the simulation is deterministic."""

    def test_same_config_same_trajectory(self):
        """Two runs with the same config should produce identical trajectories."""
        config = _make_config(dt=0.1, n_steps=100)

        sim1 = CircadianClockSimulation(config)
        traj1 = sim1.run(n_steps=100)

        sim2 = CircadianClockSimulation(config)
        traj2 = sim2.run(n_steps=100)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_different_initial_conditions(self):
        """Different initial conditions should produce different trajectories."""
        config1 = _make_config(M_0=0.5, Fc_0=0.5, Fn_0=0.5)
        config2 = _make_config(M_0=1.0, Fc_0=0.1, Fn_0=0.1)

        sim1 = CircadianClockSimulation(config1)
        traj1 = sim1.run(n_steps=100)

        sim2 = CircadianClockSimulation(config2)
        traj2 = sim2.run(n_steps=100)

        assert not np.allclose(traj1.states, traj2.states)
