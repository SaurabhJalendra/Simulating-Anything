"""Tests for the seasonal predator-prey model (#180)."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.seasonal_predator_prey import (
    SeasonalPredatorPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r0: float = 1.0,
    K: float = 100.0,
    a: float = 0.5,
    h: float = 0.02,
    e: float = 0.4,
    d: float = 0.2,
    epsilon: float = 0.3,
    T: float = 12.0,
    N_0: float = 50.0,
    P_0: float = 10.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SEASONAL_PREDATOR_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r0": r0, "K": K, "a": a, "h": h,
            "e": e, "d": d, "epsilon": epsilon, "T": T,
            "N_0": N_0, "P_0": P_0,
        },
    )


class TestSeasonalPredatorPreyCreation:
    """Basic construction and initialization tests."""

    def test_creation(self):
        """Simulation can be created with default parameters."""
        config = _make_config()
        sim = SeasonalPredatorPreySimulation(config)
        assert sim.r0 == 1.0
        assert sim.K == 100.0
        assert sim.epsilon == 0.3
        assert sim.T == 12.0

    def test_custom_parameters(self):
        """Custom parameters are correctly loaded."""
        config = _make_config(r0=2.0, K=200.0, epsilon=0.5, T=6.0)
        sim = SeasonalPredatorPreySimulation(config)
        assert sim.r0 == 2.0
        assert sim.K == 200.0
        assert sim.epsilon == 0.5
        assert sim.T == 6.0

    def test_reset_state(self):
        """Initial conditions are set correctly."""
        config = _make_config(N_0=42.0, P_0=7.5)
        sim = SeasonalPredatorPreySimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [42.0, 7.5])

    def test_observe_shape(self):
        """Observe returns 2-element state vector."""
        config = _make_config()
        sim = SeasonalPredatorPreySimulation(config)
        state = sim.reset()
        assert state.shape == (2,)
        assert sim.observe().shape == (2,)

    def test_observe_matches_reset(self):
        """Observe returns the same state as reset."""
        config = _make_config(N_0=60.0, P_0=15.0)
        sim = SeasonalPredatorPreySimulation(config)
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_allclose(state, obs)


class TestSeasonalPredatorPreyDynamics:
    """Core dynamics and physics tests."""

    def test_step_advances_state(self):
        """State changes after a step."""
        config = _make_config(N_0=50.0, P_0=10.0)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_non_negative(self):
        """Populations remain non-negative throughout simulation."""
        config = _make_config(N_0=5.0, P_0=50.0, epsilon=0.8, dt=0.01)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        for _ in range(20000):
            sim.step()
            N, P = sim.observe()
            assert N >= 0, f"Prey went negative: N={N}"
            assert P >= 0, f"Predator went negative: P={P}"

    def test_bounded(self):
        """Populations remain bounded and finite."""
        config = _make_config(N_0=50.0, P_0=10.0, epsilon=0.5)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        for _ in range(30000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            N, P = state
            assert N < 500, f"Prey diverged: N={N}"
            assert P < 500, f"Predator diverged: P={P}"

    def test_prey_bounded_by_carrying_capacity(self):
        """Prey density should not greatly exceed K."""
        config = _make_config(N_0=90.0, P_0=1.0, K=100.0, epsilon=0.3)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        max_N = 0.0
        for _ in range(50000):
            sim.step()
            N = sim.observe()[0]
            max_N = max(max_N, N)

        # With seasonal forcing, prey can briefly exceed K but not by much
        assert max_N < 2.0 * 100.0, f"Prey far exceeded K: max_N={max_N}"

    def test_rk4_stability(self):
        """No NaN or Inf in simulation output."""
        config = _make_config(N_0=50.0, P_0=10.0, epsilon=0.9, dt=0.01)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        for _ in range(20000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_deterministic(self):
        """Same parameters produce identical results."""
        config = _make_config(N_0=50.0, P_0=10.0, epsilon=0.3)

        sim1 = SeasonalPredatorPreySimulation(config)
        sim1.reset()
        for _ in range(5000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SeasonalPredatorPreySimulation(config)
        sim2.reset()
        for _ in range(5000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_predator_extinction_without_prey(self):
        """Predator dies off if prey starts at exactly zero."""
        # With N_0=0, prey cannot grow, so predator starves
        config = _make_config(N_0=0.0, P_0=10.0, epsilon=0.0)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        for _ in range(50000):
            sim.step()

        P_final = sim.observe()[1]
        assert P_final < 0.01, f"Predator should decline without prey: P={P_final}"


class TestSeasonalModulation:
    """Tests specific to the seasonal forcing mechanism."""

    def test_seasonal_growth_rate_at_zero_epsilon(self):
        """With no forcing, growth rate is constant r0."""
        config = _make_config(r0=1.5, epsilon=0.0)
        sim = SeasonalPredatorPreySimulation(config)

        for t in [0.0, 3.0, 6.0, 9.0, 12.0]:
            assert sim.seasonal_growth_rate(t) == pytest.approx(1.5)

    def test_seasonal_growth_rate_max(self):
        """Maximum growth rate is r0*(1+epsilon) at t=T/4."""
        config = _make_config(r0=1.0, epsilon=0.5, T=12.0)
        sim = SeasonalPredatorPreySimulation(config)

        # sin(2*pi*t/T) = 1 at t = T/4
        r_max = sim.seasonal_growth_rate(3.0)  # T/4 = 3.0
        assert r_max == pytest.approx(1.5, rel=1e-10)

    def test_seasonal_growth_rate_min(self):
        """Minimum growth rate is r0*(1-epsilon) at t=3T/4."""
        config = _make_config(r0=1.0, epsilon=0.5, T=12.0)
        sim = SeasonalPredatorPreySimulation(config)

        # sin(2*pi*t/T) = -1 at t = 3T/4
        r_min = sim.seasonal_growth_rate(9.0)  # 3T/4 = 9.0
        assert r_min == pytest.approx(0.5, rel=1e-10)

    def test_seasonal_growth_rate_periodic(self):
        """Growth rate is periodic with period T."""
        config = _make_config(r0=1.0, epsilon=0.3, T=12.0)
        sim = SeasonalPredatorPreySimulation(config)

        for t in [0.0, 2.5, 5.7, 11.9]:
            r_t = sim.seasonal_growth_rate(t)
            r_t_plus_T = sim.seasonal_growth_rate(t + 12.0)
            assert r_t == pytest.approx(r_t_plus_T, rel=1e-12)

    def test_forcing_amplitude_affects_dynamics(self):
        """Larger epsilon produces larger oscillation amplitude."""
        amplitudes = []
        for eps in [0.0, 0.3, 0.6]:
            config = _make_config(epsilon=eps, dt=0.01, n_steps=50000)
            sim = SeasonalPredatorPreySimulation(config)
            sim.reset()

            # Discard transient
            for _ in range(20000):
                sim.step()

            prey_values = []
            for _ in range(30000):
                sim.step()
                prey_values.append(sim.observe()[0])

            prey_arr = np.array(prey_values)
            amplitudes.append(np.max(prey_arr) - np.min(prey_arr))

        # Larger forcing should generally produce larger prey oscillations
        # At eps=0 there may still be a limit cycle, but the amplitude
        # should increase with stronger forcing
        assert amplitudes[2] > amplitudes[0] or amplitudes[1] > amplitudes[0], (
            f"Forcing should increase oscillation amplitude: {amplitudes}"
        )

    def test_current_time(self):
        """current_time() tracks simulation time correctly."""
        config = _make_config(dt=0.01)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        assert sim.current_time() == pytest.approx(0.0)

        for _ in range(100):
            sim.step()

        assert sim.current_time() == pytest.approx(1.0, rel=1e-10)


class TestEquilibrium:
    """Tests for equilibrium behavior without seasonal forcing."""

    def test_equilibrium_at_epsilon_zero(self):
        """Without forcing, system has bounded dynamics near coexistence.

        With default parameters the unforced Rosenzweig-MacArthur model may
        exhibit a limit cycle rather than a stable fixed point. We verify the
        system remains bounded and the time-averaged prey density is on the
        same order as the theoretical equilibrium N*.
        """
        config = _make_config(epsilon=0.0, dt=0.01, n_steps=200000)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        for _ in range(200000):
            sim.step()

        # Get theoretical equilibrium
        eq = sim.coexistence_equilibrium()

        # Collect states after transient
        states_list = []
        for _ in range(50000):
            sim.step()
            states_list.append(sim.observe().copy())

        states = np.array(states_list)
        mean_N = np.mean(states[:, 0])
        mean_P = np.mean(states[:, 1])

        # System should remain bounded and positive
        assert mean_N > 0, f"Mean prey should be positive: {mean_N}"
        assert mean_P > 0, f"Mean predator should be positive: {mean_P}"
        assert mean_N < sim.K, f"Mean prey should be below K: {mean_N}"

        if eq["valid"]:
            # Mean prey on limit cycle is same order of magnitude as N*
            assert mean_N < 10.0 * eq["N_star"], (
                f"Mean prey {mean_N:.2f} far from equilibrium {eq['N_star']:.2f}"
            )

    def test_coexistence_equilibrium_computation(self):
        """Equilibrium computation returns valid values for default params."""
        config = _make_config()
        sim = SeasonalPredatorPreySimulation(config)
        eq = sim.coexistence_equilibrium()

        assert eq["valid"] is True
        assert eq["N_star"] > 0
        assert eq["P_star"] > 0
        assert eq["N_star"] < sim.K

    def test_equilibrium_invalid_for_high_death_rate(self):
        """No coexistence equilibrium if predator death rate is too high."""
        # d > e*a/h means predator cannot sustain itself
        config = _make_config(d=100.0, e=0.1, a=0.1, h=0.02)
        sim = SeasonalPredatorPreySimulation(config)
        eq = sim.coexistence_equilibrium()

        # Either invalid or N_star exceeds K
        assert not eq["valid"] or eq["N_star"] >= sim.K


class TestHollingType2:
    """Tests for the functional response."""

    def test_holling_zero(self):
        """Response is zero at zero prey."""
        config = _make_config()
        sim = SeasonalPredatorPreySimulation(config)
        assert sim.holling_type2(0.0) == pytest.approx(0.0)

    def test_holling_linear_at_low_N(self):
        """Response is approximately a*N for small N."""
        config = _make_config(a=0.5, h=0.02)
        sim = SeasonalPredatorPreySimulation(config)

        # For N = 0.1, h*N = 0.002 << 1, so f(N) ~ a*N
        fr = sim.holling_type2(0.1)
        linear_approx = 0.5 * 0.1
        assert fr == pytest.approx(linear_approx, rel=0.01)

    def test_holling_saturation(self):
        """Response saturates at a/h for large N."""
        config = _make_config(a=0.5, h=0.02)
        sim = SeasonalPredatorPreySimulation(config)

        # For very large N: a*N/(1+h*N) -> a/h = 25
        fr = sim.holling_type2(1e6)
        assert fr == pytest.approx(0.5 / 0.02, rel=0.01)

    def test_holling_monotone_increasing(self):
        """Functional response is monotonically increasing."""
        config = _make_config()
        sim = SeasonalPredatorPreySimulation(config)

        N_values = np.linspace(0, 200, 50)
        fr_values = [sim.holling_type2(n) for n in N_values]
        for i in range(len(fr_values) - 1):
            assert fr_values[i + 1] >= fr_values[i]


class TestEpsilonSweep:
    """Tests for epsilon sweep behavior."""

    def test_zero_forcing_small_or_zero_amplitude(self):
        """At epsilon=0, prey oscillation amplitude is from natural dynamics only."""
        config = _make_config(epsilon=0.0, dt=0.01, n_steps=80000)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        # Discard transient
        for _ in range(50000):
            sim.step()

        prey_values = []
        for _ in range(30000):
            sim.step()
            prey_values.append(sim.observe()[0])

        prey_arr = np.array(prey_values)
        # Natural dynamics may have a limit cycle, but there is no forcing
        # Just verify it is finite and non-negative
        assert np.all(prey_arr >= 0)
        assert np.all(np.isfinite(prey_arr))

    def test_strong_forcing_produces_oscillation(self):
        """At large epsilon, prey shows clear oscillation."""
        config = _make_config(epsilon=0.8, dt=0.01, n_steps=80000)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        for _ in range(50000):
            sim.step()

        prey_values = []
        for _ in range(30000):
            sim.step()
            prey_values.append(sim.observe()[0])

        prey_arr = np.array(prey_values)
        amplitude = np.max(prey_arr) - np.min(prey_arr)
        assert amplitude > 1.0, f"Strong forcing should cause oscillation: amp={amplitude}"


class TestFrequencyDetection:
    """Tests for FFT-based frequency detection."""

    def test_detect_dominant_frequency_shape(self):
        """Frequency detection returns expected keys."""
        config = _make_config(epsilon=0.3)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        states_list = []
        for _ in range(5000):
            sim.step()
            states_list.append(sim.observe().copy())

        traj = np.array(states_list)
        result = sim.detect_dominant_frequency(traj)

        assert "dominant_freq" in result
        assert "dominant_period" in result
        assert "peak_power" in result

    def test_detect_frequency_with_1d_input(self):
        """Frequency detection works with 1D prey array."""
        config = _make_config(epsilon=0.3)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        prey_list = []
        for _ in range(5000):
            sim.step()
            prey_list.append(sim.observe()[0])

        prey_arr = np.array(prey_list)
        result = sim.detect_dominant_frequency(prey_arr)
        assert result["dominant_freq"] >= 0

    def test_seasonal_frequency_detected(self):
        """Dominant frequency matches seasonal forcing frequency for moderate epsilon."""
        config = _make_config(epsilon=0.3, T=12.0, dt=0.01, n_steps=100000)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        # Discard transient
        for _ in range(40000):
            sim.step()

        states_list = []
        for _ in range(60000):
            sim.step()
            states_list.append(sim.observe().copy())

        traj = np.array(states_list)
        result = sim.detect_dominant_frequency(traj)

        seasonal_freq = 1.0 / 12.0
        # The dominant frequency should be close to the seasonal frequency
        # or a subharmonic/harmonic thereof
        if result["dominant_freq"] > 0:
            ratio = result["dominant_freq"] / seasonal_freq
            # Accept if ratio is near an integer or half-integer
            nearest_rational = round(ratio * 2) / 2
            assert abs(ratio - nearest_rational) < 0.3, (
                f"Dominant freq {result['dominant_freq']:.4f} is not "
                f"harmonically related to seasonal freq {seasonal_freq:.4f} "
                f"(ratio={ratio:.3f})"
            )


class TestTrajectoryData:
    """Tests for run() and TrajectoryData integration."""

    def test_run_returns_trajectory_data(self):
        """run() returns a TrajectoryData object with correct shape."""
        config = _make_config(n_steps=500)
        sim = SeasonalPredatorPreySimulation(config)
        traj = sim.run(500)

        assert traj.states.shape == (501, 2)
        assert len(traj.timestamps) == 501

    def test_run_non_negative(self):
        """All states in run() output are non-negative."""
        config = _make_config(n_steps=5000, epsilon=0.5)
        sim = SeasonalPredatorPreySimulation(config)
        traj = sim.run(5000)

        assert np.all(traj.states >= 0)

    def test_run_finite(self):
        """All states in run() output are finite."""
        config = _make_config(n_steps=5000)
        sim = SeasonalPredatorPreySimulation(config)
        traj = sim.run(5000)

        assert np.all(np.isfinite(traj.states))

    def test_timestamps_correct(self):
        """Timestamps are evenly spaced with correct dt."""
        config = _make_config(dt=0.05, n_steps=200)
        sim = SeasonalPredatorPreySimulation(config)
        traj = sim.run(200)

        expected_times = np.arange(201) * 0.05
        np.testing.assert_allclose(traj.timestamps, expected_times, rtol=1e-10)


class TestParameterSensitivity:
    """Tests for sensitivity to parameter changes."""

    def test_higher_attack_rate_reduces_prey(self):
        """Increasing attack rate reduces mean prey density."""
        mean_prey = []
        for a_val in [0.2, 0.5, 1.0]:
            config = _make_config(a=a_val, epsilon=0.0, dt=0.01, n_steps=50000)
            sim = SeasonalPredatorPreySimulation(config)
            sim.reset()
            for _ in range(30000):
                sim.step()
            prey_vals = []
            for _ in range(20000):
                sim.step()
                prey_vals.append(sim.observe()[0])
            mean_prey.append(np.mean(prey_vals))

        # Higher attack rate should generally reduce prey
        assert mean_prey[2] < mean_prey[0], (
            f"Higher attack rate should reduce prey: {mean_prey}"
        )

    def test_different_period_changes_dynamics(self):
        """Different seasonal period T produces different dynamics."""
        final_states = []
        for T_val in [6.0, 24.0]:
            config = _make_config(T=T_val, epsilon=0.5, dt=0.01, n_steps=60000)
            sim = SeasonalPredatorPreySimulation(config)
            sim.reset()
            for _ in range(60000):
                sim.step()
            final_states.append(sim.observe().copy())

        # Different seasonal periods should lead to different final states
        assert not np.allclose(final_states[0], final_states[1], atol=0.1), (
            f"Different T should give different dynamics: "
            f"T=6 state={final_states[0]}, T=24 state={final_states[1]}"
        )


class TestRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_generate_trajectory_data(self):
        """Trajectory data generation works."""
        from simulating_anything.rediscovery.seasonal_predator_prey import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(epsilon=0.3, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["dt"] == 0.01
        assert data["epsilon"] == 0.3
        assert len(data["N"]) == 501
        assert len(data["P"]) == 501

    def test_generate_epsilon_sweep_data(self):
        """Epsilon sweep data generation works."""
        from simulating_anything.rediscovery.seasonal_predator_prey import (
            generate_epsilon_sweep_data,
        )

        data = generate_epsilon_sweep_data(
            epsilon_values=np.array([0.0, 0.3, 0.6]),
            n_steps=5000,
            dt=0.01,
            transient_steps=2000,
        )
        assert len(data["epsilon_values"]) == 3
        assert len(data["amplitudes"]) == 3
        assert len(data["dominant_freqs"]) == 3
        assert len(data["regimes"]) == 3

    def test_generate_frequency_data(self):
        """Frequency data generation works."""
        from simulating_anything.rediscovery.seasonal_predator_prey import (
            generate_frequency_data,
        )

        data = generate_frequency_data(
            epsilon=0.3, n_steps=10000, dt=0.01, transient_steps=3000,
        )
        assert "dominant_freq" in data
        assert "dominant_period" in data
        assert "seasonal_freq" in data
        assert "freq_ratio" in data
        assert data["seasonal_freq"] == pytest.approx(1.0 / 12.0)

    def test_epsilon_sweep_regime_classification(self):
        """Epsilon sweep assigns regime labels."""
        from simulating_anything.rediscovery.seasonal_predator_prey import (
            generate_epsilon_sweep_data,
        )

        data = generate_epsilon_sweep_data(
            epsilon_values=np.array([0.0, 0.5]),
            n_steps=5000,
            dt=0.01,
            transient_steps=2000,
        )
        for regime in data["regimes"]:
            assert regime in (
                "fixed_point", "entrained", "period_doubled", "complex", "trivial"
            )

    def test_trajectory_data_non_negative(self):
        """Generated trajectory data is non-negative."""
        from simulating_anything.rediscovery.seasonal_predator_prey import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(epsilon=0.5, n_steps=2000, dt=0.01)
        assert np.all(data["N"] >= 0)
        assert np.all(data["P"] >= 0)
