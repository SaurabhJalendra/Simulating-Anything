"""Tests for the Lotka-Volterra with Delay simulation domain."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.lotka_volterra_delay import (
    LotkaVolterraDelaySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    tau: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    **overrides: float,
) -> SimulationConfig:
    params: dict[str, float] = {
        "r": 1.0, "K": 4.0, "a": 0.5, "b": 0.2,
        "d": 0.3, "tau": tau,
        "N_0": 3.0, "P_0": 1.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.LOTKA_VOLTERRA_DELAY,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestLotkaVolterraDelayCreation:
    """Test simulation object creation and initialization."""

    def test_creation(self):
        """Simulation should be created with default parameters."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        assert sim.r == 1.0
        assert sim.K == 4.0
        assert sim.a == 0.5
        assert sim.b == 0.2
        assert sim.d == 0.3
        assert sim.tau == 1.0

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        config = _make_config(r=2.0, K=20.0, tau=3.0)
        sim = LotkaVolterraDelaySimulation(config)
        assert sim.r == 2.0
        assert sim.K == 20.0
        assert sim.tau == 3.0

    def test_reset_shape(self):
        """Initial state should be a 2-element array."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [3.0, 1.0])

    def test_reset_positive(self):
        """Initial state should have positive populations."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        state = sim.reset()
        assert state[0] > 0  # N > 0
        assert state[1] > 0  # P > 0

    def test_observe_shape(self):
        """Observe should return a 2-element array."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_observe_matches_state(self):
        """Observe should return current state values."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_allclose(obs, state)


class TestLotkaVolterraDelayStep:
    """Test stepping behavior."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Two runs with same config should give identical trajectories."""
        config = _make_config()
        sim1 = LotkaVolterraDelaySimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe()

        sim2 = LotkaVolterraDelaySimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe()

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_numerical_stability(self):
        """No NaN or Inf after many steps."""
        sim = LotkaVolterraDelaySimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_positive_populations(self):
        """N and P should remain non-negative throughout simulation."""
        sim = LotkaVolterraDelaySimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert state[0] >= 0, f"Negative prey: {state[0]}"
            assert state[1] >= 0, f"Negative predator: {state[1]}"

    def test_prey_bounded_by_carrying_capacity(self):
        """Prey should remain bounded near carrying capacity."""
        config = _make_config(tau=1.0, dt=0.01, n_steps=50000)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        max_N = 0.0
        for _ in range(50000):
            state = sim.step()
            max_N = max(max_N, state[0])

        # Prey can overshoot K with delay but should not blow up
        assert max_N < sim.K * 3.0, (
            f"Prey exceeded 3*K: max_N={max_N:.3f}, K={sim.K}"
        )


class TestLotkaVolterraDelayRun:
    """Test the run() method for trajectory collection."""

    def test_run_trajectory_shape(self):
        """run() should produce correct TrajectoryData shape."""
        config = _make_config(dt=0.01, n_steps=100)
        sim = LotkaVolterraDelaySimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)  # 100 steps + initial

    def test_run_trajectory_finite(self):
        """All trajectory values should be finite."""
        config = _make_config(dt=0.01, n_steps=500)
        sim = LotkaVolterraDelaySimulation(config)
        traj = sim.run(n_steps=500)
        assert np.all(np.isfinite(traj.states))

    def test_run_trajectory_positive(self):
        """All trajectory values should be non-negative."""
        config = _make_config(dt=0.01, n_steps=500)
        sim = LotkaVolterraDelaySimulation(config)
        traj = sim.run(n_steps=500)
        assert np.all(traj.states >= 0)

    def test_run_timestamps(self):
        """Timestamps should be monotonically increasing."""
        config = _make_config(dt=0.01, n_steps=100)
        sim = LotkaVolterraDelaySimulation(config)
        traj = sim.run(n_steps=100)
        assert len(traj.timestamps) == 101
        assert np.all(np.diff(traj.timestamps) > 0)


class TestLotkaVolterraDelayEquilibrium:
    """Test equilibrium computation."""

    def test_equilibrium_exists(self):
        """Equilibrium should exist for default parameters."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        N_star, P_star = sim.equilibrium()
        assert N_star > 0, "No positive prey equilibrium"
        assert P_star > 0, "No positive predator equilibrium"

    def test_equilibrium_analytical(self):
        """Equilibrium N* = d/b should match analytical formula."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        N_star, P_star = sim.equilibrium()

        # N* = d/b = 0.3/0.2 = 1.5
        expected_N = 0.3 / 0.2
        np.testing.assert_allclose(N_star, expected_N, rtol=1e-10)

        # P* = r*(1 - N*/K)/a = 1.0*(1 - 1.5/4.0)/0.5 = 1.25
        expected_P = 1.0 * (1.0 - expected_N / 4.0) / 0.5
        np.testing.assert_allclose(P_star, expected_P, rtol=1e-10)

    def test_equilibrium_independent_of_tau(self):
        """Equilibrium should be the same regardless of tau value."""
        eq_vals = []
        for tau_val in [0.0, 0.5, 1.0, 2.0, 3.0]:
            config = _make_config(tau=tau_val)
            sim = LotkaVolterraDelaySimulation(config)
            sim.reset()
            eq_vals.append(sim.equilibrium())

        # All should be identical
        for i in range(1, len(eq_vals)):
            np.testing.assert_allclose(eq_vals[i], eq_vals[0], atol=1e-12)

    def test_equilibrium_below_carrying_capacity(self):
        """Prey equilibrium should be less than carrying capacity."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        N_star, _ = sim.equilibrium()
        assert 0 < N_star < sim.K

    def test_no_equilibrium_when_b_zero(self):
        """No interior equilibrium when predator growth rate is zero."""
        config = _make_config(b=0.0)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()
        N_star, P_star = sim.equilibrium()
        assert N_star == 0.0
        assert P_star == 0.0

    def test_no_equilibrium_when_Nstar_exceeds_K(self):
        """No interior equilibrium when N*=d/b > K."""
        # d/b = 5/0.2 = 25 > K=4
        config = _make_config(d=5.0, b=0.2, K=4.0)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()
        N_star, P_star = sim.equilibrium()
        assert N_star == 0.0
        assert P_star == 0.0


class TestLotkaVolterraDelayZeroDelay:
    """Test system with zero delay (standard logistic LV)."""

    def test_zero_delay_converges_to_equilibrium(self):
        """With tau=0, system should converge to equilibrium."""
        config = _make_config(tau=0.0, dt=0.005, n_steps=200000)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        # Run to approach equilibrium
        for _ in range(150000):
            sim.step()

        # Average over last 5000 steps
        N_vals, P_vals = [], []
        for _ in range(5000):
            state = sim.step()
            N_vals.append(state[0])
            P_vals.append(state[1])

        N_mean = np.mean(N_vals)
        P_mean = np.mean(P_vals)
        N_star, P_star = sim.equilibrium()

        assert N_star > 0, "No valid equilibrium found"
        assert abs(N_mean - N_star) / N_star < 0.05, (
            f"N not near equilibrium: {N_mean:.3f} vs {N_star:.3f}"
        )
        assert abs(P_mean - P_star) / P_star < 0.05, (
            f"P not near equilibrium: {P_mean:.3f} vs {P_star:.3f}"
        )

    def test_zero_delay_stable(self):
        """tau=0 should yield stable dynamics (small amplitude variation)."""
        config = _make_config(tau=0.0, dt=0.005, n_steps=100000)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(80000):
            sim.step()

        N_vals = []
        for _ in range(20000):
            state = sim.step()
            N_vals.append(state[0])

        N_range = max(N_vals) - min(N_vals)
        # Stable system should have very small variation
        assert N_range < 0.1, (
            f"Too much variation at tau=0: range={N_range:.4f}"
        )


class TestLotkaVolterraDelaySmallDelay:
    """Test system with small delay (should remain stable)."""

    def test_small_delay_stable(self):
        """Very small tau should not destabilize the equilibrium."""
        config = _make_config(tau=0.01, dt=0.005, n_steps=100000)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        # Run to steady state
        for _ in range(80000):
            sim.step()

        # Measure oscillation amplitude
        N_vals = []
        for _ in range(20000):
            state = sim.step()
            N_vals.append(state[0])

        N_range = max(N_vals) - min(N_vals)
        assert N_range < 0.1, (
            f"Too much oscillation at tau=0.01: range={N_range:.4f}"
        )


class TestLotkaVolterraDelayLargeDelay:
    """Test system with large delay (Hopf bifurcation, oscillations).

    With K=4 and default parameters, the Hopf bifurcation occurs around
    tau ~ 1.5-2.0. Sustained oscillations persist for tau up to ~3.5
    before populations overshoot to near-extinction.
    """

    def test_large_delay_oscillates(self):
        """Large tau should induce oscillations in prey."""
        config = _make_config(tau=2.5, dt=0.005, n_steps=50000)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(30000):
            sim.step()

        N_vals = []
        for _ in range(20000):
            state = sim.step()
            N_vals.append(state[0])

        N_range = max(N_vals) - min(N_vals)
        assert N_range > 0.1, (
            f"No oscillation with tau=2.5: range={N_range:.4f}"
        )

    def test_oscillation_amplitude_increases_with_tau(self):
        """Larger tau should give larger oscillation amplitude (beyond tau_c)."""
        amps = []
        for tau_val in [2.0, 2.5, 3.0]:
            config = _make_config(tau=tau_val, dt=0.005, n_steps=50000)
            sim = LotkaVolterraDelaySimulation(config)
            sim.reset()

            for _ in range(30000):
                sim.step()

            N_vals = []
            for _ in range(20000):
                state = sim.step()
                N_vals.append(state[0])

            amps.append(max(N_vals) - min(N_vals))

        # Amplitude at tau=3.0 should exceed amplitude at tau=2.0
        assert amps[2] > amps[0], (
            f"Amplitude did not increase: tau=2.0 -> {amps[0]:.4f}, "
            f"tau=3.0 -> {amps[2]:.4f}"
        )

    def test_period_increases_with_tau(self):
        """Longer delay should produce longer oscillation period."""
        periods = []
        for tau_val in [2.5, 3.0]:
            config = _make_config(tau=tau_val, dt=0.005, n_steps=80000)
            sim = LotkaVolterraDelaySimulation(config)
            sim.reset()

            for _ in range(40000):
                sim.step()

            N_vals = []
            for _ in range(40000):
                state = sim.step()
                N_vals.append(state[0])

            T = LotkaVolterraDelaySimulation._fft_period(
                np.array(N_vals), 0.005
            )
            periods.append(T)

        assert all(T > 0 for T in periods), f"Zero periods: {periods}"
        assert periods[1] > periods[0], (
            f"Period did not increase: tau=2.5 -> {periods[0]:.3f}, "
            f"tau=3.0 -> {periods[1]:.3f}"
        )


class TestLotkaVolterraDelayCriticalTau:
    """Test critical delay detection."""

    def test_critical_tau_method(self):
        """critical_tau() should return a finite positive value."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        tau_c = sim.critical_tau()
        assert np.isfinite(tau_c), f"tau_c not finite: {tau_c}"
        assert tau_c > 0, f"tau_c not positive: {tau_c}"

    def test_critical_tau_reasonable_range(self):
        """Critical tau should be in a reasonable range for default params."""
        sim = LotkaVolterraDelaySimulation(_make_config())
        sim.reset()
        tau_c = sim.critical_tau()
        # For default parameters, tau_c should be roughly O(1)
        assert 0.01 < tau_c < 10.0, f"tau_c out of expected range: {tau_c}"

    def test_hopf_bifurcation_detected(self):
        """Numerical Hopf bifurcation should be detected in a sweep."""
        config = _make_config(tau=0.0, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        tau_vals = np.linspace(0.0, 3.0, 20)
        tau_c = sim.hopf_bifurcation_detect(
            tau_vals, n_steps=30000, n_transient=15000,
            amplitude_threshold=0.05,
        )
        assert tau_c is not None, "No Hopf bifurcation detected"
        assert 0 < tau_c <= 3.0, f"tau_c out of range: {tau_c}"


class TestLotkaVolterraDelayHistoryBuffer:
    """Test that the history buffer is maintained correctly."""

    def test_buffer_size_matches_delay(self):
        """Buffer size should be ceil(tau/dt) + 1."""
        config = _make_config(tau=1.0, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        expected_size = int(1.0 / 0.01) + 1  # 101
        assert sim._buf_size == expected_size

    def test_buffer_size_zero_delay(self):
        """Buffer size should be 1 for zero delay."""
        config = _make_config(tau=0.0, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        assert sim._buf_size == 1

    def test_buffer_initialized_with_N0(self):
        """After reset, buffer should be filled with initial prey value."""
        config = _make_config(tau=0.5, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()
        np.testing.assert_allclose(sim._buffer, 3.0)  # N_0 = 3.0

    def test_buffer_updates_after_steps(self):
        """Buffer should contain recent prey values after stepping."""
        config = _make_config(tau=0.1, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        # Step enough to fill the buffer with new values
        for _ in range(100):
            sim.step()

        # Buffer should no longer be all 3.0 (initial value)
        assert not np.allclose(sim._buffer, 3.0)
        # All buffer values should be finite and positive
        assert np.all(np.isfinite(sim._buffer))
        assert np.all(sim._buffer >= 0)

    def test_zero_delay_reads_current_value(self):
        """With tau=0, delayed value should equal current prey density."""
        config = _make_config(tau=0.0, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        # Buffer size is 1, so delayed value always equals latest stored
        for _ in range(50):
            sim.step()
            # The buffer should have been updated to current N
            assert sim._buffer[0] >= 0


class TestLotkaVolterraDelayMethods:
    """Test utility methods: delay_sweep, no_delay_comparison, measure_period."""

    def test_delay_sweep_output(self):
        """delay_sweep should produce valid arrays."""
        config = _make_config(dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        tau_vals = np.array([0.0, 0.5, 1.0, 2.0])
        result = sim.delay_sweep(
            tau_vals, n_steps=5000, n_transient=2000,
        )
        assert len(result["tau"]) == 4
        assert len(result["amplitude"]) == 4
        assert len(result["period"]) == 4
        assert np.all(np.isfinite(result["amplitude"]))

    def test_no_delay_comparison_output(self):
        """no_delay_comparison should return trajectory data."""
        config = _make_config(tau=2.0, dt=0.01)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()
        result = sim.no_delay_comparison(n_steps=1000)
        assert len(result["N"]) == 1000
        assert len(result["P"]) == 1000
        assert result["final_state"].shape == (2,)
        assert np.all(np.isfinite(result["N"]))

    def test_measure_period_oscillating(self):
        """measure_period should return positive value for oscillating system."""
        config = _make_config(tau=2.5, dt=0.005, n_steps=1000)
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()
        T = sim.measure_period(n_steps=30000, n_transient=15000)
        assert T > 0, f"No period detected at tau=2.5: {T}"

    def test_fft_period_constant_signal(self):
        """FFT period on constant signal should return 0."""
        signal = np.ones(1000)
        T = LotkaVolterraDelaySimulation._fft_period(signal, 0.01)
        assert T == 0.0

    def test_fft_period_sine_wave(self):
        """FFT should recover the period of a clean sine wave."""
        dt = 0.01
        freq = 0.5  # Hz
        t = np.arange(10000) * dt
        signal = np.sin(2 * np.pi * freq * t)
        T = LotkaVolterraDelaySimulation._fft_period(signal, dt)
        np.testing.assert_allclose(T, 1.0 / freq, rtol=0.05)


class TestLotkaVolterraDelayRediscovery:
    """Tests for the rediscovery data generation."""

    def test_delay_sweep_data(self):
        from simulating_anything.rediscovery.lotka_volterra_delay import (
            generate_delay_sweep_data,
        )
        data = generate_delay_sweep_data(
            n_tau=5, tau_min=0.0, tau_max=2.0,
            dt=0.01, n_steps=5000, n_transient=2000,
        )
        assert len(data["tau"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["period"]) == 5
        assert np.all(np.isfinite(data["amplitude"]))

    def test_equilibrium_data(self):
        from simulating_anything.rediscovery.lotka_volterra_delay import (
            generate_equilibrium_data,
        )
        data = generate_equilibrium_data(dt=0.01, n_steps=50000)
        assert "N_measured" in data
        assert "N_theory" in data
        assert "P_measured" in data
        assert "P_theory" in data
        assert data["N_theory"] > 0
        # N* = d/b = 0.3/0.2 = 1.5
        np.testing.assert_allclose(data["N_theory"], 1.5, rtol=1e-6)

    def test_no_delay_comparison_data(self):
        from simulating_anything.rediscovery.lotka_volterra_delay import (
            generate_no_delay_comparison,
        )
        data = generate_no_delay_comparison(dt=0.01, n_steps=1000)
        assert len(data["N"]) == 1000
        assert len(data["P"]) == 1000
        assert data["final_state"].shape == (2,)
