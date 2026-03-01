"""Tests for the Delayed Predator-Prey simulation domain."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.delayed_predator_prey import (
    DelayedPredatorPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    tau: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    **overrides: float,
) -> SimulationConfig:
    params: dict[str, float] = {
        "r": 1.0, "K": 3.0, "a": 0.5, "h": 0.1,
        "e": 0.6, "m": 0.4, "tau": tau,
        "N_0": 2.0, "P_0": 1.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.DELAYED_PREDATOR_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestDelayedPredatorPreySimulation:
    """Core simulation tests."""

    def test_reset(self):
        """Initial state should have positive populations."""
        sim = DelayedPredatorPreySimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] > 0  # N > 0
        assert state[1] > 0  # P > 0
        np.testing.assert_allclose(state, [2.0, 1.0])

    def test_observe_shape(self):
        """Observe should return a 2-element array."""
        sim = DelayedPredatorPreySimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_step_advances(self):
        """State should change after a step."""
        sim = DelayedPredatorPreySimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Two runs with same config should give identical trajectories."""
        config = _make_config()
        sim1 = DelayedPredatorPreySimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe()

        sim2 = DelayedPredatorPreySimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe()

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_stability(self):
        """No NaN or Inf after 10000 steps."""
        sim = DelayedPredatorPreySimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_positive_populations(self):
        """N and P should remain non-negative throughout simulation."""
        sim = DelayedPredatorPreySimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert state[0] >= 0, f"Negative prey: {state[0]}"
            assert state[1] >= 0, f"Negative predator: {state[1]}"

    def test_no_delay_steady_state(self):
        """With tau=0 and K=3, system should converge near equilibrium."""
        config = _make_config(tau=0.0, dt=0.005, n_steps=200000)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        # Run to approach equilibrium
        for _ in range(100000):
            sim.step()

        # Average over last 5000 steps
        N_vals, P_vals = [], []
        for _ in range(5000):
            state = sim.step()
            N_vals.append(state[0])
            P_vals.append(state[1])

        N_mean = np.mean(N_vals)
        N_star, P_star = sim.find_equilibrium()

        # With K=3 (N*/K ~ 0.48), tau=0 system is stable
        assert N_star > 0, "No valid equilibrium found"
        assert abs(N_mean - N_star) / N_star < 0.05, (
            f"N not near equilibrium: {N_mean:.3f} vs {N_star:.3f}"
        )

    def test_delay_causes_oscillation(self):
        """Large tau should induce oscillations in prey."""
        config = _make_config(tau=4.0, dt=0.005, n_steps=50000)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(30000):
            sim.step()

        # Measure oscillation
        N_vals = []
        for _ in range(20000):
            state = sim.step()
            N_vals.append(state[0])

        N_range = max(N_vals) - min(N_vals)
        # With K=3 and tau=4.0, should see clear oscillation
        assert N_range > 0.1, f"No oscillation with tau=4: range={N_range:.4f}"

    def test_oscillation_amplitude_increases(self):
        """Larger tau should give larger oscillation amplitude."""
        amps = []
        for tau_val in [2.5, 3.5, 5.0]:
            config = _make_config(tau=tau_val, dt=0.005, n_steps=50000)
            sim = DelayedPredatorPreySimulation(config)
            sim.reset()

            for _ in range(30000):
                sim.step()

            N_vals = []
            for _ in range(20000):
                state = sim.step()
                N_vals.append(state[0])

            amps.append(max(N_vals) - min(N_vals))

        # Amplitude at tau=5 should exceed amplitude at tau=2.5
        assert amps[2] > amps[0], (
            f"Amplitude did not increase: tau=2.5 -> {amps[0]:.4f}, "
            f"tau=5 -> {amps[2]:.4f}"
        )

    def test_period_increases_with_tau(self):
        """Longer delay should produce longer oscillation period."""
        periods = []
        for tau_val in [3.0, 5.0]:
            config = _make_config(tau=tau_val, dt=0.005, n_steps=80000)
            sim = DelayedPredatorPreySimulation(config)
            sim.reset()

            for _ in range(40000):
                sim.step()

            N_vals = []
            for _ in range(40000):
                state = sim.step()
                N_vals.append(state[0])

            T = DelayedPredatorPreySimulation._fft_period(
                np.array(N_vals), 0.005
            )
            periods.append(T)

        # Both should have detectable period (both are past Hopf bifurcation)
        assert all(T > 0 for T in periods), f"Zero periods: {periods}"
        # Larger tau -> longer period
        assert periods[1] > periods[0], (
            f"Period did not increase: tau=3 -> {periods[0]:.3f}, "
            f"tau=5 -> {periods[1]:.3f}"
        )

    def test_functional_response(self):
        """Holling Type II should be saturating."""
        sim = DelayedPredatorPreySimulation(_make_config())
        sim.reset()

        # Should be 0 at N=0
        assert sim.functional_response(0.0) == 0.0

        # Should increase with N
        fr1 = sim.functional_response(1.0)
        fr5 = sim.functional_response(5.0)
        fr100 = sim.functional_response(100.0)
        assert 0 < fr1 < fr5 < fr100

        # Should saturate toward 1/h = 10
        assert fr100 < 1.0 / sim.h
        # At very large N, should approach 1/h
        fr_large = sim.functional_response(1e6)
        np.testing.assert_allclose(fr_large, 1.0 / sim.h, rtol=0.01)

    def test_equilibrium_exists(self):
        """Equilibrium should exist for default parameters."""
        sim = DelayedPredatorPreySimulation(_make_config())
        sim.reset()
        N_star, P_star = sim.find_equilibrium()
        assert N_star > 0, "No positive prey equilibrium"
        assert P_star > 0, "No positive predator equilibrium"
        assert N_star < sim.K, "Prey equilibrium exceeds carrying capacity"

    def test_prey_bounded(self):
        """Prey should remain bounded by carrying capacity (approximately)."""
        config = _make_config(tau=3.0, dt=0.005, n_steps=50000)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        max_N = 0.0
        for _ in range(50000):
            state = sim.step()
            max_N = max(max_N, state[0])

        # Prey can overshoot K slightly with delay, but not hugely
        assert max_N < sim.K * 2.0, (
            f"Prey exceeded 2*K: max_N={max_N:.3f}, K={sim.K}"
        )

    def test_delay_sweep(self):
        """delay_sweep should produce valid amplitude/period arrays."""
        config = _make_config(dt=0.01)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        tau_vals = np.array([0.0, 1.0, 2.0, 3.0])
        result = sim.delay_sweep(
            tau_vals, n_steps=5000, n_transient=2000,
        )
        assert len(result["tau"]) == 4
        assert len(result["amplitude"]) == 4
        assert len(result["period"]) == 4
        assert np.all(np.isfinite(result["amplitude"]))

    def test_hopf_bifurcation(self):
        """Critical tau should be detected for default parameters."""
        config = _make_config(tau=0.0, dt=0.01)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        tau_vals = np.linspace(0.0, 5.0, 20)
        tau_c = sim.hopf_bifurcation_detect(
            tau_vals, n_steps=30000, n_transient=15000,
            amplitude_threshold=0.05,
        )
        # With K=3.0, Hopf bifurcation should occur around tau ~ 2-4
        assert tau_c is not None, "No Hopf bifurcation detected"
        assert 0 < tau_c <= 5.0, f"tau_c out of range: {tau_c}"

    def test_small_delay_stable(self):
        """Very small tau should not destabilize the equilibrium."""
        config = _make_config(tau=0.01, dt=0.005, n_steps=100000)
        sim = DelayedPredatorPreySimulation(config)
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
        N_star, _ = sim.find_equilibrium()

        # Small delay should keep system near equilibrium
        assert N_range < 0.01, (
            f"Too much oscillation at tau=0.01: range={N_range:.4f}"
        )

    def test_run_trajectory(self):
        """The run() method should produce valid TrajectoryData."""
        config = _make_config(dt=0.01, n_steps=100)
        sim = DelayedPredatorPreySimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert np.all(np.isfinite(traj.states))
        assert np.all(traj.states >= 0)

    def test_compute_period(self):
        """compute_period should return a positive value for oscillating system."""
        config = _make_config(tau=4.0, dt=0.005, n_steps=1000)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()
        T = sim.compute_period(n_steps=30000)
        # tau=4 with K=3 should oscillate
        assert T > 0, f"No period detected at tau=4: {T}"

    def test_no_delay_comparison(self):
        """no_delay_comparison should return trajectory data."""
        config = _make_config(tau=2.0, dt=0.01)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()
        result = sim.no_delay_comparison(n_steps=1000)
        assert len(result["N"]) == 1000
        assert len(result["P"]) == 1000
        assert result["final_state"].shape == (2,)
        assert np.all(np.isfinite(result["N"]))

    def test_equilibrium_analytical(self):
        """Equilibrium N* = m/(a*(e-m*h)) should match analytical formula."""
        sim = DelayedPredatorPreySimulation(_make_config())
        sim.reset()
        N_star, P_star = sim.find_equilibrium()

        # N* = 0.4 / (0.5 * (0.6 - 0.04)) = 0.4 / 0.28 = 1.4286
        expected_N = 0.4 / (0.5 * (0.6 - 0.4 * 0.1))
        np.testing.assert_allclose(N_star, expected_N, rtol=1e-6)


class TestDelayedPredatorPreyRediscovery:
    """Tests for the rediscovery data generation."""

    def test_delay_sweep_data(self):
        from simulating_anything.rediscovery.delayed_predator_prey import (
            generate_delay_sweep_data,
        )
        data = generate_delay_sweep_data(
            n_tau=5, tau_min=0.0, tau_max=3.0,
            dt=0.01, n_steps=5000, n_transient=2000,
        )
        assert len(data["tau"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["period"]) == 5
        assert np.all(np.isfinite(data["amplitude"]))

    def test_equilibrium_data(self):
        from simulating_anything.rediscovery.delayed_predator_prey import (
            generate_equilibrium_data,
        )
        data = generate_equilibrium_data(dt=0.01, n_steps=50000)
        assert "N_measured" in data
        assert "N_theory" in data
        assert "P_measured" in data
        assert "P_theory" in data
        assert data["N_theory"] > 0

    def test_fft_period_no_oscillation(self):
        """FFT period on constant signal should return 0."""
        signal = np.ones(1000)
        T = DelayedPredatorPreySimulation._fft_period(signal, 0.01)
        assert T == 0.0

    def test_fft_period_sine(self):
        """FFT should recover the period of a clean sine wave."""
        dt = 0.01
        freq = 0.5  # Hz
        t = np.arange(10000) * dt
        signal = np.sin(2 * np.pi * freq * t)
        T = DelayedPredatorPreySimulation._fft_period(signal, dt)
        # Period should be approximately 1/freq = 2.0
        np.testing.assert_allclose(T, 1.0 / freq, rtol=0.05)
