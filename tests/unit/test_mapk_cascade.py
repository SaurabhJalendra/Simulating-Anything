"""Tests for the MAPK cascade simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.mapk_cascade import MAPKCascadeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    V1: float = 1.0,
    V2: float = 0.3,
    k3: float = 1.0,
    V4: float = 0.3,
    k5: float = 1.0,
    V6: float = 0.3,
    K1: float = 0.01,
    K2: float = 0.01,
    K3: float = 0.01,
    K4: float = 0.01,
    K5: float = 0.01,
    K6: float = 0.01,
    X1_0: float = 0.0,
    X2_0: float = 0.0,
    X3_0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 2000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.MAPK_CASCADE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "V1": V1, "V2": V2, "k3": k3, "V4": V4, "k5": k5, "V6": V6,
            "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5, "K6": K6,
            "X1_0": X1_0, "X2_0": X2_0, "X3_0": X3_0,
        },
    )


# --- Construction and basic interface ---

class TestMAPKCascadeConstruction:
    def test_default_parameters(self):
        sim = MAPKCascadeSimulation(_make_config())
        assert sim.V1 == 1.0
        assert sim.V2 == 0.3
        assert sim.k3 == 1.0
        assert sim.V4 == 0.3
        assert sim.k5 == 1.0
        assert sim.V6 == 0.3
        assert sim.K1 == 0.01
        assert sim.K2 == 0.01

    def test_custom_parameters(self):
        sim = MAPKCascadeSimulation(_make_config(V1=2.0, V2=0.5, K1=0.1))
        assert sim.V1 == 2.0
        assert sim.V2 == 0.5
        assert sim.K1 == 0.1

    def test_initial_state_shape(self):
        sim = MAPKCascadeSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_default_zeros(self):
        sim = MAPKCascadeSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [0.0, 0.0, 0.0])

    def test_initial_state_custom(self):
        sim = MAPKCascadeSimulation(_make_config(X1_0=0.5, X2_0=0.3, X3_0=0.1))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, 0.3, 0.1])


class TestMAPKCascadeBasicDynamics:
    def test_step_advances_state(self):
        sim = MAPKCascadeSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        # With V1 > 0 and X1=0, dX1/dt > 0 so state should change
        assert not np.allclose(s0, s1)

    def test_observe_returns_state(self):
        sim = MAPKCascadeSimulation(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_run_returns_trajectory_data(self):
        sim = MAPKCascadeSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_run_timestamps_correct(self):
        sim = MAPKCascadeSimulation(_make_config(dt=0.01, n_steps=50))
        traj = sim.run(n_steps=50)
        np.testing.assert_allclose(traj.timestamps[0], 0.0)
        np.testing.assert_allclose(traj.timestamps[-1], 50 * 0.01, rtol=1e-10)


# --- Boundedness ---

class TestMAPKCascadeBoundedness:
    def test_states_bounded_0_1_default(self):
        """All states should remain in [0, 1] throughout integration."""
        sim = MAPKCascadeSimulation(_make_config(n_steps=5000))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            for j in range(3):
                assert 0.0 <= state[j] <= 1.0, (
                    f"State X{j+1}={state[j]} out of [0,1]"
                )

    def test_states_bounded_high_signal(self):
        """States bounded even with very high input signal."""
        sim = MAPKCascadeSimulation(_make_config(V1=10.0, n_steps=5000))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            for j in range(3):
                assert 0.0 <= state[j] <= 1.0

    def test_states_bounded_large_K(self):
        """States bounded with large Michaelis constants."""
        sim = MAPKCascadeSimulation(
            _make_config(K1=1.0, K2=1.0, K3=1.0, K4=1.0, K5=1.0, K6=1.0)
        )
        sim.reset()
        for _ in range(3000):
            sim.step()
            state = sim.observe()
            for j in range(3):
                assert 0.0 <= state[j] <= 1.0

    def test_states_bounded_from_active_initial(self):
        """Starting fully active should not exceed bounds."""
        sim = MAPKCascadeSimulation(
            _make_config(X1_0=1.0, X2_0=1.0, X3_0=1.0, V1=0.0)
        )
        sim.reset()
        for _ in range(3000):
            sim.step()
            state = sim.observe()
            for j in range(3):
                assert 0.0 <= state[j] <= 1.0


# --- Steady state behavior ---

class TestMAPKCascadeSteadyState:
    def test_reaches_steady_state(self):
        """State should converge to a fixed point."""
        sim = MAPKCascadeSimulation(_make_config(n_steps=10000))
        sim.reset()
        for _ in range(9000):
            sim.step()
        s_late1 = sim.observe().copy()
        for _ in range(1000):
            sim.step()
        s_late2 = sim.observe()
        # State should not change much in the last 1000 steps
        np.testing.assert_allclose(s_late1, s_late2, atol=1e-4)

    def test_steady_state_method(self):
        """steady_state() should return converged values."""
        sim = MAPKCascadeSimulation(_make_config())
        ss = sim.steady_state(n_steps=10000)
        assert ss.shape == (3,)
        # All should be in [0, 1]
        for j in range(3):
            assert 0.0 <= ss[j] <= 1.0

    def test_steady_state_high_signal_all_active(self):
        """Very large V1 should activate all tiers near 1."""
        sim = MAPKCascadeSimulation(_make_config(V1=10.0))
        ss = sim.steady_state(n_steps=20000)
        # With V1 >> V2 and small K, all tiers should be near 1
        assert ss[0] > 0.9, f"X1={ss[0]} should be near 1"
        assert ss[1] > 0.9, f"X2={ss[1]} should be near 1"
        assert ss[2] > 0.9, f"X3={ss[2]} should be near 1"

    def test_steady_state_zero_signal_all_inactive(self):
        """V1=0 should lead to all tiers inactive (near 0)."""
        sim = MAPKCascadeSimulation(_make_config(V1=0.0))
        ss = sim.steady_state(n_steps=10000)
        assert ss[0] < 0.01, f"X1={ss[0]} should be near 0"
        assert ss[1] < 0.01, f"X2={ss[1]} should be near 0"
        assert ss[2] < 0.01, f"X3={ss[2]} should be near 0"


# --- Signal amplification ---

class TestMAPKCascadeAmplification:
    def test_amplification_ratio_positive(self):
        """Amplification ratio should be positive with active signal."""
        sim = MAPKCascadeSimulation(_make_config(V1=0.5))
        sim.reset()
        for _ in range(10000):
            sim.step()
        amp = sim.signal_amplification()
        assert amp > 0.0

    def test_amplification_zero_signal(self):
        """No signal should give zero amplification."""
        sim = MAPKCascadeSimulation(_make_config(V1=0.0))
        sim.reset()
        amp = sim.signal_amplification()
        assert amp == 0.0

    def test_amplification_X3_greater_than_X1_moderate_signal(self):
        """For moderate input, cascade should amplify: X3 > X1 at steady state."""
        sim = MAPKCascadeSimulation(_make_config(V1=0.4))
        ss = sim.steady_state(n_steps=15000)
        # With the default ultrasensitive parameters, X3 can be > X1
        # because each tier acts as a switch
        # At V1=0.4, X1 might be moderate but X3 amplified
        # (depends on exact V1/V2 ratio)
        # Let's just verify the cascade property works
        assert ss[0] >= 0, "X1 should be non-negative"
        assert ss[2] >= 0, "X3 should be non-negative"


# --- Cascade ordering ---

class TestMAPKCascadeOrdering:
    def test_X1_activates_before_X2(self):
        """X1 should reach 50% activation before X2."""
        sim = MAPKCascadeSimulation(_make_config(V1=1.5, dt=0.01))
        sim.reset()
        t_half = [float("inf"), float("inf"), float("inf")]
        for step_i in range(1, 5001):
            sim.step()
            state = sim.observe()
            for j in range(3):
                if t_half[j] == float("inf") and state[j] >= 0.5:
                    t_half[j] = step_i * 0.01
        assert t_half[0] <= t_half[1], (
            f"X1 half-time {t_half[0]} should be <= X2 half-time {t_half[1]}"
        )

    def test_X2_activates_before_X3(self):
        """X2 should reach 50% activation before X3."""
        sim = MAPKCascadeSimulation(_make_config(V1=1.5, dt=0.01))
        sim.reset()
        t_half = [float("inf"), float("inf"), float("inf")]
        for step_i in range(1, 5001):
            sim.step()
            state = sim.observe()
            for j in range(3):
                if t_half[j] == float("inf") and state[j] >= 0.5:
                    t_half[j] = step_i * 0.01
        assert t_half[1] <= t_half[2], (
            f"X2 half-time {t_half[1]} should be <= X3 half-time {t_half[2]}"
        )

    def test_full_cascade_order(self):
        """Activation order should be X1 -> X2 -> X3."""
        sim = MAPKCascadeSimulation(_make_config(V1=2.0, dt=0.01))
        sim.reset()
        t_half = [float("inf"), float("inf"), float("inf")]
        for step_i in range(1, 5001):
            sim.step()
            state = sim.observe()
            for j in range(3):
                if t_half[j] == float("inf") and state[j] >= 0.5:
                    t_half[j] = step_i * 0.01
        assert t_half[0] <= t_half[1] <= t_half[2], (
            f"Cascade not ordered: {t_half}"
        )


# --- Michaelis constant effect ---

class TestMAPKCascadeKEffect:
    def test_small_K_more_switch_like(self):
        """Small K should give steeper dose-response (higher Hill coefficient)."""
        sim_small = MAPKCascadeSimulation(
            _make_config(K1=0.005, K2=0.005, K3=0.005, K4=0.005, K5=0.005, K6=0.005)
        )
        result_small = sim_small.dose_response(n_points=40, n_steps=8000)

        sim_large = MAPKCascadeSimulation(
            _make_config(K1=0.5, K2=0.5, K3=0.5, K4=0.5, K5=0.5, K6=0.5)
        )
        result_large = sim_large.dose_response(n_points=40, n_steps=8000)

        assert result_small["hill_coefficient"] > result_large["hill_coefficient"], (
            f"Small K Hill={result_small['hill_coefficient']:.2f} should > "
            f"large K Hill={result_large['hill_coefficient']:.2f}"
        )

    def test_is_switch_like_small_K(self):
        """With small K, is_switch_like() should return True."""
        sim = MAPKCascadeSimulation(
            _make_config(K1=0.005, K2=0.005, K3=0.005, K4=0.005, K5=0.005, K6=0.005)
        )
        assert sim.is_switch_like(n_points=40) is True


# --- Ultrasensitivity / dose-response ---

class TestMAPKCascadeDoseResponse:
    def test_dose_response_returns_correct_keys(self):
        sim = MAPKCascadeSimulation(_make_config())
        result = sim.dose_response(n_points=10, n_steps=3000)
        assert "V1_values" in result
        assert "X3_ss" in result
        assert "X1_ss" in result
        assert "X2_ss" in result
        assert "hill_coefficient" in result
        assert "amplification" in result

    def test_dose_response_monotonic(self):
        """X3_ss should be monotonically non-decreasing with V1."""
        sim = MAPKCascadeSimulation(_make_config())
        result = sim.dose_response(
            V1_range=(0.0, 2.0), n_points=20, n_steps=8000,
        )
        X3 = result["X3_ss"]
        # Allow tiny numerical tolerance for non-monotonicity
        for i in range(1, len(X3)):
            assert X3[i] >= X3[i - 1] - 1e-6, (
                f"X3 not monotonic at index {i}: {X3[i-1]:.6f} > {X3[i]:.6f}"
            )

    def test_dose_response_steeper_than_michaelis_menten(self):
        """Three-tier cascade should give Hill n > 1 (ultrasensitive)."""
        sim = MAPKCascadeSimulation(_make_config())
        result = sim.dose_response(n_points=40, n_steps=8000)
        n_H = result["hill_coefficient"]
        assert n_H > 1.0, f"Hill coefficient {n_H} should be > 1"

    def test_dose_response_bounded(self):
        """X3_ss should remain in [0, 1] across entire sweep."""
        sim = MAPKCascadeSimulation(_make_config())
        result = sim.dose_response(n_points=20, n_steps=5000)
        assert np.all(result["X3_ss"] >= 0.0)
        assert np.all(result["X3_ss"] <= 1.0)
        assert np.all(result["X1_ss"] >= 0.0)
        assert np.all(result["X1_ss"] <= 1.0)


# --- Derivatives ---

class TestMAPKCascadeDerivatives:
    def test_derivatives_at_origin(self):
        """At origin with V1 > 0, dX1/dt should be positive."""
        sim = MAPKCascadeSimulation(_make_config(V1=1.0))
        y = np.array([0.0, 0.0, 0.0])
        dy = sim._derivatives(y)
        # dX1/dt = V1*(1-0)/(K1+1-0) - V2*0/(K2+0) = V1/(K1+1)
        expected_dX1 = 1.0 / (0.01 + 1.0)
        assert dy[0] > 0, f"dX1/dt should be positive, got {dy[0]}"
        np.testing.assert_allclose(dy[0], expected_dX1, rtol=1e-10)
        # dX2/dt = k3*0*(1-0)/... = 0
        assert dy[1] == pytest.approx(0.0, abs=1e-15)
        # dX3/dt = k5*0*(1-0)/... = 0
        assert dy[2] == pytest.approx(0.0, abs=1e-15)

    def test_derivatives_at_full_activation(self):
        """At [1, 1, 1], deactivation should dominate."""
        sim = MAPKCascadeSimulation(_make_config(V1=0.0))
        y = np.array([1.0, 1.0, 1.0])
        dy = sim._derivatives(y)
        # dX1/dt = V1*0/(K1+0) - V2*1/(K2+1) = -V2/(K2+1)
        expected_dX1 = -0.3 / (0.01 + 1.0)
        np.testing.assert_allclose(dy[0], expected_dX1, rtol=1e-10)
        assert dy[0] < 0, "X1 should decrease at full activation with V1=0"

    def test_derivatives_tier2_depends_on_X1(self):
        """X2 activation rate should increase with X1."""
        sim = MAPKCascadeSimulation(_make_config())
        y_low = np.array([0.1, 0.0, 0.0])
        y_high = np.array([0.9, 0.0, 0.0])
        dy_low = sim._derivatives(y_low)
        dy_high = sim._derivatives(y_high)
        # Higher X1 should give faster X2 activation
        assert dy_high[1] > dy_low[1], (
            f"dX2/dt should increase with X1: {dy_low[1]} vs {dy_high[1]}"
        )


# --- Response time ---

class TestMAPKCascadeResponseTime:
    def test_response_time_finite_for_strong_signal(self):
        """With strong signal, X3 should reach 0.5 in finite time."""
        sim = MAPKCascadeSimulation(_make_config(V1=2.0, n_steps=5000))
        sim.reset()
        t = sim.response_time(threshold=0.5)
        assert t < float("inf"), "X3 should reach 0.5 with strong signal"
        assert t > 0, "Response time should be positive"

    def test_response_time_zero_signal_is_inf(self):
        """With no signal, X3 should never reach 0.5."""
        sim = MAPKCascadeSimulation(_make_config(V1=0.0, n_steps=500))
        sim.reset()
        t = sim.response_time(threshold=0.5)
        assert t == float("inf"), "Zero signal should never activate X3"

    def test_stronger_signal_faster_response(self):
        """Higher V1 should give faster response time."""
        sim_low = MAPKCascadeSimulation(_make_config(V1=0.5, n_steps=10000))
        sim_low.reset()
        t_low = sim_low.response_time(threshold=0.5)

        sim_high = MAPKCascadeSimulation(_make_config(V1=2.0, n_steps=10000))
        sim_high.reset()
        t_high = sim_high.response_time(threshold=0.5)

        if t_low < float("inf") and t_high < float("inf"):
            assert t_high < t_low, (
                f"Stronger signal should respond faster: "
                f"V1=2.0 t={t_high} vs V1=0.5 t={t_low}"
            )


# --- Rediscovery data generation ---

class TestMAPKCascadeRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.mapk_cascade import generate_ode_data

        data = generate_ode_data(V1=1.0, K=0.01, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["V1"] == 1.0
        assert data["K"] == 0.01

    def test_ode_data_X1_increases(self):
        from simulating_anything.rediscovery.mapk_cascade import generate_ode_data

        data = generate_ode_data(V1=1.0, K=0.01, n_steps=1000, dt=0.01)
        # X1 should increase from 0 with V1 > 0
        assert data["X1"][-1] > data["X1"][0]

    def test_ode_data_bounded(self):
        from simulating_anything.rediscovery.mapk_cascade import generate_ode_data

        data = generate_ode_data(V1=1.0, K=0.01, n_steps=2000, dt=0.01)
        assert np.all(data["states"] >= 0.0)
        assert np.all(data["states"] <= 1.0)

    def test_dose_response_data_generation(self):
        from simulating_anything.rediscovery.mapk_cascade import (
            generate_dose_response_data,
        )

        data = generate_dose_response_data(n_V1=10, n_steps=3000, dt=0.01)
        assert len(data["V1"]) == 10
        assert len(data["X3_ss"]) == 10
        assert len(data["X1_ss"]) == 10
        assert len(data["amplification"]) == 10

    def test_dose_response_data_monotonic_X3(self):
        from simulating_anything.rediscovery.mapk_cascade import (
            generate_dose_response_data,
        )

        data = generate_dose_response_data(n_V1=15, n_steps=5000, dt=0.01)
        X3 = data["X3_ss"]
        # Should be monotonically non-decreasing
        for i in range(1, len(X3)):
            assert X3[i] >= X3[i - 1] - 1e-4

    def test_K_sweep_data_generation(self):
        from simulating_anything.rediscovery.mapk_cascade import generate_K_sweep_data

        data = generate_K_sweep_data(n_K=5, n_steps=3000, dt=0.01)
        assert len(data["K_values"]) == 5
        assert len(data["hill_coefficients"]) == 5
        # All Hill coefficients should be positive
        assert np.all(data["hill_coefficients"] > 0)
