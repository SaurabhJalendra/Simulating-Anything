"""Tests for the semiconductor laser rate equations simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.laser_rate import LaserRateSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(**kwargs) -> LaserRateSimulation:
    """Create a LaserRateSimulation with default semiconductor laser params."""
    defaults = {
        "P": 2.0,
        "gamma_N": 1.0,
        "gamma_S": 100.0,
        "g": 1000.0,
        "N_tr": 0.5,
        "Gamma": 0.3,
        "beta": 1e-4,
        "N_0": 0.5,
        "S_0": 0.01,
    }
    defaults.update(kwargs)
    config = SimulationConfig(
        domain=Domain.LASER_RATE,
        dt=defaults.pop("dt", 0.001),
        n_steps=defaults.pop("n_steps", 1000),
        parameters=defaults,
    )
    return LaserRateSimulation(config)


class TestLaserRateSimulation:
    """Core simulation interface tests."""

    def test_initial_state(self):
        """reset() returns shape (2,) with values [N_0, S_0]."""
        sim = _make_sim(N_0=0.5, S_0=0.01)
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [0.5, 0.01])

    def test_step_advances_state(self):
        """A single step should change the state vector."""
        sim = _make_sim()
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1), "State should change after a step"

    def test_trajectory_shape(self):
        """run(n_steps) returns TrajectoryData with states shape (n_steps+1, 2)."""
        sim = _make_sim(n_steps=200)
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201

    def test_state_dim(self):
        """State dimension is 2: [N, S]."""
        sim = _make_sim()
        state = sim.reset()
        assert state.shape == (2,)
        assert sim.step().shape == (2,)

    def test_observe(self):
        """observe() returns the same array as the last reset/step."""
        sim = _make_sim()
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_non_negative_densities(self):
        """Carrier and photon densities must remain >= 0 throughout."""
        sim = _make_sim(P=0.1, N_0=0.01, S_0=0.001)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"N went negative: {state[0]}"
            assert state[1] >= 0, f"S went negative: {state[1]}"

    def test_below_threshold_dynamics(self):
        """Below threshold (P < P_th), photon density S stays small."""
        # P_th = 1.0*0.5 + 100.0/(0.3*1000.0) = 0.8333
        sim = _make_sim(P=0.3, N_0=0.3, S_0=0.001)
        sim.reset()
        for _ in range(20000):
            sim.step()
        _, S = sim.observe()
        assert S < 0.01, f"Below threshold S should stay near zero, got {S}"

    def test_above_threshold_dynamics(self):
        """Above threshold (P > P_th), photon density S grows significantly."""
        sim = _make_sim(P=2.0, N_0=0.5, S_0=0.001)
        sim.reset()
        for _ in range(50000):
            sim.step()
        _, S = sim.observe()
        assert S > 0.001, f"Above threshold S should be positive, got {S}"


class TestThresholdBehavior:
    """Tests for the lasing threshold property and checks."""

    def test_threshold_pump_formula(self):
        """P_th = gamma_N * N_tr + gamma_S / (Gamma * g) for default params."""
        sim = _make_sim(gamma_N=1.0, N_tr=0.5, gamma_S=100.0, Gamma=0.3, g=1000.0)
        expected = 1.0 * 0.5 + 100.0 / (0.3 * 1000.0)
        assert sim.threshold_pump == pytest.approx(expected, rel=1e-10)

    def test_threshold_pump_custom_params(self):
        """Threshold formula holds for non-default parameters."""
        sim = _make_sim(gamma_N=2.0, N_tr=0.3, gamma_S=150.0, Gamma=0.5, g=500.0)
        expected = 2.0 * 0.3 + 150.0 / (0.5 * 500.0)
        assert sim.threshold_pump == pytest.approx(expected, rel=1e-10)

    def test_above_threshold_check(self):
        """Default P=2.0 is above threshold (~0.833)."""
        sim = _make_sim(P=2.0)
        assert sim.P > sim.threshold_pump

    def test_below_threshold_check(self):
        """P=0.3 is below threshold (~0.833)."""
        sim = _make_sim(P=0.3)
        assert sim.P < sim.threshold_pump


class TestSteadyState:
    """Tests for the steady_state property and convergence."""

    def test_steady_state_above_threshold(self):
        """Above threshold, steady_state returns (N_ss, S_ss) with S_ss > 0."""
        sim = _make_sim(P=3.0)
        N_ss, S_ss = sim.steady_state
        assert S_ss > 0, f"S_ss should be positive above threshold, got {S_ss}"
        # N_ss = N_tr + gamma_S/(Gamma*g)
        expected_N = 0.5 + 100.0 / (0.3 * 1000.0)
        assert N_ss == pytest.approx(expected_N, rel=1e-10)
        # S_ss = Gamma*(P - P_th)/gamma_S
        P_th = sim.threshold_pump
        expected_S = 0.3 * (3.0 - P_th) / 100.0
        assert S_ss == pytest.approx(expected_S, rel=1e-10)

    def test_steady_state_below_threshold(self):
        """Below threshold, S_ss is clipped to 0."""
        sim = _make_sim(P=0.3)
        _, S_ss = sim.steady_state
        assert S_ss == 0.0

    def test_convergence_to_steady_state(self):
        """Long simulation converges to the analytical steady state."""
        sim = _make_sim(P=3.0, N_0=0.5, S_0=0.001)
        sim.reset()
        for _ in range(100000):
            sim.step()
        N, S = sim.observe()
        N_ss, S_ss = sim.steady_state
        # Allow tolerance for spontaneous emission (beta) perturbation
        np.testing.assert_allclose(N, N_ss, rtol=0.05)
        np.testing.assert_allclose(S, S_ss, rtol=0.1)


class TestRelaxationOscillation:
    """Tests for the relaxation_frequency property."""

    def test_relaxation_frequency_above_threshold(self):
        """Above threshold, relaxation_frequency > 0."""
        sim = _make_sim(P=2.0)
        assert sim.relaxation_frequency > 0

    def test_relaxation_frequency_below_threshold(self):
        """Below threshold, relaxation_frequency = 0."""
        sim = _make_sim(P=0.3)
        assert sim.relaxation_frequency == 0.0

    def test_relaxation_frequency_formula(self):
        """omega_r = sqrt(gamma_S * g * S_ss)."""
        sim = _make_sim(P=3.0, gamma_S=100.0, g=1000.0)
        omega = sim.relaxation_frequency
        _, S_ss = sim.steady_state
        expected = np.sqrt(100.0 * 1000.0 * S_ss)
        assert omega == pytest.approx(expected, rel=1e-10)


class TestDeterminism:
    """Determinism and reproducibility tests."""

    def test_reproducible(self):
        """Two fresh simulations with the same config produce identical trajectories."""
        sim1 = _make_sim(P=2.0, n_steps=500)
        traj1 = sim1.run(n_steps=500)

        sim2 = _make_sim(P=2.0, n_steps=500)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-14)


class TestRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_ode_data(self):
        """generate_ode_data returns states, time, N, S arrays."""
        from simulating_anything.rediscovery.laser_rate import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.001, P=2.0)
        assert data["states"].shape == (201, 2)
        assert len(data["time"]) == 201
        assert data["P"] == 2.0
        np.testing.assert_array_equal(data["N"], data["states"][:, 0])
        np.testing.assert_array_equal(data["S"], data["states"][:, 1])
        assert np.all(np.isfinite(data["states"]))

    def test_generate_threshold_sweep(self):
        """generate_threshold_sweep returns P, S_steady, N_steady, P_th_theory."""
        from simulating_anything.rediscovery.laser_rate import generate_threshold_sweep

        data = generate_threshold_sweep(n_P=5, dt=0.001, n_settle=5000)
        assert len(data["P"]) == 5
        assert len(data["S_steady"]) == 5
        assert len(data["N_steady"]) == 5
        expected_P_th = 1.0 * 0.5 + 100.0 / (0.3 * 1000.0)
        assert data["P_th_theory"] == pytest.approx(expected_P_th, rel=1e-10)

    def test_threshold_detection(self):
        """Threshold sweep should show S jumping from near-zero to positive."""
        from simulating_anything.rediscovery.laser_rate import generate_threshold_sweep

        data = generate_threshold_sweep(n_P=20, dt=0.001, n_settle=10000)
        P_th = data["P_th_theory"]
        S_steady = data["S_steady"]
        P_values = data["P"]
        # Points well below threshold should have very small S
        below_mask = P_values < P_th * 0.7
        if np.any(below_mask):
            assert np.all(S_steady[below_mask] < 0.01)
        # Points well above threshold should have S > 0
        above_mask = P_values > P_th * 1.5
        if np.any(above_mask):
            assert np.all(S_steady[above_mask] > 1e-4)

    def test_generate_li_curve(self):
        """generate_li_curve returns P, S_steady, P_th_theory."""
        from simulating_anything.rediscovery.laser_rate import generate_li_curve

        data = generate_li_curve(n_P=10, dt=0.001, n_settle=5000)
        assert len(data["P"]) == 10
        assert len(data["S_steady"]) == 10
        assert np.all(np.isfinite(data["S_steady"]))

    def test_generate_relaxation_frequency_data(self):
        """generate_relaxation_frequency_data returns omega measurements."""
        from simulating_anything.rediscovery.laser_rate import (
            generate_relaxation_frequency_data,
        )

        data = generate_relaxation_frequency_data(
            n_P=3, dt=0.0001, n_transient=1000, n_measure=5000,
        )
        assert len(data["P"]) == 3
        assert len(data["omega_r_measured"]) == 3
        assert len(data["omega_r_theory"]) == 3

    def test_generate_parameter_sweep(self):
        """generate_parameter_sweep returns multi-parameter sweep data."""
        from simulating_anything.rediscovery.laser_rate import generate_parameter_sweep

        data = generate_parameter_sweep(n_samples=5, dt=0.001, n_settle=5000)
        assert len(data["P"]) == 5
        assert len(data["S_ss"]) == 5
        assert len(data["P_th"]) == 5
        assert len(data["gamma_S"]) == 5
        assert len(data["Gamma"]) == 5
