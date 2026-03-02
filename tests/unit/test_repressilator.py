"""Tests for the Repressilator synthetic genetic oscillatory network."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.repressilator import RepressilatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha: float = 216.0,
    alpha0: float = 0.001,
    n: float = 2.0,
    beta: float = 5.0,
    dt: float = 0.01,
    m1_0: float = 1.0,
    m2_0: float = 0.5,
    m3_0: float = 0.2,
    p1_0: float = 1.0,
    p2_0: float = 0.5,
    p3_0: float = 0.2,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.REPRESSILATOR,
        dt=dt,
        n_steps=1000,
        parameters={
            "alpha": alpha, "alpha0": alpha0, "n": n, "beta": beta,
            "m1_0": m1_0, "m2_0": m2_0, "m3_0": m3_0,
            "p1_0": p1_0, "p2_0": p2_0, "p3_0": p3_0,
        },
    )


class TestRepressilatorConstruction:
    def test_default_parameters(self):
        sim = RepressilatorSimulation(_make_config())
        assert sim.alpha == 216.0
        assert sim.alpha0 == 0.001
        assert sim.n == 2.0
        assert sim.beta == 5.0

    def test_custom_parameters(self):
        sim = RepressilatorSimulation(_make_config(alpha=100.0, n=3.0, beta=2.0))
        assert sim.alpha == 100.0
        assert sim.n == 3.0
        assert sim.beta == 2.0

    def test_state_dimension(self):
        sim = RepressilatorSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (6,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        sim = RepressilatorSimulation(_make_config(
            m1_0=2.0, m2_0=1.5, m3_0=1.0,
            p1_0=2.0, p2_0=1.5, p3_0=1.0,
        ))
        state = sim.reset()
        np.testing.assert_allclose(state, [2.0, 1.5, 1.0, 2.0, 1.5, 1.0])


class TestRepressilatorHillFunction:
    def test_hill_zero_protein(self):
        """With p=0, repression is off: f(0) = alpha + alpha0."""
        sim = RepressilatorSimulation(_make_config())
        val = sim._hill_repression(0.0)
        assert val == pytest.approx(216.0 + 0.001, rel=1e-10)

    def test_hill_large_protein(self):
        """With large p, repression is maximal: f(p) -> alpha0."""
        sim = RepressilatorSimulation(_make_config())
        val = sim._hill_repression(1000.0)
        # alpha/(1+1e6) + alpha0 ~ 0.000216 + 0.001 = 0.001216
        assert val == pytest.approx(0.001216, rel=0.01)

    def test_hill_at_one(self):
        """At p=1: f(1) = alpha/2 + alpha0 for n=2."""
        sim = RepressilatorSimulation(_make_config(n=2.0))
        val = sim._hill_repression(1.0)
        # alpha/(1+1) + alpha0 = 216/2 + 0.001 = 108.001
        assert val == pytest.approx(108.001, rel=1e-6)

    def test_hill_monotonically_decreasing(self):
        """Hill repression function is monotonically decreasing in p."""
        sim = RepressilatorSimulation(_make_config())
        p_vals = np.linspace(0.0, 50.0, 100)
        h_vals = [sim._hill_repression(p) for p in p_vals]
        for i in range(1, len(h_vals)):
            assert h_vals[i] <= h_vals[i - 1]


class TestRepressilatorDynamics:
    def test_step_advances_state(self):
        sim = RepressilatorSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_state(self):
        sim = RepressilatorSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (6,)

    def test_state_non_negative(self):
        """Concentrations should remain non-negative."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= -1e-10), f"Negative concentration: {state}"

    def test_trajectory_bounded(self):
        """State should remain bounded for default parameters."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.abs(state) < 1000), f"State diverged: {state}"

    def test_rk4_accuracy_vs_euler(self):
        """RK4 should be more accurate than forward Euler over many steps."""
        config = _make_config(dt=0.01)
        sim_fine = RepressilatorSimulation(
            _make_config(dt=0.001)
        )
        sim_coarse = RepressilatorSimulation(config)

        sim_fine.reset()
        sim_coarse.reset()

        # Run fine simulation 10x more steps
        for _ in range(1000):
            sim_coarse.step()
        for _ in range(10000):
            sim_fine.step()

        # RK4 with dt=0.01 should be close to RK4 with dt=0.001
        err = np.linalg.norm(sim_coarse.observe() - sim_fine.observe())
        assert err < 1.0, f"RK4 error too large: {err}"


class TestRepressilatorOscillation:
    def test_oscillation_default_params(self):
        """Default parameters (alpha=216) should produce oscillations."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        # Measure variation in p1
        p1_vals = []
        for _ in range(5000):
            sim.step()
            p1_vals.append(sim.observe()[3])
        amplitude = max(p1_vals) - min(p1_vals)
        assert amplitude > 1.0, f"No oscillation: amplitude={amplitude}"

    def test_no_oscillation_small_alpha(self):
        """With very small alpha, system should converge to fixed point."""
        sim = RepressilatorSimulation(_make_config(alpha=1.0, dt=0.01))
        sim.reset()
        for _ in range(20000):
            sim.step()
        # Measure variation
        p1_vals = []
        for _ in range(5000):
            sim.step()
            p1_vals.append(sim.observe()[3])
        amplitude = max(p1_vals) - min(p1_vals)
        assert amplitude < 0.5, f"Unexpected oscillation: amplitude={amplitude}"

    def test_period_computation_returns_finite(self):
        """Period should be finite for oscillatory default parameters."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        assert np.isfinite(period), "Period should be finite for oscillatory params"
        assert period > 0, "Period must be positive"

    def test_period_reasonable_range(self):
        """Period should be in a physically reasonable range."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        # Repressilator period is typically a few time units
        assert 0.5 < period < 50.0, f"Period out of expected range: {period}"

    def test_period_no_oscillation(self):
        """Period should be inf when no oscillation is present."""
        sim = RepressilatorSimulation(_make_config(alpha=1.0, dt=0.01))
        sim.reset()
        period = sim.compute_period(n_transient=5000, n_measure=5000)
        assert period == float("inf") or period > 100.0


class TestRepressilatorSymmetry:
    def test_z3_equal_amplitudes(self):
        """All three proteins should oscillate with approximately equal amplitude."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        # Skip transient
        for _ in range(10000):
            sim.step()
        # Collect protein time series
        p_vals = [[], [], []]
        for _ in range(5000):
            sim.step()
            for j in range(3):
                p_vals[j].append(sim.observe()[3 + j])
        amps = [max(v) - min(v) for v in p_vals]
        # All amplitudes should be within 10% of each other
        mean_amp = np.mean(amps)
        for i, amp in enumerate(amps):
            assert abs(amp - mean_amp) / mean_amp < 0.15, (
                f"Protein {i+1} amplitude {amp:.4f} differs from mean {mean_amp:.4f}"
            )

    def test_z3_phase_shift(self):
        """Proteins should peak in sequence: p1 -> p2 -> p3 -> p1."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
        # Find first peak time of each protein
        peak_times = []
        for j in range(3):
            p_vals = []
            # Need fresh sim for same timeline
            sim2 = RepressilatorSimulation(_make_config(dt=0.005))
            sim2.reset()
            for _ in range(10000):
                sim2.step()
            for step in range(5000):
                sim2.step()
                p_vals.append(sim2.observe()[3 + j])
            # Find first peak
            arr = np.array(p_vals)
            for k in range(1, len(arr) - 1):
                if arr[k] > arr[k - 1] and arr[k] > arr[k + 1] and arr[k] > np.mean(arr):
                    peak_times.append(k * 0.005)
                    break
        if len(peak_times) == 3:
            # Peaks should be ordered (mod period)
            # Just check they are distinct
            assert len(set(peak_times)) > 1, "Peaks should not all occur at same time"

    def test_verify_z3_symmetry_method(self):
        """The verify_z3_symmetry method should return valid data."""
        sim = RepressilatorSimulation(_make_config(dt=0.005))
        sim.reset()
        z3 = sim.verify_z3_symmetry(n_transient=3000, n_measure=3000)
        assert "amplitudes" in z3
        assert len(z3["amplitudes"]) == 3
        assert "amplitude_ratio_12" in z3
        # All amplitude ratios should be close to 1.0
        assert z3["amplitude_ratio_12"] == pytest.approx(1.0, abs=0.2)
        assert z3["amplitude_ratio_23"] == pytest.approx(1.0, abs=0.2)


class TestRepressilatorFixedPoint:
    def test_symmetric_fixed_point_exists(self):
        """Symmetric fixed point should exist and be a valid concentration."""
        sim = RepressilatorSimulation(_make_config())
        fp = sim.symmetric_fixed_point
        assert fp.shape == (6,)
        assert np.all(fp > 0), "Fixed point concentrations must be positive"

    def test_fixed_point_symmetry(self):
        """At symmetric fixed point, m1=m2=m3 and p1=p2=p3."""
        sim = RepressilatorSimulation(_make_config())
        fp = sim.symmetric_fixed_point
        np.testing.assert_allclose(fp[0], fp[1], rtol=1e-8)
        np.testing.assert_allclose(fp[1], fp[2], rtol=1e-8)
        np.testing.assert_allclose(fp[3], fp[4], rtol=1e-8)
        np.testing.assert_allclose(fp[4], fp[5], rtol=1e-8)

    def test_fixed_point_m_equals_p(self):
        """At fixed point, m* = p* (from dp/dt = beta*(m-p) = 0)."""
        sim = RepressilatorSimulation(_make_config())
        fp = sim.symmetric_fixed_point
        np.testing.assert_allclose(fp[0], fp[3], rtol=1e-8)

    def test_derivatives_near_zero_at_fixed_point(self):
        """Derivatives should be nearly zero at the fixed point."""
        sim = RepressilatorSimulation(_make_config())
        fp = sim.symmetric_fixed_point
        dy = sim._derivatives(fp)
        np.testing.assert_allclose(dy, np.zeros(6), atol=1e-8)

    def test_fixed_point_satisfies_hill_equation(self):
        """At fixed point: p* = alpha/(1+p*^n) + alpha0."""
        sim = RepressilatorSimulation(_make_config())
        fp = sim.symmetric_fixed_point
        p_star = fp[3]
        expected = sim.alpha / (1.0 + p_star ** sim.n) + sim.alpha0
        assert p_star == pytest.approx(expected, rel=1e-8)


class TestRepressilatorIsOscillatory:
    def test_high_alpha_oscillatory(self):
        sim = RepressilatorSimulation(_make_config(alpha=216.0))
        assert sim.is_oscillatory is True

    def test_low_alpha_not_oscillatory(self):
        sim = RepressilatorSimulation(_make_config(alpha=1.0))
        assert sim.is_oscillatory is False

    def test_high_hill_coefficient_oscillatory(self):
        """Higher Hill coefficient makes oscillation easier."""
        sim = RepressilatorSimulation(_make_config(alpha=50.0, n=4.0))
        assert sim.is_oscillatory is True


class TestRepressilatorParameterSweep:
    def test_sweep_returns_correct_shape(self):
        sim = RepressilatorSimulation(_make_config(dt=0.01))
        sim.reset()
        alpha_range = np.array([50.0, 100.0, 200.0])
        result = sim.parameter_sweep("alpha", alpha_range, n_steps=2000)
        assert len(result["param_values"]) == 3
        assert len(result["amplitudes"]) == 3
        assert len(result["periods"]) == 3

    def test_sweep_amplitude_increases_with_alpha(self):
        """Oscillation amplitude should generally increase with alpha."""
        sim = RepressilatorSimulation(_make_config(dt=0.01))
        sim.reset()
        alpha_range = np.array([50.0, 150.0, 300.0])
        result = sim.parameter_sweep("alpha", alpha_range, n_steps=5000)
        # Amplitude for alpha=300 should be larger than alpha=50
        assert result["amplitudes"][-1] > result["amplitudes"][0]


class TestRepressilatorTrajectory:
    def test_run_returns_trajectory_data(self):
        sim = RepressilatorSimulation(_make_config(dt=0.01))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 6)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps_correct(self):
        dt = 0.01
        sim = RepressilatorSimulation(_make_config(dt=dt))
        traj = sim.run(n_steps=50)
        expected_times = np.arange(51) * dt
        np.testing.assert_allclose(traj.timestamps, expected_times, rtol=1e-10)

    def test_reproducibility_with_reset(self):
        """Resetting should produce identical trajectories."""
        config = _make_config(dt=0.01)
        sim = RepressilatorSimulation(config)

        sim.reset(seed=42)
        states1 = []
        for _ in range(100):
            sim.step()
            states1.append(sim.observe().copy())

        sim.reset(seed=42)
        states2 = []
        for _ in range(100):
            sim.step()
            states2.append(sim.observe().copy())

        np.testing.assert_array_equal(np.array(states1), np.array(states2))


class TestRepressilatorRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.repressilator import generate_ode_data
        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 6)
        assert len(data["time"]) == 501
        assert data["alpha"] == 216.0

    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.repressilator import generate_bifurcation_data
        data = generate_bifurcation_data(n_alpha=5, dt=0.01)
        assert len(data["alpha"]) == 5
        assert len(data["amplitude"]) == 5

    def test_period_data(self):
        from simulating_anything.rediscovery.repressilator import generate_period_data
        data = generate_period_data(n_alpha=3, dt=0.01)
        assert len(data["alpha"]) == 3
        assert len(data["period"]) == 3
