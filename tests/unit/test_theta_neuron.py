"""Tests for the theta neuron (Ermentrout-Kopell) model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.theta_neuron import ThetaNeuronSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I: float = -0.1,
    theta_0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.THETA_NEURON,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={"I": I, "theta_0": theta_0},
    )


# ---------------------------------------------------------------------------
# Construction and initialization
# ---------------------------------------------------------------------------

class TestThetaNeuronConstruction:
    def test_default_parameters(self):
        sim = ThetaNeuronSimulation(_make_config())
        assert sim.I == -0.1
        assert sim.theta_0 == 0.0

    def test_custom_current(self):
        sim = ThetaNeuronSimulation(_make_config(I=0.5))
        assert sim.I == 0.5

    def test_custom_theta_0(self):
        sim = ThetaNeuronSimulation(_make_config(theta_0=1.0))
        assert sim.theta_0 == 1.0

    def test_initial_state_shape(self):
        sim = ThetaNeuronSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state.dtype == np.float64

    def test_initial_state_value(self):
        sim = ThetaNeuronSimulation(_make_config(theta_0=0.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5])

    def test_reset_clears_spikes(self):
        sim = ThetaNeuronSimulation(_make_config(I=1.0))
        sim.reset()
        for _ in range(100):
            sim.step()
        sim.reset()
        assert sim._spike_count == 0
        assert sim._spike_times == []


# ---------------------------------------------------------------------------
# Subthreshold behavior (I < 0)
# ---------------------------------------------------------------------------

class TestSubthresholdBehavior:
    def test_no_spiking_property(self):
        sim = ThetaNeuronSimulation(_make_config(I=-0.1))
        assert sim.is_spiking is False

    def test_converges_to_fixed_point(self):
        """For I < 0, theta should converge to stable fixed point."""
        sim = ThetaNeuronSimulation(_make_config(I=-0.5, dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
        theta_final = sim.observe()[0]

        # Should be near the analytic fixed point
        fp = ThetaNeuronSimulation.stable_fixed_point(-0.5)
        assert fp is not None
        assert abs(theta_final - fp) < 0.05, (
            f"theta={theta_final:.4f}, expected~{fp:.4f}"
        )

    def test_fixed_point_bounded(self):
        """Fixed point should be in [-pi, pi]."""
        sim = ThetaNeuronSimulation(_make_config(I=-0.3))
        sim.reset()
        for _ in range(5000):
            sim.step()
        theta = sim.observe()[0]
        assert -np.pi <= theta <= np.pi

    def test_resting_no_spikes(self):
        """Subthreshold: no spikes should be detected."""
        sim = ThetaNeuronSimulation(_make_config(I=-0.5, dt=0.01))
        sim.reset()
        for _ in range(2000):
            sim.step()
        assert sim._spike_count == 0

    def test_derivative_at_fixed_point_near_zero(self):
        """At the stable fixed point, dtheta/dt should be ~0."""
        I = -0.3
        fp = ThetaNeuronSimulation.stable_fixed_point(I)
        assert fp is not None
        sim = ThetaNeuronSimulation(_make_config(I=I))
        dy = sim._derivative(np.array([fp]))
        assert abs(dy[0]) < 1e-10


# ---------------------------------------------------------------------------
# Superthreshold behavior (I > 0)
# ---------------------------------------------------------------------------

class TestSuperthresholdBehavior:
    def test_spiking_property(self):
        sim = ThetaNeuronSimulation(_make_config(I=0.5))
        assert sim.is_spiking is True

    def test_produces_spikes(self):
        """For I > 0, should produce spikes."""
        sim = ThetaNeuronSimulation(_make_config(I=1.0, dt=0.01))
        sim.reset()
        for _ in range(2000):
            sim.step()
        assert sim._spike_count > 0

    def test_theta_stays_bounded(self):
        """Theta should remain in [-pi, pi] during spiking."""
        sim = ThetaNeuronSimulation(_make_config(I=1.0, dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            theta = sim.observe()[0]
            assert -np.pi <= theta <= np.pi + 0.01, f"theta out of range: {theta}"

    def test_spike_times_increase(self):
        """Spike times should be monotonically increasing."""
        sim = ThetaNeuronSimulation(_make_config(I=1.0, dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
        if len(sim._spike_times) >= 2:
            diffs = np.diff(sim._spike_times)
            assert np.all(diffs > 0)

    def test_higher_current_more_spikes(self):
        """Higher current should produce more spikes."""
        spikes = []
        for I in [0.5, 2.0]:
            sim = ThetaNeuronSimulation(_make_config(I=I, dt=0.01, n_steps=3000))
            sim.reset()
            for _ in range(3000):
                sim.step()
            spikes.append(sim._spike_count)
        assert spikes[1] > spikes[0], f"Expected more spikes at I=2.0: {spikes}"


# ---------------------------------------------------------------------------
# Analytical frequency
# ---------------------------------------------------------------------------

class TestAnalyticalFrequency:
    def test_negative_current_zero_freq(self):
        assert ThetaNeuronSimulation.analytical_frequency(-1.0) == 0.0

    def test_zero_current_zero_freq(self):
        assert ThetaNeuronSimulation.analytical_frequency(0.0) == 0.0

    def test_positive_current_formula(self):
        # f = sqrt(I) / pi
        I = 1.0
        expected = np.sqrt(I) / np.pi
        assert ThetaNeuronSimulation.analytical_frequency(I) == pytest.approx(expected)

    def test_frequency_increases_with_current(self):
        f1 = ThetaNeuronSimulation.analytical_frequency(0.5)
        f2 = ThetaNeuronSimulation.analytical_frequency(2.0)
        assert f2 > f1

    def test_frequency_at_I_4(self):
        # f = sqrt(4)/pi = 2/pi
        expected = 2.0 / np.pi
        assert ThetaNeuronSimulation.analytical_frequency(4.0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Analytical period
# ---------------------------------------------------------------------------

class TestAnalyticalPeriod:
    def test_negative_current_inf_period(self):
        assert ThetaNeuronSimulation.analytical_period(-1.0) == float("inf")

    def test_zero_current_inf_period(self):
        assert ThetaNeuronSimulation.analytical_period(0.0) == float("inf")

    def test_positive_current_formula(self):
        # T = pi / sqrt(I)
        I = 1.0
        expected = np.pi / np.sqrt(I)
        assert ThetaNeuronSimulation.analytical_period(I) == pytest.approx(expected)

    def test_period_decreases_with_current(self):
        T1 = ThetaNeuronSimulation.analytical_period(0.5)
        T2 = ThetaNeuronSimulation.analytical_period(2.0)
        assert T2 < T1

    def test_frequency_period_reciprocal(self):
        I = 2.5
        f = ThetaNeuronSimulation.analytical_frequency(I)
        T = ThetaNeuronSimulation.analytical_period(I)
        assert f * T == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stable fixed point
# ---------------------------------------------------------------------------

class TestStableFixedPoint:
    def test_positive_current_no_fixed_point(self):
        assert ThetaNeuronSimulation.stable_fixed_point(0.5) is None

    def test_zero_current_no_fixed_point(self):
        assert ThetaNeuronSimulation.stable_fixed_point(0.0) is None

    def test_negative_current_has_fixed_point(self):
        fp = ThetaNeuronSimulation.stable_fixed_point(-0.5)
        assert fp is not None
        assert -np.pi <= fp <= np.pi

    def test_fixed_point_satisfies_ode(self):
        """dtheta/dt should be 0 at the fixed point."""
        I = -0.3
        fp = ThetaNeuronSimulation.stable_fixed_point(I)
        assert fp is not None
        dtheta = 1.0 - np.cos(fp) + (1.0 + np.cos(fp)) * I
        assert abs(dtheta) < 1e-10


# ---------------------------------------------------------------------------
# Spike counting
# ---------------------------------------------------------------------------

class TestSpikeCounting:
    def test_count_spikes_no_wraps(self):
        """Trajectory with no wrapping: 0 spikes."""
        sim = ThetaNeuronSimulation(_make_config())
        trajectory = np.linspace(-1.0, 1.0, 100)
        assert sim.count_spikes(trajectory) == 0

    def test_count_spikes_with_wraps(self):
        """Manually create a trajectory with known wraps."""
        sim = ThetaNeuronSimulation(_make_config())
        # Create wrap: value goes from near +pi to near -pi
        trajectory = np.array([0.0, 1.0, 2.0, 2.5, -2.5, -2.0, -1.0, 0.0,
                               1.0, 2.0, 2.5, -2.5, -2.0])
        spikes = sim.count_spikes(trajectory)
        assert spikes == 2

    def test_count_spikes_from_simulation(self):
        """Running sim with I > 0 should detect spikes."""
        sim = ThetaNeuronSimulation(_make_config(I=1.0, dt=0.01))
        sim.reset()
        trajectory = [sim.observe()[0]]
        for _ in range(3000):
            sim.step()
            trajectory.append(sim.observe()[0])
        trajectory = np.array(trajectory)
        spikes = sim.count_spikes(trajectory)
        assert spikes > 0


# ---------------------------------------------------------------------------
# Measured frequency vs analytical
# ---------------------------------------------------------------------------

class TestMeasuredFrequency:
    def test_subthreshold_zero_frequency(self):
        sim = ThetaNeuronSimulation(_make_config(I=-0.5, dt=0.01))
        sim.reset()
        freq = sim.measure_frequency(t_max=20.0, transient=5.0)
        assert freq == 0.0

    def test_superthreshold_positive_frequency(self):
        sim = ThetaNeuronSimulation(_make_config(I=1.0, dt=0.01))
        sim.reset()
        freq = sim.measure_frequency(t_max=50.0, transient=10.0)
        assert freq > 0

    def test_frequency_matches_theory(self):
        """Measured frequency should approximate sqrt(I)/pi."""
        I = 1.0
        sim = ThetaNeuronSimulation(_make_config(I=I, dt=0.005))
        sim.reset()
        freq_meas = sim.measure_frequency(t_max=100.0, transient=20.0)
        freq_theory = ThetaNeuronSimulation.analytical_frequency(I)
        assert freq_meas == pytest.approx(freq_theory, rel=0.05), (
            f"measured={freq_meas:.4f}, theory={freq_theory:.4f}"
        )

    def test_frequency_matches_theory_high_I(self):
        """At higher I, frequency should still match theory."""
        I = 4.0
        sim = ThetaNeuronSimulation(_make_config(I=I, dt=0.005))
        sim.reset()
        freq_meas = sim.measure_frequency(t_max=50.0, transient=10.0)
        freq_theory = ThetaNeuronSimulation.analytical_frequency(I)
        assert freq_meas == pytest.approx(freq_theory, rel=0.05), (
            f"measured={freq_meas:.4f}, theory={freq_theory:.4f}"
        )


# ---------------------------------------------------------------------------
# f-I curve
# ---------------------------------------------------------------------------

class TestFICurve:
    def test_fi_curve_output_shapes(self):
        sim = ThetaNeuronSimulation(_make_config(I=0.5, dt=0.01))
        sim.reset()
        I_vals = np.array([0.1, 0.5, 1.0, 2.0])
        result = sim.compute_fi_curve(I_vals, t_max=30.0, transient=5.0)
        assert len(result["I"]) == 4
        assert len(result["frequency"]) == 4
        assert len(result["frequency_theory"]) == 4

    def test_fi_curve_monotonic_increasing(self):
        """Frequency should increase with current for I > 0."""
        sim = ThetaNeuronSimulation(_make_config(I=0.5, dt=0.01))
        sim.reset()
        I_vals = np.array([0.5, 1.0, 2.0, 4.0])
        result = sim.compute_fi_curve(I_vals, t_max=50.0, transient=10.0)
        freqs = result["frequency"]
        # All should be positive
        assert np.all(freqs > 0), f"Got non-positive freqs: {freqs}"
        # Should be monotonically increasing
        assert np.all(np.diff(freqs) > 0), f"Not monotonic: {freqs}"

    def test_fi_curve_theory_included(self):
        """Theory values should match analytical formula."""
        sim = ThetaNeuronSimulation(_make_config())
        I_vals = np.array([1.0, 4.0])
        result = sim.compute_fi_curve(I_vals, t_max=30.0, transient=5.0)
        expected_theory = np.array([
            np.sqrt(1.0) / np.pi,
            np.sqrt(4.0) / np.pi,
        ])
        np.testing.assert_allclose(result["frequency_theory"], expected_theory)


# ---------------------------------------------------------------------------
# ODE structure
# ---------------------------------------------------------------------------

class TestODEStructure:
    def test_derivative_at_theta_zero(self):
        """At theta=0: dtheta/dt = 1-1+(1+1)*I = 2I."""
        sim = ThetaNeuronSimulation(_make_config(I=0.5))
        dy = sim._derivative(np.array([0.0]))
        expected = 2.0 * 0.5  # 2*I
        assert dy[0] == pytest.approx(expected, abs=1e-10)

    def test_derivative_at_theta_pi(self):
        """At theta=pi: dtheta/dt = 1-(-1)+(1+(-1))*I = 2."""
        sim = ThetaNeuronSimulation(_make_config(I=0.5))
        dy = sim._derivative(np.array([np.pi]))
        expected = 2.0  # 1 - cos(pi) + (1+cos(pi))*I = 2
        assert dy[0] == pytest.approx(expected, abs=1e-10)

    def test_derivative_independent_of_I_at_pi(self):
        """At theta=pi, derivative = 2 regardless of I."""
        for I in [-1.0, 0.0, 0.5, 5.0]:
            sim = ThetaNeuronSimulation(_make_config(I=I))
            dy = sim._derivative(np.array([np.pi]))
            assert dy[0] == pytest.approx(2.0, abs=1e-10)

    def test_step_changes_state(self):
        sim = ThetaNeuronSimulation(_make_config(I=0.5))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = ThetaNeuronSimulation(_make_config())
        sim.reset()
        assert np.allclose(sim.observe(), sim._state)


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------

class TestRediscoveryDataGeneration:
    def test_ode_data_shapes(self):
        from simulating_anything.rediscovery.theta_neuron import generate_ode_data
        data = generate_ode_data(I=0.5, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 1)
        assert len(data["time"]) == 501
        assert data["I"] == 0.5

    def test_fi_curve_data(self):
        from simulating_anything.rediscovery.theta_neuron import generate_fi_curve
        data = generate_fi_curve(n_I=5, dt=0.01, t_max=20.0, transient=5.0)
        assert len(data["I"]) == 5
        assert len(data["frequency"]) == 5
        assert len(data["frequency_theory"]) == 5
        assert np.all(data["frequency"] >= 0)

    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.theta_neuron import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_I=10, dt=0.01)
        assert len(data["I"]) == 10
        assert len(data["frequency"]) == 10
        assert len(data["resting_theta"]) == 10
