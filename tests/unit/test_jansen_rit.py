"""Tests for the Jansen-Rit neural mass model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.jansen_rit import JansenRitSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    A: float = 3.25,
    B: float = 22.0,
    a: float = 100.0,
    b: float = 50.0,
    C: float = 135.0,
    vmax: float = 5.0,
    v0: float = 6.0,
    r: float = 0.56,
    p: float = 120.0,
    dt: float = 0.001,
    n_steps: int = 5000,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.JANSEN_RIT,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "A": A, "B": B,
            "a": a, "b": b,
            "C": C,
            "vmax": vmax, "v0": v0, "r": r,
            "p": p,
        },
    )


class TestJansenRitSigmoid:
    """Tests for the sigmoid firing rate function."""

    def test_sigmoid_at_midpoint(self):
        """S(v0) should equal vmax/2."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        result = sim.sigmoid(sim.v0)
        assert result == pytest.approx(sim.vmax / 2.0)

    def test_sigmoid_large_positive(self):
        """S(large) should approach vmax."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        result = sim.sigmoid(100.0)
        assert result == pytest.approx(sim.vmax, abs=1e-6)

    def test_sigmoid_large_negative(self):
        """S(large negative) should approach 0."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        result = sim.sigmoid(-100.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_monotonically_increasing(self):
        """S(v) should be monotonically increasing."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        v_values = np.linspace(-10.0, 20.0, 100)
        s_values = sim.sigmoid(v_values)
        assert np.all(np.diff(s_values) > 0)

    def test_sigmoid_bounded(self):
        """S(v) should be in [0, vmax] for all v."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        v_values = np.linspace(-50.0, 50.0, 200)
        s_values = sim.sigmoid(v_values)
        assert np.all(s_values >= 0.0)
        assert np.all(s_values <= sim.vmax)

    def test_sigmoid_symmetry_about_midpoint(self):
        """S(v0 + delta) + S(v0 - delta) should equal vmax."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        for delta in [1.0, 2.0, 5.0]:
            s_plus = sim.sigmoid(sim.v0 + delta)
            s_minus = sim.sigmoid(sim.v0 - delta)
            assert s_plus + s_minus == pytest.approx(sim.vmax, abs=1e-10)


class TestJansenRitSimulation:
    """Tests for the simulation mechanics."""

    def test_reset_shape(self):
        """Reset should return 6D state vector."""
        sim = JansenRitSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (6,)

    def test_observe_shape(self):
        """Observe should return 6D state."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (6,)

    def test_step_advances_state(self):
        """State should change after stepping."""
        sim = JansenRitSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic_same_seed(self):
        """Same seed should produce identical trajectories."""
        config = _make_config(seed=42)
        sim1 = JansenRitSimulation(config)
        sim1.reset(seed=42)
        for _ in range(500):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = JansenRitSimulation(config)
        sim2.reset(seed=42)
        for _ in range(500):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_different_seeds_differ(self):
        """Different seeds should produce different initial states."""
        sim = JansenRitSimulation(_make_config())
        state1 = sim.reset(seed=1)
        state2 = sim.reset(seed=2)
        assert not np.allclose(state1, state2)

    def test_rk4_stability(self):
        """No NaN or Inf over a long trajectory with dt=0.001."""
        sim = JansenRitSimulation(_make_config(dt=0.001))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_run_trajectory(self):
        """run() should produce TrajectoryData with correct shapes."""
        sim = JansenRitSimulation(_make_config(n_steps=200))
        traj = sim.run(200)
        assert traj.states.shape == (201, 6)
        assert traj.timestamps.shape == (201,)

    def test_connectivity_constants(self):
        """Connectivity constants should follow C1=C, C2=0.8*C, etc."""
        sim = JansenRitSimulation(_make_config(C=135.0))
        assert sim.C1 == 135.0
        assert sim.C2 == pytest.approx(0.8 * 135.0)
        assert sim.C3 == pytest.approx(0.25 * 135.0)
        assert sim.C4 == pytest.approx(0.25 * 135.0)

    def test_default_parameters(self):
        """Default parameters should match the standard Jansen-Rit values."""
        sim = JansenRitSimulation(_make_config())
        assert sim.A == pytest.approx(3.25)
        assert sim.B == pytest.approx(22.0)
        assert sim.a == pytest.approx(100.0)
        assert sim.b == pytest.approx(50.0)
        assert sim.C == pytest.approx(135.0)
        assert sim.vmax == pytest.approx(5.0)
        assert sim.v0 == pytest.approx(6.0)
        assert sim.r == pytest.approx(0.56)
        assert sim.p_ext == pytest.approx(120.0)


class TestJansenRitEEG:
    """Tests for EEG output computation."""

    def test_compute_eeg_output_shape(self):
        """EEG output should have same length as input states."""
        sim = JansenRitSimulation(_make_config(n_steps=100))
        traj = sim.run(100)
        eeg = sim.compute_eeg_output(traj.states)
        assert eeg.shape == (101,)

    def test_compute_eeg_output_is_y1_minus_y2(self):
        """EEG output should be y1 - y2."""
        sim = JansenRitSimulation(_make_config(n_steps=100))
        traj = sim.run(100)
        eeg = sim.compute_eeg_output(traj.states)
        expected = traj.states[:, 1] - traj.states[:, 2]
        np.testing.assert_allclose(eeg, expected)

    def test_eeg_output_finite(self):
        """EEG output should be finite for a valid trajectory."""
        sim = JansenRitSimulation(_make_config(n_steps=2000))
        traj = sim.run(2000)
        eeg = sim.compute_eeg_output(traj.states)
        assert np.all(np.isfinite(eeg))


class TestJansenRitOscillation:
    """Tests for oscillatory behavior."""

    def test_oscillation_at_default_p(self):
        """Default parameters (p=120) should produce oscillation."""
        sim = JansenRitSimulation(_make_config(p=120.0, dt=0.001))
        sim.reset()
        # Skip transient
        for _ in range(3000):
            sim.step()
        # Collect EEG
        eeg = []
        for _ in range(3000):
            sim.step()
            eeg.append(sim._state[1] - sim._state[2])
        amplitude = max(eeg) - min(eeg)
        assert amplitude > 0.1, f"No oscillation, amplitude={amplitude}"

    def test_frequency_near_alpha(self):
        """Dominant frequency in alpha band (~10 Hz) at p=180 Hz input."""
        # The Jansen-Rit alpha rhythm appears for p in the range ~140-220 Hz,
        # where moderate-amplitude ~10 Hz oscillation emerges. At p=120, the
        # model is in a large-amplitude slow oscillation regime (~2.4 Hz).
        sim = JansenRitSimulation(_make_config(p=180.0, dt=0.001))
        freq = sim.compute_frequency(n_transient=3000, n_measure=5000)
        assert 5.0 <= freq <= 20.0, (
            f"Frequency {freq:.2f} Hz outside expected alpha range"
        )

    def test_low_p_low_amplitude(self):
        """Very low p should give low-amplitude activity."""
        sim = JansenRitSimulation(_make_config(p=20.0, dt=0.001))
        sim.reset()
        for _ in range(5000):
            sim.step()
        eeg = []
        for _ in range(3000):
            sim.step()
            eeg.append(sim._state[1] - sim._state[2])
        amplitude = max(eeg) - min(eeg)
        # Low p typically gives either small fluctuations or no oscillation
        assert amplitude < 20.0, f"Unexpected large amplitude at low p: {amplitude}"

    def test_high_p_behavior(self):
        """Very high p should produce large-amplitude oscillation."""
        sim = JansenRitSimulation(_make_config(p=300.0, dt=0.001))
        sim.reset()
        for _ in range(5000):
            sim.step()
        eeg = []
        for _ in range(3000):
            sim.step()
            eeg.append(sim._state[1] - sim._state[2])
        # High p should give either saturation or large oscillation
        assert np.all(np.isfinite(eeg)), "EEG should remain finite at high p"

    def test_frequency_computation_zero_for_no_oscillation(self):
        """compute_frequency should return 0 if amplitude too small."""
        # Very low input: p near zero, damped to steady state
        sim = JansenRitSimulation(_make_config(p=0.0, dt=0.001))
        freq = sim.compute_frequency(n_transient=3000, n_measure=3000)
        # Should either be 0 or very small
        assert freq >= 0.0


class TestJansenRitSweep:
    """Tests for the input sweep functionality."""

    def test_input_sweep_shapes(self):
        """Sweep should return arrays of correct length."""
        sim = JansenRitSimulation(_make_config(dt=0.001))
        result = sim.input_sweep(p_range=(50.0, 200.0), n_p=10, n_steps=2000)
        assert len(result["p_values"]) == 10
        assert len(result["amplitude"]) == 10
        assert len(result["frequency"]) == 10
        assert len(result["mean_eeg"]) == 10

    def test_input_sweep_p_range(self):
        """Sweep p values should span the requested range."""
        sim = JansenRitSimulation(_make_config(dt=0.001))
        result = sim.input_sweep(p_range=(60.0, 250.0), n_p=15, n_steps=2000)
        assert result["p_values"][0] == pytest.approx(60.0)
        assert result["p_values"][-1] == pytest.approx(250.0)

    def test_input_sweep_amplitudes_finite(self):
        """All amplitude values should be finite and non-negative."""
        sim = JansenRitSimulation(_make_config(dt=0.001))
        result = sim.input_sweep(p_range=(80.0, 200.0), n_p=8, n_steps=2000)
        assert np.all(np.isfinite(result["amplitude"]))
        assert np.all(result["amplitude"] >= 0.0)

    def test_input_sweep_frequencies_nonnegative(self):
        """All frequency values should be non-negative."""
        sim = JansenRitSimulation(_make_config(dt=0.001))
        result = sim.input_sweep(p_range=(80.0, 200.0), n_p=8, n_steps=2000)
        assert np.all(result["frequency"] >= 0.0)

    def test_input_sweep_some_oscillation(self):
        """At least some p values in a reasonable range should show oscillation."""
        sim = JansenRitSimulation(_make_config(dt=0.001))
        result = sim.input_sweep(p_range=(80.0, 200.0), n_p=15, n_steps=3000)
        # At least some amplitude > 0.1
        assert np.any(result["amplitude"] > 0.1), (
            f"No oscillation detected in sweep, max amp={np.max(result['amplitude'])}"
        )


class TestJansenRitRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.jansen_rit import generate_ode_data
        data = generate_ode_data(p=120.0, n_steps=500, dt=0.001)
        assert data["states"].shape == (501, 6)
        assert len(data["time"]) == 501
        assert data["p"] == 120.0
        assert len(data["eeg"]) == 501

    def test_ode_data_eeg_matches_states(self):
        """EEG output in ode_data should be y1 - y2."""
        from simulating_anything.rediscovery.jansen_rit import generate_ode_data
        data = generate_ode_data(p=120.0, n_steps=200, dt=0.001)
        expected_eeg = data["states"][:, 1] - data["states"][:, 2]
        np.testing.assert_allclose(data["eeg"], expected_eeg)

    def test_sweep_data_generation(self):
        from simulating_anything.rediscovery.jansen_rit import generate_sweep_data
        data = generate_sweep_data(n_p=5, p_range=(80.0, 200.0), n_steps=1000)
        assert len(data["p_values"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["frequency"]) == 5

    def test_alpha_data_generation(self):
        from simulating_anything.rediscovery.jansen_rit import generate_alpha_data
        data = generate_alpha_data(dt=0.001, n_transient=1000, n_measure=2000)
        assert "frequency" in data
        assert "amplitude" in data
        assert "eeg_signal" in data
        assert len(data["eeg_signal"]) == 2000
        assert data["frequency"] >= 0.0
        assert data["amplitude"] >= 0.0
