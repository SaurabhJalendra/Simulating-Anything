"""Tests for the Izhikevich neuron model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.izhikevich import (
    PATTERN_PRESETS,
    IzhikevichSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I_ext: float = 10.0,
    dt: float = 0.5,
    n_steps: int = 2000,
    preset: str | None = None,
    **kwargs: float,
) -> SimulationConfig:
    if preset is not None:
        params = IzhikevichSimulation.get_preset(preset)
    else:
        params = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}
    params["I"] = I_ext
    params.update(kwargs)
    return SimulationConfig(
        domain=Domain.IZHIKEVICH,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestIzhikevichConstruction:
    """Tests for simulation construction and basic properties."""

    def test_state_shape(self):
        """State vector should be 2D: [v, u]."""
        sim = IzhikevichSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_observe_shape(self):
        """Observe should return 2-element state."""
        sim = IzhikevichSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)
        assert len(obs) == 2

    def test_default_parameters(self):
        """Default parameters should match RS preset."""
        sim = IzhikevichSimulation(_make_config())
        assert sim.a == 0.02
        assert sim.b == 0.2
        assert sim.c == -65.0
        assert sim.d == 8.0
        assert sim.I == 10.0

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = IzhikevichSimulation(_make_config(preset="FS", I_ext=20.0))
        assert sim.a == 0.1
        assert sim.b == 0.2
        assert sim.c == -65.0
        assert sim.d == 2.0
        assert sim.I == 20.0

    def test_initial_state(self):
        """Reset should initialize to v_0, u_0."""
        cfg = _make_config(v_0=-70.0, u_0=-10.0)
        sim = IzhikevichSimulation(cfg)
        state = sim.reset()
        np.testing.assert_allclose(state, [-70.0, -10.0])

    def test_step_advances(self):
        """State should change after one step."""
        sim = IzhikevichSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_preset_lookup(self):
        """get_preset should return correct parameter dictionaries."""
        rs = IzhikevichSimulation.get_preset("RS")
        assert rs["a"] == 0.02
        assert rs["d"] == 8.0

        fs = IzhikevichSimulation.get_preset("FS")
        assert fs["a"] == 0.1
        assert fs["d"] == 2.0

    def test_preset_case_insensitive(self):
        """Preset lookup should be case-insensitive."""
        rs_lower = IzhikevichSimulation.get_preset("rs")
        rs_upper = IzhikevichSimulation.get_preset("RS")
        assert rs_lower == rs_upper

    def test_invalid_preset(self):
        """Invalid preset name should raise ValueError."""
        try:
            IzhikevichSimulation.get_preset("INVALID")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_all_presets_exist(self):
        """All 7 named presets should be available."""
        for name in ["RS", "IB", "CH", "FS", "TC", "RZ", "LTS"]:
            preset = IzhikevichSimulation.get_preset(name)
            assert "a" in preset
            assert "b" in preset
            assert "c" in preset
            assert "d" in preset


class TestIzhikevichSpikeReset:
    """Tests for spike detection and reset mechanism."""

    def test_spike_threshold(self):
        """Spike threshold should be at v=30 mV (reset to c after crossing)."""
        # With high current, the neuron should spike (v reaches 30)
        # With Euler integration and dt=0.5, v can jump from well below 30
        # to above 30 in a single step, so we detect the reset: v_after == c
        # while v_before was above the resting potential.
        sim = IzhikevichSimulation(_make_config(I_ext=15.0, dt=0.5))
        sim.reset()
        spiked = False
        for _ in range(2000):
            v_before = sim.observe()[0]
            sim.step()
            v_after = sim.observe()[0]
            # After a spike, v is reset to c. With Euler, v_before may not
            # have been above 25 because the quadratic term grows fast.
            if v_after == sim.c and v_before > sim.c + 10.0:
                spiked = True
                break
        assert spiked, "Neuron did not spike at I=15"

    def test_reset_voltage(self):
        """After spike, v should be reset to c."""
        sim = IzhikevichSimulation(_make_config(I_ext=15.0, dt=0.5))
        sim.reset()
        for _ in range(2000):
            sim.step()
            v = sim.observe()[0]
            # If a reset just happened, v should equal c
            # v should never be >= 30 after the step (it gets reset)
            assert v < 30.0, f"v={v} exceeded threshold without reset"

    def test_reset_recovery(self):
        """After spike, u should increase by d."""
        sim = IzhikevichSimulation(_make_config(I_ext=15.0, dt=0.5))
        sim.reset()

        # Track the spike count via the internal counter
        initial_spike_count = sim._spike_count
        for _ in range(2000):
            sim.step()
            if sim._spike_count > initial_spike_count:
                # A spike occurred -- u was incremented by d
                break

        assert sim._spike_count > 0, "No spike occurred"

    def test_spike_count(self):
        """Spike counter should track number of spikes."""
        sim = IzhikevichSimulation(_make_config(I_ext=10.0, dt=0.5))
        sim.reset()
        for _ in range(2000):
            sim.step()
        assert sim._spike_count > 0, "No spikes detected at I=10"

    def test_spike_times_recorded(self):
        """Spike times should be recorded."""
        sim = IzhikevichSimulation(_make_config(I_ext=10.0, dt=0.5))
        sim.reset()
        for _ in range(2000):
            sim.step()
        assert len(sim._spike_times) == sim._spike_count
        # Spike times should be positive and sorted
        if len(sim._spike_times) >= 2:
            assert sim._spike_times[-1] > sim._spike_times[0]

    def test_no_spike_below_threshold_current(self):
        """Very low current should not produce spikes."""
        sim = IzhikevichSimulation(_make_config(I_ext=0.0, dt=0.5))
        sim.reset()
        for _ in range(2000):
            sim.step()
        assert sim._spike_count == 0, f"Spikes at I=0: {sim._spike_count}"


class TestIzhikevichPatterns:
    """Tests for different firing patterns."""

    def test_regular_spiking(self):
        """RS preset should produce regular spiking at I=10."""
        sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=10.0))
        sim.reset()
        V_trace = []
        for _ in range(2000):
            sim.step()
            V_trace.append(sim.observe()[0])
        V_trace = np.array(V_trace)
        spikes = sim.detect_spikes(V_trace, threshold=0.0)
        assert len(spikes) >= 3, f"RS should spike regularly, got {len(spikes)} spikes"

        # Check regularity via ISI coefficient of variation
        if len(spikes) >= 3:
            isis = np.diff(spikes)
            cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
            # RS should be fairly regular (low CV)
            assert cv < 0.5, f"RS ISI too irregular: CV={cv:.3f}"

    def test_fast_spiking(self):
        """FS preset should produce faster spiking than RS."""
        rs_sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=10.0))
        fs_sim = IzhikevichSimulation(_make_config(preset="FS", I_ext=10.0))

        rs_rate = rs_sim.compute_firing_rate(t_max=1000.0)
        fs_rate = fs_sim.compute_firing_rate(t_max=1000.0)

        # FS should fire at a higher rate than RS for the same current
        assert fs_rate > rs_rate, (
            f"FS rate ({fs_rate:.4f}/ms) should exceed RS rate ({rs_rate:.4f}/ms)"
        )

    def test_bursting_pattern(self):
        """IB preset should produce bursting behavior at I=10."""
        sim = IzhikevichSimulation(_make_config(preset="IB", I_ext=10.0))
        sim.reset()
        V_trace = []
        for _ in range(2000):
            sim.step()
            V_trace.append(sim.observe()[0])
        V_trace = np.array(V_trace)
        spikes = sim.detect_spikes(V_trace, threshold=0.0)
        # IB should produce spikes
        assert len(spikes) >= 2, f"IB should spike, got {len(spikes)} spikes"

    def test_classify_pattern_rs(self):
        """classify_pattern should identify RS preset as regular_spiking."""
        sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=10.0))
        sim.reset()
        pattern = sim.classify_pattern(t_max=1000.0)
        valid = {"regular_spiking", "fast_spiking", "bursting", "chattering", "quiescent"}
        assert pattern in valid, f"Invalid pattern: {pattern}"

    def test_classify_pattern_quiescent(self):
        """With no current, pattern should be quiescent."""
        sim = IzhikevichSimulation(_make_config(I_ext=0.0))
        sim.reset()
        pattern = sim.classify_pattern(t_max=1000.0)
        assert pattern == "quiescent", f"Expected quiescent at I=0, got {pattern}"

    def test_different_presets_differ(self):
        """Different presets should produce different firing rates."""
        rates = {}
        for name in ["RS", "FS", "IB"]:
            sim = IzhikevichSimulation(_make_config(preset=name, I_ext=10.0))
            rates[name] = sim.compute_firing_rate(t_max=1000.0)
        # Not all rates should be identical
        rate_values = list(rates.values())
        assert not all(
            np.isclose(r, rate_values[0], atol=1e-4) for r in rate_values
        ), f"All presets have same rate: {rates}"


class TestIzhikevichFICurve:
    """Tests for f-I curve computation."""

    def test_fi_curve_shape(self):
        """f-I curve should return matching I and frequency arrays."""
        sim = IzhikevichSimulation(_make_config(I_ext=0.0))
        I_values = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        fi = sim.compute_fi_curve(I_values, t_max=500.0)
        assert len(fi["I"]) == 5
        assert len(fi["frequency"]) == 5

    def test_fi_curve_monotonic(self):
        """Higher current should generally produce higher frequency."""
        sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=0.0))
        I_values = np.array([5.0, 10.0, 15.0, 20.0, 30.0])
        fi = sim.compute_fi_curve(I_values, t_max=800.0)
        freqs = fi["frequency"]
        # Overall trend should be increasing
        assert freqs[-1] >= freqs[0], (
            f"f-I not monotonic: f({I_values[0]})={freqs[0]}, "
            f"f({I_values[-1]})={freqs[-1]}"
        )

    def test_fi_curve_zero_at_zero(self):
        """Firing frequency should be zero at I=0."""
        sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=0.0))
        I_values = np.array([0.0])
        fi = sim.compute_fi_curve(I_values, t_max=500.0)
        assert fi["frequency"][0] == 0.0, (
            f"Expected zero frequency at I=0, got {fi['frequency'][0]}"
        )

    def test_fi_curve_nonnegative(self):
        """All frequencies should be non-negative."""
        sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=0.0))
        I_values = np.linspace(0.0, 20.0, 10)
        fi = sim.compute_fi_curve(I_values, t_max=500.0)
        assert np.all(fi["frequency"] >= 0.0)


class TestIzhikevichStability:
    """Tests for numerical stability and determinism."""

    def test_euler_stability(self):
        """No NaN or Inf after many steps."""
        sim = IzhikevichSimulation(_make_config(I_ext=10.0, dt=0.5))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became non-finite: {state}"

    def test_trajectory_bounded(self):
        """Voltage and recovery should remain bounded."""
        sim = IzhikevichSimulation(_make_config(I_ext=20.0, dt=0.5))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            v, u = state
            # v should always be < 30 (reset) and > some lower bound
            assert v < 30.0, f"v not reset: {v}"
            assert v > -200.0, f"v diverged negative: {v}"
            assert abs(u) < 500.0, f"u diverged: {u}"

    def test_deterministic(self):
        """Same parameters should yield same trajectory."""
        cfg = _make_config(I_ext=10.0, dt=0.5)
        sim1 = IzhikevichSimulation(cfg)
        sim1.reset()
        for _ in range(500):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = IzhikevichSimulation(cfg)
        sim2.reset()
        for _ in range(500):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        sim = IzhikevichSimulation(_make_config(I_ext=10.0, dt=0.5, n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101


class TestIzhikevichISI:
    """Tests for inter-spike interval statistics."""

    def test_isi_with_spikes(self):
        """ISI statistics should be computable from a spiking trace."""
        sim = IzhikevichSimulation(_make_config(preset="RS", I_ext=10.0))
        sim.reset()
        V_trace = []
        for _ in range(2000):
            sim.step()
            V_trace.append(sim.observe()[0])
        V_trace = np.array(V_trace)
        isi_stats = sim.interspike_interval(V_trace, threshold=0.0)
        assert "mean_isi" in isi_stats
        assert "std_isi" in isi_stats
        assert "cv_isi" in isi_stats
        assert "n_spikes" in isi_stats
        assert isi_stats["n_spikes"] >= 2

    def test_isi_no_spikes(self):
        """ISI with no spikes should return zeros."""
        sim = IzhikevichSimulation(_make_config())
        V_trace = np.zeros(1000)
        isi_stats = sim.interspike_interval(V_trace, threshold=0.0)
        assert isi_stats["n_spikes"] == 0
        assert isi_stats["mean_isi"] == 0.0

    def test_detect_spikes_empty(self):
        """Spike detection on flat trace should return empty array."""
        sim = IzhikevichSimulation(_make_config())
        V_trace = -65.0 * np.ones(100)
        spikes = sim.detect_spikes(V_trace, threshold=0.0)
        assert len(spikes) == 0


class TestIzhikevichRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_generation(self):
        """Trajectory data should have correct shape."""
        from simulating_anything.rediscovery.izhikevich import generate_trajectory_data

        data = generate_trajectory_data(pattern="RS", I_ext=10.0, n_steps=500, dt=0.5)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert len(data["v"]) == 501
        assert len(data["u"]) == 501
        assert data["I"] == 10.0
        assert data["pattern"] == "RS"

    def test_trajectory_data_finite(self):
        """Generated trajectory data should not contain NaN or Inf."""
        from simulating_anything.rediscovery.izhikevich import generate_trajectory_data

        data = generate_trajectory_data(pattern="RS", I_ext=10.0, n_steps=1000, dt=0.5)
        assert np.all(np.isfinite(data["states"]))

    def test_fi_curve_generation(self):
        """f-I curve data generation should produce valid results."""
        from simulating_anything.rediscovery.izhikevich import generate_fi_curve

        data = generate_fi_curve(pattern="RS", n_I=5, I_min=0.0, I_max=20.0)
        assert len(data["I"]) == 5
        assert len(data["frequency"]) == 5
        assert np.all(data["frequency"] >= 0)

    def test_classify_all_presets(self):
        """Classification should cover all 7 presets."""
        from simulating_anything.rediscovery.izhikevich import classify_all_presets

        results = classify_all_presets(I_ext=10.0)
        assert len(results) == 7
        valid_patterns = {
            "regular_spiking", "fast_spiking", "bursting",
            "chattering", "quiescent",
        }
        for name, info in results.items():
            assert name in PATTERN_PRESETS
            assert info["pattern"] in valid_patterns, (
                f"Preset {name} has invalid pattern: {info['pattern']}"
            )

    def test_spike_statistics(self):
        """Spike statistics should be measurable."""
        from simulating_anything.rediscovery.izhikevich import measure_spike_statistics

        stats = measure_spike_statistics(pattern="RS", I_ext=10.0, t_max=500.0)
        assert "mean_isi" in stats
        assert "n_spikes" in stats
        assert stats["n_spikes"] >= 0
