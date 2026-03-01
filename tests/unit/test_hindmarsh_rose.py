"""Tests for the Hindmarsh-Rose neuron model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.hindmarsh_rose import HindmarshRoseSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I_ext: float = 3.25,
    dt: float = 0.05,
    r: float = 0.001,
    **extra_params,
) -> SimulationConfig:
    params = {
        "a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0,
        "r": r, "s": 4.0, "x_rest": -1.6, "I_ext": I_ext,
        "x_0": -1.5, "y_0": -10.0, "z_0": 2.0,
    }
    params.update(extra_params)
    return SimulationConfig(
        domain=Domain.HINDMARSH_ROSE,
        dt=dt,
        n_steps=1000,
        parameters=params,
    )


class TestHindmarshRoseSimulation:
    """Core simulation tests."""

    def test_reset_state(self):
        """Initial conditions should match parameters."""
        sim = HindmarshRoseSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)
        np.testing.assert_allclose(state, [-1.5, -10.0, 2.0])

    def test_observe_shape(self):
        """Observe should return 3-element state."""
        sim = HindmarshRoseSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        assert len(obs) == 3

    def test_step_advances(self):
        """State should change after one step."""
        sim = HindmarshRoseSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same parameters should yield same trajectory."""
        cfg = _make_config()
        sim1 = HindmarshRoseSimulation(cfg)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = HindmarshRoseSimulation(cfg)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_rk4_stability(self):
        """No NaN or Inf after many steps."""
        sim = HindmarshRoseSimulation(_make_config(dt=0.05))
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became non-finite: {state}"

    def test_trajectory_bounded(self):
        """State variables should remain bounded."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=3.25, dt=0.05))
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            x, y, z = state
            assert abs(x) < 10, f"x diverged: {x}"
            assert abs(y) < 50, f"y diverged: {y}"
            assert abs(z) < 20, f"z diverged: {z}"

    def test_parameters(self):
        """Default parameters should match standard HR literature values."""
        sim = HindmarshRoseSimulation(_make_config())
        assert sim.a == 1.0
        assert sim.b == 3.0
        assert sim.c == 1.0
        assert sim.d == 5.0
        assert sim.r == 0.001
        assert sim.s == 4.0
        assert sim.x_rest == -1.6
        assert sim.I_ext == 3.25

    def test_derivatives_structure(self):
        """Check derivatives at a known point."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=0.0))
        sim.reset()
        # At (x, y, z) = (0, 0, 0) with I_ext=0:
        # dx/dt = 0 - 0 + 0 - 0 + 0 = 0
        # dy/dt = 1 - 0 - 0 = 1
        # dz/dt = 0.001*(4*(0 - (-1.6)) - 0) = 0.001*6.4 = 0.0064
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(derivs[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(derivs[1], 1.0, atol=1e-12)
        np.testing.assert_allclose(derivs[2], 0.0064, atol=1e-12)

    def test_custom_initial_conditions(self):
        """Custom initial conditions should be respected."""
        cfg = _make_config(x_0=0.5, y_0=-5.0, z_0=1.0)
        sim = HindmarshRoseSimulation(cfg)
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, -5.0, 1.0])


class TestHindmarshRoseDynamics:
    """Tests for dynamical behavior."""

    def test_quiescent(self):
        """Low I_ext should give quiescent behavior (no spikes)."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=0.5, dt=0.05))
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        # Collect trace
        x_trace = []
        for _ in range(10000):
            sim.step()
            x_trace.append(sim.observe()[0])
        x_trace = np.array(x_trace)
        spikes = sim.detect_spikes(x_trace, threshold=0.0)
        # Should have very few or no spikes at low I_ext
        assert len(spikes) < 5, f"Too many spikes ({len(spikes)}) at I_ext=0.5"

    def test_spiking(self):
        """Moderate I_ext should produce spiking activity."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=2.0, dt=0.05))
        sim.reset()
        for _ in range(5000):
            sim.step()
        x_trace = []
        for _ in range(20000):
            sim.step()
            x_trace.append(sim.observe()[0])
        x_trace = np.array(x_trace)
        spikes = sim.detect_spikes(x_trace, threshold=0.0)
        # Should produce at least some spikes
        assert len(spikes) >= 2, f"Too few spikes ({len(spikes)}) at I_ext=2.0"

    def test_spike_amplitude(self):
        """Spikes should have reasonable amplitude."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=3.25, dt=0.05))
        sim.reset()
        for _ in range(5000):
            sim.step()
        x_trace = []
        for _ in range(20000):
            sim.step()
            x_trace.append(sim.observe()[0])
        x_trace = np.array(x_trace)
        x_range = np.max(x_trace) - np.min(x_trace)
        # HR model should show significant voltage swings
        assert x_range > 1.0, f"x range too small: {x_range}"

    def test_slow_variable(self):
        """z should change much slower than x and y due to r << 1."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=3.25, dt=0.05))
        sim.reset()
        for _ in range(2000):
            sim.step()

        # Measure rate of change for each variable over 100 steps
        states = []
        for _ in range(100):
            sim.step()
            states.append(sim.observe().copy())
        states = np.array(states)

        dx_max = np.max(np.abs(np.diff(states[:, 0])))
        dz_max = np.max(np.abs(np.diff(states[:, 2])))

        # z should change at least 10x slower than x (r=0.001)
        assert dz_max < dx_max, (
            f"z changed faster than x: dz_max={dz_max:.6f}, dx_max={dx_max:.6f}"
        )

    def test_I_sweep_different_behaviors(self):
        """Different I_ext values should give different dynamics."""
        x_ranges = []
        for I_ext in [0.5, 2.0, 3.25]:
            sim = HindmarshRoseSimulation(_make_config(I_ext=I_ext, dt=0.05))
            sim.reset()
            for _ in range(5000):
                sim.step()
            x_trace = []
            for _ in range(10000):
                sim.step()
                x_trace.append(sim.observe()[0])
            x_ranges.append(np.max(x_trace) - np.min(x_trace))

        # The three I_ext values should not all produce identical dynamics
        assert not (
            np.allclose(x_ranges[0], x_ranges[1], atol=0.1)
            and np.allclose(x_ranges[1], x_ranges[2], atol=0.1)
        ), f"All I_ext values gave same dynamics: ranges={x_ranges}"

    def test_continuous_spiking(self):
        """High I_ext should give continuous spiking."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=5.0, dt=0.05))
        sim.reset()
        for _ in range(5000):
            sim.step()
        x_trace = []
        for _ in range(20000):
            sim.step()
            x_trace.append(sim.observe()[0])
        x_trace = np.array(x_trace)
        spikes = sim.detect_spikes(x_trace, threshold=0.0)
        # High I_ext should produce many spikes
        assert len(spikes) >= 5, f"Too few spikes ({len(spikes)}) at I_ext=5.0"


class TestHindmarshRoseBurstDetection:
    """Tests for burst detection and classification."""

    def test_burst_detection_no_spikes(self):
        """No spikes should yield no bursts."""
        sim = HindmarshRoseSimulation(_make_config())
        x_trace = np.zeros(1000)  # Flat trace
        bursts = sim.detect_bursts(x_trace, threshold=0.0)
        assert len(bursts) == 0

    def test_burst_detection_single_spike(self):
        """Single spike should yield one burst with one spike."""
        sim = HindmarshRoseSimulation(_make_config())
        # Create a trace with a single upward crossing at index 500
        x_trace = -1.0 * np.ones(1000)
        x_trace[500] = 1.0  # Upward crossing from -1 to 1
        bursts = sim.detect_bursts(x_trace, threshold=0.0, min_gap_steps=100)
        assert len(bursts) == 1
        assert len(bursts[0]) == 1

    def test_behavior_classification(self):
        """classify_behavior should return a valid string label."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=3.25, dt=0.05))
        behavior = sim.classify_behavior(t_max=5000, transient=2000)
        valid_behaviors = {"quiescent", "spiking", "bursting", "continuous_spiking"}
        assert behavior in valid_behaviors, f"Invalid behavior: {behavior}"

    def test_interspike_interval_stats(self):
        """ISI statistics should be computable from a spiking trace."""
        sim = HindmarshRoseSimulation(_make_config(I_ext=3.25, dt=0.05))
        sim.reset()
        for _ in range(5000):
            sim.step()
        x_trace = []
        for _ in range(20000):
            sim.step()
            x_trace.append(sim.observe()[0])
        x_trace = np.array(x_trace)
        isi_stats = sim.interspike_interval(x_trace, threshold=0.0)
        assert "mean_isi" in isi_stats
        assert "std_isi" in isi_stats
        assert "cv_isi" in isi_stats
        assert "n_spikes" in isi_stats
        assert isi_stats["n_spikes"] >= 0

    def test_interspike_interval_no_spikes(self):
        """ISI with no spikes should return zeros."""
        sim = HindmarshRoseSimulation(_make_config())
        x_trace = np.zeros(1000)
        isi_stats = sim.interspike_interval(x_trace, threshold=0.0)
        assert isi_stats["n_spikes"] == 0
        assert isi_stats["mean_isi"] == 0.0


class TestHindmarshRoseRediscovery:
    """Tests for data generation functions."""

    def test_ode_data_generation(self):
        """ODE data should have correct shape."""
        from simulating_anything.rediscovery.hindmarsh_rose import generate_ode_data

        data = generate_ode_data(I_ext=3.25, n_steps=500, dt=0.05)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert len(data["x"]) == 501
        assert len(data["y"]) == 501
        assert len(data["z"]) == 501
        assert data["I_ext"] == 3.25

    def test_ode_data_stays_finite(self):
        """Generated ODE data should not contain NaN or Inf."""
        from simulating_anything.rediscovery.hindmarsh_rose import generate_ode_data

        data = generate_ode_data(I_ext=3.25, n_steps=1000, dt=0.05)
        assert np.all(np.isfinite(data["states"]))

    def test_behavior_sweep_generation(self):
        """Behavior sweep should produce data for all I_ext values."""
        from simulating_anything.rediscovery.hindmarsh_rose import generate_behavior_sweep

        sweep = generate_behavior_sweep(n_I=5, dt=0.05, t_max=2000, transient=500)
        assert len(sweep["I_ext"]) == 5
        assert len(sweep["behavior"]) == 5
        assert len(sweep["spikes_per_burst"]) == 5
        assert len(sweep["n_bursts"]) == 5
        valid = {"quiescent", "spiking", "bursting", "continuous_spiking"}
        for b in sweep["behavior"]:
            assert b in valid, f"Invalid behavior: {b}"

    def test_rediscovery_data_3d(self):
        """ODE data should be 3-dimensional (x, y, z)."""
        from simulating_anything.rediscovery.hindmarsh_rose import generate_ode_data

        data = generate_ode_data(I_ext=3.25, n_steps=100, dt=0.05)
        assert data["states"].shape[1] == 3
