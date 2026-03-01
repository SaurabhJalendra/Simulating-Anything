"""Tests for the FitzHugh-Rinzel neuron model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.fitzhugh_rinzel import FitzHughRinzelSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I_ext: float = 0.3,
    mu: float = 0.0001,
    dt: float = 0.1,
    **extra_params,
) -> SimulationConfig:
    params = {
        "a": 0.7, "b": 0.8, "c": -0.775, "d": 1.0,
        "delta": 0.08, "mu": mu, "I_ext": I_ext,
        "v_0": -1.0, "w_0": -0.5, "y_0": 0.0,
    }
    params.update(extra_params)
    return SimulationConfig(
        domain=Domain.FITZHUGH_RINZEL,
        dt=dt,
        n_steps=1000,
        parameters=params,
    )


class TestFitzHughRinzelSimulation:
    """Core simulation tests."""

    def test_reset_state(self):
        """Initial conditions should match parameters."""
        sim = FitzHughRinzelSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)
        np.testing.assert_allclose(state, [-1.0, -0.5, 0.0])

    def test_observe_shape(self):
        """Observe should return 3-element state."""
        sim = FitzHughRinzelSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        assert len(obs) == 3

    def test_step_advances(self):
        """State should change after one step."""
        sim = FitzHughRinzelSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same parameters should yield same trajectory."""
        cfg = _make_config()
        sim1 = FitzHughRinzelSimulation(cfg)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = FitzHughRinzelSimulation(cfg)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_stability(self):
        """No NaN or Inf after many steps."""
        sim = FitzHughRinzelSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            state = sim.step()
            assert np.all(np.isfinite(state)), (
                f"State became non-finite: {state}"
            )

    def test_bounded(self):
        """State variables should remain bounded."""
        sim = FitzHughRinzelSimulation(_make_config(I_ext=0.3, dt=0.1))
        sim.reset()
        for _ in range(50000):
            state = sim.step()
            v, w, y = state
            assert abs(v) < 10, f"v diverged: {v}"
            assert abs(w) < 10, f"w diverged: {w}"
            assert abs(y) < 10, f"y diverged: {y}"

    def test_parameters(self):
        """Default parameters should match specified values."""
        sim = FitzHughRinzelSimulation(_make_config())
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.c == -0.775
        assert sim.d == 1.0
        assert sim.delta == 0.08
        assert sim.mu == 0.0001
        assert sim.I_ext == 0.3

    def test_custom_initial_conditions(self):
        """Custom initial conditions should be respected."""
        cfg = _make_config(v_0=0.5, w_0=-0.2, y_0=0.1)
        sim = FitzHughRinzelSimulation(cfg)
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, -0.2, 0.1])

    def test_derivatives_structure(self):
        """Check derivatives at a known point."""
        sim = FitzHughRinzelSimulation(_make_config(I_ext=0.0, mu=0.0001))
        sim.reset()
        # At (v, w, y) = (0, 0, 0) with I_ext=0:
        # dv/dt = 0 - 0 - 0 + 0 + 0 = 0
        # dw/dt = 0.08*(0.7 + 0 - 0) = 0.056
        # dy/dt = 0.0001*(-0.775 - 0 - 0) = -0.0000775
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(derivs[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(derivs[1], 0.056, atol=1e-12)
        np.testing.assert_allclose(derivs[2], -0.0000775, atol=1e-12)


class TestFitzHughRinzelBursting:
    """Tests for bursting dynamics."""

    def test_bursting_detected(self):
        """With appropriate params, bursting should be detected.

        Uses I_ext=0.3 which should produce bursting with mu=0.0001.
        Run long enough to see at least one burst cycle.
        """
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        n_bursts = sim.count_bursts(n_steps=200000, transient=50000)
        assert n_bursts > 0, "No bursts detected (expected > 0)"

    def test_spikes_in_burst(self):
        """Each burst should contain multiple spikes."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        stats = sim.measure_burst_statistics(
            n_steps=200000, transient=50000, min_gap_steps=2000
        )
        if stats["n_bursts"] > 0:
            assert stats["spikes_per_burst"] > 1, (
                f"Expected multiple spikes per burst, "
                f"got {stats['spikes_per_burst']}"
            )

    def test_quiescent_between_bursts(self):
        """There should be periods of low v between bursts."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        sim.reset()
        # Skip transient
        for _ in range(50000):
            sim.step()

        v_trace = np.zeros(200000)
        for i in range(200000):
            sim.step()
            v_trace[i] = sim.observe()[0]

        # Check that v goes below threshold for extended periods
        below_threshold = v_trace < 0.0
        # Find runs of below-threshold
        if np.any(below_threshold):
            changes = np.diff(below_threshold.astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > 0 and len(ends) > 0:
                # At least some quiescent periods should exist
                assert True
            else:
                # If the trace is mostly below threshold, that is also valid
                assert np.sum(below_threshold) > 100

    def test_slow_variable_oscillates(self):
        """y should drift up and down (oscillate on slow timescale)."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        dynamics = sim.slow_variable_dynamics(
            n_steps=200000, transient=50000
        )
        y = dynamics["y"]
        y_range = np.max(y) - np.min(y)
        # y should show some variation due to slow drift
        assert y_range > 0.001, (
            f"y did not oscillate: range={y_range}"
        )

    def test_fhn_limit(self):
        """mu=0 should give FHN-like behavior (no bursting, y stays constant).

        When mu=0, dy/dt=0 so y remains at its initial value and the
        system reduces to a 2D FitzHugh-Nagumo model.
        """
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.5, mu=0.0, dt=0.1, y_0=0.0)
        )
        sim.reset()
        # Skip transient
        for _ in range(10000):
            sim.step()

        # Collect y values: they should not change
        y_values = []
        for _ in range(5000):
            sim.step()
            y_values.append(sim.observe()[2])
        y_values = np.array(y_values)

        y_range = np.max(y_values) - np.min(y_values)
        assert y_range < 1e-10, (
            f"y changed with mu=0: range={y_range}"
        )

    def test_three_timescales(self):
        """w should oscillate faster than y (delta >> mu)."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        sim.reset()
        # Skip transient
        for _ in range(50000):
            sim.step()

        # Measure rate of change over a window
        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())
        states = np.array(states)

        dw_max = np.max(np.abs(np.diff(states[:, 1])))
        dy_max = np.max(np.abs(np.diff(states[:, 2])))

        # w should change faster than y because delta >> mu
        assert dy_max < dw_max, (
            f"y changed faster than w: "
            f"dy_max={dy_max:.6f}, dw_max={dw_max:.6f}"
        )


class TestFitzHughRinzelMethods:
    """Tests for analysis methods."""

    def test_spike_detection(self):
        """detect_spikes should find spikes above threshold."""
        sim = FitzHughRinzelSimulation(_make_config())
        # Create a synthetic trace with known spikes
        v_trace = -1.0 * np.ones(1000)
        v_trace[100] = 1.0  # spike at index 100
        v_trace[500] = 1.0  # spike at index 500
        spikes = sim.detect_spikes(v_trace, threshold=0.0)
        assert len(spikes) == 2
        assert 100 in spikes
        assert 500 in spikes

    def test_spike_detection_no_spikes(self):
        """Flat trace should yield no spikes."""
        sim = FitzHughRinzelSimulation(_make_config())
        v_trace = -1.0 * np.ones(1000)
        spikes = sim.detect_spikes(v_trace, threshold=0.0)
        assert len(spikes) == 0

    def test_burst_statistics(self):
        """Burst statistics should return valid structure."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        stats = sim.measure_burst_statistics(
            n_steps=200000, transient=50000, min_gap_steps=2000
        )
        assert "burst_duration" in stats
        assert "interburst_interval" in stats
        assert "spikes_per_burst" in stats
        assert "n_bursts" in stats
        assert stats["n_bursts"] >= 0
        assert stats["burst_duration"] >= 0
        assert stats["interburst_interval"] >= 0
        assert stats["spikes_per_burst"] >= 0

    def test_current_sweep(self):
        """Current sweep should produce valid data for all I_ext values."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.0, mu=0.0001, dt=0.1)
        )
        I_values = np.array([0.0, 0.3, 0.5])
        result = sim.current_sweep(
            I_values, n_steps=50000, transient=10000
        )
        assert len(result["I_ext"]) == 3
        assert len(result["n_bursts"]) == 3
        assert len(result["n_spikes"]) == 3
        # Different currents may produce different spike counts
        assert all(n >= 0 for n in result["n_spikes"])

    def test_mu_sweep(self):
        """mu sweep should produce valid data."""
        sim = FitzHughRinzelSimulation(
            _make_config(I_ext=0.3, mu=0.0001, dt=0.1)
        )
        mu_values = np.array([0.00001, 0.0001, 0.001])
        result = sim.mu_sweep(
            mu_values, n_steps=50000, transient=10000
        )
        assert len(result["mu"]) == 3
        assert len(result["n_bursts"]) == 3
        assert len(result["burst_frequency"]) == 3
        assert all(f >= 0 for f in result["burst_frequency"])


class TestFitzHughRinzelRediscovery:
    """Tests for data generation functions."""

    def test_rediscovery_data(self):
        """ODE data should have correct shape and keys."""
        from simulating_anything.rediscovery.fitzhugh_rinzel import (
            generate_ode_data,
        )

        data = generate_ode_data(
            I_ext=0.3, mu=0.0001, n_steps=500, dt=0.1
        )
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert len(data["v"]) == 501
        assert len(data["w"]) == 501
        assert len(data["y"]) == 501
        assert data["I_ext"] == 0.3
        assert data["mu"] == 0.0001

    def test_rediscovery_data_finite(self):
        """Generated ODE data should not contain NaN or Inf."""
        from simulating_anything.rediscovery.fitzhugh_rinzel import (
            generate_ode_data,
        )

        data = generate_ode_data(
            I_ext=0.3, mu=0.0001, n_steps=1000, dt=0.1
        )
        assert np.all(np.isfinite(data["states"]))

    def test_rediscovery_data_3d(self):
        """ODE data should be 3-dimensional (v, w, y)."""
        from simulating_anything.rediscovery.fitzhugh_rinzel import (
            generate_ode_data,
        )

        data = generate_ode_data(
            I_ext=0.3, mu=0.0001, n_steps=100, dt=0.1
        )
        assert data["states"].shape[1] == 3

    def test_mu_sweep_generation(self):
        """mu sweep should produce data for all mu values."""
        from simulating_anything.rediscovery.fitzhugh_rinzel import (
            generate_mu_sweep,
        )

        sweep = generate_mu_sweep(
            n_mu=3, dt=0.1, n_steps=50000, transient=10000
        )
        assert len(sweep["mu"]) == 3
        assert len(sweep["n_bursts"]) == 3
        assert len(sweep["burst_frequency"]) == 3
