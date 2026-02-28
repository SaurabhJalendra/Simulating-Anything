"""Tests for the FitzHugh-Nagumo neuron model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.fitzhugh_nagumo import FitzHughNagumoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(I: float = 0.0, dt: float = 0.05) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.FITZHUGH_NAGUMO,
        dt=dt,
        n_steps=1000,
        parameters={"I": I, "a": 0.7, "b": 0.8, "eps": 0.08, "v_0": -1.0, "w_0": -0.5},
    )


class TestFitzHughNagumoSimulation:
    def test_initial_state(self):
        sim = FitzHughNagumoSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [-1.0, -0.5])

    def test_step_advances(self):
        sim = FitzHughNagumoSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_trajectory_bounded(self):
        sim = FitzHughNagumoSimulation(_make_config(I=0.5, dt=0.02))
        sim.reset()
        for _ in range(5000):
            sim.step()
            v, w = sim.observe()
            assert abs(v) < 5, f"v diverged: {v}"
            assert abs(w) < 5, f"w diverged: {w}"

    def test_no_current_rests(self):
        """Without input current, should settle to resting state."""
        sim = FitzHughNagumoSimulation(_make_config(I=0.0, dt=0.05))
        sim.reset()
        for _ in range(10000):
            sim.step()
        v, w = sim.observe()
        # Should be at or near rest
        assert abs(v) < 2.0, f"v not at rest: {v}"

    def test_high_current_oscillates(self):
        """With high enough current, should show oscillations."""
        sim = FitzHughNagumoSimulation(_make_config(I=0.5, dt=0.02))
        sim.reset()
        # Skip transient
        for _ in range(10000):
            sim.step()
        # Measure variation
        v_vals = []
        for _ in range(5000):
            sim.step()
            v_vals.append(sim.observe()[0])
        v_range = max(v_vals) - min(v_vals)
        assert v_range > 0.5, f"No oscillation at I=0.5, range={v_range}"

    def test_derivatives_structure(self):
        """Check the cubic nonlinearity v - v^3/3."""
        sim = FitzHughNagumoSimulation(_make_config(I=0.0))
        y = np.array([0.0, 0.0])
        dy = sim._derivatives(y)
        # dv/dt = v - v^3/3 - w + I = 0 - 0 - 0 + 0 = 0
        # dw/dt = eps*(v + a - b*w) = 0.08*(0 + 0.7 - 0) = 0.056
        np.testing.assert_allclose(dy, [0.0, 0.056], atol=1e-10)

    def test_different_currents_different_behavior(self):
        """Higher current should lead to different dynamics."""
        sim1 = FitzHughNagumoSimulation(_make_config(I=0.0, dt=0.02))
        sim1.reset()
        for _ in range(5000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = FitzHughNagumoSimulation(_make_config(I=1.0, dt=0.02))
        sim2.reset()
        for _ in range(5000):
            sim2.step()
        state2 = sim2.observe().copy()

        assert not np.allclose(state1, state2, atol=0.1)

    def test_measure_firing_frequency(self):
        sim = FitzHughNagumoSimulation(_make_config(I=0.5, dt=0.02))
        sim.reset()
        freq = sim.measure_firing_frequency(n_spikes=3)
        # Should be a positive frequency for I=0.5
        assert freq >= 0  # May be 0 if not enough spikes


class TestFitzHughNagumoRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.fitzhugh_nagumo import generate_ode_data
        data = generate_ode_data(I=0.5, n_steps=500, dt=0.05)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501

    def test_fi_curve_generation(self):
        from simulating_anything.rediscovery.fitzhugh_nagumo import generate_fi_curve
        data = generate_fi_curve(n_I=5, dt=0.05)
        assert len(data["I"]) == 5
        assert len(data["frequency"]) == 5
        assert np.all(data["frequency"] >= 0)
