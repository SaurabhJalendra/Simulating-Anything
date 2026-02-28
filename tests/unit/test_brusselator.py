"""Tests for the Brusselator chemical oscillator."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.brusselator import BrusselatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(a: float = 1.0, b: float = 3.0, dt: float = 0.01) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BRUSSELATOR,
        dt=dt,
        n_steps=1000,
        parameters={"a": a, "b": b, "u_0": 1.5, "v_0": 1.5},
    )


class TestBrusselatorSimulation:
    def test_initial_state(self):
        sim = BrusselatorSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [1.5, 1.5])

    def test_step_advances(self):
        sim = BrusselatorSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_fixed_point(self):
        sim = BrusselatorSimulation(_make_config(a=1.0, b=3.0))
        u_star, v_star = sim.fixed_point
        assert u_star == pytest.approx(1.0)
        assert v_star == pytest.approx(3.0)

    def test_hopf_threshold(self):
        sim = BrusselatorSimulation(_make_config(a=1.0))
        assert sim.hopf_threshold == pytest.approx(2.0)

        sim2 = BrusselatorSimulation(_make_config(a=2.0))
        assert sim2.hopf_threshold == pytest.approx(5.0)

    def test_below_hopf_converges(self):
        """Below Hopf bifurcation, should converge to fixed point."""
        sim = BrusselatorSimulation(_make_config(a=1.0, b=1.5, dt=0.01))
        sim.reset()
        for _ in range(20000):
            sim.step()
        u, v = sim.observe()
        np.testing.assert_allclose([u, v], [1.0, 1.5], atol=0.1)

    def test_above_hopf_oscillates(self):
        """Above Hopf bifurcation, should oscillate."""
        sim = BrusselatorSimulation(_make_config(a=1.0, b=3.0, dt=0.005))
        sim.reset()
        # Skip transient
        for _ in range(20000):
            sim.step()
        # Measure variation
        u_vals = []
        for _ in range(5000):
            sim.step()
            u_vals.append(sim.observe()[0])
        amplitude = max(u_vals) - min(u_vals)
        assert amplitude > 0.5, f"No oscillation detected, amplitude={amplitude}"

    def test_trajectory_bounded(self):
        sim = BrusselatorSimulation(_make_config(a=1.0, b=3.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
            u, v = sim.observe()
            assert abs(u) < 20, f"u diverged: {u}"
            assert abs(v) < 20, f"v diverged: {v}"

    def test_derivatives_at_fixed_point(self):
        sim = BrusselatorSimulation(_make_config(a=1.0, b=2.0))
        y = np.array([1.0, 2.0])  # Fixed point for a=1, b=2
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_is_oscillatory_flag(self):
        sim_stable = BrusselatorSimulation(_make_config(a=1.0, b=1.5))
        assert not sim_stable.is_oscillatory

        sim_osc = BrusselatorSimulation(_make_config(a=1.0, b=3.0))
        assert sim_osc.is_oscillatory


class TestBrusselatorRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.brusselator import generate_ode_data
        data = generate_ode_data(a=1.0, b=3.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501

    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.brusselator import generate_bifurcation_data
        data = generate_bifurcation_data(a=1.0, n_b=5, dt=0.01)
        assert len(data["b"]) == 5
        assert len(data["amplitude"]) == 5
        assert data["b_c_theory"] == pytest.approx(2.0)
