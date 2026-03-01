"""Tests for the Oregonator model of the Belousov-Zhabotinsky reaction."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.oregonator import Oregonator
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    eps: float = 0.04,
    f: float = 1.0,
    q: float = 0.002,
    kw: float = 0.5,
    dt: float = 0.001,
    u_0: float = 0.5,
    v_0: float = 0.5,
    w_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.OREGONATOR,
        dt=dt,
        n_steps=1000,
        parameters={
            "eps": eps, "f": f, "q": q, "kw": kw,
            "u_0": u_0, "v_0": v_0, "w_0": w_0,
        },
    )


class TestOregonatorCreation:
    def test_initial_state_shape(self):
        sim = Oregonator(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_values(self):
        sim = Oregonator(_make_config(u_0=0.5, v_0=0.5, w_0=0.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, 0.5, 0.5])

    def test_default_parameters(self):
        sim = Oregonator(_make_config())
        assert sim.eps == pytest.approx(0.04)
        assert sim.f == pytest.approx(1.0)
        assert sim.q == pytest.approx(0.002)
        assert sim.kw == pytest.approx(0.5)


class TestOregonatorDynamics:
    def test_step_advances_state(self):
        sim = Oregonator(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_state_remains_positive(self):
        """Concentrations must remain non-negative."""
        sim = Oregonator(_make_config(dt=0.001))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative concentration: {state}"

    def test_trajectory_bounded(self):
        """State should remain bounded (no blow-up)."""
        sim = Oregonator(_make_config(dt=0.001))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"
            assert np.all(state < 100), f"State diverged: {state}"

    def test_observe_returns_current(self):
        sim = Oregonator(_make_config())
        sim.reset()
        sim.step()
        s1 = sim.observe()
        s2 = sim.observe()
        np.testing.assert_array_equal(s1, s2)

    def test_trajectory_reproducibility(self):
        """Same config should produce identical trajectories."""
        config = _make_config()
        sim1 = Oregonator(config)
        sim2 = Oregonator(config)
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            sim1.step()
            sim2.step()
        np.testing.assert_array_equal(sim1.observe(), sim2.observe())


class TestOregonatorProperties:
    def test_fixed_point_calculation(self):
        """Fixed point should satisfy the ODE at zero derivatives."""
        sim = Oregonator(_make_config())
        u_star, v_star, w_star = sim.fixed_point
        assert np.isfinite(u_star)
        assert np.isfinite(v_star)
        assert np.isfinite(w_star)
        # Verify derivatives are near zero at fixed point
        y = np.array([u_star, v_star, w_star])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0, 0, 0], atol=1e-10)

    def test_fixed_point_symmetry(self):
        """At steady state, u* = v* = w*."""
        sim = Oregonator(_make_config())
        u_star, v_star, w_star = sim.fixed_point
        assert u_star == pytest.approx(v_star, abs=1e-12)
        assert u_star == pytest.approx(w_star, abs=1e-12)

    def test_total_concentration(self):
        sim = Oregonator(_make_config(u_0=0.1, v_0=0.2, w_0=0.3))
        sim.reset()
        assert sim.total_concentration == pytest.approx(0.6)

    def test_oscillation_detection_default(self):
        """Default parameters (f=1.0) should produce oscillations."""
        sim = Oregonator(_make_config(f=1.0, dt=0.001))
        sim.reset()
        assert sim.is_oscillating is True

    def test_period_measurement(self):
        """Period should be finite and positive for oscillating parameters."""
        sim = Oregonator(_make_config(f=1.0, dt=0.001))
        sim.reset()
        period = sim.measure_period(n_periods=3)
        assert np.isfinite(period), "Period should be finite for f=1.0"
        assert period > 0, "Period should be positive"
        # Oregonator period is typically a few time units
        assert period < 20.0, f"Period too large: {period}"

    def test_amplitude_measurement(self):
        """Amplitude should be positive for oscillating parameters."""
        sim = Oregonator(_make_config(f=1.0, dt=0.001))
        sim.reset()
        amp_u, amp_v, amp_w = sim.measure_amplitude(n_periods=2)
        assert amp_u > 0.01, f"u amplitude too small: {amp_u}"
        assert amp_v > 0, f"v amplitude should be positive: {amp_v}"
        assert amp_w > 0, f"w amplitude should be positive: {amp_w}"

    def test_derivatives_at_fixed_point(self):
        """Derivatives should be zero at the fixed point."""
        sim = Oregonator(_make_config())
        u_star, v_star, w_star = sim.fixed_point
        y = np.array([u_star, v_star, w_star])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0, 0, 0], atol=1e-10)


class TestOregonatorBifurcation:
    def test_different_f_different_dynamics(self):
        """Changing f should change the dynamics."""
        config1 = _make_config(f=0.5, dt=0.001)
        config2 = _make_config(f=2.0, dt=0.001)
        sim1 = Oregonator(config1)
        sim2 = Oregonator(config2)
        sim1.reset()
        sim2.reset()
        for _ in range(5000):
            sim1.step()
            sim2.step()
        assert not np.allclose(sim1.observe(), sim2.observe(), atol=0.01)


class TestOregonatorRediscovery:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.oregonator import generate_trajectory_data
        data = generate_trajectory_data(n_steps=500, dt=0.001)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert "u" in data
        assert "v" in data
        assert "w" in data

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.oregonator import generate_bifurcation_data
        data = generate_bifurcation_data(n_f=5, dt=0.001)
        assert len(data["f"]) == 5
        assert len(data["amplitude"]) == 5

    def test_oscillation_data_generation(self):
        from simulating_anything.rediscovery.oregonator import generate_oscillation_data
        data = generate_oscillation_data(n_f=3, dt=0.001)
        assert len(data["f"]) == 3
        assert len(data["period"]) == 3
        assert len(data["amplitude_u"]) == 3
