"""Tests for the Van der Pol oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.van_der_pol import VanDerPolSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(mu: float = 1.0, dt: float = 0.005, x_0: float = 0.1) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.VAN_DER_POL,
        dt=dt,
        n_steps=1000,
        parameters={"mu": mu, "x_0": x_0, "v_0": 0.0},
    )


class TestVanDerPolSimulation:
    def test_initial_state(self):
        sim = VanDerPolSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [0.1, 0.0])

    def test_step_advances(self):
        sim = VanDerPolSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_trajectory_bounded(self):
        sim = VanDerPolSimulation(_make_config(mu=1.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
            x, v = sim.observe()
            assert abs(x) < 10, f"x diverged: {x}"
            assert abs(v) < 30, f"v diverged: {v}"

    def test_limit_cycle_exists(self):
        """After transient, trajectory should oscillate near limit cycle."""
        sim = VanDerPolSimulation(_make_config(mu=1.0, dt=0.005, x_0=0.01))
        sim.reset()
        # Run through transient
        for _ in range(10000):
            sim.step()
        # Measure amplitude over several cycles
        x_max = 0.0
        for _ in range(5000):
            sim.step()
            x_max = max(x_max, abs(sim.observe()[0]))
        # Limit cycle amplitude should be near 2
        assert 1.5 < x_max < 2.5, f"Limit cycle amplitude {x_max} not near 2"

    def test_small_mu_near_harmonic(self):
        """For small mu, period should be near 2*pi."""
        sim = VanDerPolSimulation(_make_config(mu=0.1, dt=0.005, x_0=0.5))
        sim.reset()
        T = sim.measure_period(n_periods=3)
        assert abs(T - 2 * np.pi) < 0.5, f"Period {T} not near 2*pi for small mu"

    def test_large_mu_longer_period(self):
        """For larger mu, period should be longer than 2*pi."""
        sim = VanDerPolSimulation(_make_config(mu=5.0, dt=0.002, x_0=0.5))
        sim.reset()
        T = sim.measure_period(n_periods=3)
        assert T > 2 * np.pi, f"Period {T} should be > 2*pi for mu=5"

    def test_different_mu_different_period(self):
        sim1 = VanDerPolSimulation(_make_config(mu=0.5, dt=0.005, x_0=0.5))
        sim1.reset()
        T1 = sim1.measure_period(n_periods=3)

        sim2 = VanDerPolSimulation(_make_config(mu=3.0, dt=0.005, x_0=0.5))
        sim2.reset()
        T2 = sim2.measure_period(n_periods=3)

        assert T2 > T1, f"Period should increase with mu: T1={T1}, T2={T2}"

    def test_energy_not_conserved(self):
        """Van der Pol is dissipative/driven: energy should not be constant."""
        sim = VanDerPolSimulation(_make_config(mu=1.0, x_0=2.0))
        sim.reset()
        E0 = sim.total_energy()
        for _ in range(1000):
            sim.step()
        E1 = sim.total_energy()
        assert E0 != pytest.approx(E1, rel=0.1)

    def test_derivatives_correct(self):
        """Check derivatives at a known point."""
        sim = VanDerPolSimulation(_make_config(mu=2.0))
        sim.reset()
        y = np.array([1.0, 0.5])
        dy = sim._derivatives(y)
        # dx/dt = v = 0.5
        # dv/dt = mu*(1-x^2)*v - x = 2*(1-1)*0.5 - 1 = -1
        np.testing.assert_allclose(dy, [0.5, -1.0])


class TestVanDerPolRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.van_der_pol import generate_ode_data
        data = generate_ode_data(mu=1.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["mu"] == 1.0

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.van_der_pol import generate_period_data
        data = generate_period_data(n_mu=5, dt=0.01)
        assert len(data["mu"]) == 5
        assert len(data["period"]) == 5
        assert len(data["amplitude"]) == 5
        # All periods should be positive and finite
        valid = np.isfinite(data["period"])
        assert np.sum(valid) >= 3
        assert np.all(data["period"][valid] > 0)

    def test_amplitude_near_two(self):
        from simulating_anything.rediscovery.van_der_pol import generate_period_data
        data = generate_period_data(n_mu=5, dt=0.005)
        valid = np.isfinite(data["amplitude"]) & (data["amplitude"] > 0)
        mean_amp = np.mean(data["amplitude"][valid])
        assert 1.5 < mean_amp < 2.5, f"Mean amplitude {mean_amp} not near 2"
