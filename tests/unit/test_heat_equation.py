"""Tests for the 1D heat equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.heat_equation import HeatEquation1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(D: float = 0.1, dt: float = 0.01, N: int = 128) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.HEAT_EQUATION_1D,
        dt=dt,
        n_steps=100,
        parameters={"D": D, "N": float(N), "L": 2 * np.pi},
    )


class TestHeatEquation1D:
    def test_initial_state_shape(self):
        sim = HeatEquation1DSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (128,)

    def test_step_advances(self):
        sim = HeatEquation1DSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_gaussian_init(self):
        sim = HeatEquation1DSimulation(_make_config())
        sim.init_type = "gaussian"
        state = sim.reset()
        # Gaussian should peak in the middle
        assert np.argmax(state) == 64 or np.argmax(state) == 63

    def test_sine_init(self):
        sim = HeatEquation1DSimulation(_make_config())
        sim.init_type = "sine"
        state = sim.reset()
        # Sine wave should oscillate
        assert np.max(state) > 0.9
        assert np.min(state) < -0.9

    def test_step_init(self):
        sim = HeatEquation1DSimulation(_make_config())
        sim.init_type = "step"
        state = sim.reset()
        assert np.max(state) == 1.0
        assert np.min(state) == 0.0

    def test_heat_conserved(self):
        """Total heat should be conserved (periodic BCs)."""
        sim = HeatEquation1DSimulation(_make_config())
        sim.reset()
        h0 = sim.total_heat
        for _ in range(100):
            sim.step()
        h1 = sim.total_heat
        np.testing.assert_allclose(h0, h1, rtol=1e-10)

    def test_max_temperature_decreases(self):
        """Max temperature should decrease as heat diffuses."""
        sim = HeatEquation1DSimulation(_make_config())
        sim.reset()
        T_max_0 = sim.max_temperature
        for _ in range(50):
            sim.step()
        T_max_1 = sim.max_temperature
        assert T_max_1 < T_max_0

    def test_higher_D_faster_diffusion(self):
        """Higher diffusion coefficient should spread heat faster."""
        sim1 = HeatEquation1DSimulation(_make_config(D=0.01))
        sim1.reset()
        for _ in range(100):
            sim1.step()
        max1 = sim1.max_temperature

        sim2 = HeatEquation1DSimulation(_make_config(D=1.0))
        sim2.reset()
        for _ in range(100):
            sim2.step()
        max2 = sim2.max_temperature

        assert max2 < max1  # Higher D -> more spread -> lower peak

    def test_sine_exact_decay(self):
        """Sine mode should decay as exp(-D*k^2*t) (spectral solver is exact)."""
        D = 0.1
        sim = HeatEquation1DSimulation(_make_config(D=D))
        sim.init_type = "sine"
        sim.reset()

        a0 = np.max(np.abs(sim.observe()))

        n_steps = 100
        dt = 0.01
        for _ in range(n_steps):
            sim.step()

        af = np.max(np.abs(sim.observe()))
        t = n_steps * dt
        k = 1.0  # mode 1, L=2*pi
        theory = a0 * np.exp(-D * k**2 * t)
        np.testing.assert_allclose(af, theory, rtol=1e-10)

    def test_decay_rate_formula(self):
        sim = HeatEquation1DSimulation(_make_config(D=0.5))
        rate = sim.decay_rate_of_mode(1)
        # k = 2*pi*1/(2*pi) = 1, so rate = D*1 = 0.5
        assert rate == pytest.approx(0.5)


class TestHeatEquationRediscovery:
    def test_decay_data_generation(self):
        from simulating_anything.rediscovery.heat_equation import generate_decay_data
        data = generate_decay_data(n_D=5, n_steps=50, dt=0.01, N=64)
        assert len(data["D"]) == 5
        assert len(data["decay_rate"]) == 5
        valid = np.isfinite(data["decay_rate"])
        assert np.sum(valid) >= 3
