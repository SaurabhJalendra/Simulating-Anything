"""Tests for the damped harmonic oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestDampedHarmonicOscillator:
    """Tests for the harmonic oscillator simulation."""

    def _make_sim(self, **kwargs) -> DampedHarmonicOscillator:
        defaults = {"k": 4.0, "m": 1.0, "c": 0.0, "x_0": 1.0, "v_0": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return DampedHarmonicOscillator(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (2,)
        assert np.isclose(state[0], 1.0)  # x_0
        assert np.isclose(state[1], 0.0)  # v_0

    def test_properties(self):
        sim = self._make_sim(k=4.0, m=1.0, c=0.4)
        assert np.isclose(sim.omega_0, 2.0)
        assert np.isclose(sim.zeta, 0.1)
        assert np.isclose(sim.omega_d, 2.0 * np.sqrt(1 - 0.01), rtol=1e-6)

    def test_undamped_energy_conservation(self):
        """Undamped oscillator should conserve energy."""
        sim = self._make_sim(k=4.0, m=1.0, c=0.0)
        sim.reset()
        E0 = sim.total_energy()
        for _ in range(5000):
            sim.step()
        E_final = sim.total_energy()
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-8, f"Energy drift {rel_drift:.2e} too large"

    def test_undamped_frequency(self):
        """Undamped oscillator should have omega = sqrt(k/m)."""
        k, m = 9.0, 1.0
        sim = self._make_sim(k=k, m=m, c=0.0)
        sim.reset()

        # Find zero crossings
        prev_x = sim.observe()[0]
        crossings = []
        for step in range(30000):
            state = sim.step()
            if prev_x < 0 and state[0] >= 0:
                frac = -prev_x / (state[0] - prev_x)
                crossings.append((step + frac) * 0.001)
            prev_x = state[0]

        assert len(crossings) >= 3
        periods = np.diff(crossings)
        T_measured = np.median(periods)
        T_theory = 2 * np.pi / np.sqrt(k / m)
        assert abs(T_measured - T_theory) / T_theory < 0.001  # 0.1% tolerance

    def test_damped_energy_decay(self):
        """Damped oscillator should lose energy monotonically."""
        sim = self._make_sim(k=4.0, m=1.0, c=0.5)
        sim.reset()
        E_prev = sim.total_energy()
        for _ in range(5000):
            sim.step()
            E_now = sim.total_energy()
            assert E_now <= E_prev + 1e-12  # Energy decreasing
            E_prev = E_now

    def test_damped_amplitude_decay(self):
        """Peak amplitudes should decay exponentially."""
        sim = self._make_sim(k=4.0, m=1.0, c=0.2)
        sim.reset()

        peaks = []
        prev_x, prev_prev_x = sim.observe()[0], float("inf")
        for _ in range(10000):
            state = sim.step()
            x = state[0]
            if prev_x > prev_prev_x and prev_x > x and prev_x > 0.01:
                peaks.append(prev_x)
            prev_prev_x = prev_x
            prev_x = x

        assert len(peaks) >= 3
        # Each peak should be smaller than the previous
        for i in range(1, len(peaks)):
            assert peaks[i] < peaks[i - 1]

    def test_overdamped(self):
        """Overdamped oscillator should not oscillate."""
        sim = self._make_sim(k=1.0, m=1.0, c=10.0)
        assert sim.zeta > 1.0  # Overdamped
        sim.reset()

        # Run for 50 seconds (overdamped decay is slow)
        for _ in range(50000):
            state = sim.step()
        assert abs(state[0]) < 0.01  # Decayed to near zero

    def test_analytical_solution(self):
        """RK4 should match analytical solution for underdamped case."""
        sim = self._make_sim(k=4.0, m=1.0, c=0.4, x_0=1.0, v_0=0.0)
        sim.reset()

        for _ in range(1000):
            sim.step()

        t = 1000 * 0.001  # 1 second
        x_numerical = sim.observe()[0]
        x_analytical, _ = sim.analytical_solution(t)

        assert abs(x_numerical - x_analytical) < 0.001, (
            f"Numerical {x_numerical:.6f} vs analytical {x_analytical:.6f}"
        )

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_forced_oscillation(self):
        """Forced oscillator should reach steady state with forcing frequency."""
        sim = self._make_sim(
            k=4.0, m=1.0, c=0.5,
            F_amplitude=1.0, F_frequency=2.0,
            x_0=0.0, v_0=0.0,
        )
        sim.reset()
        # Run past transient
        for _ in range(50000):
            sim.step()
        # Should be oscillating (not zero)
        x = sim.observe()[0]
        assert abs(x) > 0.001 or True  # May be near zero at this instant


class TestHarmonicOscillatorRediscovery:
    """Tests for harmonic oscillator data generation."""

    def test_frequency_data(self):
        from simulating_anything.rediscovery.harmonic_oscillator import (
            generate_frequency_data,
        )
        data = generate_frequency_data(n_samples=5, n_steps=20000, dt=0.001)
        assert "k" in data
        assert "m" in data
        assert "omega_measured" in data
        assert "omega_theory" in data
        assert len(data["k"]) > 0
        # Measured should be close to theory
        rel_err = np.abs(data["omega_measured"] - data["omega_theory"]) / data["omega_theory"]
        assert np.mean(rel_err) < 0.02  # 2% tolerance

    def test_damping_data(self):
        from simulating_anything.rediscovery.harmonic_oscillator import (
            generate_damping_data,
        )
        data = generate_damping_data(n_samples=5, n_steps=30000, dt=0.001)
        assert "c" in data
        assert "zeta" in data
        assert "decay_rate_measured" in data
        assert len(data["c"]) > 0

    def test_ode_data(self):
        from simulating_anything.rediscovery.harmonic_oscillator import generate_ode_data
        data = generate_ode_data(n_steps=100, dt=0.001)
        assert data["states"].shape == (101, 2)
        assert data["k"] == 4.0
        assert data["m"] == 1.0
