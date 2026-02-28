"""Tests for V2 simulation domains: SIR epidemic and double pendulum."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.epidemiological import SIRSimulation
from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestSIRSimulation:
    """Tests for the SIR epidemic model."""

    def _make_sim(self, beta=0.3, gamma=0.1, S_0=0.99, I_0=0.01) -> SIRSimulation:
        config = SimulationConfig(
            domain=Domain.EPIDEMIOLOGICAL, dt=0.1, n_steps=1000,
            parameters={"beta": beta, "gamma": gamma, "S_0": S_0, "I_0": I_0},
        )
        return SIRSimulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state.sum(), 1.0)
        assert state[0] > 0.98  # S_0

    def test_conservation(self):
        """S + I + R = 1 at all times."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(500):
            state = sim.step()
            assert np.isclose(state.sum(), 1.0, atol=1e-10)
            assert np.all(state >= 0)

    def test_epidemic_occurs(self):
        """With R0 > 1, infected should rise then fall."""
        sim = self._make_sim(beta=0.3, gamma=0.1)  # R0 = 3
        sim.reset()
        max_I = 0
        for _ in range(5000):
            state = sim.step()
            max_I = max(max_I, state[1])
        assert max_I > 0.1  # Significant epidemic
        assert state[1] < 0.001  # Epidemic ends

    def test_no_epidemic(self):
        """With R0 < 1, infected should decay monotonically."""
        sim = self._make_sim(beta=0.05, gamma=0.1)  # R0 = 0.5
        sim.reset()
        prev_I = 0.01
        for _ in range(500):
            state = sim.step()
            assert state[1] <= prev_I + 1e-10  # I decreasing
            prev_I = state[1]

    def test_R0_property(self):
        sim = self._make_sim(beta=0.6, gamma=0.2)
        assert np.isclose(sim.R0, 3.0)

    def test_final_size_property(self):
        sim = self._make_sim(beta=0.3, gamma=0.1)
        # R0 = 3, so final size should be significant
        assert sim.final_size > 0.5

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        assert obs is sim._state

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        # States should differ
        assert not np.allclose(state0, state1)


class TestDoublePendulum:
    """Tests for the double pendulum simulation."""

    def _make_sim(self, **kwargs) -> DoublePendulumSimulation:
        defaults = {
            "m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
            "theta1_0": np.pi / 4, "theta2_0": np.pi / 4,
            "omega1_0": 0.0, "omega2_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CHAOTIC_ODE, dt=0.001, n_steps=10000,
            parameters=defaults,
        )
        return DoublePendulumSimulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)
        assert np.isclose(state[0], np.pi / 4)
        assert np.isclose(state[2], 0.0)

    def test_energy_conservation(self):
        """Total energy should be conserved (within RK4 accuracy)."""
        sim = self._make_sim()
        sim.reset()
        E0 = sim.total_energy()
        for _ in range(5000):
            sim.step()
        E_final = sim.total_energy()
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_small_angle_period(self):
        """Small-angle pendulum 1 should have period T = 2*pi*sqrt(L/g)."""
        L1 = 1.0
        g = 9.81
        sim = self._make_sim(
            m1=10.0, m2=0.01, L1=L1, g=g,
            theta1_0=0.05, theta2_0=0.0,
        )
        sim.reset()

        # Run and find zero crossings
        prev_th1 = sim.observe()[0]
        crossings = []
        for step in range(50000):
            state = sim.step()
            if prev_th1 < 0 and state[0] >= 0:
                frac = -prev_th1 / (state[0] - prev_th1)
                crossings.append((step + frac) * 0.001)
            prev_th1 = state[0]

        assert len(crossings) >= 3
        periods = np.diff(crossings)
        T_measured = np.median(periods)
        T_theory = 2 * np.pi * np.sqrt(L1 / g)
        assert abs(T_measured - T_theory) / T_theory < 0.01  # 1% tolerance

    def test_cartesian_positions(self):
        sim = self._make_sim(theta1_0=0.0, theta2_0=0.0)
        sim.reset()
        (x1, y1), (x2, y2) = sim.cartesian_positions()
        assert np.isclose(x1, 0.0, atol=1e-10)
        assert np.isclose(y1, -1.0)  # Hanging down
        assert np.isclose(x2, 0.0, atol=1e-10)
        assert np.isclose(y2, -2.0)  # Both hanging down

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)


class TestSIRDataGeneration:
    """Tests for SIR rediscovery data generation."""

    def test_generate_sweep(self):
        from simulating_anything.rediscovery.sir_epidemic import generate_sir_sweep_data
        data = generate_sir_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "gamma" in data
        assert "R0" in data
        assert "peak_I" in data
        assert "final_size" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)

    def test_generate_ode_data(self):
        from simulating_anything.rediscovery.sir_epidemic import generate_sir_ode_data
        data = generate_sir_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 3)
        assert np.allclose(data["states"].sum(axis=1), 1.0, atol=1e-8)


class TestDoublePendulumDataGeneration:
    """Tests for double pendulum rediscovery data generation."""

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.double_pendulum import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(n_trajectories=3, n_steps=1000, dt=0.001)
        assert len(data["final_drift"]) == 3
        assert np.all(data["final_drift"] < 1e-4)

    def test_period_data(self):
        from simulating_anything.rediscovery.double_pendulum import (
            generate_small_angle_period_data,
        )
        data = generate_small_angle_period_data(n_samples=3, n_steps=20000, dt=0.001)
        assert len(data["L1"]) > 0
        assert len(data["T_measured"]) == len(data["T_theory"])
        # Period should be close to theory
        rel_err = np.abs(data["T_measured"] - data["T_theory"]) / data["T_theory"]
        assert np.all(rel_err < 0.05)  # 5% tolerance
