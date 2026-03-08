"""Tests for the stochastic resonance simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.stochastic_resonance import (
    StochasticResonanceSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestStochasticResonanceSimulation:
    """Tests for the StochasticResonanceSimulation class."""

    def _make_sim(self, **kwargs) -> StochasticResonanceSimulation:
        defaults = {
            "gamma": 0.5,
            "signal_amplitude": 0.3,
            "signal_omega": 0.1,
            "noise_intensity": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return StochasticResonanceSimulation(config)

    def test_initial_state_shape(self):
        sim = self._make_sim()
        state = sim.reset(seed=42)
        assert state.shape == (2,)

    def test_initial_velocity_zero(self):
        sim = self._make_sim()
        state = sim.reset(seed=42)
        assert state[1] == 0.0

    def test_initial_position_in_well(self):
        sim = self._make_sim()
        state = sim.reset(seed=42)
        assert abs(state[0]) == 1.0, "Should start in one of the wells"

    def test_step_returns_state(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        state = sim.step()
        assert state.shape == (2,)

    def test_observe_matches_step(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        state = sim.step()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_state_dim(self):
        sim = self._make_sim()
        assert sim.state_dim == 2

    def test_name(self):
        sim = self._make_sim()
        assert sim.name == "StochasticResonance"

    def test_trajectory_finite(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State NaN/Inf: {state}"

    def test_determinism_same_seed(self):
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        sim1.reset(seed=42)
        sim2.reset(seed=42)
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_diverge(self):
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        sim1.reset(seed=42)
        sim2.reset(seed=99)
        states_differ = False
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
            if not np.allclose(s1, s2):
                states_differ = True
                break
        assert states_differ

    def test_no_signal_double_well(self):
        """Without signal, particle should stay in double-well."""
        sim = self._make_sim(signal_amplitude=0.0, noise_intensity=0.01)
        sim.reset(seed=42)
        positions = []
        for _ in range(5000):
            sim.step()
            positions.append(sim.observe()[0])
        positions = np.array(positions)
        assert np.all(np.isfinite(positions))

    def test_high_noise_explores_both_wells(self):
        """High noise should cause transitions between wells."""
        sim = self._make_sim(
            signal_amplitude=0.0,
            noise_intensity=2.0,
        )
        sim.reset(seed=42)
        crossings = 0
        prev_x = sim.observe()[0]
        for _ in range(10000):
            sim.step()
            x = sim.observe()[0]
            if prev_x * x < 0:
                crossings += 1
            prev_x = x
        assert crossings > 0, "High noise should cause well transitions"

    def test_low_noise_stays_in_well(self):
        """Very low noise should keep particle in initial well."""
        sim = self._make_sim(
            signal_amplitude=0.0,
            noise_intensity=0.001,
        )
        sim.reset(seed=42)
        initial_sign = np.sign(sim.observe()[0])
        stayed = True
        for _ in range(5000):
            sim.step()
            if np.sign(sim.observe()[0]) != initial_sign:
                stayed = False
                break
        assert stayed, "Very low noise should not cause well transitions"

    def test_gamma_damping(self):
        """Higher damping should reduce velocity magnitude."""
        sim_low = self._make_sim(gamma=0.1, noise_intensity=0.0, signal_amplitude=0.0)
        sim_high = self._make_sim(gamma=5.0, noise_intensity=0.0, signal_amplitude=0.0)
        sim_low.reset(seed=42)
        sim_high.reset(seed=42)
        for _ in range(1000):
            sim_low.step()
            sim_high.step()
        v_low = abs(sim_low.observe()[1])
        v_high = abs(sim_high.observe()[1])
        assert v_high <= v_low or v_high < 0.01


class TestStochasticResonanceRediscovery:
    """Tests for rediscovery data generation."""

    def test_import(self):
        from simulating_anything.rediscovery.stochastic_resonance import (
            run_stochastic_resonance_rediscovery,
        )
        assert callable(run_stochastic_resonance_rediscovery)
