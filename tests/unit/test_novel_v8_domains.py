"""Tests for V8 novel coupled domains: PredatorPreyClimate, EpidemicEconomy, NeuralEcosystem."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.types.simulation import Domain, SimulationConfig


# ============================================================================
# PredatorPreyClimate tests
# ============================================================================

class TestPredatorPreyClimate:
    def _make_sim(self, **kwargs):
        from simulating_anything.simulation.predator_prey_climate import (
            PredatorPreyClimateSimulation,
        )
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=100,
            parameters=kwargs,
        )
        return PredatorPreyClimateSimulation(config)

    def test_init_default(self):
        sim = self._make_sim()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_shape(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        s = sim.step()
        assert s.shape == (4,)

    def test_state_changes(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_no_nan(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        for _ in range(1000):
            s = sim.step()
        assert not np.any(np.isnan(s))
        assert not np.any(np.isinf(s))

    def test_populations_positive(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(500):
            s = sim.step()
        assert s[0] >= 0, "Prey should be non-negative"
        assert s[1] >= 0, "Predator should be non-negative"

    def test_reset_deterministic(self):
        sim = self._make_sim()
        s1 = sim.reset(seed=42)
        s2 = sim.reset(seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_coupling_effect(self):
        sim0 = self._make_sim(coupling_TK=0.0)
        sim0.reset(seed=0)
        for _ in range(500):
            sim0.step()

        sim1 = self._make_sim(coupling_TK=0.5)
        sim1.reset(seed=0)
        for _ in range(500):
            sim1.step()

        # Different coupling should produce different dynamics
        assert not np.allclose(sim0.observe(), sim1.observe(), atol=0.01)

    def test_run_returns_correct_length(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(100):
            states.append(sim.step().copy())
        assert len(states) == 101


# ============================================================================
# EpidemicEconomy tests
# ============================================================================

class TestEpidemicEconomy:
    def _make_sim(self, **kwargs):
        from simulating_anything.simulation.epidemic_economy import (
            EpidemicEconomySimulation,
        )
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=100,
            parameters=kwargs,
        )
        return EpidemicEconomySimulation(config)

    def test_init_default(self):
        sim = self._make_sim()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_shape(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        s = sim.step()
        assert s.shape == (4,)

    def test_sir_bounds(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(500):
            s = sim.step()
        assert 0 <= s[0] <= 1, "S should be in [0,1]"
        assert 0 <= s[1] <= 1, "I should be in [0,1]"

    def test_economic_bounds(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(500):
            s = sim.step()
        assert s[2] > 0, "Wage share should be positive"
        assert s[3] > 0, "Employment should be positive"

    def test_no_nan(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        for _ in range(1000):
            s = sim.step()
        assert not np.any(np.isnan(s))

    def test_coupling_effect(self):
        sim0 = self._make_sim(coupling_uS=0.0)
        sim0.reset(seed=0)
        for _ in range(300):
            sim0.step()

        sim1 = self._make_sim(coupling_uS=0.8)
        sim1.reset(seed=0)
        for _ in range(300):
            sim1.step()

        assert not np.allclose(sim0.observe(), sim1.observe(), atol=0.01)

    def test_reset_deterministic(self):
        sim = self._make_sim()
        s1 = sim.reset(seed=42)
        s2 = sim.reset(seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_epidemic_peaks(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        max_I = 0
        for _ in range(1000):
            s = sim.step()
            max_I = max(max_I, s[1])
        assert max_I > sim.I_0, "Epidemic should grow initially"


# ============================================================================
# NeuralEcosystem tests
# ============================================================================

class TestNeuralEcosystem:
    def _make_sim(self, **kwargs):
        from simulating_anything.simulation.neural_ecosystem import (
            NeuralEcosystemSimulation,
        )
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=100,
            parameters=kwargs,
        )
        return NeuralEcosystemSimulation(config)

    def test_init_default(self):
        sim = self._make_sim()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_shape(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        s = sim.step()
        assert s.shape == (4,)

    def test_neural_bounds(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(500):
            s = sim.step()
        assert 0 <= s[0] <= 1, "E should be in [0,1]"
        assert 0 <= s[1] <= 1, "I_n should be in [0,1]"

    def test_populations_positive(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(500):
            s = sim.step()
        assert s[2] >= 0, "Prey should be non-negative"
        assert s[3] >= 0, "Predator should be non-negative"

    def test_no_nan(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        for _ in range(1000):
            s = sim.step()
        assert not np.any(np.isnan(s))

    def test_coupling_effect(self):
        sim0 = self._make_sim(coupling_EN=0.0, coupling_NE=0.0)
        sim0.reset(seed=0)
        for _ in range(500):
            sim0.step()

        sim1 = self._make_sim(coupling_EN=0.5, coupling_NE=0.3)
        sim1.reset(seed=0)
        for _ in range(500):
            sim1.step()

        assert not np.allclose(sim0.observe(), sim1.observe(), atol=0.01)

    def test_reset_deterministic(self):
        sim = self._make_sim()
        s1 = sim.reset(seed=42)
        s2 = sim.reset(seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_sigmoid(self):
        from simulating_anything.simulation.neural_ecosystem import (
            NeuralEcosystemSimulation,
        )
        assert abs(NeuralEcosystemSimulation._sigmoid(0) - 0.5) < 1e-10
        assert NeuralEcosystemSimulation._sigmoid(10) > 0.99
        assert NeuralEcosystemSimulation._sigmoid(-10) < 0.01
