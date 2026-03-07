"""Tests for the hypothesis tester."""

from __future__ import annotations

import numpy as np

from simulating_anything.analysis.hypothesis_tester import HypothesisTester
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.campaign import ConfidenceLevel
from simulating_anything.types.discovery import Discovery
from simulating_anything.types.simulation import Domain, SimulationConfig


# Simple linear simulation for testing
class LinearSimulation(SimulationEnvironment):
    """State tracks k*t linearly."""

    def __init__(self, config):
        super().__init__(config)
        self.k = config.parameters.get("k", 1.0)

    def reset(self, seed=None):
        self._state = np.array([0.0])
        self._step_count = 0
        return self._state

    def step(self):
        self._step_count += 1
        self._state = np.array([self.k * self._step_count * self.config.dt])
        return self._state

    def observe(self):
        return self._state


# Oscillator for testing
class OscillatorSim(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.k = config.parameters.get("k", 1.0)

    def reset(self, seed=None):
        self._state = np.array([1.0, 0.0])
        self._step_count = 0
        return self._state

    def step(self):
        dt = self.config.dt
        x, v = self._state
        a = -self.k * x
        self._state = np.array([x + v * dt, v + a * dt])
        self._step_count += 1
        return self._state

    def observe(self):
        return self._state


class TestHypothesisTester:
    """Tests for HypothesisTester."""

    def setup_method(self):
        self.tester = HypothesisTester(
            n_sweep_points=10,
            n_sim_steps=50,
        )

    def test_init_defaults(self):
        t = HypothesisTester()
        assert t.extrapolation_factor == 2.0
        assert t.holdout_fraction == 0.2
        assert t.r2_threshold == 0.9

    def test_init_custom(self):
        t = HypothesisTester(r2_threshold=0.95, n_sweep_points=30)
        assert t.r2_threshold == 0.95
        assert t.n_sweep_points == 30

    def test_test_empty_expression(self):
        discovery = Discovery(id="d1", expression="")
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        result = self.tester.test(discovery, LinearSimulation, config)
        assert result.confidence_level == ConfidenceLevel.UNTESTED
        assert "No expression" in result.details

    def test_test_with_discovery(self):
        discovery = Discovery(id="d1", expression="x * 2.0")
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        result = self.tester.test(discovery, LinearSimulation, config)

        assert result.discovery_id == "d1"
        assert isinstance(result.interpolation_r2, float)
        assert isinstance(result.extrapolation_r2, float)
        assert result.confidence_level in list(ConfidenceLevel)

    def test_r_squared_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert HypothesisTester._r_squared(y_true, y_pred) == 1.0

    def test_r_squared_zero(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # Mean prediction
        r2 = HypothesisTester._r_squared(y_true, y_pred)
        assert abs(r2) < 1e-10

    def test_r_squared_negative(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        r2 = HypothesisTester._r_squared(y_true, y_pred)
        assert r2 < 0

    def test_r_squared_constant_true(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert HypothesisTester._r_squared(y_true, y_pred) == 1.0

    def test_evaluate_expression_simple(self):
        result = self.tester._evaluate_expression("x * 2", np.array([1.0, 2.0, 3.0]))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])

    def test_evaluate_expression_numpy(self):
        result = self.tester._evaluate_expression("np.sqrt(x)", np.array([1.0, 4.0, 9.0]))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_evaluate_expression_invalid(self):
        result = self.tester._evaluate_expression("undefined_var * 2", np.array([1.0]))
        # Should return None for invalid expressions
        assert result is None

    def test_evaluate_expression_constant(self):
        result = self.tester._evaluate_expression("42.0", np.array([1.0, 2.0]))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [42.0, 42.0])

    def test_test_extreme_values_safe(self):
        discovery = Discovery(expression="x * 2")
        assert self.tester._test_extreme_values(discovery) is True

    def test_test_extreme_values_empty(self):
        discovery = Discovery(expression="")
        assert self.tester._test_extreme_values(discovery) is True

    def test_test_dimensional_consistency_good(self):
        discovery = Discovery(expression="(a + b) * c")
        assert self.tester._test_dimensional_consistency(discovery) is True

    def test_test_dimensional_consistency_unbalanced(self):
        discovery = Discovery(expression="(a + b * c")
        assert self.tester._test_dimensional_consistency(discovery) is False

    def test_generate_sweep_data_with_params(self):
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        params, obs = self.tester._generate_sweep_data(
            LinearSimulation, config, "k", (0.5, 2.0), 5
        )
        assert len(params) == 5
        assert len(obs) == 5

    def test_generate_sweep_data_no_params(self):
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={}, dt=0.01, n_steps=50)
        params, obs = self.tester._generate_sweep_data(
            LinearSimulation, config, n_points=5
        )
        assert len(params) == 5

    def test_interpolation_with_linear_sim(self):
        discovery = Discovery(id="d1", expression="x * 2")
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        r2 = self.tester._test_interpolation(discovery, LinearSimulation, config)
        assert isinstance(r2, float)
        assert 0.0 <= r2 <= 1.0 or r2 < 0  # R^2 can be negative

    def test_extrapolation_no_params(self):
        discovery = Discovery(id="d1", expression="x")
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={}, dt=0.01, n_steps=50)
        r2 = self.tester._test_extrapolation(discovery, LinearSimulation, config)
        assert r2 == 0.0  # No params to sweep

    def test_confidence_assignment(self):
        # Test HIGH confidence path
        discovery = Discovery(id="d1", expression="x * 2")
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        result = self.tester.test(discovery, LinearSimulation, config)

        assert result.confidence_level in list(ConfidenceLevel)
        if result.passed:
            assert result.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)

    def test_failure_modes_recorded(self):
        discovery = Discovery(id="d1", expression="x * 2")
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        result = self.tester.test(discovery, LinearSimulation, config)

        if not result.passed:
            assert len(result.failure_modes) > 0
