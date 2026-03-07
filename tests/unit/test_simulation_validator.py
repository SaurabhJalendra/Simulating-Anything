"""Tests for the simulation validator."""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import Domain, SimulationConfig
from simulating_anything.verification.simulation_validator import SimulationValidator


# Good simulation for testing
class GoodSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)
        p = config.parameters
        self.k = p.get("k", 1.0)

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


# NaN-producing simulation
class NaNSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None):
        self._state = np.array([1.0])
        self._step_count = 0
        return self._state

    def step(self):
        self._state = np.array([float('nan')])
        self._step_count += 1
        return self._state

    def observe(self):
        return self._state


# Unbounded simulation
class UnboundedSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None):
        self._state = np.array([1.0])
        self._step_count = 0
        return self._state

    def step(self):
        self._state = self._state * 2.0  # Exponential growth
        self._step_count += 1
        return self._state

    def observe(self):
        return self._state


# Crash simulation
class CrashSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None):
        self._state = np.array([1.0])
        self._step_count = 0
        return self._state

    def step(self):
        raise RuntimeError("Simulation crashed!")

    def observe(self):
        return self._state


# Non-deterministic simulation
class NonDeterministicSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None):
        self._state = np.array([np.random.random()])
        self._step_count = 0
        return self._state

    def step(self):
        self._state = np.array([np.random.random()])
        self._step_count += 1
        return self._state

    def observe(self):
        return self._state


# Bad reset (returns list, not ndarray)
class BadResetSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None):
        self._state = np.array([1.0])
        self._step_count = 0
        return [1.0]  # Wrong type!

    def step(self):
        return self._state

    def observe(self):
        return self._state


class TestSimulationValidator:
    """Tests for SimulationValidator."""

    def setup_method(self):
        self.validator = SimulationValidator(n_test_steps=50, bound_threshold=1e10)

    def test_init_defaults(self):
        v = SimulationValidator()
        assert v.n_test_steps == 100
        assert v.bound_threshold == 1e10
        assert v.sensitivity_eps == 1e-4

    def test_init_custom(self):
        v = SimulationValidator(n_test_steps=50, bound_threshold=1e5)
        assert v.n_test_steps == 50
        assert v.bound_threshold == 1e5

    def test_validate_good_simulation(self):
        report = self.validator.validate(GoodSimulation)
        assert report.passed is True
        assert report.all_critical_passed is True
        assert len(report.checks) == 7  # All 7 checks run
        assert all(c.passed for c in report.checks)

    def test_validate_nan_simulation(self):
        report = self.validator.validate(NaNSimulation)
        assert report.passed is False
        nan_check = next(c for c in report.checks if c.name == "nan_inf")
        assert nan_check.passed is False

    def test_validate_unbounded_simulation(self):
        report = self.validator.validate(UnboundedSimulation)
        assert report.passed is False
        bound_check = next(c for c in report.checks if c.name == "boundedness")
        assert bound_check.passed is False

    def test_validate_crash_simulation(self):
        report = self.validator.validate(CrashSimulation)
        assert report.passed is False
        step_check = next(c for c in report.checks if c.name == "step")
        assert step_check.passed is False

    def test_validate_nondeterministic_simulation(self):
        report = self.validator.validate(NonDeterministicSimulation)
        det_check = next(c for c in report.checks if c.name == "determinism")
        assert det_check.passed is False

    def test_validate_bad_reset(self):
        report = self.validator.validate(BadResetSimulation)
        reset_check = next(c for c in report.checks if c.name == "reset")
        assert reset_check.passed is False
        assert "expected np.ndarray" in reset_check.message

    def test_validate_with_custom_config(self):
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 2.0}, dt=0.005, n_steps=50)
        report = self.validator.validate(GoodSimulation, config)
        assert report.passed is True

    def test_validate_instantiation_failure(self):
        class BadInit:
            def __init__(self, config):
                raise TypeError("Bad init")

        report = self.validator.validate(BadInit)
        assert report.passed is False
        inst_check = next(c for c in report.checks if c.name == "instantiation")
        assert inst_check.passed is False
        assert "Bad init" in inst_check.message

    def test_critical_vs_warnings(self):
        report = self.validator.validate(NonDeterministicSimulation)
        # Determinism is a warning, not critical
        warn_names = ("sensitivity", "determinism")
        critical = [c.name for c in report.checks if not c.passed and c.name not in warn_names]
        assert "determinism" not in critical

    def test_get_fix_prompt_all_passed(self):
        report = self.validator.validate(GoodSimulation)
        prompt = self.validator.get_fix_prompt(report)
        assert "All checks passed" in prompt

    def test_get_fix_prompt_with_failures(self):
        report = self.validator.validate(NaNSimulation)
        prompt = self.validator.get_fix_prompt(report)
        assert "nan_inf" in prompt
        assert "NaN" in prompt

    def test_sensitivity_check_with_params(self):
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)
        report = self.validator.validate(GoodSimulation, config)
        sens_check = next(c for c in report.checks if c.name == "sensitivity")
        assert sens_check.passed is True

    def test_sensitivity_check_no_params(self):
        config = SimulationConfig(domain=Domain.CUSTOM, parameters={}, dt=0.01, n_steps=50)
        report = self.validator.validate(GoodSimulation, config)
        sens_check = next(c for c in report.checks if c.name == "sensitivity")
        assert sens_check.passed is True  # Skipped = passed
        assert "skipped" in sens_check.message.lower()

    def test_determinism_check_good(self):
        report = self.validator.validate(GoodSimulation)
        det_check = next(c for c in report.checks if c.name == "determinism")
        assert det_check.passed is True
        assert det_check.value < 1e-12

    def test_check_results_have_names(self):
        report = self.validator.validate(GoodSimulation)
        expected_names = {
            "instantiation", "reset", "step", "nan_inf", "boundedness", "determinism", "sensitivity",
        }
        actual_names = {c.name for c in report.checks}
        assert expected_names == actual_names

    def test_build_report_structure(self):
        report = self.validator.validate(GoodSimulation)
        assert hasattr(report, "checks")
        assert hasattr(report, "passed")
        assert hasattr(report, "warnings")
        assert hasattr(report, "critical_failures")
