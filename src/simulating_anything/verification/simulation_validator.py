"""Automated validation pipeline for generated simulations.

Runs a series of checks to verify that a dynamically generated simulation
class produces physically reasonable behavior.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.discovery import CheckResult, ValidationReport
from simulating_anything.types.simulation import SimulationConfig

logger = logging.getLogger(__name__)


class SimulationValidator:
    """Validates generated simulation classes.

    Runs a battery of checks (instantiation, NaN, boundedness, determinism,
    sensitivity) and returns a ValidationReport.
    """

    def __init__(
        self,
        n_test_steps: int = 100,
        bound_threshold: float = 1e10,
        sensitivity_eps: float = 1e-4,
        sensitivity_ratio_max: float = 1e6,
    ) -> None:
        self.n_test_steps = n_test_steps
        self.bound_threshold = bound_threshold
        self.sensitivity_eps = sensitivity_eps
        self.sensitivity_ratio_max = sensitivity_ratio_max

    def validate(
        self,
        sim_class: type,
        config: SimulationConfig | None = None,
    ) -> ValidationReport:
        """Run all validation checks on a simulation class.

        Args:
            sim_class: The simulation class to validate.
            config: Optional config; a default is used if None.

        Returns:
            ValidationReport with results of all checks.
        """
        if config is None:
            config = SimulationConfig(parameters={}, dt=0.01, n_steps=self.n_test_steps)

        checks: list[CheckResult] = []

        # 1. Instantiation check
        sim = self._check_instantiation(sim_class, config, checks)
        if sim is None:
            return self._build_report(checks)

        # 2. Reset check
        state = self._check_reset(sim, checks)
        if state is None:
            return self._build_report(checks)

        # 3. Step check (run n_test_steps)
        states = self._check_step(sim, checks)
        if states is None:
            return self._build_report(checks)

        # 4. NaN/Inf check
        self._check_nan_inf(states, checks)

        # 5. Boundedness check
        self._check_boundedness(states, checks)

        # 6. Determinism check
        self._check_determinism(sim_class, config, checks)

        # 7. Sensitivity check
        self._check_sensitivity(sim_class, config, checks)

        return self._build_report(checks)

    def get_fix_prompt(self, report: ValidationReport) -> str:
        """Generate a prompt describing validation failures for the LLM to fix.

        Args:
            report: The validation report with failures.

        Returns:
            A prompt string describing what went wrong.
        """
        failures = [c for c in report.checks if not c.passed]
        if not failures:
            return "All checks passed."

        lines = ["The simulation failed the following validation checks:\n"]
        for check in failures:
            lines.append(f"- {check.name}: {check.message}")
            if check.value != 0.0:
                lines.append(f"  Measured value: {check.value}, threshold: {check.threshold}")

        lines.append(
            "\nFix the simulation code to pass these checks. Common issues:"
        )
        lines.append("- NaN: timestep too large, division by zero, or sqrt of negative")
        lines.append("- Unbounded: missing clipping or exponential blowup")
        lines.append("- Non-deterministic: using random without seed")
        return "\n".join(lines)

    def _check_instantiation(
        self,
        sim_class: type,
        config: SimulationConfig,
        checks: list[CheckResult],
    ) -> SimulationEnvironment | None:
        """Check that the class can be instantiated."""
        try:
            sim = sim_class(config)
            checks.append(CheckResult(
                name="instantiation",
                passed=True,
                value=1.0,
                threshold=1.0,
                message="Class instantiated successfully",
            ))
            return sim
        except Exception as e:
            checks.append(CheckResult(
                name="instantiation",
                passed=False,
                value=0.0,
                threshold=1.0,
                message=f"Failed to instantiate: {e}",
            ))
            return None

    def _check_reset(
        self,
        sim: SimulationEnvironment,
        checks: list[CheckResult],
    ) -> np.ndarray | None:
        """Check that reset() returns a valid numpy array."""
        try:
            state = sim.reset(seed=42)
            if not isinstance(state, np.ndarray):
                checks.append(CheckResult(
                    name="reset",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    message=f"reset() returned {type(state)}, expected np.ndarray",
                ))
                return None
            if state.size == 0:
                checks.append(CheckResult(
                    name="reset",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    message="reset() returned empty array",
                ))
                return None
            checks.append(CheckResult(
                name="reset",
                passed=True,
                value=float(state.size),
                threshold=1.0,
                message=f"reset() returned array of shape {state.shape}",
            ))
            return state
        except Exception as e:
            checks.append(CheckResult(
                name="reset",
                passed=False,
                value=0.0,
                threshold=1.0,
                message=f"reset() raised: {e}",
            ))
            return None

    def _check_step(
        self,
        sim: SimulationEnvironment,
        checks: list[CheckResult],
    ) -> np.ndarray | None:
        """Check that step() runs for n_test_steps without crashing."""
        try:
            states = [sim.observe().copy()]
            for i in range(self.n_test_steps):
                state = sim.step()
                if not isinstance(state, np.ndarray):
                    checks.append(CheckResult(
                        name="step",
                        passed=False,
                        value=float(i),
                        threshold=float(self.n_test_steps),
                        message=f"step() returned {type(state)} at step {i}",
                    ))
                    return None
                states.append(state.copy())

            checks.append(CheckResult(
                name="step",
                passed=True,
                value=float(self.n_test_steps),
                threshold=float(self.n_test_steps),
                message=f"Completed {self.n_test_steps} steps without error",
            ))
            return np.array(states)
        except Exception as e:
            checks.append(CheckResult(
                name="step",
                passed=False,
                value=0.0,
                threshold=float(self.n_test_steps),
                message=f"step() raised at step {len(states)-1}: {e}",
            ))
            return None

    def _check_nan_inf(
        self, states: np.ndarray, checks: list[CheckResult]
    ) -> None:
        """Check for NaN or Inf values in the trajectory."""
        n_nan = int(np.sum(np.isnan(states)))
        n_inf = int(np.sum(np.isinf(states)))
        total_bad = n_nan + n_inf
        checks.append(CheckResult(
            name="nan_inf",
            passed=total_bad == 0,
            value=float(total_bad),
            threshold=0.0,
            message=(
                "No NaN or Inf values" if total_bad == 0
                else f"Found {n_nan} NaN and {n_inf} Inf values"
            ),
        ))

    def _check_boundedness(
        self, states: np.ndarray, checks: list[CheckResult]
    ) -> None:
        """Check that states remain bounded."""
        max_abs = float(np.max(np.abs(states[np.isfinite(states)]))) if np.any(np.isfinite(states)) else float("inf")
        checks.append(CheckResult(
            name="boundedness",
            passed=max_abs < self.bound_threshold,
            value=max_abs,
            threshold=self.bound_threshold,
            message=(
                f"States bounded: max |state| = {max_abs:.2e}"
                if max_abs < self.bound_threshold
                else f"States unbounded: max |state| = {max_abs:.2e} > {self.bound_threshold:.2e}"
            ),
        ))

    def _check_determinism(
        self,
        sim_class: type,
        config: SimulationConfig,
        checks: list[CheckResult],
    ) -> None:
        """Check that same seed produces same trajectory."""
        try:
            sim1 = sim_class(config)
            sim1.reset(seed=42)
            states1 = [sim1.step().copy() for _ in range(10)]

            sim2 = sim_class(config)
            sim2.reset(seed=42)
            states2 = [sim2.step().copy() for _ in range(10)]

            diff = max(float(np.max(np.abs(s1 - s2))) for s1, s2 in zip(states1, states2))
            checks.append(CheckResult(
                name="determinism",
                passed=diff < 1e-12,
                value=diff,
                threshold=1e-12,
                message=(
                    "Deterministic: same seed gives identical trajectories"
                    if diff < 1e-12
                    else f"Non-deterministic: max diff = {diff:.2e}"
                ),
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="determinism",
                passed=False,
                value=0.0,
                threshold=0.0,
                message=f"Determinism check failed: {e}",
            ))

    def _check_sensitivity(
        self,
        sim_class: type,
        config: SimulationConfig,
        checks: list[CheckResult],
    ) -> None:
        """Check that small parameter perturbation gives small output change."""
        if not config.parameters:
            checks.append(CheckResult(
                name="sensitivity",
                passed=True,
                value=0.0,
                threshold=0.0,
                message="No parameters to perturb (skipped)",
            ))
            return

        try:
            # Run baseline
            sim1 = sim_class(config)
            sim1.reset(seed=42)
            for _ in range(min(20, self.n_test_steps)):
                sim1.step()
            baseline = sim1.observe().copy()

            # Perturb first parameter
            perturbed_params = dict(config.parameters)
            first_key = next(iter(perturbed_params))
            original_val = perturbed_params[first_key]
            perturbed_params[first_key] = original_val * (1 + self.sensitivity_eps)

            perturbed_config = SimulationConfig(
                parameters=perturbed_params,
                dt=config.dt,
                n_steps=config.n_steps,
            )
            sim2 = sim_class(perturbed_config)
            sim2.reset(seed=42)
            for _ in range(min(20, self.n_test_steps)):
                sim2.step()
            perturbed = sim2.observe().copy()

            diff = float(np.max(np.abs(baseline - perturbed)))
            baseline_scale = float(np.max(np.abs(baseline))) + 1e-10
            ratio = diff / (abs(original_val * self.sensitivity_eps) + 1e-10)

            checks.append(CheckResult(
                name="sensitivity",
                passed=ratio < self.sensitivity_ratio_max,
                value=ratio,
                threshold=self.sensitivity_ratio_max,
                message=(
                    f"Sensitivity OK: perturbation ratio = {ratio:.2e}"
                    if ratio < self.sensitivity_ratio_max
                    else f"Oversensitive: ratio = {ratio:.2e} > {self.sensitivity_ratio_max:.2e}"
                ),
            ))
        except Exception as e:
            checks.append(CheckResult(
                name="sensitivity",
                passed=False,
                value=0.0,
                threshold=0.0,
                message=f"Sensitivity check failed: {e}",
            ))

    def _build_report(self, checks: list[CheckResult]) -> ValidationReport:
        """Build a ValidationReport from check results."""
        passed = all(c.passed for c in checks)
        warnings = [c.message for c in checks if not c.passed and c.name in ("sensitivity", "determinism")]
        critical_failures = [c.message for c in checks if not c.passed and c.name not in ("sensitivity", "determinism")]
        return ValidationReport(
            checks=checks,
            passed=passed,
            warnings=warnings,
            critical_failures=critical_failures,
        )
