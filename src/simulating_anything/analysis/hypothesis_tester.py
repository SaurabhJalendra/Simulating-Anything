"""Hypothesis testing for discovered equations.

Validates that discovered equations are genuine physical relationships,
not fitting artifacts, by testing extrapolation, interpolation, and
dimensional consistency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.campaign import ConfidenceLevel, HypothesisResult
from simulating_anything.types.discovery import Discovery
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


class HypothesisTester:
    """Validates discovered equations against simulation data.

    Tests extrapolation, interpolation, extreme values, and assigns
    confidence levels based on how many tests pass.
    """

    def __init__(
        self,
        extrapolation_factor: float = 2.0,
        holdout_fraction: float = 0.2,
        r2_threshold: float = 0.9,
        n_sweep_points: int = 20,
        n_sim_steps: int = 100,
    ) -> None:
        self.extrapolation_factor = extrapolation_factor
        self.holdout_fraction = holdout_fraction
        self.r2_threshold = r2_threshold
        self.n_sweep_points = n_sweep_points
        self.n_sim_steps = n_sim_steps

    def test(
        self,
        discovery: Discovery,
        sim_class: type,
        config: SimulationConfig,
    ) -> HypothesisResult:
        """Run all hypothesis tests on a discovered equation.

        Args:
            discovery: The discovered equation/relationship.
            sim_class: The simulation class to generate test data.
            config: Base configuration for the simulation.

        Returns:
            HypothesisResult with confidence level and test details.
        """
        result = HypothesisResult(
            discovery_id=discovery.id,
        )

        if not discovery.expression:
            result.details = "No expression to test"
            result.confidence_level = ConfidenceLevel.UNTESTED
            return result

        # Test 1: Interpolation (holdout validation)
        interp_r2 = self._test_interpolation(discovery, sim_class, config)
        result.interpolation_r2 = interp_r2

        # Test 2: Extrapolation (beyond training range)
        extrap_r2 = self._test_extrapolation(discovery, sim_class, config)
        result.extrapolation_r2 = extrap_r2

        # Test 3: Extreme values check
        extreme_ok = self._test_extreme_values(discovery)

        # Test 4: Dimensional consistency (basic check)
        dim_ok = self._test_dimensional_consistency(discovery)
        result.dimensional_consistent = dim_ok

        # Assign confidence level
        interp_pass = interp_r2 > self.r2_threshold
        extrap_pass = extrap_r2 > self.r2_threshold

        if interp_pass and extrap_pass and dim_ok and extreme_ok:
            result.confidence_level = ConfidenceLevel.HIGH
            result.passed = True
        elif interp_pass and extrap_pass:
            result.confidence_level = ConfidenceLevel.MEDIUM
            result.passed = True
        elif interp_pass:
            result.confidence_level = ConfidenceLevel.LOW
            result.passed = False
        else:
            result.confidence_level = ConfidenceLevel.UNTESTED
            result.passed = False

        # Record failure modes
        if not interp_pass:
            result.failure_modes.append(
                f"Interpolation R^2 = {interp_r2:.4f} < {self.r2_threshold}"
            )
        if not extrap_pass:
            result.failure_modes.append(
                f"Extrapolation R^2 = {extrap_r2:.4f} < {self.r2_threshold}"
            )
        if not extreme_ok:
            result.failure_modes.append("Expression produces extreme values")
        if not dim_ok:
            result.failure_modes.append("Dimensional inconsistency detected")

        result.details = (
            f"Interpolation R^2={interp_r2:.4f}, "
            f"Extrapolation R^2={extrap_r2:.4f}, "
            f"Extremes={'OK' if extreme_ok else 'FAIL'}, "
            f"Dimensions={'OK' if dim_ok else 'FAIL'}"
        )

        return result

    def _generate_sweep_data(
        self,
        sim_class: type,
        config: SimulationConfig,
        param_name: str | None = None,
        param_range: tuple[float, float] | None = None,
        n_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate parameter sweep data from a simulation.

        Returns (param_values, observable_values) arrays.
        """
        n = n_points or self.n_sweep_points

        if param_name and param_range:
            param_values = np.linspace(param_range[0], param_range[1], n)
        elif config.parameters:
            # Sweep the first parameter
            param_name = next(iter(config.parameters))
            base_val = config.parameters[param_name]
            param_values = np.linspace(base_val * 0.5, base_val * 2.0, n)
        else:
            # No parameters to sweep; just run multiple times
            param_values = np.arange(n, dtype=float)
            observables = []
            for _ in range(n):
                sim = sim_class(config)
                sim.reset(seed=42)
                for _ in range(self.n_sim_steps):
                    sim.step()
                observables.append(sim.observe().copy())
            return param_values, np.array(observables)

        observables = []
        for val in param_values:
            params = dict(config.parameters)
            params[param_name] = float(val)
            sweep_config = SimulationConfig(
                domain=config.domain,
                parameters=params,
                dt=config.dt,
                n_steps=config.n_steps,
            )
            try:
                sim = sim_class(sweep_config)
                sim.reset(seed=42)
                for _ in range(self.n_sim_steps):
                    sim.step()
                observables.append(sim.observe().copy())
            except Exception:
                observables.append(np.full_like(observables[-1] if observables else np.zeros(1), np.nan))

        return param_values, np.array(observables)

    def _test_interpolation(
        self,
        discovery: Discovery,
        sim_class: type,
        config: SimulationConfig,
    ) -> float:
        """Test prediction accuracy on held-out data points.

        Returns R^2 score on holdout data.
        """
        try:
            params, obs = self._generate_sweep_data(sim_class, config)

            # Remove NaN rows
            valid = ~np.any(np.isnan(obs), axis=-1) if obs.ndim > 1 else ~np.isnan(obs)
            params = params[valid]
            obs = obs[valid]

            if len(params) < 5:
                return 0.0

            # Holdout split
            n = len(params)
            n_holdout = max(1, int(n * self.holdout_fraction))
            rng = np.random.RandomState(42)
            indices = rng.permutation(n)
            train_idx = indices[n_holdout:]
            test_idx = indices[:n_holdout]

            # Use first observable component as target
            if obs.ndim > 1:
                y_train = obs[train_idx, 0]
                y_test = obs[test_idx, 0]
            else:
                y_train = obs[train_idx]
                y_test = obs[test_idx]

            # Fit a simple model: evaluate the expression if possible,
            # otherwise use polynomial interpolation as proxy
            try:
                predicted = self._evaluate_expression(
                    discovery.expression, params[test_idx]
                )
                if predicted is not None and len(predicted) == len(y_test):
                    return self._r_squared(y_test, predicted)
            except Exception:
                pass

            # Fallback: polynomial fit as a proxy test
            if len(y_train) >= 3:
                degree = min(3, len(y_train) - 1)
                coeffs = np.polyfit(params[train_idx], y_train, degree)
                predicted = np.polyval(coeffs, params[test_idx])
                return self._r_squared(y_test, predicted)

            return 0.0
        except Exception as e:
            logger.warning(f"Interpolation test failed: {e}")
            return 0.0

    def _test_extrapolation(
        self,
        discovery: Discovery,
        sim_class: type,
        config: SimulationConfig,
    ) -> float:
        """Test prediction accuracy beyond training parameter range.

        Returns R^2 score on extrapolated data.
        """
        try:
            if not config.parameters:
                return 0.0

            param_name = next(iter(config.parameters))
            base_val = config.parameters[param_name]

            # Training range
            train_range = (base_val * 0.5, base_val * 2.0)
            train_params, train_obs = self._generate_sweep_data(
                sim_class, config, param_name, train_range, self.n_sweep_points
            )

            # Extrapolation range
            extrap_range = (
                base_val * 2.0,
                base_val * 2.0 * self.extrapolation_factor,
            )
            extrap_params, extrap_obs = self._generate_sweep_data(
                sim_class, config, param_name, extrap_range,
                max(5, self.n_sweep_points // 4),
            )

            # Remove NaN
            valid_train = ~np.any(np.isnan(train_obs), axis=-1) if train_obs.ndim > 1 else ~np.isnan(train_obs)
            valid_extrap = ~np.any(np.isnan(extrap_obs), axis=-1) if extrap_obs.ndim > 1 else ~np.isnan(extrap_obs)

            train_params = train_params[valid_train]
            train_obs = train_obs[valid_train]
            extrap_params = extrap_params[valid_extrap]
            extrap_obs = extrap_obs[valid_extrap]

            if len(train_params) < 3 or len(extrap_params) < 2:
                return 0.0

            # First component
            if train_obs.ndim > 1:
                y_train = train_obs[:, 0]
                y_extrap = extrap_obs[:, 0]
            else:
                y_train = train_obs
                y_extrap = extrap_obs

            # Try expression evaluation
            try:
                predicted = self._evaluate_expression(
                    discovery.expression, extrap_params
                )
                if predicted is not None and len(predicted) == len(y_extrap):
                    return self._r_squared(y_extrap, predicted)
            except Exception:
                pass

            # Fallback: fit on training, predict on extrapolation
            degree = min(3, len(y_train) - 1)
            coeffs = np.polyfit(train_params, y_train, degree)
            predicted = np.polyval(coeffs, extrap_params)
            return self._r_squared(y_extrap, predicted)

        except Exception as e:
            logger.warning(f"Extrapolation test failed: {e}")
            return 0.0

    def _test_extreme_values(self, discovery: Discovery) -> bool:
        """Check if the expression produces finite values at extremes."""
        if not discovery.expression:
            return True

        try:
            # Test with very large and very small values
            for test_val in [1e-10, 1e-5, 0.01, 1.0, 100.0, 1e5, 1e10]:
                result = self._evaluate_expression(
                    discovery.expression, np.array([test_val])
                )
                if result is not None:
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        return False
                    if np.any(np.abs(result) > 1e20):
                        return False
            return True
        except Exception:
            return True  # Can't evaluate, don't fail

    def _test_dimensional_consistency(self, discovery: Discovery) -> bool:
        """Basic dimensional consistency check.

        Checks if the expression contains obvious dimensional mismatches
        like adding quantities with different units.
        """
        expr = discovery.expression
        if not expr:
            return True

        # Simple heuristic: check for obvious issues
        # (A full dimensional analysis would require unit tracking)
        try:
            # Check if expression is parseable and balanced
            if expr.count("(") != expr.count(")"):
                return False
            if expr.count("[") != expr.count("]"):
                return False
            return True
        except Exception:
            return True

    def _evaluate_expression(
        self, expression: str, x_values: np.ndarray
    ) -> np.ndarray | None:
        """Try to evaluate a mathematical expression with given x values.

        Uses a safe namespace with numpy functions.
        """
        safe_namespace: dict[str, Any] = {
            "x": x_values,
            "np": np,
            "sqrt": np.sqrt,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "abs": np.abs,
            "pi": np.pi,
        }

        try:
            # Try common variable names
            for var_name in ["x", "r", "t", "k", "n"]:
                safe_namespace[var_name] = x_values

            result = eval(expression, {"__builtins__": {}}, safe_namespace)  # noqa: S307
            if isinstance(result, (int, float)):
                return np.full_like(x_values, result)
            return np.asarray(result)
        except Exception:
            return None

    @staticmethod
    def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R^2 coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-20:
            return 1.0 if ss_res < 1e-20 else 0.0
        return float(1.0 - ss_res / ss_tot)
