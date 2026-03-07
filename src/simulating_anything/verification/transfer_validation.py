"""Sim-to-real transfer validation framework.

Validates that discoveries from simulated data generalize to real-world
observations. Compares simulation predictions against experimental data
using multiple metrics and statistical tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransferMetrics:
    """Metrics for sim-to-real transfer validation."""

    # Core accuracy metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    max_error: float = 0.0

    # Relative metrics
    mape: float = 0.0  # Mean absolute percentage error
    normalized_rmse: float = 0.0  # RMSE / data range

    # Statistical tests
    ks_statistic: float = 0.0  # Kolmogorov-Smirnov
    ks_pvalue: float = 0.0
    correlation: float = 0.0
    spearman_rho: float = 0.0

    # Transfer quality assessment
    transfer_score: float = 0.0  # 0-1 composite
    confidence_level: str = "unknown"  # high/medium/low/failed
    failure_modes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Transfer Validation: {self.confidence_level.upper()}",
            f"  R²: {self.r_squared:.4f}",
            f"  RMSE: {self.rmse:.6f}",
            f"  MAE: {self.mae:.6f}",
            f"  MAPE: {self.mape:.2f}%",
            f"  Normalized RMSE: {self.normalized_rmse:.4f}",
            f"  Correlation: {self.correlation:.4f}",
            f"  KS p-value: {self.ks_pvalue:.4f}",
            f"  Transfer score: {self.transfer_score:.4f}",
        ]
        if self.failure_modes:
            lines.append(f"  Failures: {', '.join(self.failure_modes)}")
        return "\n".join(lines)


@dataclass
class TransferReport:
    """Complete transfer validation report."""

    domain: str = ""
    equation: str = ""
    metrics: TransferMetrics = field(default_factory=TransferMetrics)
    sim_data_points: int = 0
    real_data_points: int = 0
    parameter_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class TransferValidator:
    """Validates sim-to-real transfer of discovered equations.

    Workflow:
    1. Provide discovered equation/model from simulation
    2. Provide real-world experimental data
    3. Evaluate prediction accuracy across metrics
    4. Assess transfer quality with confidence level
    """

    def __init__(
        self,
        r2_threshold: float = 0.90,
        mape_threshold: float = 10.0,
        nrmse_threshold: float = 0.15,
        ks_alpha: float = 0.05,
    ) -> None:
        self.r2_threshold = r2_threshold
        self.mape_threshold = mape_threshold
        self.nrmse_threshold = nrmse_threshold
        self.ks_alpha = ks_alpha

    def validate_prediction(
        self,
        sim_predictions: np.ndarray,
        real_observations: np.ndarray,
        domain: str = "",
        equation: str = "",
    ) -> TransferReport:
        """Compare simulation predictions against real observations.

        Args:
            sim_predictions: Model predictions (from simulation/equation).
            real_observations: Real-world measurements.
            domain: Domain name for the report.
            equation: Discovered equation string.

        Returns:
            TransferReport with full metrics.
        """
        sim = np.asarray(sim_predictions, dtype=np.float64).ravel()
        real = np.asarray(real_observations, dtype=np.float64).ravel()

        n = min(len(sim), len(real))
        sim = sim[:n]
        real = real[:n]

        metrics = self._compute_metrics(sim, real)
        metrics = self._assess_quality(metrics)

        return TransferReport(
            domain=domain,
            equation=equation,
            metrics=metrics,
            sim_data_points=len(sim),
            real_data_points=len(real),
        )

    def validate_equation(
        self,
        equation_fn: Callable,
        real_inputs: np.ndarray,
        real_outputs: np.ndarray,
        domain: str = "",
        equation: str = "",
    ) -> TransferReport:
        """Validate a discovered equation against real input-output pairs.

        Args:
            equation_fn: Callable mapping inputs to predicted outputs.
            real_inputs: Real-world input values.
            real_outputs: Real-world output measurements.
            domain: Domain name.
            equation: Equation string.

        Returns:
            TransferReport.
        """
        try:
            sim_predictions = np.array([equation_fn(x) for x in real_inputs])
        except Exception as e:
            logger.error(f"Equation evaluation failed: {e}")
            metrics = TransferMetrics(
                confidence_level="failed",
                failure_modes=[f"Equation evaluation error: {e}"],
            )
            return TransferReport(domain=domain, equation=equation, metrics=metrics)

        return self.validate_prediction(
            sim_predictions, real_outputs, domain=domain, equation=equation
        )

    def validate_trajectory(
        self,
        sim_trajectory: np.ndarray,
        real_trajectory: np.ndarray,
        dt: float = 1.0,
        domain: str = "",
    ) -> TransferReport:
        """Validate full trajectory match (time series).

        Compares trajectories point-by-point and also evaluates
        statistical distribution match.

        Args:
            sim_trajectory: (T_sim, D) simulated trajectory.
            real_trajectory: (T_real, D) real trajectory.
            dt: Time step for reporting.
            domain: Domain name.

        Returns:
            TransferReport with trajectory-specific metrics.
        """
        sim = np.asarray(sim_trajectory, dtype=np.float64)
        real = np.asarray(real_trajectory, dtype=np.float64)

        if sim.ndim == 1:
            sim = sim.reshape(-1, 1)
        if real.ndim == 1:
            real = real.reshape(-1, 1)

        n_steps = min(sim.shape[0], real.shape[0])
        n_dims = min(sim.shape[1], real.shape[1])
        sim = sim[:n_steps, :n_dims]
        real = real[:n_steps, :n_dims]

        # Per-dimension metrics, then average
        all_metrics = []
        for d in range(n_dims):
            m = self._compute_metrics(sim[:, d], real[:, d])
            all_metrics.append(m)

        # Average across dimensions
        avg_metrics = TransferMetrics(
            mse=np.mean([m.mse for m in all_metrics]),
            rmse=np.mean([m.rmse for m in all_metrics]),
            mae=np.mean([m.mae for m in all_metrics]),
            r_squared=np.mean([m.r_squared for m in all_metrics]),
            max_error=np.max([m.max_error for m in all_metrics]),
            mape=np.mean([m.mape for m in all_metrics]),
            normalized_rmse=np.mean([m.normalized_rmse for m in all_metrics]),
            ks_statistic=np.mean([m.ks_statistic for m in all_metrics]),
            ks_pvalue=np.min([m.ks_pvalue for m in all_metrics]),
            correlation=np.mean([m.correlation for m in all_metrics]),
            spearman_rho=np.mean([m.spearman_rho for m in all_metrics]),
        )
        avg_metrics = self._assess_quality(avg_metrics)

        report = TransferReport(
            domain=domain,
            metrics=avg_metrics,
            sim_data_points=n_steps,
            real_data_points=n_steps,
        )
        report.notes.append(f"Trajectory comparison: {n_steps} steps x {n_dims} dims")
        return report

    def validate_parameter_sweep(
        self,
        equation_fn: Callable,
        param_values: np.ndarray,
        real_outputs: np.ndarray,
        param_name: str = "param",
        domain: str = "",
        equation: str = "",
    ) -> TransferReport:
        """Validate equation across a parameter sweep.

        Tests that the equation generalizes across different parameter values,
        not just at the training point.

        Args:
            equation_fn: f(param_value) -> prediction.
            param_values: Parameter values used in experiments.
            real_outputs: Real measured outputs at each parameter value.
            param_name: Name of the swept parameter.
            domain: Domain name.
            equation: Equation string.

        Returns:
            TransferReport with parameter range info.
        """
        report = self.validate_equation(
            equation_fn, param_values, real_outputs,
            domain=domain, equation=equation,
        )
        report.parameter_range[param_name] = (
            float(np.min(param_values)),
            float(np.max(param_values)),
        )
        return report

    def _compute_metrics(
        self, sim: np.ndarray, real: np.ndarray
    ) -> TransferMetrics:
        """Compute all transfer metrics."""
        residuals = sim - real
        n = len(real)

        # Basic error metrics
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))
        max_error = float(np.max(np.abs(residuals)))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((real - np.mean(real)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # MAPE (avoid division by zero)
        nonzero = np.abs(real) > 1e-10
        if np.any(nonzero):
            mape = float(np.mean(np.abs(residuals[nonzero] / real[nonzero])) * 100)
        else:
            mape = 0.0

        # Normalized RMSE
        data_range = float(np.max(real) - np.min(real))
        normalized_rmse = rmse / data_range if data_range > 1e-10 else 0.0

        # Correlation
        if np.std(sim) > 1e-10 and np.std(real) > 1e-10:
            correlation = float(np.corrcoef(sim, real)[0, 1])
        else:
            correlation = 0.0

        # Spearman rank correlation
        spearman_rho = self._spearman(sim, real)

        # Kolmogorov-Smirnov test (distribution comparison)
        ks_stat, ks_p = self._ks_test(sim, real)

        return TransferMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            max_error=max_error,
            mape=mape,
            normalized_rmse=normalized_rmse,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            correlation=correlation,
            spearman_rho=spearman_rho,
        )

    def _assess_quality(self, metrics: TransferMetrics) -> TransferMetrics:
        """Assign confidence level and identify failure modes."""
        failures = []
        scores = []

        # R² check
        if metrics.r_squared >= self.r2_threshold:
            scores.append(1.0)
        elif metrics.r_squared >= self.r2_threshold * 0.8:
            scores.append(0.6)
        else:
            scores.append(0.2)
            failures.append(f"Low R² ({metrics.r_squared:.3f} < {self.r2_threshold})")

        # MAPE check
        if metrics.mape <= self.mape_threshold:
            scores.append(1.0)
        elif metrics.mape <= self.mape_threshold * 2:
            scores.append(0.5)
        else:
            scores.append(0.1)
            failures.append(f"High MAPE ({metrics.mape:.1f}% > {self.mape_threshold}%)")

        # NRMSE check
        if metrics.normalized_rmse <= self.nrmse_threshold:
            scores.append(1.0)
        elif metrics.normalized_rmse <= self.nrmse_threshold * 2:
            scores.append(0.5)
        else:
            scores.append(0.1)
            failures.append(f"High NRMSE ({metrics.normalized_rmse:.3f})")

        # KS test check
        if metrics.ks_pvalue >= self.ks_alpha:
            scores.append(1.0)
        else:
            scores.append(0.3)
            failures.append(f"KS test rejected (p={metrics.ks_pvalue:.4f})")

        # Correlation check
        if metrics.correlation >= 0.95:
            scores.append(1.0)
        elif metrics.correlation >= 0.8:
            scores.append(0.6)
        else:
            scores.append(0.2)
            failures.append(f"Low correlation ({metrics.correlation:.3f})")

        transfer_score = float(np.mean(scores))
        if transfer_score >= 0.8 and not failures:
            confidence = "high"
        elif transfer_score >= 0.6:
            confidence = "medium"
        elif transfer_score >= 0.3:
            confidence = "low"
        else:
            confidence = "failed"

        metrics.transfer_score = transfer_score
        metrics.confidence_level = confidence
        metrics.failure_modes = failures
        return metrics

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman rank correlation without scipy."""
        n = len(x)
        if n < 3:
            return 0.0
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        d = rx - ry
        rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
        return float(rho)

    @staticmethod
    def _ks_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Two-sample KS test without scipy."""
        n1 = len(x)
        n2 = len(y)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        all_vals = np.sort(np.concatenate([x, y]))
        cdf1 = np.searchsorted(np.sort(x), all_vals, side="right") / n1
        cdf2 = np.searchsorted(np.sort(y), all_vals, side="right") / n2
        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Approximate p-value (asymptotic formula)
        n_eff = np.sqrt(n1 * n2 / (n1 + n2))
        lam = (n_eff + 0.12 + 0.11 / n_eff) * ks_stat
        # Kolmogorov distribution approximation
        if lam < 0.01:
            p_value = 1.0
        else:
            p_value = 2 * np.exp(-2 * lam ** 2)
            p_value = float(min(max(p_value, 0.0), 1.0))
        return ks_stat, p_value


def validate_rediscovery(
    equation_fn: Callable,
    analytical_fn: Callable,
    param_range: tuple[float, float],
    n_points: int = 100,
    domain: str = "",
    equation: str = "",
) -> TransferReport:
    """Convenience function: validate a rediscovered equation against analytics.

    Useful for validating that PySR/SINDy rediscoveries match known formulas
    across a parameter range.

    Args:
        equation_fn: Discovered equation f(param) -> value.
        analytical_fn: Known analytical solution f(param) -> value.
        param_range: (min, max) parameter range.
        n_points: Number of test points.
        domain: Domain name.
        equation: Equation string.

    Returns:
        TransferReport.
    """
    params = np.linspace(param_range[0], param_range[1], n_points)
    real_values = np.array([analytical_fn(p) for p in params])

    validator = TransferValidator()
    return validator.validate_equation(
        equation_fn, params, real_values, domain=domain, equation=equation
    )
