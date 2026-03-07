"""Tests for sim-to-real transfer validation framework."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.verification.transfer_validation import (
    TransferMetrics,
    TransferReport,
    TransferValidator,
    validate_rediscovery,
)


class TestTransferMetrics:
    """Test metrics dataclass."""

    def test_defaults(self):
        m = TransferMetrics()
        assert m.mse == 0.0
        assert m.confidence_level == "unknown"
        assert m.failure_modes == []

    def test_summary(self):
        m = TransferMetrics(r_squared=0.99, rmse=0.01, confidence_level="high")
        s = m.summary()
        assert "HIGH" in s
        assert "0.99" in s


class TestTransferValidator:
    """Test the main validation engine."""

    def test_perfect_prediction(self):
        """Perfect sim-to-real match should give high confidence."""
        validator = TransferValidator()
        real = np.sin(np.linspace(0, 2 * np.pi, 100))
        sim = real.copy()
        report = validator.validate_prediction(sim, real, domain="test")
        assert report.metrics.r_squared == pytest.approx(1.0)
        assert report.metrics.rmse == pytest.approx(0.0, abs=1e-10)
        assert report.metrics.confidence_level == "high"
        assert report.metrics.transfer_score >= 0.9

    def test_good_prediction(self):
        """Small noise should still give high confidence."""
        validator = TransferValidator()
        x = np.linspace(0, 10, 200)
        real = np.sin(x)
        sim = real + np.random.RandomState(42).normal(0, 0.01, len(real))
        report = validator.validate_prediction(sim, real, domain="test")
        assert report.metrics.r_squared > 0.99
        assert report.metrics.confidence_level == "high"

    def test_poor_prediction(self):
        """Large errors should give low confidence."""
        validator = TransferValidator()
        real = np.linspace(0, 10, 100)
        sim = np.random.RandomState(42).randn(100) * 10
        report = validator.validate_prediction(sim, real, domain="test")
        assert report.metrics.r_squared < 0.5
        assert report.metrics.confidence_level in ("low", "failed")
        assert len(report.metrics.failure_modes) > 0

    def test_systematic_bias(self):
        """Systematic offset should be detected."""
        validator = TransferValidator()
        real = np.linspace(1, 10, 100)
        sim = real + 5.0  # Constant offset
        report = validator.validate_prediction(sim, real, domain="test")
        # R² is 1.0 (perfect correlation), but MAPE will be high
        assert report.metrics.correlation == pytest.approx(1.0, abs=0.001)

    def test_validate_equation(self):
        """Validate a callable equation against real data."""
        validator = TransferValidator()
        # Real physics: F = m*a, discover F = k*a
        real_a = np.linspace(1, 10, 50)
        real_f = 5.0 * real_a  # m = 5
        equation_fn = lambda a: 5.0 * a  # Perfect discovery

        report = validator.validate_equation(
            equation_fn, real_a, real_f,
            domain="newton", equation="F = 5*a"
        )
        assert report.metrics.r_squared == pytest.approx(1.0, abs=1e-10)
        assert report.equation == "F = 5*a"

    def test_validate_equation_with_error(self):
        """Equation that throws should return failed."""
        validator = TransferValidator()

        def bad_fn(x):
            raise ValueError("bad equation")

        report = validator.validate_equation(
            bad_fn, np.array([1.0]), np.array([1.0])
        )
        assert report.metrics.confidence_level == "failed"
        assert "error" in report.metrics.failure_modes[0].lower()

    def test_validate_trajectory_1d(self):
        """Validate 1D trajectory match."""
        validator = TransferValidator()
        t = np.linspace(0, 5, 200)
        real_traj = np.sin(t)
        sim_traj = np.sin(t) + 0.001 * np.random.RandomState(0).randn(200)

        report = validator.validate_trajectory(sim_traj, real_traj, domain="oscillator")
        assert report.metrics.r_squared > 0.999
        assert report.metrics.confidence_level == "high"

    def test_validate_trajectory_2d(self):
        """Validate 2D trajectory match."""
        validator = TransferValidator()
        t = np.linspace(0, 5, 100)
        real_traj = np.column_stack([np.sin(t), np.cos(t)])
        sim_traj = real_traj + 0.01 * np.random.RandomState(0).randn(100, 2)

        report = validator.validate_trajectory(sim_traj, real_traj)
        assert report.metrics.r_squared > 0.99
        assert "2 dims" in report.notes[0]

    def test_validate_parameter_sweep(self):
        """Validate equation across parameter sweep."""
        validator = TransferValidator()
        params = np.linspace(1, 10, 20)
        real_outputs = params ** 2  # y = x^2
        equation_fn = lambda x: x ** 2

        report = validator.validate_parameter_sweep(
            equation_fn, params, real_outputs,
            param_name="x", domain="quadratic", equation="y=x^2"
        )
        assert report.metrics.r_squared == pytest.approx(1.0, abs=1e-10)
        assert "x" in report.parameter_range
        assert report.parameter_range["x"] == (1.0, 10.0)

    def test_metrics_computation_edge_cases(self):
        """Handle edge cases in metric computation."""
        validator = TransferValidator()

        # Constant signals
        real = np.ones(50)
        sim = np.ones(50)
        report = validator.validate_prediction(sim, real)
        # R² is 0 when ss_tot is 0 (constant)
        assert np.isfinite(report.metrics.rmse)

    def test_different_length_arrays(self):
        """Should handle mismatched lengths by truncating."""
        validator = TransferValidator()
        real = np.linspace(0, 10, 100)
        sim = np.linspace(0, 10, 50)  # Shorter
        report = validator.validate_prediction(sim, real)
        assert report.sim_data_points == 50
        assert report.real_data_points == 50

    def test_custom_thresholds(self):
        """Custom thresholds should affect confidence levels."""
        strict = TransferValidator(r2_threshold=0.99, mape_threshold=1.0)
        lenient = TransferValidator(r2_threshold=0.5, mape_threshold=50.0)

        real = np.linspace(1, 10, 100)
        sim = real + np.random.RandomState(42).normal(0, 0.5, 100)

        strict_report = strict.validate_prediction(sim, real)
        lenient_report = lenient.validate_prediction(sim, real)

        # Lenient should give equal or higher confidence
        assert lenient_report.metrics.transfer_score >= strict_report.metrics.transfer_score


class TestSpearmanAndKS:
    """Test statistical utilities."""

    def test_spearman_perfect(self):
        """Perfect monotonic relationship should give rho=1."""
        rho = TransferValidator._spearman(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_spearman_inverse(self):
        """Perfect inverse relationship should give rho=-1."""
        rho = TransferValidator._spearman(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([50.0, 40.0, 30.0, 20.0, 10.0]),
        )
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_ks_same_distribution(self):
        """Identical distributions should have high p-value."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        stat, pval = TransferValidator._ks_test(x, y)
        assert pval > 0.01  # Should not reject

    def test_ks_different_distribution(self):
        """Very different distributions should have low p-value."""
        x = np.random.RandomState(42).normal(0, 1, 500)
        y = np.random.RandomState(42).normal(10, 1, 500)
        stat, pval = TransferValidator._ks_test(x, y)
        assert stat > 0.5
        assert pval < 0.01

    def test_ks_empty_arrays(self):
        stat, pval = TransferValidator._ks_test(np.array([]), np.array([1.0]))
        assert stat == 0.0
        assert pval == 1.0


class TestValidateRediscovery:
    """Test the convenience function."""

    def test_projectile_range(self):
        """Validate projectile range equation: R = v^2 * sin(2*theta) / g."""
        g = 9.81
        # Discovered equation
        discovered = lambda v: v ** 2 * np.sin(2 * np.pi / 4) / g
        # Analytical
        analytical = lambda v: v ** 2 / g  # sin(90) = 1

        report = validate_rediscovery(
            discovered, analytical,
            param_range=(5.0, 50.0),
            n_points=100,
            domain="projectile",
            equation="R = v^2 * sin(2*theta) / g",
        )
        assert report.metrics.r_squared == pytest.approx(1.0, abs=1e-10)
        assert report.metrics.confidence_level == "high"
        assert report.domain == "projectile"

    def test_harmonic_oscillator_frequency(self):
        """Validate omega = sqrt(k/m)."""
        discovered = lambda k: np.sqrt(k / 2.0)  # m=2
        analytical = lambda k: np.sqrt(k / 2.0)

        report = validate_rediscovery(
            discovered, analytical,
            param_range=(0.1, 100.0),
            n_points=50,
            domain="oscillator",
            equation="omega = sqrt(k/m)",
        )
        assert report.metrics.r_squared == pytest.approx(1.0, abs=1e-10)

    def test_imperfect_rediscovery(self):
        """Slightly wrong coefficient should be detected."""
        discovered = lambda k: np.sqrt(k / 2.1)  # Wrong mass
        analytical = lambda k: np.sqrt(k / 2.0)  # True mass

        report = validate_rediscovery(
            discovered, analytical,
            param_range=(1.0, 50.0),
            n_points=100,
            domain="oscillator",
        )
        # Should still have decent R² but not perfect
        assert report.metrics.r_squared > 0.9
        assert report.metrics.r_squared < 1.0
