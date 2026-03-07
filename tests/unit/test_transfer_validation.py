"""Tests for the sim-to-real transfer validation framework."""

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
    """Tests for the TransferMetrics dataclass."""

    def test_defaults(self):
        """All fields should have sensible defaults."""
        m = TransferMetrics()
        assert m.mse == 0.0
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.r_squared == 0.0
        assert m.max_error == 0.0
        assert m.mape == 0.0
        assert m.normalized_rmse == 0.0
        assert m.ks_statistic == 0.0
        assert m.ks_pvalue == 0.0
        assert m.correlation == 0.0
        assert m.spearman_rho == 0.0
        assert m.transfer_score == 0.0
        assert m.confidence_level == "unknown"
        assert m.failure_modes == []

    def test_custom_construction(self):
        """Explicit values should be stored correctly."""
        m = TransferMetrics(
            mse=0.01,
            r_squared=0.99,
            confidence_level="high",
            failure_modes=["test failure"],
        )
        assert m.mse == 0.01
        assert m.r_squared == 0.99
        assert m.confidence_level == "high"
        assert m.failure_modes == ["test failure"]

    def test_summary_format(self):
        """Summary should be a human-readable multi-line string."""
        m = TransferMetrics(
            r_squared=0.95,
            rmse=0.001,
            mae=0.0008,
            mape=1.5,
            normalized_rmse=0.02,
            correlation=0.98,
            ks_pvalue=0.45,
            transfer_score=0.92,
            confidence_level="high",
        )
        s = m.summary()
        assert "HIGH" in s
        assert "0.9500" in s
        assert "Transfer score" in s
        assert "0.9200" in s

    def test_summary_includes_failures(self):
        """Summary should list failure modes when present."""
        m = TransferMetrics(
            confidence_level="low",
            failure_modes=["Low R²", "High MAPE"],
        )
        s = m.summary()
        assert "Failures:" in s
        assert "High MAPE" in s

    def test_summary_no_failures(self):
        """Summary should omit the failures line when there are none."""
        m = TransferMetrics(confidence_level="high", failure_modes=[])
        s = m.summary()
        assert "Failures:" not in s

    def test_failure_modes_list_is_independent(self):
        """Each instance should have its own failure_modes list (no sharing)."""
        m1 = TransferMetrics()
        m2 = TransferMetrics()
        m1.failure_modes.append("oops")
        assert m2.failure_modes == []


class TestTransferReport:
    """Tests for TransferReport dataclass."""

    def test_default_construction(self):
        r = TransferReport()
        assert r.domain == ""
        assert r.equation == ""
        assert isinstance(r.metrics, TransferMetrics)
        assert r.sim_data_points == 0
        assert r.real_data_points == 0
        assert r.parameter_range == {}
        assert r.notes == []

    def test_notes_independent(self):
        """Each report instance should have its own notes list."""
        r1 = TransferReport()
        r2 = TransferReport()
        r1.notes.append("note1")
        assert r2.notes == []


class TestTransferValidatorInit:
    """Tests for TransferValidator instantiation and thresholds."""

    def test_default_thresholds(self):
        v = TransferValidator()
        assert v.r2_threshold == 0.90
        assert v.mape_threshold == 10.0
        assert v.nrmse_threshold == 0.15
        assert v.ks_alpha == 0.05

    def test_custom_thresholds(self):
        v = TransferValidator(
            r2_threshold=0.95,
            mape_threshold=5.0,
            nrmse_threshold=0.10,
            ks_alpha=0.01,
        )
        assert v.r2_threshold == 0.95
        assert v.mape_threshold == 5.0
        assert v.nrmse_threshold == 0.10
        assert v.ks_alpha == 0.01


class TestValidatePrediction:
    """Tests for validate_prediction with various data quality levels."""

    def test_perfect_match(self):
        """Identical sim and real data should yield high confidence."""
        v = TransferValidator()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        report = v.validate_prediction(data, data, domain="test", equation="y=x")

        m = report.metrics
        assert m.r_squared == pytest.approx(1.0)
        assert m.mse == pytest.approx(0.0, abs=1e-12)
        assert m.rmse == pytest.approx(0.0, abs=1e-12)
        assert m.mae == pytest.approx(0.0, abs=1e-12)
        assert m.max_error == pytest.approx(0.0, abs=1e-12)
        assert m.correlation == pytest.approx(1.0, abs=1e-6)
        assert m.confidence_level == "high"
        assert m.transfer_score >= 0.8
        assert m.failure_modes == []

    def test_perfect_sinusoidal(self):
        """Perfect sin match should give high confidence."""
        v = TransferValidator()
        real = np.sin(np.linspace(0, 2 * np.pi, 100))
        sim = real.copy()
        report = v.validate_prediction(sim, real, domain="test")
        assert report.metrics.r_squared == pytest.approx(1.0)
        assert report.metrics.rmse == pytest.approx(0.0, abs=1e-10)
        assert report.metrics.confidence_level == "high"
        assert report.metrics.transfer_score >= 0.9

    def test_small_noise_high_confidence(self):
        """Very small noise should still give high confidence."""
        v = TransferValidator()
        x = np.linspace(0, 10, 200)
        real = np.sin(x)
        sim = real + np.random.RandomState(42).normal(0, 0.01, len(real))
        report = v.validate_prediction(sim, real, domain="test")
        assert report.metrics.r_squared > 0.99
        assert report.metrics.confidence_level == "high"

    def test_moderate_noise_medium_confidence(self):
        """Moderate noise should produce medium or high confidence."""
        rng = np.random.default_rng(42)
        v = TransferValidator()
        x = np.linspace(1, 20, 100)
        sim = x
        real = x + rng.normal(0, 1.5, size=100)
        report = v.validate_prediction(sim, real)
        assert report.metrics.confidence_level in ("high", "medium")

    def test_large_noise_low_confidence(self):
        """Very large noise should get low or failed confidence."""
        v = TransferValidator()
        real = np.linspace(0, 10, 100)
        sim = np.random.RandomState(42).randn(100) * 10
        report = v.validate_prediction(sim, real, domain="test")
        assert report.metrics.r_squared < 0.5
        assert report.metrics.confidence_level in ("low", "failed")
        assert len(report.metrics.failure_modes) > 0

    def test_anticorrelated_data_fails(self):
        """Anticorrelated data should produce negative R² and 'failed'."""
        v = TransferValidator()
        x = np.linspace(1, 50, 100)
        sim = x
        real = -x
        report = v.validate_prediction(sim, real)
        assert report.metrics.r_squared < 0.0
        assert report.metrics.confidence_level == "failed"
        assert len(report.metrics.failure_modes) >= 2

    def test_mismatched_lengths_truncates(self):
        """When sim and real have different lengths, use the shorter."""
        v = TransferValidator()
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        real = np.array([1.0, 2.0, 3.0])
        report = v.validate_prediction(sim, real)
        assert report.sim_data_points == 3
        assert report.real_data_points == 3

    def test_systematic_bias_detected(self):
        """Constant offset: high correlation but MAPE/NRMSE problems."""
        v = TransferValidator()
        real = np.linspace(1, 10, 100)
        sim = real + 5.0
        report = v.validate_prediction(sim, real, domain="test")
        assert report.metrics.correlation == pytest.approx(1.0, abs=0.001)
        # MAPE should be nonzero because of the offset
        assert report.metrics.mape > 0.0

    def test_report_metadata(self):
        """Domain and equation should appear in the report."""
        v = TransferValidator()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        report = v.validate_prediction(
            data, data, domain="projectile", equation="v^2 sin(2t)/g"
        )
        assert report.domain == "projectile"
        assert report.equation == "v^2 sin(2t)/g"

    def test_large_dataset(self):
        """Should handle 10k points without issue."""
        v = TransferValidator()
        rng = np.random.default_rng(42)
        n = 10000
        x = np.linspace(0, 100, n)
        sim = np.sin(x)
        real = sim + rng.normal(0, 0.01, size=n)
        report = v.validate_prediction(sim, real)
        assert report.metrics.r_squared > 0.99
        assert report.sim_data_points == n


class TestValidateEquation:
    """Tests for validate_equation with callable equations."""

    def test_linear_equation_perfect(self):
        """A correct linear equation should get high confidence."""
        v = TransferValidator()
        inputs = np.linspace(0, 10, 50)
        real_outputs = 2.0 * inputs + 3.0

        report = v.validate_equation(
            lambda x: 2.0 * x + 3.0,
            inputs,
            real_outputs,
            domain="linear",
            equation="2x+3",
        )
        assert report.metrics.r_squared == pytest.approx(1.0)
        assert report.metrics.confidence_level == "high"

    def test_wrong_equation(self):
        """An incorrect equation should get low or failed confidence."""
        v = TransferValidator()
        inputs = np.linspace(1, 10, 50)
        real_outputs = inputs ** 2

        report = v.validate_equation(
            lambda x: x * 0.5,
            inputs,
            real_outputs,
        )
        assert report.metrics.r_squared < 0.5
        assert report.metrics.confidence_level in ("low", "failed")

    def test_equation_evaluation_error(self):
        """If the equation callable raises, report should show 'failed'."""
        v = TransferValidator()
        inputs = np.linspace(0, 10, 20)
        real_outputs = inputs * 2

        def broken_fn(x):
            raise ValueError("broken equation")

        report = v.validate_equation(broken_fn, inputs, real_outputs)
        assert report.metrics.confidence_level == "failed"
        assert len(report.metrics.failure_modes) == 1
        assert "Equation evaluation error" in report.metrics.failure_modes[0]
        assert "broken equation" in report.metrics.failure_modes[0]

    def test_quadratic_equation_perfect(self):
        """A correct quadratic equation should match perfectly."""
        v = TransferValidator()
        inputs = np.linspace(-5, 5, 100)
        real_outputs = inputs ** 2 - 3 * inputs + 2

        report = v.validate_equation(
            lambda x: x ** 2 - 3 * x + 2,
            inputs,
            real_outputs,
            equation="x^2-3x+2",
        )
        assert report.metrics.r_squared == pytest.approx(1.0)
        assert report.metrics.mse == pytest.approx(0.0, abs=1e-12)

    def test_newtons_second_law(self):
        """Validate F = m*a equation (physical law)."""
        v = TransferValidator()
        real_a = np.linspace(1, 10, 50)
        real_f = 5.0 * real_a  # m = 5
        equation_fn = lambda a: 5.0 * a

        report = v.validate_equation(
            equation_fn, real_a, real_f,
            domain="newton", equation="F = 5*a"
        )
        assert report.metrics.r_squared == pytest.approx(1.0, abs=1e-10)
        assert report.equation == "F = 5*a"


class TestValidateTrajectory:
    """Tests for validate_trajectory with time-series data."""

    def test_identical_1d_trajectory(self):
        """Identical 1D trajectories should give perfect scores."""
        v = TransferValidator()
        traj = np.sin(np.linspace(0, 2 * np.pi, 100))
        report = v.validate_trajectory(traj, traj, domain="oscillator")
        assert report.metrics.r_squared == pytest.approx(1.0)
        assert report.metrics.confidence_level == "high"
        assert "1 dims" in report.notes[0]

    def test_identical_2d_trajectory(self):
        """Identical 2D trajectories should work across dimensions."""
        v = TransferValidator()
        t = np.linspace(0, 4 * np.pi, 200)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        report = v.validate_trajectory(traj, traj)
        assert report.metrics.r_squared == pytest.approx(1.0)
        assert "2 dims" in report.notes[0]

    def test_slightly_noisy_1d(self):
        """Slightly noisy trajectory should still score well."""
        v = TransferValidator()
        t = np.linspace(0, 5, 200)
        real_traj = np.sin(t)
        sim_traj = np.sin(t) + 0.001 * np.random.RandomState(0).randn(200)

        report = v.validate_trajectory(sim_traj, real_traj, domain="oscillator")
        assert report.metrics.r_squared > 0.999
        assert report.metrics.confidence_level == "high"

    def test_noisy_2d_trajectory(self):
        """Noisy 2D trajectory should still produce valid metrics."""
        v = TransferValidator()
        t = np.linspace(0, 5, 100)
        real_traj = np.column_stack([np.sin(t), np.cos(t)])
        sim_traj = real_traj + 0.01 * np.random.RandomState(0).randn(100, 2)

        report = v.validate_trajectory(sim_traj, real_traj)
        assert report.metrics.r_squared > 0.99
        assert "2 dims" in report.notes[0]

    def test_trajectory_length_mismatch(self):
        """Mismatched lengths should be truncated to the shorter."""
        v = TransferValidator()
        sim = np.zeros((50, 2))
        real = np.zeros((30, 2))
        report = v.validate_trajectory(sim, real)
        assert report.sim_data_points == 30

    def test_trajectory_dim_mismatch(self):
        """Mismatched dimensions should be truncated to the fewer."""
        v = TransferValidator()
        sim = np.zeros((50, 3))
        real = np.zeros((50, 2))
        report = v.validate_trajectory(sim, real)
        assert "2 dims" in report.notes[0]

    def test_trajectory_max_error_across_dims(self):
        """max_error should be the max across all dimensions."""
        v = TransferValidator()
        # dim 0: small error, dim 1: large error
        t = np.linspace(0, 2 * np.pi, 100)
        real = np.column_stack([np.sin(t), np.cos(t)])
        sim = real.copy()
        sim[:, 1] += 2.0  # Large offset on dim 1 only
        report = v.validate_trajectory(sim, real)
        assert report.metrics.max_error == pytest.approx(2.0, abs=0.01)

    def test_trajectory_ks_pvalue_min_across_dims(self):
        """KS p-value should be the minimum across dimensions."""
        v = TransferValidator()
        t = np.linspace(0, 2 * np.pi, 200)
        # dim 0: identical, dim 1: shifted (different distribution)
        real = np.column_stack([np.sin(t), np.cos(t)])
        sim = real.copy()
        sim[:, 1] += 10.0  # shift distribution drastically
        report = v.validate_trajectory(sim, real)
        # The min p-value across dims should be low (from the shifted dim)
        assert report.metrics.ks_pvalue < 0.05


class TestValidateParameterSweep:
    """Tests for validate_parameter_sweep."""

    def test_parameter_sweep_records_range(self):
        """Parameter range should be stored in the report."""
        v = TransferValidator()
        params = np.linspace(1.0, 10.0, 50)
        outputs = params ** 2

        report = v.validate_parameter_sweep(
            lambda x: x ** 2,
            params,
            outputs,
            param_name="velocity",
            domain="projectile",
            equation="y=x^2",
        )
        assert "velocity" in report.parameter_range
        assert report.parameter_range["velocity"] == pytest.approx((1.0, 10.0))
        assert report.metrics.r_squared == pytest.approx(1.0)

    def test_parameter_sweep_with_equation_string(self):
        """Equation and domain should propagate through to the report."""
        v = TransferValidator()
        params = np.linspace(1, 10, 20)
        real_outputs = params ** 2
        report = v.validate_parameter_sweep(
            lambda x: x ** 2, params, real_outputs,
            param_name="x", domain="quadratic", equation="y=x^2"
        )
        assert report.metrics.r_squared == pytest.approx(1.0, abs=1e-10)
        assert "x" in report.parameter_range
        assert report.parameter_range["x"] == (1.0, 10.0)
        assert report.domain == "quadratic"
        assert report.equation == "y=x^2"


class TestConfidenceLevels:
    """Tests for the confidence level assignment logic in _assess_quality."""

    def test_high_confidence_all_pass(self):
        """All metrics passing should yield 'high' with score=1.0."""
        v = TransferValidator()
        m = TransferMetrics(
            r_squared=0.99,
            mape=1.0,
            normalized_rmse=0.01,
            ks_pvalue=0.5,
            correlation=0.99,
        )
        m = v._assess_quality(m)
        assert m.transfer_score == pytest.approx(1.0)
        assert m.confidence_level == "high"
        assert m.failure_modes == []

    def test_failed_confidence_all_fail(self):
        """Metrics that fail all checks should get 'failed'."""
        v = TransferValidator()
        m = TransferMetrics(
            r_squared=-0.5,
            mape=500.0,
            normalized_rmse=5.0,
            ks_pvalue=0.001,
            correlation=-0.5,
        )
        m = v._assess_quality(m)
        assert m.transfer_score < 0.3
        assert m.confidence_level == "failed"
        assert len(m.failure_modes) >= 4

    def test_borderline_r2_gives_medium_score(self):
        """R^2 between 0.8*threshold and threshold gives 0.6 sub-score."""
        v = TransferValidator(r2_threshold=0.90)
        # 0.72 = 0.8 * 0.9, so 0.75 is in the borderline region
        m = TransferMetrics(
            r_squared=0.75,
            mape=1.0,
            normalized_rmse=0.01,
            ks_pvalue=0.5,
            correlation=0.99,
        )
        m = v._assess_quality(m)
        # (0.6 + 1 + 1 + 1 + 1) / 5 = 0.92
        assert m.transfer_score == pytest.approx(0.92)
        # No R^2 failure appended in borderline region
        assert not any("R" in f and "Low" in f for f in m.failure_modes)

    def test_borderline_mape_gives_half_score(self):
        """MAPE between threshold and 2*threshold gives 0.5 sub-score."""
        v = TransferValidator(mape_threshold=10.0)
        m = TransferMetrics(
            r_squared=0.95,
            mape=15.0,  # Between 10 and 20
            normalized_rmse=0.01,
            ks_pvalue=0.5,
            correlation=0.99,
        )
        m = v._assess_quality(m)
        # No MAPE failure in borderline
        assert not any("MAPE" in f for f in m.failure_modes)

    def test_borderline_nrmse(self):
        """NRMSE between threshold and 2*threshold gives 0.5."""
        v = TransferValidator(nrmse_threshold=0.15)
        m = TransferMetrics(
            r_squared=0.95,
            mape=1.0,
            normalized_rmse=0.20,  # Between 0.15 and 0.30
            ks_pvalue=0.5,
            correlation=0.99,
        )
        m = v._assess_quality(m)
        assert not any("NRMSE" in f for f in m.failure_modes)

    def test_ks_rejection_adds_failure(self):
        """When KS p-value < alpha, a failure mode should be recorded."""
        v = TransferValidator(ks_alpha=0.05)
        m = TransferMetrics(
            r_squared=0.95,
            mape=1.0,
            normalized_rmse=0.01,
            ks_pvalue=0.01,  # < 0.05
            correlation=0.99,
        )
        m = v._assess_quality(m)
        assert any("KS test" in f for f in m.failure_modes)

    def test_low_correlation_adds_failure(self):
        """Correlation below 0.8 should add a failure mode."""
        v = TransferValidator()
        m = TransferMetrics(
            r_squared=0.95,
            mape=1.0,
            normalized_rmse=0.01,
            ks_pvalue=0.5,
            correlation=0.3,
        )
        m = v._assess_quality(m)
        assert any("correlation" in f.lower() for f in m.failure_modes)

    def test_medium_correlation_no_failure(self):
        """Correlation between 0.8 and 0.95 gives 0.6 sub-score, no failure."""
        v = TransferValidator()
        m = TransferMetrics(
            r_squared=0.95,
            mape=1.0,
            normalized_rmse=0.01,
            ks_pvalue=0.5,
            correlation=0.85,
        )
        m = v._assess_quality(m)
        assert not any("correlation" in f.lower() for f in m.failure_modes)

    def test_medium_confidence_score_range(self):
        """Transfer score 0.6-0.8 should give 'medium' (with failures present)."""
        v = TransferValidator()
        # Construct metrics where score falls in [0.6, 0.8)
        # R^2 borderline (0.6), MAPE pass (1.0), NRMSE pass (1.0),
        # KS fail (0.3), correlation borderline (0.6)
        # Mean = (0.6 + 1.0 + 1.0 + 0.3 + 0.6) / 5 = 0.7
        m = TransferMetrics(
            r_squared=0.75,  # borderline: 0.6
            mape=1.0,       # pass: 1.0
            normalized_rmse=0.01,  # pass: 1.0
            ks_pvalue=0.01,  # fail: 0.3
            correlation=0.85,  # borderline: 0.6
        )
        m = v._assess_quality(m)
        assert m.transfer_score == pytest.approx(0.7)
        # Has failures (KS rejected), so even though >= 0.6, confidence = medium
        assert m.confidence_level == "medium"

    def test_low_confidence_score_range(self):
        """Transfer score 0.3-0.6 should give 'low'."""
        v = TransferValidator()
        m = TransferMetrics(
            r_squared=0.5,     # fail: 0.2
            mape=15.0,         # borderline: 0.5
            normalized_rmse=0.20,  # borderline: 0.5
            ks_pvalue=0.01,    # fail: 0.3
            correlation=0.85,  # borderline: 0.6
        )
        m = v._assess_quality(m)
        # (0.2 + 0.5 + 0.5 + 0.3 + 0.6) / 5 = 0.42
        assert m.transfer_score == pytest.approx(0.42)
        assert m.confidence_level == "low"

    def test_custom_thresholds_affect_confidence(self):
        """Stricter thresholds should make the same data fail more."""
        strict = TransferValidator(r2_threshold=0.99, mape_threshold=1.0)
        lenient = TransferValidator(r2_threshold=0.5, mape_threshold=50.0)

        real = np.linspace(1, 10, 100)
        sim = real + np.random.RandomState(42).normal(0, 0.5, 100)

        strict_report = strict.validate_prediction(sim, real)
        lenient_report = lenient.validate_prediction(sim, real)
        assert lenient_report.metrics.transfer_score >= strict_report.metrics.transfer_score


class TestTransferScore:
    """Tests for transfer_score computation range and behavior."""

    def test_score_perfect(self):
        """Perfect data should give score of 1.0."""
        v = TransferValidator()
        data = np.linspace(1, 50, 100)
        report = v.validate_prediction(data, data)
        assert report.metrics.transfer_score == pytest.approx(1.0)

    def test_score_always_in_0_1_range(self):
        """Transfer score should always be between 0 and 1."""
        rng = np.random.default_rng(42)
        v = TransferValidator()
        for i in range(10):
            sim = rng.normal(0, 5, size=50)
            real = rng.normal(0, 5, size=50)
            report = v.validate_prediction(sim, real)
            assert 0.0 <= report.metrics.transfer_score <= 1.0


class TestSpearmanAndKS:
    """Tests for KS test, correlation, and Spearman rank correlation."""

    def test_spearman_perfect_positive(self):
        """Perfect monotonic relationship should give rho=1."""
        rho = TransferValidator._spearman(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        )
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_spearman_perfect_inverse(self):
        """Perfect inverse relationship should give rho=-1."""
        rho = TransferValidator._spearman(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([50.0, 40.0, 30.0, 20.0, 10.0]),
        )
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_spearman_too_few_points(self):
        """With fewer than 3 points, Spearman should return 0."""
        rho = TransferValidator._spearman(
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        )
        assert rho == 0.0

    def test_spearman_single_point(self):
        """Single point returns 0."""
        rho = TransferValidator._spearman(np.array([1.0]), np.array([2.0]))
        assert rho == 0.0

    def test_ks_identical_arrays(self):
        """KS statistic should be 0 and p-value ~1 for identical arrays."""
        data = np.linspace(0, 10, 100)
        stat, pval = TransferValidator._ks_test(data, data)
        assert stat == pytest.approx(0.0)
        assert pval == pytest.approx(1.0)

    def test_ks_same_distribution(self):
        """Samples from the same distribution should have high p-value."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        stat, pval = TransferValidator._ks_test(x, y)
        assert pval > 0.01

    def test_ks_different_distribution(self):
        """Very different distributions should have low p-value."""
        x = np.random.RandomState(42).normal(0, 1, 500)
        y = np.random.RandomState(42).normal(10, 1, 500)
        stat, pval = TransferValidator._ks_test(x, y)
        assert stat > 0.5
        assert pval < 0.01

    def test_ks_empty_first(self):
        stat, pval = TransferValidator._ks_test(np.array([]), np.array([1.0]))
        assert stat == 0.0
        assert pval == 1.0

    def test_ks_empty_second(self):
        stat, pval = TransferValidator._ks_test(np.array([1.0]), np.array([]))
        assert stat == 0.0
        assert pval == 1.0

    def test_ks_both_empty(self):
        stat, pval = TransferValidator._ks_test(np.array([]), np.array([]))
        assert stat == 0.0
        assert pval == 1.0

    def test_ks_pvalue_clipped_to_01(self):
        """KS p-value should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.normal(0, 1, size=50)
            y = rng.normal(rng.uniform(-5, 5), 1, size=50)
            stat, pval = TransferValidator._ks_test(x, y)
            assert 0.0 <= pval <= 1.0

    def test_correlation_perfect_positive(self):
        """Perfectly correlated data should yield correlation ~1."""
        v = TransferValidator()
        x = np.linspace(1, 100, 100)
        report = v.validate_prediction(x, x)
        assert report.metrics.correlation == pytest.approx(1.0, abs=1e-6)

    def test_correlation_scaled_linear(self):
        """Linear scaling preserves perfect correlation."""
        v = TransferValidator()
        x = np.linspace(1, 50, 100)
        report = v.validate_prediction(2 * x, x)
        assert report.metrics.correlation == pytest.approx(1.0, abs=1e-6)

    def test_correlation_zero_variance(self):
        """Constant arrays should yield correlation 0 (zero std)."""
        v = TransferValidator()
        sim = np.full(50, 3.0)
        real = np.full(50, 3.0)
        report = v.validate_prediction(sim, real)
        assert report.metrics.correlation == 0.0


class TestEdgeCases:
    """Edge cases: empty, single-point, constant data."""

    def test_single_point_data(self):
        """Single data point should not crash."""
        v = TransferValidator()
        sim = np.array([5.0])
        real = np.array([5.0])
        report = v.validate_prediction(sim, real)
        assert isinstance(report, TransferReport)
        assert report.sim_data_points == 1

    def test_two_point_data(self):
        """Two data points should produce valid metrics."""
        v = TransferValidator()
        sim = np.array([1.0, 2.0])
        real = np.array([1.0, 2.0])
        report = v.validate_prediction(sim, real)
        assert report.metrics.mse == pytest.approx(0.0, abs=1e-12)

    def test_constant_data_r2_zero(self):
        """Constant arrays have zero variance; R^2 should be 0 (ss_tot=0)."""
        v = TransferValidator()
        sim = np.full(50, 3.0)
        real = np.full(50, 3.0)
        report = v.validate_prediction(sim, real)
        assert report.metrics.r_squared == 0.0
        assert report.metrics.mse == pytest.approx(0.0, abs=1e-12)

    def test_constant_sim_varying_real(self):
        """Constant predictions against varying real: low R^2."""
        v = TransferValidator()
        sim = np.full(100, 5.0)
        real = np.linspace(0, 10, 100)
        report = v.validate_prediction(sim, real)
        assert report.metrics.r_squared < 0.5

    def test_all_zero_real_mape_safe(self):
        """All-zero real data: MAPE should be 0 (avoids division by zero)."""
        v = TransferValidator()
        sim = np.array([0.1, 0.2, 0.3])
        real = np.array([0.0, 0.0, 0.0])
        report = v.validate_prediction(sim, real)
        assert report.metrics.mape == 0.0

    def test_near_zero_real_mape(self):
        """Values near zero (< 1e-10) should be excluded from MAPE."""
        v = TransferValidator()
        sim = np.array([0.0, 1.0, 2.0])
        real = np.array([1e-15, 1.0, 2.0])
        report = v.validate_prediction(sim, real)
        # MAPE should be computed only on the non-near-zero entries
        assert np.isfinite(report.metrics.mape)

    def test_normalized_rmse_zero_range(self):
        """If data range is 0, normalized RMSE should be 0."""
        v = TransferValidator()
        sim = np.array([1.0, 1.1, 0.9])
        real = np.array([1.0, 1.0, 1.0])
        report = v.validate_prediction(sim, real)
        # Range of real is 0, so normalized_rmse = 0
        assert report.metrics.normalized_rmse == 0.0

    def test_lists_accepted_as_input(self):
        """Plain Python lists should be accepted (converted via np.asarray)."""
        v = TransferValidator()
        report = v.validate_prediction([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert report.metrics.r_squared == pytest.approx(1.0)


class TestValidateRediscovery:
    """Tests for the validate_rediscovery convenience function."""

    def test_projectile_range(self):
        """Validate projectile range equation: R = v^2 * sin(2*theta) / g."""
        g = 9.81
        discovered = lambda v: v ** 2 * np.sin(2 * np.pi / 4) / g
        analytical = lambda v: v ** 2 / g  # sin(pi/2) = 1

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
        discovered = lambda k: np.sqrt(k / 2.0)
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
        """Slightly wrong coefficient should produce R^2 < 1."""
        discovered = lambda k: np.sqrt(k / 2.1)
        analytical = lambda k: np.sqrt(k / 2.0)

        report = validate_rediscovery(
            discovered, analytical,
            param_range=(1.0, 50.0),
            n_points=100,
            domain="oscillator",
        )
        assert report.metrics.r_squared > 0.9
        assert report.metrics.r_squared < 1.0

    def test_wrong_equation_fails(self):
        """A completely wrong equation should fail validation."""
        report = validate_rediscovery(
            equation_fn=lambda x: np.log(x + 1),
            analytical_fn=lambda x: x ** 3,
            param_range=(1.0, 50.0),
            n_points=100,
        )
        assert report.metrics.r_squared < 0.5
        assert report.metrics.confidence_level in ("low", "failed")

    def test_n_points_respected(self):
        """The number of evaluation points should match n_points."""
        report = validate_rediscovery(
            equation_fn=lambda x: x,
            analytical_fn=lambda x: x,
            param_range=(0.0, 1.0),
            n_points=37,
        )
        assert report.sim_data_points == 37
        assert report.real_data_points == 37

    def test_default_validator_used(self):
        """Convenience function should use default TransferValidator thresholds."""
        report = validate_rediscovery(
            equation_fn=lambda x: x ** 2,
            analytical_fn=lambda x: x ** 2,
            param_range=(1.0, 10.0),
        )
        # Default n_points is 100
        assert report.sim_data_points == 100
        assert report.metrics.confidence_level == "high"


class TestMetricsComputation:
    """Detailed tests for specific metric values."""

    def test_mse_value(self):
        """MSE should be the mean of squared residuals."""
        v = TransferValidator()
        sim = np.array([1.0, 2.0, 3.0])
        real = np.array([1.5, 2.5, 3.5])
        report = v.validate_prediction(sim, real)
        # Residuals: [-0.5, -0.5, -0.5], MSE = mean([0.25, 0.25, 0.25]) = 0.25
        assert report.metrics.mse == pytest.approx(0.25)
        assert report.metrics.rmse == pytest.approx(0.5)
        assert report.metrics.mae == pytest.approx(0.5)
        assert report.metrics.max_error == pytest.approx(0.5)

    def test_r_squared_known_value(self):
        """R^2 for a known case."""
        v = TransferValidator()
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        report = v.validate_prediction(sim, real)
        # Manual: ss_res = sum of (0.1^2 + 0.1^2 + 0.1^2 + 0.1^2 + 0.1^2) = 0.05
        # mean(real) = 3.0, ss_tot = 1+1+0+1+4 = 10
        # R^2 = 1 - 0.05/10 = 0.995
        assert report.metrics.r_squared == pytest.approx(0.995, abs=1e-6)

    def test_mape_known_value(self):
        """MAPE for known data (no zeros)."""
        v = TransferValidator()
        real = np.array([10.0, 20.0, 30.0])
        sim = np.array([11.0, 22.0, 27.0])
        report = v.validate_prediction(sim, real)
        # |1/10|=0.1, |2/20|=0.1, |3/30|=0.1 => mean = 0.1 => 10%
        assert report.metrics.mape == pytest.approx(10.0)

    def test_normalized_rmse_known_value(self):
        """Normalized RMSE = RMSE / data_range."""
        v = TransferValidator()
        real = np.array([0.0, 10.0])
        sim = np.array([1.0, 9.0])
        report = v.validate_prediction(sim, real)
        # RMSE = sqrt(mean([1, 1])) = 1.0; range = 10; nrmse = 0.1
        assert report.metrics.normalized_rmse == pytest.approx(0.1)
