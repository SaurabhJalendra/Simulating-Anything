"""Tests for baseline comparison module."""
from __future__ import annotations

import numpy as np

from simulating_anything.analysis.baselines import (
    BaselineComparison,
    BaselineResult,
    _fit_multivariate_poly,
    _fit_polynomial,
    _fit_power_law,
    _fit_ratio,
    _fit_sqrt_ratio,
    _generate_decay_data,
    _generate_oscillator_data,
    _generate_pendulum_data,
    _generate_projectile_data,
    _generate_sir_data,
    generate_latex_table,
    run_all_baselines,
    run_decay_baselines,
    run_oscillator_baselines,
    run_pendulum_baselines,
    run_projectile_baselines,
    run_sir_baselines,
    save_results,
)


class TestDataGeneration:
    """Test data generation functions."""

    def test_projectile_data_shapes(self):
        v0, theta, R = _generate_projectile_data(100)
        assert v0.shape == theta.shape == R.shape
        assert len(v0) >= 100

    def test_projectile_data_physics(self):
        v0, theta, R = _generate_projectile_data(225)
        # Range should be positive
        assert np.all(R >= 0)
        # Range should scale with v0^2
        assert np.corrcoef(v0**2, R)[0, 1] > 0.5

    def test_projectile_noise(self):
        _, _, R_clean = _generate_projectile_data(100, noise=0.0)
        _, _, R_noisy = _generate_projectile_data(100, noise=0.1)
        assert not np.allclose(R_clean, R_noisy)

    def test_oscillator_data_shapes(self):
        k, m, omega = _generate_oscillator_data(100)
        assert k.shape == m.shape == omega.shape
        assert len(k) >= 100

    def test_oscillator_data_physics(self):
        k, m, omega = _generate_oscillator_data(100)
        expected = np.sqrt(k / m)
        np.testing.assert_allclose(omega, expected, rtol=1e-10)

    def test_sir_data_shapes(self):
        beta, gamma, R0 = _generate_sir_data(100)
        assert beta.shape == gamma.shape == R0.shape

    def test_sir_data_physics(self):
        beta, gamma, R0 = _generate_sir_data(100)
        np.testing.assert_allclose(R0, beta / gamma, rtol=1e-10)

    def test_pendulum_data_shapes(self):
        L, T = _generate_pendulum_data(50)
        assert len(L) == len(T) == 50

    def test_pendulum_data_physics(self):
        L, T = _generate_pendulum_data(50)
        expected = 2 * np.pi * np.sqrt(L / 9.81)
        np.testing.assert_allclose(T, expected, rtol=1e-10)

    def test_decay_data_shapes(self):
        D, rate = _generate_decay_data(30)
        assert len(D) == len(rate) == 30

    def test_decay_data_physics(self):
        D, rate = _generate_decay_data(30)
        np.testing.assert_allclose(rate, D, rtol=1e-10)


class TestFittingFunctions:
    """Test individual fitting functions."""

    def test_polynomial_linear(self):
        x = np.linspace(1, 10, 50)
        y = 3.0 * x + 1.0
        r2, eq = _fit_polynomial(x, y, degree=1)
        assert r2 > 0.999

    def test_polynomial_quadratic(self):
        x = np.linspace(1, 10, 50)
        y = 2.0 * x**2 - x + 0.5
        r2, eq = _fit_polynomial(x, y, degree=2)
        assert r2 > 0.999

    def test_multivariate_poly(self):
        rng = np.random.default_rng(42)
        x0 = np.linspace(1, 10, 50)
        x1 = np.linspace(2, 8, 50)
        X = np.column_stack([x0, x1])
        y = 2 * x0 + 3 * x1
        r2, eq = _fit_multivariate_poly(X, y, degree=1)
        assert r2 > 0.999

    def test_power_law(self):
        x = np.linspace(1, 10, 50)
        y = 2.5 * x**1.5
        r2, eq = _fit_power_law(x, y)
        assert r2 > 0.99
        assert "1.5" in eq

    def test_ratio_fit(self):
        x1 = np.linspace(1, 10, 50)
        x2 = np.linspace(0.5, 5, 50)
        y = x1 / x2
        r2, eq = _fit_ratio(x1, x2, y)
        assert r2 > 0.999

    def test_sqrt_ratio_fit(self):
        k = np.linspace(1, 20, 50)
        m = np.linspace(0.5, 10, 50)
        omega = np.sqrt(k / m)
        r2, eq = _fit_sqrt_ratio(k, m, omega)
        assert r2 > 0.999


class TestBaselineResult:
    """Test BaselineResult dataclass."""

    def test_creation(self):
        r = BaselineResult(
            domain="test", method="poly", r_squared=0.95,
            correct_form=True, n_samples=100,
        )
        assert r.domain == "test"
        assert r.r_squared == 0.95

    def test_defaults(self):
        r = BaselineResult(
            domain="test", method="poly", r_squared=0.95,
            correct_form=True, n_samples=100,
        )
        assert r.wall_time_s == 0.0
        assert r.equation == ""


class TestBaselineComparison:
    """Test BaselineComparison."""

    def test_best_r_squared(self):
        comp = BaselineComparison(domain="test", target_equation="y=x")
        comp.results = [
            BaselineResult("test", "a", 0.8, False, 10),
            BaselineResult("test", "b", 0.95, True, 10),
        ]
        assert comp.best_r_squared() == 0.95

    def test_best_r_squared_empty(self):
        comp = BaselineComparison(domain="test", target_equation="y=x")
        assert comp.best_r_squared() == 0.0

    def test_pipeline_advantage(self):
        comp = BaselineComparison(domain="test", target_equation="y=x")
        comp.results = [
            BaselineResult("test", "pipeline", 0.99, True, 100),
            BaselineResult("test", "baseline", 0.85, False, 100),
        ]
        assert abs(comp.pipeline_advantage() - 0.14) < 0.01

    def test_to_dict(self):
        comp = BaselineComparison(domain="test", target_equation="y=x")
        comp.results = [
            BaselineResult("test", "poly", 0.9, False, 100),
        ]
        d = comp.to_dict()
        assert d["domain"] == "test"
        assert len(d["results"]) == 1


class TestDomainBaselines:
    """Test domain-specific baseline runners."""

    def test_projectile_baselines(self):
        comp = run_projectile_baselines(sample_counts=[25, 100])
        assert comp.domain == "projectile"
        assert len(comp.results) >= 4  # 2 methods x 2 sample counts
        # Correct-form fit should have R^2 ~ 1.0
        correct = [r for r in comp.results if "correct_form" in r.method]
        assert all(r.r_squared > 0.999 for r in correct)

    def test_oscillator_baselines(self):
        comp = run_oscillator_baselines(sample_counts=[25, 100])
        assert comp.domain == "harmonic_oscillator"
        sqrt_fits = [r for r in comp.results if "sqrt_ratio" in r.method]
        assert all(r.r_squared > 0.999 for r in sqrt_fits)

    def test_sir_baselines(self):
        comp = run_sir_baselines(sample_counts=[25, 100])
        assert comp.domain == "sir_epidemic"
        ratio_fits = [r for r in comp.results if "ratio" in r.method]
        assert all(r.r_squared > 0.999 for r in ratio_fits)

    def test_pendulum_baselines(self):
        comp = run_pendulum_baselines(sample_counts=[25, 50])
        assert comp.domain == "double_pendulum"
        sqrt_fits = [r for r in comp.results if "sqrt_fit" in r.method]
        assert all(r.r_squared > 0.999 for r in sqrt_fits)

    def test_decay_baselines(self):
        comp = run_decay_baselines(sample_counts=[25])
        assert comp.domain == "heat_equation"
        linear_fits = [r for r in comp.results if "linear" in r.method]
        assert all(r.r_squared > 0.999 for r in linear_fits)

    def test_noisy_baselines(self):
        comp = run_projectile_baselines(sample_counts=[225], noise=0.01)
        correct = [r for r in comp.results if "correct_form" in r.method]
        assert all(r.r_squared > 0.99 for r in correct)


class TestAllBaselines:
    """Test the aggregate baseline runner."""

    def test_run_all(self):
        comparisons = run_all_baselines()
        assert len(comparisons) == 5
        domains = [c.domain for c in comparisons]
        assert "projectile" in domains
        assert "harmonic_oscillator" in domains
        assert "sir_epidemic" in domains


class TestOutput:
    """Test output generation."""

    def test_latex_table(self):
        comparisons = run_all_baselines()
        latex = generate_latex_table(comparisons)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "projectile" in latex
        assert "$R^2$" in latex

    def test_save_results(self, tmp_path):
        comparisons = run_all_baselines()
        path = save_results(comparisons, output_dir=tmp_path / "baselines")
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert "comparisons" in data
        assert len(data["comparisons"]) == 5
