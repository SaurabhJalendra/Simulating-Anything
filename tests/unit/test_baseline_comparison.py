"""Tests for baseline comparison benchmark."""
from __future__ import annotations

import pytest

from simulating_anything.analysis.baseline_comparison import (
    BenchmarkResult,
    DomainBenchmark,
    format_benchmark_table,
)


class TestBenchmarkResult:
    def test_creation(self):
        r = BenchmarkResult(
            domain="test",
            method="PySR",
            best_r2=0.999,
            correct_form=True,
            n_samples=100,
            compute_time_s=5.0,
            best_expression="x^2",
        )
        assert r.best_r2 == 0.999
        assert r.correct_form is True

    def test_defaults(self):
        r = BenchmarkResult(
            domain="test", method="PySR", best_r2=0.5,
            correct_form=False, n_samples=10, compute_time_s=1.0,
        )
        assert r.best_expression == ""
        assert r.notes == ""


class TestDomainBenchmark:
    def test_creation(self):
        b = DomainBenchmark(domain="projectile", target_equation="R=v^2*sin(2t)/g")
        assert b.domain == "projectile"
        assert len(b.results) == 0

    def test_add_result(self):
        b = DomainBenchmark(domain="projectile", target_equation="R=v^2*sin(2t)/g")
        b.results.append(BenchmarkResult(
            domain="projectile", method="PySR", best_r2=0.999,
            correct_form=True, n_samples=100, compute_time_s=5.0,
        ))
        assert len(b.results) == 1


class TestFormatTable:
    def test_format_produces_output(self):
        b = DomainBenchmark(domain="test", target_equation="x^2")
        b.results.append(BenchmarkResult(
            domain="test", method="PySR", best_r2=0.999,
            correct_form=True, n_samples=100, compute_time_s=5.0,
        ))
        table = format_benchmark_table({"test": b})
        assert "test" in table
        assert "PySR" in table
        assert "0.999" in table

    def test_format_multiple_methods(self):
        b = DomainBenchmark(domain="test", target_equation="x^2")
        b.results.append(BenchmarkResult(
            domain="test", method="PySR", best_r2=0.99,
            correct_form=True, n_samples=100, compute_time_s=5.0,
        ))
        b.results.append(BenchmarkResult(
            domain="test", method="SINDy", best_r2=1.0,
            correct_form=True, n_samples=200, compute_time_s=2.0,
        ))
        table = format_benchmark_table({"test": b})
        assert "PySR" in table
        assert "SINDy" in table
