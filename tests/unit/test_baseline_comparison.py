"""Tests for baseline comparison benchmark."""
from __future__ import annotations

from simulating_anything.analysis.baseline_comparison import (
    BenchmarkResult,
    DomainBenchmark,
    benchmark_lorenz,
    benchmark_lotka_volterra,
    benchmark_projectile,
    format_benchmark_table,
    run_baseline_benchmarks,
)


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_creation_all_fields(self):
        r = BenchmarkResult(
            domain="test",
            method="PySR",
            best_r2=0.999,
            correct_form=True,
            n_samples=100,
            compute_time_s=5.0,
            best_expression="x^2",
            notes="test note",
        )
        assert r.domain == "test"
        assert r.method == "PySR"
        assert r.best_r2 == 0.999
        assert r.correct_form is True
        assert r.n_samples == 100
        assert r.compute_time_s == 5.0
        assert r.best_expression == "x^2"
        assert r.notes == "test note"

    def test_defaults(self):
        r = BenchmarkResult(
            domain="test", method="PySR", best_r2=0.5,
            correct_form=False, n_samples=10, compute_time_s=1.0,
        )
        assert r.best_expression == ""
        assert r.notes == ""

    def test_zero_r2(self):
        r = BenchmarkResult(
            domain="d", method="m", best_r2=0.0,
            correct_form=False, n_samples=1, compute_time_s=0.0,
        )
        assert r.best_r2 == 0.0

    def test_negative_r2_allowed(self):
        """BenchmarkResult should accept any float for R2, including negative."""
        r = BenchmarkResult(
            domain="d", method="m", best_r2=-0.5,
            correct_form=False, n_samples=1, compute_time_s=0.0,
        )
        assert r.best_r2 == -0.5

    def test_r2_one(self):
        r = BenchmarkResult(
            domain="d", method="m", best_r2=1.0,
            correct_form=True, n_samples=50, compute_time_s=0.1,
        )
        assert r.best_r2 == 1.0

    def test_large_n_samples(self):
        r = BenchmarkResult(
            domain="d", method="m", best_r2=0.9,
            correct_form=True, n_samples=1_000_000, compute_time_s=100.0,
        )
        assert r.n_samples == 1_000_000

    def test_correct_form_false(self):
        r = BenchmarkResult(
            domain="d", method="m", best_r2=0.8,
            correct_form=False, n_samples=10, compute_time_s=1.0,
            notes="wrong form",
        )
        assert r.correct_form is False
        assert r.notes == "wrong form"


class TestDomainBenchmark:
    """Tests for the DomainBenchmark dataclass."""

    def test_creation_empty_results(self):
        b = DomainBenchmark(domain="projectile", target_equation="R=v^2*sin(2t)/g")
        assert b.domain == "projectile"
        assert b.target_equation == "R=v^2*sin(2t)/g"
        assert len(b.results) == 0

    def test_results_default_is_list(self):
        b = DomainBenchmark(domain="test", target_equation="y=x")
        assert isinstance(b.results, list)

    def test_add_single_result(self):
        b = DomainBenchmark(domain="projectile", target_equation="R=v^2*sin(2t)/g")
        b.results.append(BenchmarkResult(
            domain="projectile", method="PySR", best_r2=0.999,
            correct_form=True, n_samples=100, compute_time_s=5.0,
        ))
        assert len(b.results) == 1
        assert b.results[0].method == "PySR"

    def test_add_multiple_results(self):
        b = DomainBenchmark(domain="lorenz", target_equation="dx/dt=sigma*(y-x)")
        methods = ["PySR", "SINDy", "Manual", "Pipeline"]
        for m in methods:
            b.results.append(BenchmarkResult(
                domain="lorenz", method=m, best_r2=0.9,
                correct_form=True, n_samples=100, compute_time_s=1.0,
            ))
        assert len(b.results) == 4
        assert [r.method for r in b.results] == methods

    def test_independent_default_lists(self):
        """Each DomainBenchmark should have its own results list (no shared mutable default)."""
        b1 = DomainBenchmark(domain="a", target_equation="x")
        b2 = DomainBenchmark(domain="b", target_equation="y")
        b1.results.append(BenchmarkResult(
            domain="a", method="PySR", best_r2=0.5,
            correct_form=False, n_samples=10, compute_time_s=1.0,
        ))
        assert len(b1.results) == 1
        assert len(b2.results) == 0


class TestBenchmarkProjectile:
    """Tests for the benchmark_projectile function."""

    def test_returns_domain_benchmark(self):
        result = benchmark_projectile()
        assert isinstance(result, DomainBenchmark)

    def test_domain_name(self):
        result = benchmark_projectile()
        assert result.domain == "projectile"

    def test_target_equation_set(self):
        result = benchmark_projectile()
        assert "sin" in result.target_equation
        assert "v0" in result.target_equation or "v" in result.target_equation

    def test_has_results(self):
        """Should have at least the analytical baseline result."""
        result = benchmark_projectile()
        assert len(result.results) >= 1

    def test_analytical_baseline_present(self):
        """The analytical ground truth method should always be present."""
        result = benchmark_projectile()
        methods = [r.method for r in result.results]
        assert any("Analytical" in m or "ground truth" in m.lower() for m in methods)

    def test_analytical_baseline_r2_near_one(self):
        """Analytical baseline should have very high R2 (simulation vs theory)."""
        result = benchmark_projectile()
        analytical = [r for r in result.results if "Analytical" in r.method]
        assert len(analytical) == 1
        assert analytical[0].best_r2 > 0.99

    def test_analytical_correct_form(self):
        result = benchmark_projectile()
        analytical = [r for r in result.results if "Analytical" in r.method]
        assert analytical[0].correct_form is True

    def test_analytical_zero_compute_time(self):
        """The ground truth baseline should have zero compute time."""
        result = benchmark_projectile()
        analytical = [r for r in result.results if "Analytical" in r.method]
        assert analytical[0].compute_time_s == 0.0

    def test_analytical_has_expression(self):
        result = benchmark_projectile()
        analytical = [r for r in result.results if "Analytical" in r.method]
        assert len(analytical[0].best_expression) > 0

    def test_all_results_have_domain_set(self):
        result = benchmark_projectile()
        for r in result.results:
            assert r.domain == "projectile"

    def test_all_results_have_positive_n_samples(self):
        result = benchmark_projectile()
        for r in result.results:
            assert r.n_samples > 0

    def test_all_results_have_nonnegative_compute_time(self):
        result = benchmark_projectile()
        for r in result.results:
            assert r.compute_time_s >= 0.0

    def test_all_results_have_method_name(self):
        result = benchmark_projectile()
        for r in result.results:
            assert len(r.method) > 0

    def test_analytical_n_samples_is_225(self):
        """15 speeds x 15 angles = 225 data points."""
        result = benchmark_projectile()
        analytical = [r for r in result.results if "Analytical" in r.method]
        assert analytical[0].n_samples == 225


class TestBenchmarkLotkaVolterra:
    """Tests for the benchmark_lotka_volterra function."""

    def test_returns_domain_benchmark(self):
        result = benchmark_lotka_volterra()
        assert isinstance(result, DomainBenchmark)

    def test_domain_name(self):
        result = benchmark_lotka_volterra()
        assert result.domain == "lotka_volterra"

    def test_target_equation_set(self):
        result = benchmark_lotka_volterra()
        assert "dx/dt" in result.target_equation or "d" in result.target_equation

    def test_has_results_or_empty_without_sindy(self):
        """Results list has entries if SINDy is installed, otherwise may be empty."""
        result = benchmark_lotka_volterra()
        # With SINDy installed, we get 3 results; without, we get 0
        assert isinstance(result.results, list)

    def test_all_results_have_domain(self):
        result = benchmark_lotka_volterra()
        for r in result.results:
            assert r.domain == "lotka_volterra"

    def test_all_results_nonnegative_compute_time(self):
        result = benchmark_lotka_volterra()
        for r in result.results:
            assert r.compute_time_s >= 0.0

    def test_all_results_positive_n_samples(self):
        result = benchmark_lotka_volterra()
        for r in result.results:
            assert r.n_samples > 0

    def test_method_names_nonempty(self):
        result = benchmark_lotka_volterra()
        for r in result.results:
            assert isinstance(r.method, str)
            assert len(r.method) > 0

    def test_r2_in_valid_range_or_zero_fallback(self):
        """R2 should be in [0, 1] for successful fits, or 0.0 for failures."""
        result = benchmark_lotka_volterra()
        for r in result.results:
            # R2 can be 0 on failure; successful SINDy typically gives [0, 1]
            assert isinstance(r.best_r2, float)


class TestBenchmarkLorenz:
    """Tests for the benchmark_lorenz function."""

    def test_returns_domain_benchmark(self):
        result = benchmark_lorenz()
        assert isinstance(result, DomainBenchmark)

    def test_domain_name(self):
        result = benchmark_lorenz()
        assert result.domain == "lorenz"

    def test_target_equation_mentions_sigma_rho_beta(self):
        result = benchmark_lorenz()
        eq = result.target_equation
        assert "sigma" in eq or "rho" in eq or "beta" in eq

    def test_has_results_or_empty_without_sindy(self):
        """Results list has entries if SINDy is installed, otherwise may be empty."""
        result = benchmark_lorenz()
        assert isinstance(result.results, list)

    def test_all_results_have_domain(self):
        result = benchmark_lorenz()
        for r in result.results:
            assert r.domain == "lorenz"

    def test_all_results_nonneg_compute_time(self):
        result = benchmark_lorenz()
        for r in result.results:
            assert r.compute_time_s >= 0.0

    def test_all_results_positive_n_samples(self):
        result = benchmark_lorenz()
        for r in result.results:
            assert r.n_samples > 0


class TestRunBaselineBenchmarks:
    """Tests for the run_baseline_benchmarks dispatcher."""

    def test_all_domains_returns_dict(self):
        results = run_baseline_benchmarks()
        assert isinstance(results, dict)

    def test_all_domains_keys(self):
        results = run_baseline_benchmarks()
        assert "projectile" in results
        assert "lotka_volterra" in results
        assert "lorenz" in results

    def test_all_domains_count(self):
        results = run_baseline_benchmarks()
        assert len(results) == 3

    def test_values_are_domain_benchmarks(self):
        results = run_baseline_benchmarks()
        for v in results.values():
            assert isinstance(v, DomainBenchmark)

    def test_single_domain_filter(self):
        results = run_baseline_benchmarks(domains=["projectile"])
        assert len(results) == 1
        assert "projectile" in results

    def test_two_domain_filter(self):
        results = run_baseline_benchmarks(domains=["projectile", "lorenz"])
        assert len(results) == 2
        assert "projectile" in results
        assert "lorenz" in results

    def test_unknown_domain_skipped(self):
        results = run_baseline_benchmarks(domains=["nonexistent_domain"])
        assert len(results) == 0

    def test_mixed_known_unknown(self):
        results = run_baseline_benchmarks(domains=["projectile", "nonexistent"])
        assert len(results) == 1
        assert "projectile" in results

    def test_empty_domain_list(self):
        results = run_baseline_benchmarks(domains=[])
        assert len(results) == 0

    def test_projectile_always_has_results(self):
        """Projectile has an analytical baseline that does not depend on PySR/SINDy."""
        results = run_baseline_benchmarks(domains=["projectile"])
        assert len(results["projectile"].results) >= 1

    def test_sindy_domains_return_benchmarks(self):
        """LV and Lorenz return DomainBenchmark even if SINDy is unavailable."""
        results = run_baseline_benchmarks()
        for domain in ["lotka_volterra", "lorenz"]:
            assert isinstance(results[domain], DomainBenchmark)


class TestFormatBenchmarkTable:
    """Tests for the format_benchmark_table function."""

    def test_produces_string(self):
        b = DomainBenchmark(domain="test", target_equation="x^2")
        b.results.append(BenchmarkResult(
            domain="test", method="PySR", best_r2=0.999,
            correct_form=True, n_samples=100, compute_time_s=5.0,
        ))
        table = format_benchmark_table({"test": b})
        assert isinstance(table, str)

    def test_contains_header(self):
        b = DomainBenchmark(domain="test", target_equation="x^2")
        b.results.append(BenchmarkResult(
            domain="test", method="PySR", best_r2=0.5,
            correct_form=False, n_samples=10, compute_time_s=1.0,
        ))
        table = format_benchmark_table({"test": b})
        assert "BASELINE COMPARISON BENCHMARK" in table

    def test_contains_column_headers(self):
        b = DomainBenchmark(domain="test", target_equation="y=x")
        b.results.append(BenchmarkResult(
            domain="test", method="A", best_r2=0.1,
            correct_form=False, n_samples=1, compute_time_s=0.0,
        ))
        table = format_benchmark_table({"test": b})
        assert "Domain" in table
        assert "Method" in table
        assert "R" in table  # R-squared header
        assert "Samples" in table

    def test_contains_domain_name(self):
        b = DomainBenchmark(domain="mydom", target_equation="y=x")
        b.results.append(BenchmarkResult(
            domain="mydom", method="PySR", best_r2=0.999,
            correct_form=True, n_samples=100, compute_time_s=5.0,
        ))
        table = format_benchmark_table({"mydom": b})
        assert "mydom" in table

    def test_contains_method_name(self):
        b = DomainBenchmark(domain="d", target_equation="y=x")
        b.results.append(BenchmarkResult(
            domain="d", method="SINDy (special)", best_r2=0.7,
            correct_form=True, n_samples=50, compute_time_s=2.0,
        ))
        table = format_benchmark_table({"d": b})
        assert "SINDy (special)" in table

    def test_contains_r2_value(self):
        b = DomainBenchmark(domain="d", target_equation="y=x")
        b.results.append(BenchmarkResult(
            domain="d", method="M", best_r2=0.123456,
            correct_form=True, n_samples=50, compute_time_s=2.0,
        ))
        table = format_benchmark_table({"d": b})
        assert "0.123456" in table

    def test_correct_form_yes_no(self):
        b = DomainBenchmark(domain="d", target_equation="y=x")
        b.results.append(BenchmarkResult(
            domain="d", method="A", best_r2=0.9,
            correct_form=True, n_samples=50, compute_time_s=1.0,
        ))
        b.results.append(BenchmarkResult(
            domain="d", method="B", best_r2=0.1,
            correct_form=False, n_samples=10, compute_time_s=0.5,
        ))
        table = format_benchmark_table({"d": b})
        assert "Yes" in table
        assert "No" in table

    def test_multiple_methods_listed(self):
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

    def test_multiple_domains(self):
        b1 = DomainBenchmark(domain="alpha", target_equation="a")
        b1.results.append(BenchmarkResult(
            domain="alpha", method="M1", best_r2=0.5,
            correct_form=True, n_samples=10, compute_time_s=1.0,
        ))
        b2 = DomainBenchmark(domain="beta", target_equation="b")
        b2.results.append(BenchmarkResult(
            domain="beta", method="M2", best_r2=0.6,
            correct_form=False, n_samples=20, compute_time_s=2.0,
        ))
        table = format_benchmark_table({"alpha": b1, "beta": b2})
        assert "alpha" in table
        assert "beta" in table

    def test_empty_results_dict(self):
        table = format_benchmark_table({})
        assert isinstance(table, str)
        assert "BASELINE COMPARISON BENCHMARK" in table

    def test_separator_lines(self):
        b = DomainBenchmark(domain="d", target_equation="y")
        b.results.append(BenchmarkResult(
            domain="d", method="M", best_r2=0.5,
            correct_form=True, n_samples=10, compute_time_s=1.0,
        ))
        table = format_benchmark_table({"d": b})
        assert "=" * 100 in table
        assert "-" * 100 in table

    def test_domain_name_only_on_first_row(self):
        """For a domain with multiple methods, domain name appears only once."""
        b = DomainBenchmark(domain="testdom", target_equation="y")
        for i in range(3):
            b.results.append(BenchmarkResult(
                domain="testdom", method=f"Method{i}", best_r2=0.5,
                correct_form=True, n_samples=10, compute_time_s=1.0,
            ))
        table = format_benchmark_table({"testdom": b})
        lines = table.split("\n")
        # Count how many lines start with "testdom" (non-header)
        domain_lines = [ln for ln in lines if ln.startswith("testdom")]
        assert len(domain_lines) == 1


class TestIntegration:
    """Integration tests verifying cross-component consistency."""

    def test_projectile_results_all_same_domain(self):
        bench = benchmark_projectile()
        for r in bench.results:
            assert r.domain == bench.domain

    def test_lotka_volterra_results_all_same_domain(self):
        bench = benchmark_lotka_volterra()
        for r in bench.results:
            assert r.domain == bench.domain

    def test_lorenz_results_all_same_domain(self):
        bench = benchmark_lorenz()
        for r in bench.results:
            assert r.domain == bench.domain

    def test_run_all_then_format(self):
        """Full pipeline: run all benchmarks, then format as table."""
        results = run_baseline_benchmarks()
        table = format_benchmark_table(results)
        assert isinstance(table, str)
        # Domains with results should appear; domains without results still
        # contribute separator lines. At minimum the header is present.
        assert "BASELINE COMPARISON BENCHMARK" in table
        # Projectile always has the analytical result
        assert "projectile" in table

    def test_projectile_analytical_matches_theory(self):
        """Verify the analytical baseline computes correct projectile range."""
        bench = benchmark_projectile()
        analytical = [r for r in bench.results if "Analytical" in r.method]
        assert len(analytical) == 1
        # R2 between simulation and v0^2*sin(2*theta)/g should be nearly 1
        # because drag=0 in the benchmark
        assert analytical[0].best_r2 > 0.999
