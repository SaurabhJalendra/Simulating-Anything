"""Tests for computational cost analysis module."""
from __future__ import annotations

from simulating_anything.analysis.computational_cost import (
    CostReport,
    DomainCost,
    StageCost,
    measure_fitting_cost,
    measure_scaling,
    measure_simulation_cost,
    run_cost_analysis,
)


class TestStageCost:
    """Test StageCost dataclass."""

    def test_creation(self):
        c = StageCost(stage="fit", domain="test", wall_time_s=0.5)
        assert c.stage == "fit"
        assert c.wall_time_s == 0.5
        assert c.peak_memory_mb == 0.0

    def test_with_samples(self):
        c = StageCost(
            stage="sweep", domain="test", wall_time_s=1.0,
            n_samples=1000,
        )
        assert c.n_samples == 1000


class TestDomainCost:
    """Test DomainCost dataclass."""

    def test_total_time(self):
        dc = DomainCost(
            domain="test",
            stages=[
                StageCost("a", "test", 1.0),
                StageCost("b", "test", 2.0),
                StageCost("c", "test", 0.5),
            ],
        )
        assert abs(dc.total_time_s - 3.5) < 0.01

    def test_empty(self):
        dc = DomainCost(domain="test")
        assert dc.total_time_s == 0.0
        assert dc.total_memory_mb == 0.0


class TestCostReport:
    """Test CostReport."""

    def test_total_time(self):
        report = CostReport(domains=[
            DomainCost("a", [StageCost("s1", "a", 1.0)]),
            DomainCost("b", [StageCost("s1", "b", 2.0)]),
        ])
        assert abs(report.total_time_s - 3.0) < 0.01

    def test_summary_table(self):
        report = CostReport(domains=[
            DomainCost("a", [
                StageCost("import", "a", 0.1),
                StageCost("single_run", "a", 0.5),
            ]),
        ])
        table = report.summary_table()
        assert len(table) == 1
        assert table[0]["domain"] == "a"
        assert "import_s" in table[0]
        assert "single_run_s" in table[0]


class TestMeasureFittingCost:
    """Test fitting cost measurement."""

    def test_returns_costs(self):
        costs = measure_fitting_cost(n_samples=100)
        assert len(costs) >= 3
        assert all(c.wall_time_s >= 0 for c in costs)

    def test_stages_labeled(self):
        costs = measure_fitting_cost(n_samples=100)
        stages = [c.stage for c in costs]
        assert "poly_d4" in stages
        assert "correct_form" in stages
        assert "power_law" in stages

    def test_correct_form_fastest(self):
        costs = measure_fitting_cost(n_samples=225)
        correct = [c for c in costs if c.stage == "correct_form"][0]
        poly = [c for c in costs if c.stage == "poly_d4"][0]
        # Correct form should be very fast
        assert correct.wall_time_s < 1.0


class TestMeasureScaling:
    """Test scaling measurements."""

    def test_returns_costs(self):
        costs = measure_scaling(n_values=[25, 100, 400])
        assert len(costs) == 3

    def test_samples_match(self):
        costs = measure_scaling(n_values=[25])
        assert costs[0].n_samples > 0

    def test_notes_contain_r2(self):
        costs = measure_scaling(n_values=[100])
        assert "R^2" in costs[0].notes


class TestMeasureSimulationCost:
    """Test simulation cost measurement."""

    def test_known_domain(self):
        cost = measure_simulation_cost("harmonic_oscillator", n_steps=100, n_sweeps=3)
        assert cost.domain == "harmonic_oscillator"
        stages = [s.stage for s in cost.stages]
        assert "import" in stages
        assert "single_run" in stages
        assert "param_sweep" in stages

    def test_unknown_domain(self):
        cost = measure_simulation_cost("nonexistent_domain")
        assert cost.domain == "nonexistent_domain"
        assert len(cost.stages) == 1
        assert "not found" in cost.stages[0].notes

    def test_timing_positive(self):
        cost = measure_simulation_cost("logistic_map", n_steps=100, n_sweeps=2)
        for s in cost.stages:
            assert s.wall_time_s >= 0


class TestRunCostAnalysis:
    """Test full cost analysis."""

    def test_default_domains(self):
        report = run_cost_analysis(n_steps=50, n_sweeps=2)
        assert len(report.domains) >= 3
        assert report.total_time_s >= 0

    def test_custom_domains(self):
        report = run_cost_analysis(
            domains=["harmonic_oscillator", "logistic_map"],
            n_steps=50, n_sweeps=2,
        )
        assert len(report.domains) == 2
        domain_names = [d.domain for d in report.domains]
        assert "harmonic_oscillator" in domain_names
        assert "logistic_map" in domain_names
