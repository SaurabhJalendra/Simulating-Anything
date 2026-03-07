"""Tests for campaign types (Pydantic models)."""

from __future__ import annotations

import json

from simulating_anything.types.campaign import (
    CampaignReport,
    ConfidenceLevel,
    Experiment,
    ExperimentStatus,
    GeneratedSimulation,
    HypothesisResult,
    NotebookEntry,
    ResearchPlan,
)
from simulating_anything.types.discovery import Discovery


class TestExperimentStatus:
    """Tests for ExperimentStatus enum."""

    def test_all_statuses(self):
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.FAILED == "failed"
        assert ExperimentStatus.SKIPPED == "skipped"

    def test_from_string(self):
        assert ExperimentStatus("pending") == ExperimentStatus.PENDING
        assert ExperimentStatus("completed") == ExperimentStatus.COMPLETED


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_all_levels(self):
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.UNTESTED == "untested"


class TestExperiment:
    """Tests for Experiment model."""

    def test_default_construction(self):
        exp = Experiment()
        assert exp.id == ""
        assert exp.status == ExperimentStatus.PENDING
        assert exp.parameters_to_sweep == {}
        assert exp.target_observables == []
        assert exp.depends_on == []

    def test_full_construction(self):
        exp = Experiment(
            id="exp_1",
            description="Test projectile",
            simulation_description="Projectile motion with drag",
            parameters_to_sweep={"v0": (1.0, 50.0), "angle": (10.0, 80.0)},
            target_observables=["range", "max_height"],
            expected_outcome="R = v^2 sin(2theta) / g",
            depends_on=["exp_0"],
            status=ExperimentStatus.RUNNING,
        )
        assert exp.id == "exp_1"
        assert exp.parameters_to_sweep["v0"] == (1.0, 50.0)
        assert len(exp.depends_on) == 1

    def test_serialization(self):
        exp = Experiment(id="exp_1", description="test")
        data = exp.model_dump()
        assert data["id"] == "exp_1"
        assert data["status"] == "pending"

        # Round-trip
        exp2 = Experiment(**data)
        assert exp2 == exp

    def test_json_serialization(self):
        exp = Experiment(id="exp_1", description="test")
        json_str = exp.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "exp_1"


class TestResearchPlan:
    """Tests for ResearchPlan model."""

    def test_default_construction(self):
        plan = ResearchPlan()
        assert plan.question == ""
        assert plan.experiments == []
        assert plan.current_step == 0
        assert plan.max_steps == 20
        assert plan.completed is False

    def test_with_experiments(self):
        plan = ResearchPlan(
            question="How do sand dunes form?",
            sub_questions=["What forces move sand?", "What shapes emerge?"],
            experiments=[
                Experiment(id="exp_1", description="Wind transport"),
                Experiment(id="exp_2", description="Pattern formation"),
            ],
            success_criteria=["Recover dune wavelength scaling"],
        )
        assert len(plan.experiments) == 2
        assert len(plan.sub_questions) == 2

    def test_completion(self):
        plan = ResearchPlan(completed=True)
        assert plan.completed is True


class TestHypothesisResult:
    """Tests for HypothesisResult model."""

    def test_default(self):
        hr = HypothesisResult()
        assert hr.passed is False
        assert hr.confidence_level == ConfidenceLevel.UNTESTED
        assert hr.failure_modes == []

    def test_passed_result(self):
        hr = HypothesisResult(
            discovery_id="disc_1",
            passed=True,
            extrapolation_r2=0.98,
            interpolation_r2=0.99,
            dimensional_consistent=True,
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert hr.passed is True
        assert hr.extrapolation_r2 == 0.98

    def test_failed_result(self):
        hr = HypothesisResult(
            discovery_id="disc_2",
            passed=False,
            failure_modes=["Extrapolation R^2 too low", "Dimensional mismatch"],
        )
        assert len(hr.failure_modes) == 2


class TestGeneratedSimulation:
    """Tests for GeneratedSimulation model."""

    def test_default(self):
        gs = GeneratedSimulation()
        assert gs.problem_id == ""
        assert gs.source_code == ""
        assert gs.validation_passed is False
        assert gs.generation_attempts == 0

    def test_successful_generation(self):
        gs = GeneratedSimulation(
            problem_id="projectile",
            source_code="class ProjectileSimulation(SimulationEnvironment): ...",
            class_name="ProjectileSimulation",
            validation_passed=True,
            validation_details=["All 7 checks passed"],
            generation_attempts=1,
        )
        assert gs.validation_passed is True
        assert gs.generation_attempts == 1


class TestNotebookEntry:
    """Tests for NotebookEntry model."""

    def test_default(self):
        entry = NotebookEntry()
        assert entry.step == 0
        assert entry.discoveries == []

    def test_with_findings(self):
        entry = NotebookEntry(
            step=3,
            experiment_id="exp_1",
            action="Ran parameter sweep",
            result="Found R = v^2/g equation",
            discoveries=[Discovery(expression="v^2/g", confidence=0.95)],
        )
        assert len(entry.discoveries) == 1
        assert entry.discoveries[0].confidence == 0.95


class TestCampaignReport:
    """Tests for CampaignReport model."""

    def test_default(self):
        report = CampaignReport()
        assert report.question == ""
        assert report.experiments_run == 0
        assert report.discoveries == []
        assert report.conclusion == ""

    def test_full_report(self):
        report = CampaignReport(
            question="How does projectile motion work?",
            experiments_run=3,
            experiments_failed=1,
            discoveries=[Discovery(expression="v^2*sin(2*theta)/g")],
            validated_discoveries=[Discovery(expression="v^2*sin(2*theta)/g")],
            generated_simulations=[GeneratedSimulation(problem_id="proj")],
            conclusion="Discovered range equation",
        )
        assert report.experiments_run == 3
        assert len(report.discoveries) == 1
        assert len(report.validated_discoveries) == 1

    def test_json_round_trip(self):
        report = CampaignReport(
            question="Test",
            experiments_run=1,
            conclusion="Done",
        )
        json_str = report.model_dump_json()
        report2 = CampaignReport.model_validate_json(json_str)
        assert report2.question == "Test"
        assert report2.experiments_run == 1
