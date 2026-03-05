"""Types for autonomous research campaigns and discovery planning."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from simulating_anything.types.discovery import Discovery


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNTESTED = "untested"


class Experiment(BaseModel):
    """A single experiment in a research plan."""

    id: str = ""
    description: str = ""
    simulation_description: str = ""
    parameters_to_sweep: dict[str, tuple[float, float]] = Field(default_factory=dict)
    target_observables: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    depends_on: list[str] = Field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.PENDING
    findings: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """A multi-step research plan for autonomous discovery."""

    question: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 20
    completed: bool = False


class HypothesisResult(BaseModel):
    """Result of testing a discovered equation."""

    discovery_id: str = ""
    passed: bool = False
    extrapolation_r2: float = 0.0
    interpolation_r2: float = 0.0
    dimensional_consistent: bool = False
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNTESTED
    failure_modes: list[str] = Field(default_factory=list)
    details: str = ""


class GeneratedSimulation(BaseModel):
    """Metadata about a dynamically generated simulation."""

    problem_id: str = ""
    source_code: str = ""
    class_name: str = ""
    validation_passed: bool = False
    validation_details: list[str] = Field(default_factory=list)
    generation_attempts: int = 0


class NotebookEntry(BaseModel):
    """A single entry in the research notebook."""

    step: int = 0
    experiment_id: str = ""
    action: str = ""
    result: str = ""
    discoveries: list[Discovery] = Field(default_factory=list)
    hypothesis_results: list[HypothesisResult] = Field(default_factory=list)


class CampaignReport(BaseModel):
    """Final report from an autonomous discovery campaign."""

    question: str = ""
    plan: ResearchPlan = Field(default_factory=ResearchPlan)
    experiments_run: int = 0
    experiments_failed: int = 0
    discoveries: list[Discovery] = Field(default_factory=list)
    validated_discoveries: list[Discovery] = Field(default_factory=list)
    hypothesis_results: list[HypothesisResult] = Field(default_factory=list)
    generated_simulations: list[GeneratedSimulation] = Field(default_factory=list)
    notebook_entries: list[NotebookEntry] = Field(default_factory=list)
    conclusion: str = ""
    research_notebook_md: str = ""
