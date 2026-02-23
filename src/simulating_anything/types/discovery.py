"""Discovery and validation report types."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DiscoveryType(str, Enum):
    GOVERNING_EQUATION = "governing_equation"
    PHASE_BOUNDARY = "phase_boundary"
    SCALING_LAW = "scaling_law"
    QUALITATIVE_FINDING = "qualitative_finding"
    OPTIMAL_PARAMETER = "optimal_parameter"


class DiscoveryStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    WEAKENED = "weakened"
    REJECTED = "rejected"


class Evidence(BaseModel):
    """Evidence supporting a discovery."""

    trajectory_ids: list[str] = Field(default_factory=list)
    fit_r_squared: float = 0.0
    cross_validation_score: float = 0.0
    n_supporting: int = 0


class Discovery(BaseModel):
    """A candidate scientific finding."""

    id: str = ""
    type: DiscoveryType = DiscoveryType.GOVERNING_EQUATION
    confidence: float = 0.0
    expression: str = ""
    description: str = ""
    domain: str = ""
    evidence: Evidence = Field(default_factory=Evidence)
    assumptions: list[str] = Field(default_factory=list)
    status: DiscoveryStatus = DiscoveryStatus.PENDING


class CheckResult(BaseModel):
    """Result of a single validation check."""

    name: str
    passed: bool
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""


class ValidationReport(BaseModel):
    """Output of the Simulation Validator Agent."""

    checks: list[CheckResult] = Field(default_factory=list)
    passed: bool = False
    warnings: list[str] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)

    @property
    def all_critical_passed(self) -> bool:
        return len(self.critical_failures) == 0


class AblationResult(BaseModel):
    """Result of ablating a single factor."""

    factor_name: str
    original_value: float = 0.0
    ablated_value: float = 0.0
    effect_size: float = 0.0
    is_essential: bool = False
    description: str = ""


class DiscoveryReport(BaseModel):
    """Full analysis output from the Analyst Agent."""

    discoveries: list[Discovery] = Field(default_factory=list)
    ablation_results: list[AblationResult] = Field(default_factory=list)
    summary: str = ""
    n_trajectories_analyzed: int = 0
    parameter_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
