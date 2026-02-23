"""Structured problem specification produced by the Problem Architect Agent."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    SCALAR = "scalar"
    SCALAR_FIELD = "scalar_field"
    VECTOR = "vector"
    VECTOR_FIELD = "vector_field"
    MESH = "mesh"


class ObjectiveType(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    MAP = "map"
    CHARACTERIZE = "characterize"


class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    CONVECTIVE = "convective"


class Variable(BaseModel):
    """A state-space variable in the problem."""

    name: str
    type: VariableType
    domain: tuple[float | None, float | None] = (None, None)
    units: str = "dimensionless"


class Objective(BaseModel):
    """A research objective or optimization target."""

    type: ObjectiveType
    target: str = ""
    description: str = ""


class BoundaryCondition(BaseModel):
    """A boundary condition specification."""

    type: BoundaryType
    surface: str = "all"
    field: str = ""
    value: float | None = None
    parameters: dict[str, float] = Field(default_factory=dict)


class Assumption(BaseModel):
    """A simplifying assumption with justification and impact assessment."""

    id: str
    description: str
    justification: str = ""
    impact_if_wrong: str = ""
    evidence_status: str = "UNKNOWN"


class SweepParameter(BaseModel):
    """A parameter to sweep during exploration."""

    name: str
    definition: str = ""
    range: tuple[float, float] = (0.0, 1.0)
    n_points: int = 50


class Scales(BaseModel):
    """Characteristic scales for dimensional analysis."""

    length: float = 1.0
    time: float = 1.0
    concentration: float = 1.0
    temperature: float = 1.0
    velocity: float = 1.0


class ProblemSpec(BaseModel):
    """Complete structured problem specification."""

    id: str = ""
    title: str = ""
    description: str = ""
    variables: list[Variable] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    physics_domains: list[str] = Field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    scales: Scales = Field(default_factory=Scales)
    assumptions: list[Assumption] = Field(default_factory=list)
    sweep_parameters: list[SweepParameter] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    grid: tuple[int, ...] = (128, 128)
    domain_size: tuple[float, ...] = (1.0, 1.0)
    dimensions: int = 2
