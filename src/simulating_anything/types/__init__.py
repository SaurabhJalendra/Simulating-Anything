"""Core data types for the Simulating Anything pipeline."""

from simulating_anything.types.discovery import (
    AblationResult,
    CheckResult,
    Discovery,
    DiscoveryReport,
    DiscoveryStatus,
    DiscoveryType,
    Evidence,
    ValidationReport,
)
from simulating_anything.types.problem_spec import (
    Assumption,
    BoundaryCondition,
    BoundaryType,
    Objective,
    ObjectiveType,
    ProblemSpec,
    Scales,
    SweepParameter,
    Variable,
    VariableType,
)
from simulating_anything.types.simulation import (
    Domain,
    DomainClassification,
    Provenance,
    SimulationBackend,
    SimulationConfig,
    TrainingConfig,
    ValidationMetrics,
    WorldModelCheckpoint,
)
from simulating_anything.types.trajectory import (
    TrajectoryData,
    TrajectoryMetadata,
)

__all__ = [
    # problem_spec
    "VariableType",
    "ObjectiveType",
    "BoundaryType",
    "Variable",
    "Objective",
    "BoundaryCondition",
    "Assumption",
    "SweepParameter",
    "Scales",
    "ProblemSpec",
    # simulation
    "Domain",
    "SimulationBackend",
    "SimulationConfig",
    "DomainClassification",
    "Provenance",
    "TrainingConfig",
    "ValidationMetrics",
    "WorldModelCheckpoint",
    # trajectory
    "TrajectoryMetadata",
    "TrajectoryData",
    # discovery
    "DiscoveryType",
    "DiscoveryStatus",
    "Evidence",
    "Discovery",
    "CheckResult",
    "ValidationReport",
    "AblationResult",
    "DiscoveryReport",
]
