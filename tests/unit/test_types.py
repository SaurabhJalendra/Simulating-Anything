"""Tests for Pydantic data types."""

import numpy as np
import pytest

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
from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata


class TestProblemSpec:
    def test_variable_creation(self):
        v = Variable(name="u", type=VariableType.SCALAR_FIELD, units="mol/L")
        assert v.name == "u"
        assert v.type == VariableType.SCALAR_FIELD
        assert v.units == "mol/L"

    def test_objective_creation(self):
        obj = Objective(type=ObjectiveType.MAP, target="phase_diagram")
        assert obj.type == ObjectiveType.MAP

    def test_problem_spec_defaults(self):
        spec = ProblemSpec()
        assert spec.grid == (128, 128)
        assert spec.dimensions == 2
        assert spec.variables == []

    def test_problem_spec_with_fields(self):
        spec = ProblemSpec(
            id="test-1",
            title="Gray-Scott Patterns",
            variables=[Variable(name="u", type=VariableType.SCALAR_FIELD)],
            objectives=[Objective(type=ObjectiveType.MAP)],
            physics_domains=["reaction_diffusion"],
            sweep_parameters=[SweepParameter(name="f", range=(0.01, 0.08))],
        )
        assert len(spec.variables) == 1
        assert spec.sweep_parameters[0].range == (0.01, 0.08)

    def test_problem_spec_serialization(self):
        spec = ProblemSpec(id="s1", title="Test")
        data = spec.model_dump()
        restored = ProblemSpec(**data)
        assert restored.id == "s1"
        assert restored.title == "Test"

    def test_assumption(self):
        a = Assumption(id="A1", description="Incompressible flow", justification="Low Mach")
        assert a.evidence_status == "UNKNOWN"

    def test_boundary_condition(self):
        bc = BoundaryCondition(type=BoundaryType.PERIODIC)
        assert bc.surface == "all"

    def test_scales_defaults(self):
        s = Scales()
        assert s.length == 1.0
        assert s.time == 1.0


class TestSimulationTypes:
    def test_domain_enum(self):
        assert Domain.REACTION_DIFFUSION.value == "reaction_diffusion"
        assert Domain.RIGID_BODY.value == "rigid_body"
        assert Domain.AGENT_BASED.value == "agent_based"

    def test_simulation_config_defaults(self):
        config = SimulationConfig(domain=Domain.REACTION_DIFFUSION)
        assert config.dt == 0.01
        assert config.n_steps == 1000
        assert config.seed == 42

    def test_simulation_config_custom(self):
        config = SimulationConfig(
            domain=Domain.RIGID_BODY,
            backend=SimulationBackend.BRAX,
            dt=0.005,
            parameters={"gravity": 9.81},
        )
        assert config.parameters["gravity"] == 9.81

    def test_domain_classification(self):
        dc = DomainClassification(domain=Domain.AGENT_BASED, confidence=0.95)
        assert dc.confidence == 0.95

    def test_training_config_defaults(self):
        tc = TrainingConfig()
        assert tc.learning_rate == 1e-4
        assert tc.kl_free_bits == 1.0

    def test_provenance(self):
        p = Provenance(code_version="0.1.0", random_seed=42)
        assert p.hardware == ""

    def test_world_model_checkpoint(self):
        wm = WorldModelCheckpoint(model_id="m1", path="/tmp/model")
        assert wm.checkpoint_path().name == "model"

    def test_simulation_config_serialization(self):
        config = SimulationConfig(
            domain=Domain.REACTION_DIFFUSION,
            parameters={"f": 0.035, "k": 0.065},
        )
        data = config.model_dump()
        restored = SimulationConfig(**data)
        assert restored.parameters["f"] == 0.035


class TestTrajectory:
    def test_trajectory_metadata_defaults(self):
        meta = TrajectoryMetadata()
        assert meta.confidence == 0.0
        assert meta.validated is False

    def test_trajectory_data_defaults(self):
        traj = TrajectoryData()
        assert traj.tier == 1
        assert traj.n_steps == 0
        assert traj.states is None

    def test_trajectory_data_with_arrays(self):
        traj = TrajectoryData(id="t1", parameters={"f": 0.035})
        states = np.random.randn(100, 64, 64, 2)
        traj.states = states
        assert traj.n_steps == 100
        assert traj.states.shape == (100, 64, 64, 2)

    def test_trajectory_timestamps(self):
        traj = TrajectoryData()
        ts = np.linspace(0, 1, 50)
        traj.timestamps = ts
        assert len(traj.timestamps) == 50


class TestDiscovery:
    def test_discovery_creation(self):
        d = Discovery(
            id="d1",
            type=DiscoveryType.GOVERNING_EQUATION,
            confidence=0.92,
            expression="du/dt = D * nabla^2(u) - u*v^2 + f*(1-u)",
        )
        assert d.confidence == 0.92
        assert d.status == DiscoveryStatus.PENDING

    def test_evidence(self):
        e = Evidence(trajectory_ids=["t1", "t2"], fit_r_squared=0.97, n_supporting=50)
        assert len(e.trajectory_ids) == 2

    def test_check_result(self):
        cr = CheckResult(name="mass_conservation", passed=True, value=1e-8, threshold=1e-6)
        assert cr.passed

    def test_validation_report(self):
        report = ValidationReport(
            checks=[CheckResult(name="test", passed=True)],
            passed=True,
        )
        assert report.all_critical_passed

    def test_validation_report_with_failures(self):
        report = ValidationReport(
            passed=False,
            critical_failures=["Energy conservation violated"],
        )
        assert not report.all_critical_passed

    def test_ablation_result(self):
        ar = AblationResult(
            factor_name="drag", effect_size=0.75, is_essential=True
        )
        assert ar.is_essential

    def test_discovery_report(self):
        report = DiscoveryReport(
            discoveries=[
                Discovery(id="d1", type=DiscoveryType.SCALING_LAW, confidence=0.8)
            ],
            n_trajectories_analyzed=100,
            summary="Found 1 scaling law",
        )
        assert len(report.discoveries) == 1

    def test_discovery_serialization(self):
        d = Discovery(id="d1", type=DiscoveryType.PHASE_BOUNDARY, confidence=0.5)
        data = d.model_dump()
        restored = Discovery(**data)
        assert restored.type == DiscoveryType.PHASE_BOUNDARY
