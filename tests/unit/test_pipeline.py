"""Tests for pipeline orchestration (mocked agents)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simulating_anything.types.discovery import Discovery, DiscoveryType
from simulating_anything.types.problem_spec import ProblemSpec, SweepParameter
from simulating_anything.types.simulation import (
    Domain,
    DomainClassification,
    SimulationBackend,
    SimulationConfig,
)
from simulating_anything.types.trajectory import TrajectoryData
from simulating_anything.verification.conservation import (
    check_boundedness,
    check_mass_conservation,
    check_positivity,
)
from simulating_anything.verification.dimensional import (
    DIFFUSIVITY,
    DIMENSIONLESS,
    LENGTH,
    TIME,
    Dimensions,
    check_dimensional_consistency,
)


class TestDimensionalAnalysis:
    def test_dimensionless(self):
        d = DIMENSIONLESS
        assert d.is_dimensionless
        assert str(d) == "dimensionless"

    def test_multiplication(self):
        velocity = LENGTH / TIME
        assert velocity.length == 1
        assert velocity.time == -1

    def test_power(self):
        area = LENGTH**2
        assert area.length == 2

    def test_diffusivity(self):
        d = DIFFUSIVITY
        assert d.length == 2
        assert d.time == -1

    def test_consistency_check_pass(self):
        passed, msg = check_dimensional_consistency(DIFFUSIVITY, DIFFUSIVITY, "test")
        assert passed

    def test_consistency_check_fail(self):
        passed, msg = check_dimensional_consistency(LENGTH, TIME, "mismatch")
        assert not passed
        assert "mismatch" in msg


class TestConservation:
    def test_mass_conserved(self):
        states = np.ones((10, 5, 5)) * 2.0
        result = check_mass_conservation(states)
        assert result.passed

    def test_mass_not_conserved(self):
        states = np.ones((10, 5, 5))
        states[5:] *= 2.0
        result = check_mass_conservation(states, tolerance=0.01)
        assert not result.passed

    def test_positivity_pass(self):
        states = np.abs(np.random.randn(10, 4)) + 0.1
        result = check_positivity(states)
        assert result.passed

    def test_positivity_fail(self):
        states = np.random.randn(10, 4)
        states[3, 2] = -0.5
        result = check_positivity(states)
        assert not result.passed

    def test_boundedness_pass(self):
        states = np.random.uniform(0, 1, (10, 4))
        result = check_boundedness(states, 0.0, 1.0)
        assert result.passed

    def test_boundedness_fail(self):
        states = np.random.uniform(0, 2, (10, 4))
        result = check_boundedness(states, 0.0, 1.0)
        assert not result.passed


class TestPipelineStages:
    """Test individual pipeline stage logic with mocks."""

    def test_simulation_produces_trajectory(self):
        """Test that each simulation can produce a trajectory."""
        from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
        from simulating_anything.simulation.reaction_diffusion import GrayScottSimulation
        from simulating_anything.simulation.rigid_body import ProjectileSimulation

        configs = [
            (
                GrayScottSimulation,
                SimulationConfig(
                    domain=Domain.REACTION_DIFFUSION,
                    grid_resolution=(16, 16),
                    domain_size=(2.5, 2.5),
                    dt=1.0,
                    n_steps=5,
                    parameters={"D_u": 0.16, "D_v": 0.08, "f": 0.035, "k": 0.065},
                ),
            ),
            (
                ProjectileSimulation,
                SimulationConfig(
                    domain=Domain.RIGID_BODY,
                    grid_resolution=(1,),
                    dt=0.01,
                    n_steps=50,
                    parameters={
                        "gravity": 9.81, "drag_coefficient": 0.1, "mass": 1.0,
                        "initial_speed": 30.0, "launch_angle": 45.0,
                    },
                ),
            ),
            (
                LotkaVolterraSimulation,
                SimulationConfig(
                    domain=Domain.AGENT_BASED,
                    grid_resolution=(1,),
                    dt=0.01,
                    n_steps=50,
                    parameters={
                        "alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1,
                        "prey_0": 40.0, "predator_0": 9.0,
                    },
                ),
            ),
        ]

        for sim_cls, config in configs:
            sim = sim_cls(config)
            traj = sim.run()
            assert traj.states is not None
            assert traj.timestamps is not None
            assert len(traj.states) == config.n_steps + 1

    def test_ablation_runs(self):
        """Test ablation study with a simple metric function."""
        from simulating_anything.analysis.ablation import run_ablation

        def metric(params):
            return params.get("a", 0) * 2 + params.get("b", 0) * 3

        results = run_ablation(
            metric, {"a": 1.0, "b": 2.0}, metric_name="test_metric"
        )
        assert len(results) == 2
        assert all(r.factor_name in ("a", "b") for r in results)

    def test_exploration_proposes_params(self):
        """Test that explorer proposes valid parameters."""
        from simulating_anything.exploration.uncertainty_driven import (
            UncertaintyDrivenExplorer,
        )

        explorer = UncertaintyDrivenExplorer(
            sweep_ranges={"f": (0.01, 0.08), "k": (0.03, 0.07)},
            n_points_per_dim=5,
        )

        params = explorer.propose_parameters()
        assert "f" in params
        assert "k" in params
        assert 0.01 <= params["f"] <= 0.08
        assert 0.03 <= params["k"] <= 0.07

    def test_exploration_progress(self):
        from simulating_anything.exploration.uncertainty_driven import (
            UncertaintyDrivenExplorer,
        )

        explorer = UncertaintyDrivenExplorer(
            sweep_ranges={"x": (0.0, 1.0)}, n_points_per_dim=5
        )
        progress = explorer.get_exploration_progress()
        assert progress["total_grid_points"] == 5
        assert progress["visited"] == 0

        explorer.propose_parameters()
        progress = explorer.get_exploration_progress()
        assert progress["visited"] == 1


class TestKnowledgeStores:
    def test_trajectory_store_save_load(self, tmp_path):
        from simulating_anything.knowledge.trajectory_store import TrajectoryStore

        store = TrajectoryStore(tmp_path / "trajectories")
        traj = TrajectoryData(id="t1", problem_id="p1", parameters={"f": 0.035})
        traj.states = np.random.randn(10, 4)
        traj.timestamps = np.linspace(0, 1, 10)

        traj_id = store.save(traj)
        assert traj_id == "t1"

        loaded = store.load("t1")
        assert loaded.problem_id == "p1"
        assert loaded.states.shape == (10, 4)

    def test_trajectory_store_query(self, tmp_path):
        from simulating_anything.knowledge.trajectory_store import TrajectoryStore

        store = TrajectoryStore(tmp_path / "trajectories")
        for i in range(3):
            traj = TrajectoryData(id=f"t{i}", problem_id="p1", tier=i + 1)
            traj.states = np.zeros((5, 2))
            store.save(traj)

        results = store.query(tier=2)
        assert len(results) == 1
        assert results[0] == "t1"

    def test_discovery_log(self, tmp_path):
        from simulating_anything.knowledge.discovery_log import DiscoveryLog
        from simulating_anything.types.discovery import DiscoveryStatus

        log = DiscoveryLog(tmp_path / "discoveries")
        d = Discovery(type=DiscoveryType.GOVERNING_EQUATION, confidence=0.9, expression="x^2")
        did = log.add(d)
        assert did

        log.update_status(did, DiscoveryStatus.CONFIRMED)
        confirmed = log.get_confirmed()
        assert len(confirmed) == 1

    def test_discovery_log_persistence(self, tmp_path):
        from simulating_anything.knowledge.discovery_log import DiscoveryLog

        log1 = DiscoveryLog(tmp_path / "disc")
        log1.add(Discovery(id="d1", type=DiscoveryType.SCALING_LAW, confidence=0.5))

        # New instance should load from disk
        log2 = DiscoveryLog(tmp_path / "disc")
        assert len(log2.get_all()) == 1
        assert log2.get("d1").confidence == 0.5
