"""End-to-end pipeline orchestrating all agents for scientific discovery."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from simulating_anything.agents.base import ClaudeCodeBackend
from simulating_anything.agents.communicator import CommunicatorAgent
from simulating_anything.agents.domain_classifier import DomainClassifierAgent
from simulating_anything.agents.problem_architect import ProblemArchitectAgent
from simulating_anything.agents.simulation_builder import SimulationBuilderAgent
from simulating_anything.analysis.ablation import run_ablation
from simulating_anything.exploration.uncertainty_driven import UncertaintyDrivenExplorer
from simulating_anything.knowledge.discovery_log import DiscoveryLog
from simulating_anything.knowledge.trajectory_store import TrajectoryStore
from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.simulation.reaction_diffusion import GrayScottSimulation
from simulating_anything.simulation.rigid_body import ProjectileSimulation
from simulating_anything.types.discovery import (
    AblationResult,
    Discovery,
    DiscoveryReport,
    DiscoveryStatus,
)
from simulating_anything.types.problem_spec import ProblemSpec
from simulating_anything.types.simulation import (
    Domain,
    DomainClassification,
    SimulationConfig,
)
from simulating_anything.types.trajectory import TrajectoryData
from simulating_anything.utils.config import SimulatingAnythingConfig, load_config
from simulating_anything.verification.conservation import check_mass_conservation, check_positivity

logger = logging.getLogger(__name__)

_DOMAIN_SIM_MAP = {
    Domain.REACTION_DIFFUSION: GrayScottSimulation,
    Domain.RIGID_BODY: ProjectileSimulation,
    Domain.AGENT_BASED: LotkaVolterraSimulation,
}


class Pipeline:
    """Main orchestrator for the Simulating Anything discovery pipeline.

    Stages:
    1. Problem Architect: NL -> ProblemSpec
    2. Domain Classifier: ProblemSpec -> Domain
    3. Simulation Builder: ProblemSpec + Domain -> SimulationConfig
    4. Simulation: Run ground-truth trajectories
    5. Exploration: Uncertainty-driven parameter sweep
    6. Analysis: Discover equations, ablation studies
    7. Communication: Generate Markdown report
    """

    def __init__(
        self,
        config: SimulatingAnythingConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backend and agents
        self.backend = ClaudeCodeBackend()
        self.problem_architect = ProblemArchitectAgent(self.backend)
        self.domain_classifier = DomainClassifierAgent(self.backend)
        self.simulation_builder = SimulationBuilderAgent(self.backend)
        self.communicator = CommunicatorAgent(self.backend)

        # Knowledge stores
        self.trajectory_store = TrajectoryStore(self.output_dir / "trajectories")
        self.discovery_log = DiscoveryLog(self.output_dir / "discoveries")

    def run(self, problem_description: str, socratic: bool = False) -> str:
        """Execute the full pipeline from natural language to report.

        Args:
            problem_description: Natural language research question.
            socratic: If True, pause between stages for user input (not implemented in V1).

        Returns:
            Markdown report string.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE START")
        logger.info("=" * 60)
        start_time = time.time()

        # Stage 1: Problem Architect
        logger.info("[1/7] Problem Architect: parsing problem...")
        spec = self.problem_architect.run(problem_description)
        logger.info(f"  -> ProblemSpec: {spec.title}")

        # Stage 2: Domain Classifier
        logger.info("[2/7] Domain Classifier: classifying domain...")
        classification = self.domain_classifier.run(spec)
        logger.info(f"  -> Domain: {classification.domain.value} ({classification.confidence:.0%})")

        # Stage 3: Simulation Builder
        logger.info("[3/7] Simulation Builder: building config...")
        sim_config = self.simulation_builder.run(spec, classification)
        logger.info(f"  -> Config: {sim_config.backend.value}, dt={sim_config.dt}")

        # Stage 4: Ground-truth simulation
        logger.info("[4/7] Running ground-truth simulation...")
        trajectories = self._run_simulations(sim_config, spec)
        logger.info(f"  -> Collected {len(trajectories)} trajectories")

        # Stage 5: Exploration
        logger.info("[5/7] Exploring parameter space...")
        explored = self._explore(sim_config, spec)
        trajectories.extend(explored)
        logger.info(f"  -> Total trajectories: {len(trajectories)}")

        # Stage 6: Analysis
        logger.info("[6/7] Analyzing trajectories...")
        discoveries, ablations = self._analyze(trajectories, sim_config)
        logger.info(f"  -> Discoveries: {len(discoveries)}, Ablations: {len(ablations)}")

        # Stage 7: Communication
        logger.info("[7/7] Generating report...")
        report = DiscoveryReport(
            discoveries=discoveries,
            ablation_results=ablations,
            summary=f"Analysis of {spec.title} across {len(trajectories)} trajectories",
            n_trajectories_analyzed=len(trajectories),
            parameter_ranges={
                name: (sp.range[0], sp.range[1]) for sp in spec.sweep_parameters
                for name in [sp.name]
            },
        )
        markdown = self.communicator.run(report)

        # Save report
        report_path = self.output_dir / "report.md"
        report_path.write_text(markdown)

        elapsed = time.time() - start_time
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info(f"Report saved to {report_path}")

        return markdown

    def _run_simulations(
        self, config: SimulationConfig, spec: ProblemSpec
    ) -> list[TrajectoryData]:
        """Run ground-truth simulations with default parameters."""
        sim_cls = _DOMAIN_SIM_MAP.get(config.domain)
        if sim_cls is None:
            logger.error(f"No simulation for domain {config.domain}")
            return []

        sim = sim_cls(config)
        traj = sim.run()
        traj.problem_id = spec.id
        traj.tier = 2  # Ground-truth tier

        # Validate
        if traj.states is not None:
            mass_check = check_mass_conservation(traj.states)
            pos_check = check_positivity(traj.states)
            traj.metadata.validated = mass_check.passed and pos_check.passed

        # Store
        self.trajectory_store.save(traj)
        return [traj]

    def _explore(
        self, config: SimulationConfig, spec: ProblemSpec
    ) -> list[TrajectoryData]:
        """Run uncertainty-driven exploration of parameter space."""
        sweep_ranges = {}
        for sp in spec.sweep_parameters:
            sweep_ranges[sp.name] = sp.range

        if not sweep_ranges:
            # Use config defaults if no sweep params specified
            from simulating_anything.utils.config import load_domain_config
            dc = load_domain_config(config.domain)
            sweep_ranges = dict(dc.sweep_ranges)

        if not sweep_ranges:
            return []

        explorer = UncertaintyDrivenExplorer(
            sweep_ranges=sweep_ranges,
            n_points_per_dim=self.config.exploration.trajectories_per_round,
            seed=self.config.seed,
        )

        sim_cls = _DOMAIN_SIM_MAP.get(config.domain)
        if sim_cls is None:
            return []

        trajectories = []
        n_rounds = min(self.config.exploration.n_rounds, 3)  # Cap for V1

        for round_idx in range(n_rounds):
            params = explorer.propose_parameters()
            explore_config = SimulationConfig(
                domain=config.domain,
                backend=config.backend,
                grid_resolution=config.grid_resolution,
                domain_size=config.domain_size,
                dt=config.dt,
                n_steps=config.n_steps,
                parameters={**config.parameters, **params},
                boundary_conditions=config.boundary_conditions,
                seed=config.seed + round_idx,
            )

            try:
                sim = sim_cls(explore_config)
                traj = sim.run()
                traj.problem_id = spec.id
                traj.explorer_id = "uncertainty_driven"
                traj.parameters = params
                explorer.update(traj)
                self.trajectory_store.save(traj)
                trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Exploration round {round_idx} failed: {e}")

        return trajectories

    def _analyze(
        self,
        trajectories: list[TrajectoryData],
        config: SimulationConfig,
    ) -> tuple[list[Discovery], list[AblationResult]]:
        """Run analysis on collected trajectories."""
        discoveries: list[Discovery] = []
        ablations: list[AblationResult] = []

        # Try symbolic regression / SINDy if we have enough data
        if len(trajectories) >= 2 and trajectories[0].states is not None:
            try:
                from simulating_anything.analysis.equation_discovery import run_sindy
                import numpy as np

                states = trajectories[0].states
                if states.ndim <= 2:
                    sindy_results = run_sindy(states, config.dt)
                    for d in sindy_results:
                        d.domain = config.domain.value
                        discoveries.append(d)
                        self.discovery_log.add(d)
            except Exception as e:
                logger.warning(f"Equation discovery failed: {e}")

        # Ablation study
        if trajectories and config.parameters:
            import numpy as np

            def metric_fn(params: dict[str, float]) -> float:
                sim_cls = _DOMAIN_SIM_MAP.get(config.domain)
                if sim_cls is None:
                    return 0.0
                test_config = SimulationConfig(
                    domain=config.domain,
                    backend=config.backend,
                    grid_resolution=config.grid_resolution,
                    domain_size=config.domain_size,
                    dt=config.dt,
                    n_steps=min(config.n_steps, 100),  # Short run for ablation
                    parameters=params,
                    boundary_conditions=config.boundary_conditions,
                    seed=config.seed,
                )
                sim = sim_cls(test_config)
                traj = sim.run()
                if traj.states is not None:
                    return float(np.std(traj.states[-1]))
                return 0.0

            try:
                ablations = run_ablation(
                    metric_fn, config.parameters, metric_name="final_state_std"
                )
            except Exception as e:
                logger.warning(f"Ablation study failed: {e}")

        return discoveries, ablations
