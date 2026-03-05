"""Campaign manager for autonomous scientific discovery.

Orchestrates the full discovery loop: plan -> generate -> validate ->
explore -> analyze -> hypothesize -> replan, until success criteria
are met or resources are exhausted.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from simulating_anything.agents.base import ClaudeCodeBackend
from simulating_anything.agents.research_planner import ResearchPlannerAgent
from simulating_anything.agents.simulation_generator import SimulationGeneratorAgent
from simulating_anything.analysis.hypothesis_tester import HypothesisTester
from simulating_anything.campaign.notebook import ResearchNotebook
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.campaign import (
    CampaignReport,
    ExperimentStatus,
    GeneratedSimulation,
    ResearchPlan,
)
from simulating_anything.types.discovery import Discovery, DiscoveryType, Evidence
from simulating_anything.types.problem_spec import ProblemSpec
from simulating_anything.types.simulation import SimulationConfig
from simulating_anything.verification.simulation_validator import SimulationValidator

logger = logging.getLogger(__name__)


class CampaignManager:
    """Orchestrates autonomous scientific discovery campaigns.

    Given a research question, the manager:
    1. Plans experiments via ResearchPlannerAgent
    2. Generates simulations via SimulationGeneratorAgent
    3. Validates simulations via SimulationValidator
    4. Runs parameter sweeps and collects data
    5. Discovers equations via symbolic regression
    6. Tests hypotheses via HypothesisTester
    7. Replans based on findings
    8. Repeats until success or resource exhaustion
    """

    def __init__(
        self,
        backend: ClaudeCodeBackend | None = None,
        output_dir: str | Path = "output/campaigns",
        max_steps: int = 20,
        n_sweep_points: int = 30,
        n_sim_steps: int = 200,
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.n_sweep_points = n_sweep_points
        self.n_sim_steps = n_sim_steps

        # Sub-components
        self.planner = ResearchPlannerAgent(backend=backend)
        self.generator = SimulationGeneratorAgent(backend=backend, output_dir=str(self.output_dir / "sims"))
        self.validator = SimulationValidator()
        self.hypothesis_tester = HypothesisTester(n_sim_steps=n_sim_steps)
        self.notebook = ResearchNotebook()

    def run_campaign(
        self,
        question: str,
        max_steps: int | None = None,
    ) -> CampaignReport:
        """Run a full autonomous discovery campaign.

        Args:
            question: A natural language research question.
            max_steps: Override for maximum campaign steps.

        Returns:
            CampaignReport with all findings and the research notebook.
        """
        max_steps = max_steps or self.max_steps
        start_time = time.time()

        logger.info(f"Starting campaign: {question}")
        logger.info(f"Max steps: {max_steps}")

        # Phase 1: Plan
        plan = self.planner.plan(question)
        plan.max_steps = max_steps
        logger.info(f"Plan created: {len(plan.experiments)} experiments")

        all_discoveries: list[Discovery] = []
        validated_discoveries: list[Discovery] = []
        all_hypothesis_results = []
        generated_sims: list[GeneratedSimulation] = []
        experiments_run = 0
        experiments_failed = 0
        no_new_findings_count = 0

        # Phase 2: Execute experiments
        step = 0
        while step < max_steps and not plan.completed:
            # Find next experiment to run
            experiment = self._get_next_experiment(plan)
            if experiment is None:
                # All experiments done or blocked, try replanning
                logger.info("No runnable experiments, replanning...")
                self.notebook.log_replan(step, "All experiments completed or blocked")
                plan = self.planner.replan(plan, all_discoveries)

                experiment = self._get_next_experiment(plan)
                if experiment is None:
                    logger.info("No more experiments after replan, ending campaign")
                    break

            experiment.status = ExperimentStatus.RUNNING
            logger.info(f"Step {step}: Running experiment {experiment.id}")

            try:
                # Step A: Generate simulation
                spec = ProblemSpec(
                    id=experiment.id,
                    title=experiment.simulation_description or experiment.description,
                    description=experiment.description,
                    parameters=dict(experiment.parameters_to_sweep),
                )

                source_code, sim_class = self.generator.generate(spec)

                gen_sim = GeneratedSimulation(
                    problem_id=experiment.id,
                    source_code=source_code,
                    class_name=sim_class.__name__,
                    generation_attempts=1,
                )

                # Step B: Validate simulation
                config = self._make_config(experiment.parameters_to_sweep)
                report = self.validator.validate(sim_class, config)

                if not report.all_critical_passed:
                    # Try to fix via regeneration
                    fix_prompt = self.validator.get_fix_prompt(report)
                    logger.warning(f"Validation failed: {report.critical_failures}")
                    self.notebook.log_failure(
                        step, experiment.id,
                        f"Validation failed: {'; '.join(report.critical_failures)}"
                    )
                    experiment.status = ExperimentStatus.FAILED
                    experiments_failed += 1
                    gen_sim.validation_passed = False
                    gen_sim.validation_details = report.critical_failures
                    generated_sims.append(gen_sim)
                    step += 1
                    continue

                gen_sim.validation_passed = True
                gen_sim.validation_details = [c.message for c in report.checks]
                generated_sims.append(gen_sim)

                # Step C: Run parameter sweeps and collect data
                sweep_data = self._run_sweeps(sim_class, config, experiment.parameters_to_sweep)

                # Step D: Discover equations
                discoveries = self._discover_equations(sweep_data, experiment)

                # Step E: Test hypotheses
                hypothesis_results = []
                for disc in discoveries:
                    h_result = self.hypothesis_tester.test(disc, sim_class, config)
                    hypothesis_results.append(h_result)
                    all_hypothesis_results.append(h_result)

                    if h_result.passed:
                        disc.status = "confirmed"
                        validated_discoveries.append(disc)

                all_discoveries.extend(discoveries)

                # Log to notebook
                experiment.status = ExperimentStatus.COMPLETED
                experiment.findings = [d.expression or d.description for d in discoveries]
                experiments_run += 1

                self.notebook.log_experiment(
                    step=step,
                    experiment_id=experiment.id,
                    action=experiment.description,
                    result=f"Generated {sim_class.__name__}, found {len(discoveries)} equations",
                    discoveries=discoveries,
                    hypothesis_results=hypothesis_results,
                )

                # Track stagnation
                if discoveries:
                    no_new_findings_count = 0
                else:
                    no_new_findings_count += 1

                if no_new_findings_count >= 3:
                    logger.info("No new findings for 3 steps, ending campaign")
                    plan.completed = True

            except Exception as e:
                logger.error(f"Experiment {experiment.id} failed: {e}")
                experiment.status = ExperimentStatus.FAILED
                experiments_failed += 1
                self.notebook.log_failure(step, experiment.id, str(e))

            step += 1

        # Phase 3: Build report
        elapsed = time.time() - start_time
        logger.info(
            f"Campaign complete: {experiments_run} experiments, "
            f"{len(all_discoveries)} discoveries, {elapsed:.1f}s"
        )

        report = CampaignReport(
            question=question,
            plan=plan,
            experiments_run=experiments_run,
            experiments_failed=experiments_failed,
            discoveries=all_discoveries,
            validated_discoveries=validated_discoveries,
            hypothesis_results=all_hypothesis_results,
            generated_simulations=generated_sims,
            notebook_entries=self.notebook.entries,
            conclusion=self._generate_conclusion(question, validated_discoveries),
            research_notebook_md=self.notebook.to_markdown(),
        )

        # Save
        self._save_report(report)
        return report

    def _get_next_experiment(self, plan: ResearchPlan):
        """Find the next runnable experiment (pending, not blocked)."""
        completed_ids = {
            e.id for e in plan.experiments
            if e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.SKIPPED)
        }

        for exp in plan.experiments:
            if exp.status != ExperimentStatus.PENDING:
                continue
            # Check dependencies
            if all(dep in completed_ids for dep in exp.depends_on):
                return exp
        return None

    def _make_config(
        self,
        parameters_to_sweep: dict[str, tuple[float, float]],
    ) -> SimulationConfig:
        """Create a SimulationConfig with midpoint parameter values."""
        params = {}
        for name, (lo, hi) in parameters_to_sweep.items():
            params[name] = (lo + hi) / 2.0

        return SimulationConfig(
            parameters=params,
            dt=0.01,
            n_steps=self.n_sim_steps,
        )

    def _run_sweeps(
        self,
        sim_class: type,
        base_config: SimulationConfig,
        parameters_to_sweep: dict[str, tuple[float, float]],
    ) -> dict[str, Any]:
        """Run parameter sweeps and collect observable data."""
        sweep_data: dict[str, Any] = {
            "param_names": [],
            "param_values": [],
            "observables": [],
            "trajectories": [],
        }

        if not parameters_to_sweep:
            # Single run with default params
            try:
                sim = sim_class(base_config)
                sim.reset(seed=42)
                states = []
                for _ in range(self.n_sim_steps):
                    states.append(sim.step().copy())
                sweep_data["trajectories"].append(np.array(states))
                sweep_data["observables"].append(sim.observe().copy())
            except Exception as e:
                logger.warning(f"Default run failed: {e}")
            return sweep_data

        # Sweep each parameter
        for param_name, (lo, hi) in parameters_to_sweep.items():
            values = np.linspace(lo, hi, self.n_sweep_points)
            sweep_data["param_names"].append(param_name)
            sweep_data["param_values"].append(values)

            for val in values:
                params = dict(base_config.parameters)
                params[param_name] = float(val)
                config = SimulationConfig(
                    parameters=params,
                    dt=base_config.dt,
                    n_steps=base_config.n_steps,
                )

                try:
                    sim = sim_class(config)
                    sim.reset(seed=42)
                    states = []
                    for _ in range(self.n_sim_steps):
                        states.append(sim.step().copy())
                    sweep_data["trajectories"].append(np.array(states))
                    sweep_data["observables"].append(sim.observe().copy())
                except Exception as e:
                    logger.warning(f"Sweep {param_name}={val:.4f} failed: {e}")
                    sweep_data["observables"].append(None)

        return sweep_data

    def _discover_equations(
        self,
        sweep_data: dict[str, Any],
        experiment: Any,
    ) -> list[Discovery]:
        """Attempt equation discovery from sweep data."""
        discoveries: list[Discovery] = []

        # Try SINDy on trajectory data
        if sweep_data["trajectories"]:
            try:
                from simulating_anything.analysis.equation_discovery import run_sindy

                traj = sweep_data["trajectories"][0]
                if traj.ndim == 2 and traj.shape[0] > 10:
                    sindy_results = run_sindy(traj, dt=0.01)
                    for d in sindy_results:
                        d.domain = experiment.id
                        discoveries.append(d)
            except ImportError:
                logger.info("PySINDy not available, skipping SINDy")
            except Exception as e:
                logger.warning(f"SINDy failed: {e}")

        # Try PySR on parameter-observable relationships
        if sweep_data["param_values"] and sweep_data["observables"]:
            try:
                from simulating_anything.analysis.symbolic_regression import run_pysr

                param_vals = sweep_data["param_values"][0]
                obs_list = sweep_data["observables"]

                # Filter out None entries
                valid = [(p, o) for p, o in zip(param_vals, obs_list) if o is not None]
                if len(valid) >= 10:
                    x_vals = np.array([v[0] for v in valid]).reshape(-1, 1)
                    y_vals = np.array([
                        v[1][0] if isinstance(v[1], np.ndarray) and v[1].size > 0
                        else float(v[1]) for v in valid
                    ])

                    param_name = sweep_data["param_names"][0]
                    pysr_results = run_pysr(
                        x_vals, y_vals,
                        variable_names=[param_name],
                        n_iterations=20,
                    )
                    for d in pysr_results:
                        d.domain = experiment.id
                        discoveries.append(d)
            except ImportError:
                logger.info("PySR not available, skipping symbolic regression")
            except Exception as e:
                logger.warning(f"PySR failed: {e}")

        # If no equation discovery tools available, create observational findings
        if not discoveries and sweep_data["observables"]:
            valid_obs = [o for o in sweep_data["observables"] if o is not None]
            if valid_obs:
                obs_array = np.array([
                    o if isinstance(o, np.ndarray) else np.array([o])
                    for o in valid_obs
                ])
                if obs_array.ndim >= 2:
                    # Report basic statistics as a finding
                    mean_obs = np.mean(obs_array, axis=0)
                    std_obs = np.std(obs_array, axis=0)
                    discoveries.append(Discovery(
                        id=f"{experiment.id}_stats",
                        type=DiscoveryType.QUALITATIVE_FINDING,
                        confidence=0.5,
                        description=(
                            f"Observable mean: {mean_obs}, std: {std_obs} "
                            f"across {len(valid_obs)} parameter points"
                        ),
                        domain=experiment.id,
                        evidence=Evidence(n_supporting=len(valid_obs)),
                    ))

        return discoveries

    def _generate_conclusion(
        self,
        question: str,
        validated_discoveries: list[Discovery],
    ) -> str:
        """Generate a conclusion summarizing the campaign findings."""
        if not validated_discoveries:
            return (
                f"Investigation of '{question}' completed. "
                f"No validated equations were discovered. "
                f"The system may need more experiments or different parameter ranges."
            )

        equations = [d for d in validated_discoveries if d.expression]
        findings = [d for d in validated_discoveries if not d.expression]

        parts = [f"Investigation of '{question}' completed successfully."]

        if equations:
            parts.append(f"\nDiscovered {len(equations)} validated equation(s):")
            for eq in equations:
                parts.append(f"  - {eq.expression} (confidence: {eq.confidence:.2f})")

        if findings:
            parts.append(f"\n{len(findings)} qualitative finding(s):")
            for f in findings:
                parts.append(f"  - {f.description}")

        return "\n".join(parts)

    def _save_report(self, report: CampaignReport) -> None:
        """Save campaign report and notebook to disk."""
        campaign_dir = self.output_dir / report.plan.question[:50].replace(" ", "_")
        campaign_dir.mkdir(parents=True, exist_ok=True)

        # Save notebook
        self.notebook.save(campaign_dir / "research_notebook.md")

        # Save report JSON
        report_path = campaign_dir / "campaign_report.json"
        report_path.write_text(report.model_dump_json(indent=2))

        logger.info(f"Campaign report saved to {campaign_dir}")
