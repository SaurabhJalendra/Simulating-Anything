"""Simulation Builder Agent: ProblemSpec + Domain -> SimulationConfig."""

from __future__ import annotations

import logging

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.types.problem_spec import ProblemSpec
from simulating_anything.types.simulation import (
    Domain,
    DomainClassification,
    SimulationConfig,
)
from simulating_anything.utils.config import load_domain_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Simulation Builder Agent. Given a ProblemSpec and domain classification,
produce a SimulationConfig JSON. Start from the domain template defaults and customize
based on the problem's specific needs:

1. Set appropriate grid resolution for the problem scale
2. Choose dt for numerical stability (CFL condition for PDEs)
3. Map problem parameters to simulation parameters
4. Configure boundary conditions
5. Set appropriate n_steps for the phenomenon timescale

Return a SimulationConfig JSON object.
"""


class SimulationBuilderAgent(Agent):
    """Build a SimulationConfig from a ProblemSpec and domain classification.

    Loads domain template defaults, then uses LLM to customize parameters
    based on the specific problem.
    """

    def __init__(self, backend: ClaudeCodeBackend | None = None) -> None:
        super().__init__(backend or ClaudeCodeBackend())

    def run(
        self, spec: ProblemSpec, classification: DomainClassification
    ) -> SimulationConfig:
        """Build simulation configuration.

        Args:
            spec: Structured problem specification.
            classification: Domain classification result.

        Returns:
            SimulationConfig ready for simulation instantiation.
        """
        # Load domain template
        domain_config = load_domain_config(classification.domain)

        # Start with template defaults
        config = SimulationConfig(
            domain=classification.domain,
            backend=classification.backend,
            grid_resolution=domain_config.grid_resolution,
            domain_size=domain_config.domain_size,
            dt=domain_config.dt,
            n_steps=domain_config.n_steps,
            parameters=dict(domain_config.default_parameters),
            boundary_conditions=domain_config.boundary_conditions,
            seed=42,
        )

        # Override with problem-specific parameters
        for key, value in spec.parameters.items():
            if isinstance(value, (int, float)):
                config.parameters[key] = float(value)

        if spec.grid != (128, 128):
            config.grid_resolution = spec.grid
        if spec.domain_size != (1.0, 1.0):
            config.domain_size = spec.domain_size

        # Use LLM to refine if the problem has special requirements
        if self._needs_llm_refinement(spec, config):
            config = self._llm_refine(spec, classification, config)

        logger.info(
            f"Simulation Builder: {config.domain.value} "
            f"({config.grid_resolution}, dt={config.dt}, n_steps={config.n_steps})"
        )
        return config

    def _needs_llm_refinement(self, spec: ProblemSpec, config: SimulationConfig) -> bool:
        """Check if LLM refinement is needed beyond template defaults."""
        # Refine if there are constraints or unusual boundary conditions
        return bool(spec.constraints) or len(spec.boundary_conditions) > 0

    def _llm_refine(
        self,
        spec: ProblemSpec,
        classification: DomainClassification,
        base_config: SimulationConfig,
    ) -> SimulationConfig:
        """Use LLM to refine the simulation configuration."""
        prompt = (
            f"Refine this simulation configuration for the given problem.\n\n"
            f"Problem:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Current config:\n{base_config.model_dump_json(indent=2)}\n\n"
            f"Adjust parameters, dt, n_steps, or boundary conditions as needed."
        )

        try:
            refined = self.backend.ask_structured(
                prompt=prompt,
                response_model=SimulationConfig,
                system=SYSTEM_PROMPT,
            )
            return refined
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e} — using template defaults")
            return base_config
