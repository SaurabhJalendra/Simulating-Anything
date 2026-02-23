"""Problem Architect Agent: natural language -> ProblemSpec."""

from __future__ import annotations

import logging

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.types.problem_spec import ProblemSpec

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Problem Architect Agent in a scientific simulation pipeline.
Your job is to convert a natural-language research question into a structured
ProblemSpec JSON object.

You must identify:
1. State variables (names, types: scalar/scalar_field/vector/vector_field, units, domains)
2. Research objectives (maximize/minimize/map/characterize)
3. Physical constraints and conservation laws
4. Relevant physics domains (reaction_diffusion, rigid_body, agent_based)
5. Boundary conditions
6. Characteristic scales (length, time, concentration, temperature, velocity)
7. Simplifying assumptions with justifications
8. Parameters to sweep and their ranges
9. Grid resolution and domain size

Be precise about units and dimensional analysis. If information is missing,
make reasonable physics-informed defaults and document them as assumptions.
"""


class ProblemArchitectAgent(Agent):
    """Parse natural language research questions into structured ProblemSpec."""

    def __init__(self, backend: ClaudeCodeBackend | None = None) -> None:
        super().__init__(backend or ClaudeCodeBackend())

    def run(self, problem_description: str) -> ProblemSpec:
        """Convert a research question to a ProblemSpec.

        Args:
            problem_description: Natural language description of the research problem.

        Returns:
            Structured ProblemSpec ready for downstream agents.
        """
        logger.info("Problem Architect: parsing problem description")

        spec = self.backend.ask_structured(
            prompt=f"Convert this research question into a ProblemSpec:\n\n{problem_description}",
            response_model=ProblemSpec,
            system=SYSTEM_PROMPT,
        )

        logger.info(f"Problem Architect: generated spec with {len(spec.variables)} variables")
        return spec
