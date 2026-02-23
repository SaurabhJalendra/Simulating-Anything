"""Domain Classifier Agent: ProblemSpec -> DomainClassification."""

from __future__ import annotations

import logging

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.types.problem_spec import ProblemSpec
from simulating_anything.types.simulation import (
    Domain,
    DomainClassification,
    SimulationBackend,
)

logger = logging.getLogger(__name__)

# Rule-based keyword mapping for fast path
_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.REACTION_DIFFUSION: [
        "reaction", "diffusion", "concentration", "chemical", "pattern",
        "morphogen", "turing", "gray-scott", "activator", "inhibitor",
        "spatiotemporal", "field",
    ],
    Domain.RIGID_BODY: [
        "projectile", "pendulum", "rigid", "body", "trajectory", "orbit",
        "gravity", "drag", "mass", "spring", "collision", "velocity",
        "acceleration", "force", "newtonian", "mechanics",
    ],
    Domain.AGENT_BASED: [
        "population", "predator", "prey", "lotka", "volterra", "epidemic",
        "sir", "agent", "species", "ecology", "competition", "cooperation",
        "ode", "dynamical system",
    ],
}

_DOMAIN_BACKENDS: dict[Domain, SimulationBackend] = {
    Domain.REACTION_DIFFUSION: SimulationBackend.JAX_FD,
    Domain.RIGID_BODY: SimulationBackend.JAX_FD,
    Domain.AGENT_BASED: SimulationBackend.DIFFRAX,
}

SYSTEM_PROMPT = """\
You are the Domain Classifier Agent. Given a ProblemSpec, classify which
simulation domain it belongs to. Choose from:
- reaction_diffusion: PDE systems with spatial fields (concentration, temperature)
- rigid_body: Newtonian mechanics, projectiles, pendulums, orbital dynamics
- agent_based: ODE population dynamics, epidemiology, ecology

Return a DomainClassification with domain, confidence (0-1), backend, and rationale.
"""


class DomainClassifierAgent(Agent):
    """Classify a ProblemSpec into a simulation domain.

    Uses rule-based keyword matching first (fast path).
    Falls back to LLM classification when confidence is low.
    """

    def __init__(self, backend: ClaudeCodeBackend | None = None) -> None:
        super().__init__(backend or ClaudeCodeBackend())

    def run(self, spec: ProblemSpec) -> DomainClassification:
        """Classify the problem domain.

        Args:
            spec: Structured problem specification.

        Returns:
            DomainClassification with domain, backend, and confidence.
        """
        # Fast path: keyword matching
        result = self._rule_based(spec)
        if result.confidence >= 0.7:
            logger.info(
                f"Domain Classifier (rules): {result.domain.value} "
                f"(confidence={result.confidence:.2f})"
            )
            return result

        # Slow path: LLM classification
        logger.info("Domain Classifier: low rule confidence, using LLM fallback")
        return self._llm_classify(spec)

    def _rule_based(self, spec: ProblemSpec) -> DomainClassification:
        """Score domains by keyword presence in spec text."""
        text = " ".join([
            spec.title.lower(),
            spec.description.lower(),
            " ".join(spec.physics_domains),
            " ".join(v.name.lower() for v in spec.variables),
        ])

        scores: dict[Domain, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in text)

        total = sum(scores.values()) or 1
        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain] / total

        return DomainClassification(
            domain=best_domain,
            confidence=confidence,
            backend=_DOMAIN_BACKENDS[best_domain],
            rationale=f"Keyword match: {scores[best_domain]}/{total} keywords",
        )

    def _llm_classify(self, spec: ProblemSpec) -> DomainClassification:
        """Use LLM to classify when rules are uncertain."""
        result = self.backend.ask_structured(
            prompt=f"Classify this problem:\n\n{spec.model_dump_json(indent=2)}",
            response_model=DomainClassification,
            system=SYSTEM_PROMPT,
        )
        return result
