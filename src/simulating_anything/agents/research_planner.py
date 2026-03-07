"""Research planner agent for autonomous discovery campaigns.

Plans multi-step research campaigns from high-level questions,
decomposes them into testable experiments, and replans based on findings.
"""

from __future__ import annotations

import json
import logging
import textwrap

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.types.campaign import (
    Experiment,
    ExperimentStatus,
    ResearchPlan,
)
from simulating_anything.types.discovery import Discovery

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a research scientist planning computational experiments.
    Given a question about a physical, biological, or mathematical phenomenon,
    decompose it into testable sub-questions and design experiments.

    For each experiment, specify:
    1. A description of what to test
    2. A simulation description (what physical system to build)
    3. Parameters to sweep (name -> [min, max])
    4. What observables to measure
    5. What outcome you expect
    6. Dependencies on other experiments (by ID)

    Think step by step:
    - What are the key variables?
    - What are the governing equations (if known)?
    - What parameter ranges are physically reasonable?
    - What would constitute a discovery?

    Respond with ONLY valid JSON matching the schema provided.
""")

_REPLAN_PROMPT = textwrap.dedent("""\
    You are a research scientist reviewing experimental findings and deciding
    what to investigate next.

    Given the current research plan and findings so far, update the plan:
    1. Mark completed experiments
    2. Add new experiments suggested by the findings
    3. Decide if success criteria are met
    4. If no new findings for 3+ steps, recommend stopping

    Think about:
    - Do findings suggest new hypotheses to test?
    - Are there parameter regimes we haven't explored?
    - Do any equations need validation at different scales?

    Respond with ONLY valid JSON matching the schema provided.
""")


class ResearchPlannerAgent(Agent):
    """Plans multi-step research campaigns from high-level questions.

    Uses an LLM to decompose questions into testable experiments,
    then replans based on findings from completed experiments.
    """

    def __init__(
        self,
        backend: ClaudeCodeBackend | None = None,
        max_experiments: int = 10,
    ) -> None:
        super().__init__(backend)
        self.max_experiments = max_experiments

    def run(self, question: str) -> ResearchPlan:
        """Create an initial research plan from a question."""
        return self.plan(question)

    def plan(self, question: str) -> ResearchPlan:
        """Create an initial research plan from a high-level question.

        Args:
            question: A natural language question about a phenomenon.

        Returns:
            A ResearchPlan with experiments to run.
        """
        prompt = self._build_plan_prompt(question)

        if self.backend is None:
            return self._fallback_plan(question)

        try:
            raw = self.backend.ask(prompt, system=_SYSTEM_PROMPT)
            return self._parse_plan(raw, question)
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using fallback plan")
            return self._fallback_plan(question)

    def replan(
        self,
        plan: ResearchPlan,
        findings: list[Discovery],
    ) -> ResearchPlan:
        """Update a research plan based on experimental findings.

        Args:
            plan: The current research plan.
            findings: Discoveries made so far.

        Returns:
            Updated ResearchPlan with new experiments or completion status.
        """
        prompt = self._build_replan_prompt(plan, findings)

        if self.backend is None:
            return self._fallback_replan(plan, findings)

        try:
            raw = self.backend.ask(prompt, system=_REPLAN_PROMPT)
            return self._parse_replan(raw, plan)
        except Exception as e:
            logger.warning(f"LLM replanning failed: {e}, using fallback replan")
            return self._fallback_replan(plan, findings)

    def _build_plan_prompt(self, question: str) -> str:
        """Build the initial planning prompt."""
        schema = {
            "sub_questions": ["string"],
            "experiments": [
                {
                    "id": "exp_1",
                    "description": "string",
                    "simulation_description": "string",
                    "parameters_to_sweep": {"param_name": [0.0, 1.0]},
                    "target_observables": ["string"],
                    "expected_outcome": "string",
                    "depends_on": [],
                }
            ],
            "success_criteria": ["string"],
        }
        return (
            f"Research question: {question}\n\n"
            f"Create a research plan with up to {self.max_experiments} experiments.\n\n"
            f"Respond with JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )

    def _build_replan_prompt(
        self, plan: ResearchPlan, findings: list[Discovery]
    ) -> str:
        """Build the replanning prompt with current state."""
        findings_text = "\n".join(
            f"- {d.expression or d.description} (confidence: {d.confidence:.2f})"
            for d in findings
        )
        experiments_text = "\n".join(
            f"- {e.id}: {e.description} [{e.status.value}]"
            for e in plan.experiments
        )
        return (
            f"Original question: {plan.question}\n\n"
            f"Current experiments:\n{experiments_text}\n\n"
            f"Findings so far:\n{findings_text or 'None yet'}\n\n"
            f"Current step: {plan.current_step}/{plan.max_steps}\n\n"
            f"Should we continue? If yes, what new experiments? "
            f"If success criteria are met, set completed=true.\n\n"
            f"Respond with JSON: {{'new_experiments': [...], 'completed': bool, "
            f"'reasoning': 'string'}}"
        )

    def _parse_plan(self, raw: str, question: str) -> ResearchPlan:
        """Parse LLM response into a ResearchPlan."""
        text = raw.strip()
        # Extract JSON from markdown fences if present
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON, using fallback")
            return self._fallback_plan(question)

        experiments = []
        for exp_data in data.get("experiments", []):
            params = {}
            for k, v in exp_data.get("parameters_to_sweep", {}).items():
                if isinstance(v, list) and len(v) == 2:
                    params[k] = (float(v[0]), float(v[1]))
            experiments.append(Experiment(
                id=exp_data.get("id", f"exp_{len(experiments)+1}"),
                description=exp_data.get("description", ""),
                simulation_description=exp_data.get("simulation_description", ""),
                parameters_to_sweep=params,
                target_observables=exp_data.get("target_observables", []),
                expected_outcome=exp_data.get("expected_outcome", ""),
                depends_on=exp_data.get("depends_on", []),
            ))

        return ResearchPlan(
            question=question,
            sub_questions=data.get("sub_questions", []),
            experiments=experiments[:self.max_experiments],
            success_criteria=data.get("success_criteria", []),
            current_step=0,
            max_steps=max(20, len(experiments) * 2),
        )

    def _parse_replan(self, raw: str, plan: ResearchPlan) -> ResearchPlan:
        """Parse replanning response and update the plan."""
        text = raw.strip()
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse replan JSON")
            plan.current_step += 1
            return plan

        if data.get("completed", False):
            plan.completed = True
            return plan

        # Add new experiments
        for exp_data in data.get("new_experiments", []):
            params = {}
            for k, v in exp_data.get("parameters_to_sweep", {}).items():
                if isinstance(v, list) and len(v) == 2:
                    params[k] = (float(v[0]), float(v[1]))
            plan.experiments.append(Experiment(
                id=exp_data.get("id", f"exp_{len(plan.experiments)+1}"),
                description=exp_data.get("description", ""),
                simulation_description=exp_data.get("simulation_description", ""),
                parameters_to_sweep=params,
                target_observables=exp_data.get("target_observables", []),
                expected_outcome=exp_data.get("expected_outcome", ""),
                depends_on=exp_data.get("depends_on", []),
            ))

        plan.current_step += 1
        return plan

    def _fallback_plan(self, question: str) -> ResearchPlan:
        """Create a reasonable default plan without LLM.

        Generates a generic 3-experiment plan: baseline behavior,
        parameter sweep, and equation discovery.
        """
        return ResearchPlan(
            question=question,
            sub_questions=[
                "What are the governing equations of the system?",
                "How does the system respond to parameter changes?",
                "Are there critical thresholds or phase transitions?",
            ],
            experiments=[
                Experiment(
                    id="exp_1",
                    description="Baseline simulation with default parameters",
                    simulation_description=question,
                    parameters_to_sweep={},
                    target_observables=["state_trajectory", "energy"],
                    expected_outcome="Stable dynamics with interpretable behavior",
                ),
                Experiment(
                    id="exp_2",
                    description="Parameter sweep to map system behavior",
                    simulation_description=question,
                    parameters_to_sweep={},
                    target_observables=["steady_state", "oscillation_period"],
                    expected_outcome="Identify parameter dependence of key observables",
                    depends_on=["exp_1"],
                ),
                Experiment(
                    id="exp_3",
                    description="Symbolic regression for governing equations",
                    simulation_description=question,
                    parameters_to_sweep={},
                    target_observables=["equations", "r_squared"],
                    expected_outcome="Recover interpretable equations with R^2 > 0.95",
                    depends_on=["exp_2"],
                ),
            ],
            success_criteria=[
                "At least one governing equation discovered with R^2 > 0.95",
                "Parameter dependence mapped for primary variables",
            ],
            max_steps=20,
        )

    def _fallback_replan(
        self, plan: ResearchPlan, findings: list[Discovery]
    ) -> ResearchPlan:
        """Update plan without LLM based on simple heuristics."""
        plan.current_step += 1

        # Check if all experiments are done
        all_done = all(
            e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.SKIPPED)
            for e in plan.experiments
        )

        # Check for success: any high-confidence discovery
        has_discovery = any(d.confidence > 0.8 for d in findings)

        if all_done or has_discovery or plan.current_step >= plan.max_steps:
            plan.completed = True

        return plan
