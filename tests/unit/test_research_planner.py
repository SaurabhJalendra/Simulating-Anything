"""Tests for the research planner agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from simulating_anything.agents.research_planner import ResearchPlannerAgent
from simulating_anything.types.campaign import (
    Experiment,
    ExperimentStatus,
    ResearchPlan,
)
from simulating_anything.types.discovery import Discovery


class TestResearchPlannerAgent:
    """Tests for ResearchPlannerAgent."""

    def test_init_no_backend(self):
        planner = ResearchPlannerAgent(backend=None)
        assert planner.backend is None
        assert planner.max_experiments == 10

    def test_init_custom(self):
        planner = ResearchPlannerAgent(backend=None, max_experiments=5)
        assert planner.max_experiments == 5

    def test_plan_fallback_no_backend(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = planner.plan("How do projectiles fly?")
        assert plan.question == "How do projectiles fly?"
        assert len(plan.experiments) == 3
        assert len(plan.sub_questions) == 3
        assert len(plan.success_criteria) == 2

    def test_fallback_plan_structure(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = planner.plan("Sand dune formation")

        # Check experiment structure
        exp1 = plan.experiments[0]
        assert exp1.id == "exp_1"
        assert exp1.description != ""
        assert exp1.depends_on == []

        exp2 = plan.experiments[1]
        assert "exp_1" in exp2.depends_on

        exp3 = plan.experiments[2]
        assert "exp_2" in exp3.depends_on

    def test_run_alias(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = planner.run("Test question")
        assert isinstance(plan, ResearchPlan)
        assert plan.question == "Test question"

    def test_plan_with_mock_backend(self):
        mock_backend = MagicMock()
        mock_backend.ask.return_value = json.dumps({
            "sub_questions": ["What is the trajectory?"],
            "experiments": [
                {
                    "id": "exp_1",
                    "description": "Test projectile motion",
                    "simulation_description": "Projectile with gravity",
                    "parameters_to_sweep": {"v0": [1.0, 50.0]},
                    "target_observables": ["range"],
                    "expected_outcome": "R = v^2 sin(2theta)/g",
                    "depends_on": [],
                }
            ],
            "success_criteria": ["Find range equation"],
        })

        planner = ResearchPlannerAgent(backend=mock_backend)
        plan = planner.plan("Projectile motion")

        assert plan.question == "Projectile motion"
        assert len(plan.experiments) == 1
        assert plan.experiments[0].parameters_to_sweep["v0"] == (1.0, 50.0)

    def test_plan_with_markdown_response(self):
        mock_backend = MagicMock()
        data = {
            "sub_questions": ["q1"],
            "experiments": [
                {
                    "id": "exp_1",
                    "description": "Test",
                    "simulation_description": "Sim",
                    "parameters_to_sweep": {},
                    "target_observables": [],
                    "expected_outcome": "",
                    "depends_on": [],
                }
            ],
            "success_criteria": [],
        }
        mock_backend.ask.return_value = f"```json\n{json.dumps(data)}\n```"

        planner = ResearchPlannerAgent(backend=mock_backend)
        plan = planner.plan("Test")
        assert len(plan.experiments) == 1

    def test_plan_with_invalid_json_falls_back(self):
        mock_backend = MagicMock()
        mock_backend.ask.return_value = "This is not JSON"

        planner = ResearchPlannerAgent(backend=mock_backend)
        plan = planner.plan("Test question")
        # Should fallback to default plan
        assert len(plan.experiments) == 3
        assert plan.question == "Test question"

    def test_replan_fallback_no_backend(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = ResearchPlan(
            question="Test",
            experiments=[
                Experiment(id="exp_1", status=ExperimentStatus.COMPLETED),
            ],
            current_step=0,
            max_steps=5,
        )

        findings = [Discovery(expression="x^2", confidence=0.9)]
        updated = planner.replan(plan, findings)
        assert updated.completed is True  # Has high-confidence finding
        assert updated.current_step == 1

    def test_replan_not_done_without_findings(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = ResearchPlan(
            question="Test",
            experiments=[
                Experiment(id="exp_1", status=ExperimentStatus.PENDING),
            ],
            current_step=0,
            max_steps=20,
        )

        updated = planner.replan(plan, [])
        assert updated.completed is False
        assert updated.current_step == 1

    def test_replan_stops_at_max_steps(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = ResearchPlan(
            question="Test",
            experiments=[
                Experiment(id="exp_1", status=ExperimentStatus.PENDING),
            ],
            current_step=19,
            max_steps=20,
        )

        updated = planner.replan(plan, [])
        assert updated.completed is True

    def test_replan_with_mock_backend(self):
        mock_backend = MagicMock()
        mock_backend.ask.return_value = json.dumps({
            "new_experiments": [
                {
                    "id": "exp_new",
                    "description": "Follow-up test",
                    "simulation_description": "Extended sim",
                    "parameters_to_sweep": {"k": [0.5, 5.0]},
                    "target_observables": ["frequency"],
                    "expected_outcome": "omega = sqrt(k/m)",
                    "depends_on": [],
                }
            ],
            "completed": False,
            "reasoning": "Need more experiments",
        })

        planner = ResearchPlannerAgent(backend=mock_backend)
        plan = ResearchPlan(question="Test", current_step=0)
        updated = planner.replan(plan, [])

        assert len(updated.experiments) == 1
        assert updated.experiments[0].id == "exp_new"
        assert updated.completed is False

    def test_replan_completed(self):
        mock_backend = MagicMock()
        mock_backend.ask.return_value = json.dumps({
            "new_experiments": [],
            "completed": True,
            "reasoning": "Success criteria met",
        })

        planner = ResearchPlannerAgent(backend=mock_backend)
        plan = ResearchPlan(question="Test")
        updated = planner.replan(plan, [])
        assert updated.completed is True

    def test_max_experiments_limit(self):
        mock_backend = MagicMock()
        experiments = [
            {
                "id": f"exp_{i}",
                "description": f"Test {i}",
                "simulation_description": f"Sim {i}",
                "parameters_to_sweep": {},
                "target_observables": [],
                "expected_outcome": "",
                "depends_on": [],
            }
            for i in range(20)
        ]
        mock_backend.ask.return_value = json.dumps({
            "sub_questions": [],
            "experiments": experiments,
            "success_criteria": [],
        })

        planner = ResearchPlannerAgent(backend=mock_backend, max_experiments=5)
        plan = planner.plan("Test")
        assert len(plan.experiments) <= 5

    def test_build_plan_prompt(self):
        planner = ResearchPlannerAgent(backend=None)
        prompt = planner._build_plan_prompt("How do waves work?")
        assert "How do waves work?" in prompt
        assert "experiments" in prompt.lower()

    def test_build_replan_prompt(self):
        planner = ResearchPlannerAgent(backend=None)
        plan = ResearchPlan(
            question="Waves",
            experiments=[
                Experiment(id="exp_1", description="Basic wave", status=ExperimentStatus.COMPLETED),
            ],
            current_step=1,
            max_steps=10,
        )
        findings = [Discovery(expression="v = sqrt(T/mu)", confidence=0.95)]
        prompt = planner._build_replan_prompt(plan, findings)
        assert "Waves" in prompt
        assert "exp_1" in prompt
        assert "sqrt(T/mu)" in prompt
