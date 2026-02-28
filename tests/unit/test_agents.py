"""Tests for agent base classes and domain classifier rules."""

import json
from unittest.mock import MagicMock, patch

import pytest

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.agents.domain_classifier import DomainClassifierAgent
from simulating_anything.agents.communicator import CommunicatorAgent
from simulating_anything.types.discovery import (
    Discovery,
    DiscoveryReport,
    DiscoveryStatus,
    DiscoveryType,
)
from simulating_anything.types.problem_spec import (
    Objective,
    ObjectiveType,
    ProblemSpec,
    Variable,
    VariableType,
)
from simulating_anything.types.simulation import Domain


class TestClaudeCodeBackend:
    def test_init_defaults(self):
        backend = ClaudeCodeBackend()
        assert backend.max_retries == 3
        assert backend.timeout == 120

    @patch("subprocess.run")
    def test_ask_success(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"result": "Hello world"}),
            returncode=0,
        )
        backend = ClaudeCodeBackend()
        result = backend.ask("test prompt")
        assert result == "Hello world"

    @patch("subprocess.run")
    def test_ask_with_system_prompt(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"result": "response"}),
            returncode=0,
        )
        backend = ClaudeCodeBackend()
        backend.ask("test", system="You are a scientist")
        cmd = mock_run.call_args[0][0]
        # System prompt is prepended to user prompt (CLI has no --system flag)
        prompt_arg = cmd[cmd.index("-p") + 1]
        assert "You are a scientist" in prompt_arg
        assert "test" in prompt_arg

    @patch("subprocess.run")
    def test_ask_structured(self, mock_run):
        spec_json = ProblemSpec(id="t1", title="Test Problem").model_dump_json()
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"result": spec_json}),
            returncode=0,
        )
        backend = ClaudeCodeBackend()
        result = backend.ask_structured("parse this", ProblemSpec)
        assert isinstance(result, ProblemSpec)
        assert result.id == "t1"


class TestAgentBase:
    def test_concrete_agent(self):
        class TestAgent(Agent):
            def run(self):
                return "done"

        agent = TestAgent()
        assert agent.run() == "done"
        assert "TestAgent" in repr(agent)


class TestDomainClassifier:
    def test_reaction_diffusion_keywords(self):
        spec = ProblemSpec(
            title="Gray-Scott reaction-diffusion patterns",
            description="How do concentration fields evolve with diffusion?",
            physics_domains=["reaction_diffusion"],
            variables=[Variable(name="u", type=VariableType.SCALAR_FIELD)],
        )
        agent = DomainClassifierAgent(backend=MagicMock())
        result = agent._rule_based(spec)
        assert result.domain == Domain.REACTION_DIFFUSION

    def test_rigid_body_keywords(self):
        spec = ProblemSpec(
            title="Projectile trajectory with drag",
            description="Mass launched at velocity under gravity force",
            variables=[Variable(name="position", type=VariableType.VECTOR)],
        )
        agent = DomainClassifierAgent(backend=MagicMock())
        result = agent._rule_based(spec)
        assert result.domain == Domain.RIGID_BODY

    def test_agent_based_keywords(self):
        spec = ProblemSpec(
            title="Lotka-Volterra predator-prey dynamics",
            description="Population ecology with species competition",
            variables=[Variable(name="prey", type=VariableType.SCALAR)],
        )
        agent = DomainClassifierAgent(backend=MagicMock())
        result = agent._rule_based(spec)
        assert result.domain == Domain.AGENT_BASED

    def test_high_confidence_skips_llm(self):
        spec = ProblemSpec(
            title="reaction diffusion chemical concentration field pattern",
            description="diffusion activator inhibitor turing morphogen",
        )
        mock_backend = MagicMock()
        agent = DomainClassifierAgent(backend=mock_backend)
        result = agent.run(spec)
        # Should NOT call LLM if rule confidence is high enough
        mock_backend.ask_structured.assert_not_called()


class TestCommunicator:
    def test_template_report(self):
        report = DiscoveryReport(
            discoveries=[
                Discovery(
                    id="d1",
                    type=DiscoveryType.GOVERNING_EQUATION,
                    confidence=0.95,
                    expression="du/dt = D*lap(u)",
                    status=DiscoveryStatus.CONFIRMED,
                )
            ],
            summary="Found governing equation",
            n_trajectories_analyzed=50,
        )
        agent = CommunicatorAgent()
        md = agent.run(report)
        assert "# Discovery Report" in md
        assert "du/dt = D*lap(u)" in md
        assert "95.0%" in md
        assert "CONFIRMED" in md

    def test_empty_report(self):
        report = DiscoveryReport()
        agent = CommunicatorAgent()
        md = agent.run(report)
        assert "# Discovery Report" in md
        assert "Discoveries found:** 0" in md
