"""Tests for the auto-simulation generator agent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from simulating_anything.agents.simulation_generator import SimulationGeneratorAgent
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.problem_spec import ProblemSpec

# A valid simulation class source for testing
_VALID_SIM_CODE = '''
class TestSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)
        p = config.parameters
        self.k = p.get("k", 1.0)

    def reset(self, seed=None):
        self._state = np.array([1.0, 0.0])
        self._step_count = 0
        return self._state

    def step(self):
        dt = self.config.dt
        x, v = self._state
        a = -self.k * x
        self._state = np.array([x + v * dt, v + a * dt])
        self._step_count += 1
        return self._state

    def observe(self):
        return self._state
'''

# Code that produces NaN
_NAN_SIM_CODE = '''
class NaNSimulation(SimulationEnvironment):
    def __init__(self, config):
        super().__init__(config)

    def reset(self, seed=None):
        self._state = np.array([1.0])
        self._step_count = 0
        return self._state

    def step(self):
        self._state = np.array([float('nan')])
        self._step_count += 1
        return self._state

    def observe(self):
        return self._state
'''


class TestSimulationGeneratorAgent:
    """Tests for SimulationGeneratorAgent."""

    def test_init_no_backend(self):
        agent = SimulationGeneratorAgent(backend=None)
        assert agent.backend is None
        assert agent.max_retries == 5

    def test_init_with_params(self):
        agent = SimulationGeneratorAgent(
            backend=None,
            max_retries=3,
            output_dir="test_output",
        )
        assert agent.max_retries == 3

    def test_extract_code_plain(self):
        agent = SimulationGeneratorAgent(backend=None)
        code = agent._extract_code("class Foo:\n    pass")
        assert "class Foo:" in code

    def test_extract_code_markdown_python(self):
        agent = SimulationGeneratorAgent(backend=None)
        raw = "```python\nclass Foo:\n    pass\n```"
        code = agent._extract_code(raw)
        assert "class Foo:" in code
        assert "```" not in code

    def test_extract_code_markdown_generic(self):
        agent = SimulationGeneratorAgent(backend=None)
        raw = "```\nclass Foo:\n    pass\n```"
        code = agent._extract_code(raw)
        assert "class Foo:" in code

    def test_extract_code_strips_imports(self):
        agent = SimulationGeneratorAgent(backend=None)
        raw = "import numpy as np\nfrom foo import bar\nclass Foo:\n    pass"
        code = agent._extract_code(raw)
        assert "import numpy" not in code
        assert "from foo" not in code
        assert "class Foo:" in code

    def test_load_class_valid(self):
        agent = SimulationGeneratorAgent(backend=None)
        cls = agent._load_class(_VALID_SIM_CODE)
        assert issubclass(cls, SimulationEnvironment)
        assert cls.__name__ == "TestSimulation"

    def test_load_class_no_subclass(self):
        agent = SimulationGeneratorAgent(backend=None)
        with pytest.raises(ValueError, match="No SimulationEnvironment subclass"):
            agent._load_class("x = 42")

    def test_find_class_name(self):
        agent = SimulationGeneratorAgent(backend=None)
        assert agent._find_class_name(_VALID_SIM_CODE) == "TestSimulation"
        assert agent._find_class_name("x = 42") == "UnknownSimulation"

    def test_quick_smoke_test_valid(self):
        agent = SimulationGeneratorAgent(backend=None)
        cls = agent._load_class(_VALID_SIM_CODE)
        # Should not raise
        agent._quick_smoke_test(cls)

    def test_quick_smoke_test_nan(self):
        agent = SimulationGeneratorAgent(backend=None)
        cls = agent._load_class(_NAN_SIM_CODE)
        with pytest.raises(ValueError, match="NaN"):
            agent._quick_smoke_test(cls)

    def test_generate_no_backend_raises(self):
        agent = SimulationGeneratorAgent(backend=None)
        spec = ProblemSpec(title="Test", description="Test sim")
        with pytest.raises(RuntimeError, match="No LLM backend"):
            agent.generate(spec)

    def test_generate_with_mock_backend(self, tmp_path):
        mock_backend = MagicMock()
        mock_backend.ask.return_value = _VALID_SIM_CODE
        agent = SimulationGeneratorAgent(
            backend=mock_backend,
            output_dir=str(tmp_path),
        )
        spec = ProblemSpec(id="test_sim", title="Test oscillator")
        source, cls = agent.generate(spec)

        assert "TestSimulation" in source
        assert issubclass(cls, SimulationEnvironment)
        mock_backend.ask.assert_called_once()

        # Check file was saved
        saved = tmp_path / "test_sim.py"
        assert saved.exists()

    def test_generate_from_description(self, tmp_path):
        mock_backend = MagicMock()
        mock_backend.ask.return_value = _VALID_SIM_CODE
        agent = SimulationGeneratorAgent(
            backend=mock_backend,
            output_dir=str(tmp_path),
        )
        source, cls = agent.generate_from_description("harmonic oscillator")
        assert issubclass(cls, SimulationEnvironment)

    def test_retry_on_failure(self, tmp_path):
        mock_backend = MagicMock()
        # First call returns bad code, second returns good code
        mock_backend.ask.side_effect = [_NAN_SIM_CODE, _VALID_SIM_CODE]
        agent = SimulationGeneratorAgent(
            backend=mock_backend,
            max_retries=3,
            output_dir=str(tmp_path),
        )
        spec = ProblemSpec(id="retry_test", title="Test retry")
        source, cls = agent.generate(spec)
        assert mock_backend.ask.call_count == 2

    def test_build_prompt(self):
        agent = SimulationGeneratorAgent(backend=None)
        spec = ProblemSpec(
            title="SIR Epidemic",
            description="Susceptible-Infected-Recovered model",
            constraints=["Population conservation"],
            parameters={"beta": 0.3, "gamma": 0.1},
        )
        prompt = agent._build_prompt(spec)
        assert "SIR Epidemic" in prompt
        assert "Susceptible-Infected-Recovered" in prompt
        assert "Population conservation" in prompt

    def test_build_fix_prompt(self):
        agent = SimulationGeneratorAgent(backend=None)
        spec = ProblemSpec(title="Test")
        prompt = agent._build_fix_prompt(spec, "bad code", "ValueError: oops", 2)
        assert "ValueError: oops" in prompt
        assert "bad code" in prompt
        assert "attempt 2" in prompt

    def test_get_generated_simulation_info(self):
        agent = SimulationGeneratorAgent(backend=None)
        spec = ProblemSpec(id="test_id", title="Test")
        info = agent.get_generated_simulation_info(spec, _VALID_SIM_CODE, 1)
        assert info.problem_id == "test_id"
        assert info.class_name == "TestSimulation"
        assert info.generation_attempts == 1
