"""Auto-simulation generator agent.

Generates SimulationEnvironment subclasses from natural language descriptions
using an LLM, then dynamically loads them via exec().
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.campaign import GeneratedSimulation
from simulating_anything.types.problem_spec import ProblemSpec
from simulating_anything.types.simulation import SimulationConfig

logger = logging.getLogger(__name__)

# Few-shot examples for the LLM: minimal but complete simulation classes
_EXAMPLE_SIR = textwrap.dedent("""\
    class SIRSimulation(SimulationEnvironment):
        def __init__(self, config):
            super().__init__(config)
            p = config.parameters
            self.beta = p.get("beta", 0.3)
            self.gamma = p.get("gamma", 0.1)
            self.S_0 = p.get("S_0", 0.99)
            self.I_0 = p.get("I_0", 0.01)

        def reset(self, seed=None):
            self._state = np.array([self.S_0, self.I_0, 1.0 - self.S_0 - self.I_0])
            self._step_count = 0
            return self._state

        def step(self):
            dt = self.config.dt
            y = self._state
            k1 = self._derivatives(y)
            k2 = self._derivatives(y + 0.5 * dt * k1)
            k3 = self._derivatives(y + 0.5 * dt * k2)
            k4 = self._derivatives(y + dt * k3)
            self._state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

        def _derivatives(self, y):
            S, I, R = y
            dS = -self.beta * S * I
            dI = self.beta * S * I - self.gamma * I
            dR = self.gamma * I
            return np.array([dS, dI, dR])
""")

_EXAMPLE_OSCILLATOR = textwrap.dedent("""\
    class OscillatorSimulation(SimulationEnvironment):
        def __init__(self, config):
            super().__init__(config)
            p = config.parameters
            self.k = p.get("k", 1.0)
            self.m = p.get("m", 1.0)
            self.c = p.get("c", 0.0)
            self.x_0 = p.get("x_0", 1.0)
            self.v_0 = p.get("v_0", 0.0)

        def reset(self, seed=None):
            self._state = np.array([self.x_0, self.v_0])
            self._step_count = 0
            return self._state

        def step(self):
            dt = self.config.dt
            y = self._state
            k1 = self._derivatives(y)
            k2 = self._derivatives(y + 0.5 * dt * k1)
            k3 = self._derivatives(y + 0.5 * dt * k2)
            k4 = self._derivatives(y + dt * k3)
            self._state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

        def _derivatives(self, y):
            x, v = y
            a = (-self.k * x - self.c * v) / self.m
            return np.array([v, a])
""")

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a physics simulation code generator. Given a description of a
    physical phenomenon, write a Python class that simulates it.

    REQUIREMENTS:
    1. The class MUST inherit from SimulationEnvironment
    2. The class MUST implement: __init__(self, config), reset(self, seed=None),
       step(self), observe(self)
    3. Use RK4 integration for ODEs (4th-order Runge-Kutta)
    4. Use numpy ONLY (no scipy, no JAX, no external libraries)
    5. Parameters come from config.parameters dict with sensible defaults
    6. State is a 1D numpy array stored in self._state
    7. step() increments self._step_count
    8. Timestep is self.config.dt

    TEMPLATE:
    ```python
    class <Name>Simulation(SimulationEnvironment):
        def __init__(self, config):
            super().__init__(config)
            p = config.parameters
            self.param1 = p.get("param1", default_value)
            ...

        def reset(self, seed=None):
            self._state = np.array([...])
            self._step_count = 0
            return self._state

        def step(self):
            dt = self.config.dt
            y = self._state
            k1 = self._derivatives(y)
            k2 = self._derivatives(y + 0.5 * dt * k1)
            k3 = self._derivatives(y + 0.5 * dt * k2)
            k4 = self._derivatives(y + dt * k3)
            self._state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

        def _derivatives(self, y):
            # Implement the ODEs here
            return np.array([...])
    ```

    EXAMPLES:

    Example 1 (SIR epidemic):
    {sir_example}

    Example 2 (Harmonic oscillator):
    {oscillator_example}

    IMPORTANT:
    - Return ONLY the Python class code, no imports, no markdown fences
    - The class will be loaded in an environment where numpy (as np) and
      SimulationEnvironment are already available
    - Choose physically reasonable default parameter values
    - Name the class descriptively, ending with "Simulation"
""").format(sir_example=_EXAMPLE_SIR, oscillator_example=_EXAMPLE_OSCILLATOR)


class SimulationGeneratorAgent(Agent):
    """Generates simulation code from natural language descriptions.

    Uses an LLM to write a SimulationEnvironment subclass, then loads it
    dynamically via exec(). Includes retry logic for debugging failed code.
    """

    def __init__(
        self,
        backend: ClaudeCodeBackend | None = None,
        max_retries: int = 5,
        output_dir: str = "output/generated_sims",
    ) -> None:
        super().__init__(backend)
        self.max_retries = max_retries
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, spec: ProblemSpec) -> tuple[str, type]:
        """Generate a simulation class from a ProblemSpec.

        Returns:
            Tuple of (source_code, simulation_class).
        """
        return self.generate(spec)

    def generate(self, spec: ProblemSpec) -> tuple[str, type]:
        """Generate and load a simulation class.

        Args:
            spec: The problem specification describing the phenomenon.

        Returns:
            Tuple of (source_code, loaded_class).

        Raises:
            RuntimeError: If code generation fails after max_retries.
        """
        prompt = self._build_prompt(spec)
        source_code = ""
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            if attempt == 1:
                current_prompt = prompt
            else:
                current_prompt = self._build_fix_prompt(
                    spec, source_code, last_error, attempt
                )

            logger.info(f"Generation attempt {attempt}/{self.max_retries}")

            if self.backend is None:
                raise RuntimeError("No LLM backend configured for code generation")

            raw = self.backend.ask(current_prompt, system=_SYSTEM_PROMPT)
            source_code = self._extract_code(raw)

            try:
                sim_class = self._load_class(source_code)
                self._quick_smoke_test(sim_class)

                # Save successful code
                problem_id = spec.id or "unknown"
                save_path = self.output_dir / f"{problem_id}.py"
                self._save_code(source_code, save_path)

                logger.info(f"Successfully generated simulation on attempt {attempt}")
                return source_code, sim_class

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"Attempt {attempt} failed: {last_error}")

        raise RuntimeError(
            f"Failed to generate valid simulation after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_from_description(self, description: str) -> tuple[str, type]:
        """Generate a simulation from a plain text description.

        Convenience method that creates a minimal ProblemSpec.
        """
        spec = ProblemSpec(title=description, description=description)
        return self.generate(spec)

    def get_generated_simulation_info(
        self, spec: ProblemSpec, source_code: str, attempts: int
    ) -> GeneratedSimulation:
        """Create a GeneratedSimulation metadata object."""
        class_name = self._find_class_name(source_code)
        return GeneratedSimulation(
            problem_id=spec.id or "unknown",
            source_code=source_code,
            class_name=class_name,
            generation_attempts=attempts,
        )

    def _build_prompt(self, spec: ProblemSpec) -> str:
        """Build the initial generation prompt from a ProblemSpec."""
        parts = [f"Generate a simulation for: {spec.title}"]
        if spec.description:
            parts.append(f"\nDescription: {spec.description}")
        if spec.variables:
            var_names = [v.name for v in spec.variables]
            parts.append(f"\nState variables: {', '.join(var_names)}")
        if spec.parameters:
            parts.append(f"\nParameters: {spec.parameters}")
        if spec.constraints:
            parts.append(f"\nConstraints: {', '.join(spec.constraints)}")
        return "\n".join(parts)

    def _build_fix_prompt(
        self, spec: ProblemSpec, code: str, error: str, attempt: int
    ) -> str:
        """Build a prompt to fix failed code."""
        return (
            f"The previous simulation code for '{spec.title}' failed with:\n"
            f"Error: {error}\n\n"
            f"Previous code:\n```python\n{code}\n```\n\n"
            f"Fix the code. Return ONLY the corrected Python class, no imports, "
            f"no markdown. This is attempt {attempt}."
        )

    def _extract_code(self, raw: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences."""
        text = raw.strip()
        # Remove markdown code fences
        if "```python" in text:
            match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        if "```" in text:
            match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        # Remove any import lines (we provide the namespace)
        lines = text.split("\n")
        filtered = [l for l in lines if not l.strip().startswith(("import ", "from "))]
        return "\n".join(filtered).strip()

    def _load_class(self, source_code: str) -> type:
        """Dynamically load a class from source code using exec().

        Returns the first SimulationEnvironment subclass found.
        """
        namespace: dict[str, Any] = {
            "np": np,
            "numpy": np,
            "SimulationEnvironment": SimulationEnvironment,
            "SimulationConfig": SimulationConfig,
        }
        exec(source_code, namespace)  # noqa: S102

        # Find the simulation class in the namespace
        for name, obj in namespace.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, SimulationEnvironment)
                and obj is not SimulationEnvironment
            ):
                return obj

        raise ValueError("No SimulationEnvironment subclass found in generated code")

    def _find_class_name(self, source_code: str) -> str:
        """Extract the class name from source code."""
        match = re.search(r"class\s+(\w+)\s*\(", source_code)
        return match.group(1) if match else "UnknownSimulation"

    def _quick_smoke_test(self, sim_class: type) -> None:
        """Run a minimal smoke test on the generated simulation class."""
        config = SimulationConfig(
            parameters={},
            dt=0.01,
            n_steps=10,
        )
        sim = sim_class(config)
        state = sim.reset(seed=42)

        if not isinstance(state, np.ndarray):
            raise TypeError(f"reset() returned {type(state)}, expected np.ndarray")
        if state.size == 0:
            raise ValueError("reset() returned empty array")

        for _ in range(10):
            new_state = sim.step()
            if not isinstance(new_state, np.ndarray):
                raise TypeError(f"step() returned {type(new_state)}")
            if np.any(np.isnan(new_state)):
                raise ValueError("step() produced NaN")
            if np.any(np.isinf(new_state)):
                raise ValueError("step() produced Inf")

    def _save_code(self, source_code: str, path: Path) -> None:
        """Save generated code to a file for inspection."""
        full_code = (
            '"""Auto-generated simulation code."""\n\n'
            "from __future__ import annotations\n\n"
            "import numpy as np\n\n"
            "from simulating_anything.simulation.base import SimulationEnvironment\n"
            "from simulating_anything.types.simulation import SimulationConfig\n\n\n"
            f"{source_code}\n"
        )
        path.write_text(full_code)
        logger.info(f"Saved generated simulation to {path}")
