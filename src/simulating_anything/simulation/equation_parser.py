"""Equation-to-simulation auto-generator.

Parses mathematical equations (string format) into working
SimulationEnvironment subclasses automatically. Supports ODE systems,
single ODEs, and algebraic equations.

Example:
    sim = EquationSimulation.from_equations(
        equations={"x": "v", "v": "-k*x - c*v"},
        parameters={"k": 4.0, "c": 0.2},
        initial_values={"x": 1.0, "v": 0.0},
    )
    sim.reset()
    for _ in range(1000):
        sim.step()
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import Domain, SimulationConfig

# Safe math namespace for eval
_SAFE_MATH_NS: dict[str, Any] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs,
    "pi": np.pi, "e": np.e,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "max": max, "min": min,
    "pow": pow,
}


def _validate_expression(expr: str) -> None:
    """Validate that an expression doesn't contain dangerous constructs."""
    forbidden = ["import", "__", "exec", "eval", "open", "compile", "globals", "locals"]
    for word in forbidden:
        if word in expr:
            raise ValueError(f"Forbidden construct '{word}' in expression: {expr}")


def parse_equation_string(equation: str) -> tuple[str, str]:
    """Parse 'd(var)/dt = expr' format into (var_name, expression).

    Supports formats:
        d(x)/dt = -k*x
        dx/dt = -k*x
        x' = -k*x
        x_dot = -k*x
    """
    equation = equation.strip()

    # Format: d(var)/dt = expr
    m = re.match(r"d\((\w+)\)/dt\s*=\s*(.+)", equation)
    if m:
        return m.group(1), m.group(2).strip()

    # Format: dvar/dt = expr
    m = re.match(r"d(\w+)/dt\s*=\s*(.+)", equation)
    if m:
        return m.group(1), m.group(2).strip()

    # Format: var' = expr
    m = re.match(r"(\w+)'\s*=\s*(.+)", equation)
    if m:
        return m.group(1), m.group(2).strip()

    # Format: var_dot = expr
    m = re.match(r"(\w+)_dot\s*=\s*(.+)", equation)
    if m:
        return m.group(1), m.group(2).strip()

    raise ValueError(
        f"Cannot parse equation: '{equation}'. "
        f"Expected format: 'd(var)/dt = expr', 'dvar/dt = expr', "
        f"\"var' = expr\", or 'var_dot = expr'"
    )


def equations_from_strings(equation_strings: list[str]) -> dict[str, str]:
    """Parse a list of equation strings into a {var: expr} dict.

    Args:
        equation_strings: List of equations in 'd(var)/dt = expr' format.

    Returns:
        Dict mapping variable names to their rate expressions.
    """
    result = {}
    for eq in equation_strings:
        var, expr = parse_equation_string(eq)
        result[var] = expr
    return result


class EquationSimulation(SimulationEnvironment):
    """A simulation built from parsed mathematical equations.

    Evaluates rate expressions using numpy-safe eval and integrates
    with RK4.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self._var_names: list[str] = []
        self._equations: dict[str, str] = {}
        self._initial_values: dict[str, float] = {}
        self._compiled: dict[str, Any] = {}

    @classmethod
    def from_equations(
        cls,
        equations: dict[str, str] | list[str],
        parameters: dict[str, float] | None = None,
        initial_values: dict[str, float] | None = None,
        dt: float = 0.01,
        n_steps: int = 1000,
    ) -> EquationSimulation:
        """Create a simulation from mathematical equations.

        Args:
            equations: Either a dict {var_name: rate_expression} or a list
                of equation strings in 'd(var)/dt = expr' format.
            parameters: Parameter values used in expressions.
            initial_values: Initial values for state variables.
            dt: Timestep for integration.
            n_steps: Default number of simulation steps.

        Returns:
            A configured EquationSimulation instance.

        Example:
            # Dict form:
            sim = EquationSimulation.from_equations(
                equations={"x": "v", "v": "-k*x - c*v"},
                parameters={"k": 4.0, "c": 0.2},
                initial_values={"x": 1.0, "v": 0.0},
            )

            # String form:
            sim = EquationSimulation.from_equations(
                equations=["d(x)/dt = v", "d(v)/dt = -k*x - c*v"],
                parameters={"k": 4.0, "c": 0.2},
                initial_values={"x": 1.0, "v": 0.0},
            )
        """
        if isinstance(equations, list):
            equations = equations_from_strings(equations)

        parameters = parameters or {}
        initial_values = initial_values or {}

        # Validate expressions
        for var, expr in equations.items():
            _validate_expression(expr)

        config = SimulationConfig(
            domain=Domain.CUSTOM,
            parameters=parameters,
            dt=dt,
            n_steps=n_steps,
        )

        sim = cls(config)
        sim._var_names = list(equations.keys())
        sim._equations = dict(equations)
        sim._initial_values = dict(initial_values)

        # Pre-compile expressions
        for var, expr in equations.items():
            sim._compiled[var] = compile(expr, f"<equation:{var}>", "eval")

        return sim

    @classmethod
    def from_ode_string(
        cls,
        ode_string: str,
        parameters: dict[str, float] | None = None,
        initial_values: dict[str, float] | None = None,
        dt: float = 0.01,
        n_steps: int = 1000,
    ) -> EquationSimulation:
        """Create from a multi-line ODE string.

        Args:
            ode_string: Multi-line string with one equation per line.
                Lines starting with '#' are ignored.
            parameters: Parameter values.
            initial_values: Initial state.

        Example:
            sim = EquationSimulation.from_ode_string('''
                d(x)/dt = sigma * (y - x)
                d(y)/dt = x * (rho - z) - y
                d(z)/dt = x * y - beta * z
            ''', parameters={"sigma": 10, "rho": 28, "beta": 2.667})
        """
        lines = [
            line.strip()
            for line in ode_string.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        return cls.from_equations(
            equations=lines,
            parameters=parameters,
            initial_values=initial_values,
            dt=dt,
            n_steps=n_steps,
        )

    def _evaluate(self, state_dict: dict[str, float], t: float) -> np.ndarray:
        """Evaluate all rate expressions at given state."""
        # Build evaluation namespace
        ns = dict(_SAFE_MATH_NS)
        ns.update(self.config.parameters)
        ns.update(state_dict)
        ns["t"] = t

        rates = np.zeros(len(self._var_names))
        for i, var in enumerate(self._var_names):
            rates[i] = float(eval(self._compiled[var], {"__builtins__": {}}, ns))
        return rates

    def _state_to_dict(self, state: np.ndarray) -> dict[str, float]:
        return {name: float(state[i]) for i, name in enumerate(self._var_names)}

    def reset(self, seed: int | None = None) -> np.ndarray:
        state = np.array([
            self._initial_values.get(v, 0.0) for v in self._var_names
        ])
        self._state = state
        self._step_count = 0
        return state

    def step(self) -> np.ndarray:
        dt = self.config.dt
        t = self._step_count * dt
        s = self._state

        def f(state, t_val):
            return self._evaluate(self._state_to_dict(state), t_val)

        k1 = f(s, t)
        k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(s + dt * k3, t + dt)

        self._state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        return self._state

    def get_variable_names(self) -> list[str]:
        return list(self._var_names)

    def get_equations(self) -> dict[str, str]:
        return dict(self._equations)
