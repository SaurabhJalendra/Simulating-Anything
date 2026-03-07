"""Tests for equation-to-simulation auto-generator."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.equation_parser import (
    EquationSimulation,
    equations_from_strings,
    parse_equation_string,
)


class TestParseEquationString:
    """Test equation string parsing."""

    def test_d_parens_format(self):
        var, expr = parse_equation_string("d(x)/dt = -k*x")
        assert var == "x"
        assert expr == "-k*x"

    def test_d_no_parens_format(self):
        var, expr = parse_equation_string("dx/dt = v")
        assert var == "x"
        assert expr == "v"

    def test_prime_format(self):
        var, expr = parse_equation_string("x' = -k*x - c*v")
        assert var == "x"
        assert expr == "-k*x - c*v"

    def test_dot_format(self):
        var, expr = parse_equation_string("x_dot = v")
        assert var == "x"
        assert expr == "v"

    def test_whitespace_handling(self):
        var, expr = parse_equation_string("  d(y)/dt  =  x * (rho - z) - y  ")
        assert var == "y"
        assert expr == "x * (rho - z) - y"

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_equation_string("x = 5")

    def test_complex_expression(self):
        var, expr = parse_equation_string("d(z)/dt = x * y - beta * z")
        assert var == "z"
        assert expr == "x * y - beta * z"


class TestEquationsFromStrings:
    """Test batch equation parsing."""

    def test_lorenz_system(self):
        eqs = equations_from_strings([
            "d(x)/dt = sigma * (y - x)",
            "d(y)/dt = x * (rho - z) - y",
            "d(z)/dt = x * y - beta * z",
        ])
        assert len(eqs) == 3
        assert "x" in eqs
        assert eqs["x"] == "sigma * (y - x)"

    def test_mixed_formats(self):
        eqs = equations_from_strings([
            "d(x)/dt = v",
            "v' = -k*x",
        ])
        assert eqs["x"] == "v"
        assert eqs["v"] == "-k*x"


class TestEquationSimulation:
    """Test equation-based simulation."""

    def test_from_equations_dict(self):
        sim = EquationSimulation.from_equations(
            equations={"x": "v", "v": "-k*x"},
            parameters={"k": 4.0},
            initial_values={"x": 1.0, "v": 0.0},
        )
        assert isinstance(sim, EquationSimulation)
        assert sim.get_variable_names() == ["x", "v"]

    def test_from_equations_strings(self):
        sim = EquationSimulation.from_equations(
            equations=["d(x)/dt = v", "d(v)/dt = -k*x"],
            parameters={"k": 4.0},
            initial_values={"x": 1.0, "v": 0.0},
        )
        assert sim.get_variable_names() == ["x", "v"]

    def test_from_ode_string(self):
        sim = EquationSimulation.from_ode_string(
            """
            d(x)/dt = sigma * (y - x)
            d(y)/dt = x * (rho - z) - y
            d(z)/dt = x * y - beta * z
            """,
            parameters={"sigma": 10, "rho": 28, "beta": 2.667},
            initial_values={"x": 1.0, "y": 0.0, "z": 0.0},
        )
        assert len(sim.get_variable_names()) == 3

    def test_reset(self):
        sim = EquationSimulation.from_equations(
            equations={"x": "0", "v": "0"},
            initial_values={"x": 2.0, "v": 3.0},
        )
        state = sim.reset()
        assert state[0] == pytest.approx(2.0)
        assert state[1] == pytest.approx(3.0)

    def test_step_deterministic(self):
        sim = EquationSimulation.from_equations(
            equations={"x": "v", "v": "-4*x"},
            initial_values={"x": 1.0, "v": 0.0},
            dt=0.01,
        )
        sim.reset()
        s1 = sim.step().copy()
        sim.reset()
        s2 = sim.step().copy()
        np.testing.assert_array_equal(s1, s2)

    def test_harmonic_oscillator_frequency(self):
        """Equation-based SHO should have omega = sqrt(k)."""
        k = 9.0
        sim = EquationSimulation.from_equations(
            equations={"x": "v", "v": "-k*x"},
            parameters={"k": k},
            initial_values={"x": 1.0, "v": 0.0},
            dt=0.001, n_steps=10000,
        )
        sim.reset()
        positions = []
        for _ in range(10000):
            state = sim.step()
            positions.append(state[0])

        positions = np.array(positions)
        crossings = np.where(np.diff(np.sign(positions)))[0]
        if len(crossings) >= 4:
            half_periods = np.diff(crossings) * 0.001
            period = 2 * np.mean(half_periods)
            omega = 2 * np.pi / period
            assert omega == pytest.approx(np.sqrt(k), rel=0.02)

    def test_lorenz_attractor(self):
        """Equation-based Lorenz should produce bounded chaos."""
        sim = EquationSimulation.from_ode_string(
            """
            d(x)/dt = sigma * (y - x)
            d(y)/dt = x * (rho - z) - y
            d(z)/dt = x * y - beta * z
            """,
            parameters={"sigma": 10, "rho": 28, "beta": 2.667},
            initial_values={"x": 1.0, "y": 0.0, "z": 0.0},
            dt=0.01, n_steps=5000,
        )
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        assert np.all(np.isfinite(state))
        assert np.all(np.abs(state) < 100)

    def test_sir_conservation(self):
        """Equation-based SIR should conserve population."""
        sim = EquationSimulation.from_equations(
            equations={
                "S": "-beta * S * I / N",
                "I": "beta * S * I / N - gamma * I",
                "R": "gamma * I",
            },
            parameters={"beta": 0.3, "gamma": 0.1, "N": 1000.0},
            initial_values={"S": 990.0, "I": 10.0, "R": 0.0},
            dt=0.1, n_steps=2000,
        )
        sim.reset()
        for _ in range(2000):
            state = sim.step()
        total = sum(state)
        assert total == pytest.approx(1000.0, rel=0.01)

    def test_brusselator_steady_state(self):
        """Brusselator at steady state should remain there."""
        a, b = 1.0, 2.5
        sim = EquationSimulation.from_equations(
            equations={
                "u": "a - (b + 1)*u + u**2 * v",
                "v": "b*u - u**2 * v",
            },
            parameters={"a": a, "b": b},
            initial_values={"u": a, "v": b / a},
            dt=0.001, n_steps=1000,
        )
        sim.reset()
        for _ in range(1000):
            state = sim.step()
        assert state[0] == pytest.approx(a, abs=0.01)
        assert state[1] == pytest.approx(b / a, abs=0.01)

    def test_math_functions(self):
        """Trig functions should work in expressions."""
        sim = EquationSimulation.from_equations(
            equations={"x": "sin(t)"},
            initial_values={"x": 0.0},
            dt=0.01,
        )
        sim.reset()
        for _ in range(100):
            sim.step()
        # x should be approximately integral of sin(t) from 0 to 1
        # = 1 - cos(1) = 0.4597
        assert sim.observe()[0] == pytest.approx(1 - np.cos(1.0), abs=0.01)

    def test_forbidden_constructs(self):
        with pytest.raises(ValueError, match="Forbidden"):
            EquationSimulation.from_equations(
                equations={"x": "__import__('os').system('ls')"},
            )

    def test_get_equations(self):
        sim = EquationSimulation.from_equations(
            equations={"x": "v", "v": "-k*x"},
            parameters={"k": 1.0},
        )
        eqs = sim.get_equations()
        assert eqs["x"] == "v"
        assert eqs["v"] == "-k*x"

    def test_run_trajectory(self):
        sim = EquationSimulation.from_equations(
            equations={"x": "-x"},
            initial_values={"x": 1.0},
            dt=0.01, n_steps=100,
        )
        traj = sim.run(100)
        assert traj.states.shape == (101, 1)
        # Exponential decay: x(1) = e^(-1) ~ 0.368
        assert traj.states[-1, 0] == pytest.approx(np.exp(-1), abs=0.01)

    def test_time_dependent_equation(self):
        """Equations can reference 't' for time-dependent systems."""
        sim = EquationSimulation.from_equations(
            equations={"x": "cos(t)"},
            initial_values={"x": 0.0},
            dt=0.001, n_steps=1000,
        )
        sim.reset()
        for _ in range(1000):
            sim.step()
        # x(1) = integral of cos(t) from 0 to 1 = sin(1) ~ 0.841
        assert sim.observe()[0] == pytest.approx(np.sin(1.0), abs=0.01)

    def test_nonlinear_ode(self):
        """Van der Pol from equations should produce limit cycle."""
        sim = EquationSimulation.from_equations(
            equations={
                "x": "v",
                "v": "mu * (1 - x**2) * v - x",
            },
            parameters={"mu": 1.0},
            initial_values={"x": 0.1, "v": 0.0},
            dt=0.01, n_steps=5000,
        )
        sim.reset()
        # Run through transient
        for _ in range(3000):
            sim.step()
        # Check limit cycle amplitude
        max_x = 0.0
        for _ in range(2000):
            state = sim.step()
            max_x = max(max_x, abs(state[0]))
        assert max_x == pytest.approx(2.0, abs=0.3)
