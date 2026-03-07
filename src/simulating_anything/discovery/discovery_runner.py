"""Runner for new discovery attempts on open/unsolved problems.

Uses parameter sweeps + analysis to search for scaling laws,
phase boundaries, and empirical equations in problems without
known analytical solutions.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simulating_anything.discovery.open_problems import OpenProblem, get_problem
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryAttempt:
    """Record of a single discovery attempt."""

    problem_name: str
    parameter_name: str
    parameter_values: list[float] = field(default_factory=list)
    observable_name: str = ""
    observable_values: list[float] = field(default_factory=list)
    candidate_equation: str = ""
    r_squared: float = 0.0
    method: str = ""
    runtime_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Full result of a discovery campaign on an open problem."""

    problem: OpenProblem
    attempts: list[DiscoveryAttempt] = field(default_factory=list)
    best_equations: dict[str, str] = field(default_factory=dict)
    best_r_squared: dict[str, float] = field(default_factory=dict)
    total_runtime: float = 0.0
    n_simulations: int = 0

    def summary(self) -> str:
        lines = [
            f"Discovery Result: {self.problem.name}",
            f"  Domain: {self.problem.domain}",
            f"  Simulations run: {self.n_simulations}",
            f"  Runtime: {self.total_runtime:.1f}s",
            f"  Discoveries found: {len(self.best_equations)}",
        ]
        for obs, eq in self.best_equations.items():
            r2 = self.best_r_squared.get(obs, 0)
            lines.append(f"  - {obs}: {eq} (R²={r2:.4f})")
        return "\n".join(lines)


class DiscoveryRunner:
    """Runs discovery campaigns on open scientific problems.

    Pipeline:
    1. Load simulation class for the problem
    2. Run parameter sweeps to collect data
    3. Apply analysis (curve fitting, scaling laws)
    4. Report discovered relationships
    """

    def __init__(
        self,
        output_dir: str | Path = "output/discoveries",
        n_steps: int = 5000,
        dt: float = 0.01,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_steps = n_steps
        self.dt = dt

    def run_problem(
        self,
        problem_name: str,
        max_sweeps: int = 3,
    ) -> DiscoveryResult:
        """Run a full discovery campaign on an open problem.

        Args:
            problem_name: Name of the registered open problem.
            max_sweeps: Maximum number of parameter sweeps to run.

        Returns:
            DiscoveryResult with all findings.
        """
        problem = get_problem(problem_name)
        result = DiscoveryResult(problem=problem)
        start_time = time.time()

        logger.info(f"Starting discovery: {problem.name}")

        # Load simulation class
        sim_class = self._load_sim_class(problem.simulation_class)
        if sim_class is None:
            result.attempts.append(DiscoveryAttempt(
                problem_name=problem_name,
                parameter_name="",
                notes=[f"Failed to load simulation: {problem.simulation_class}"],
            ))
            return result

        # Run parameter sweeps
        sweep_count = 0
        for param_name, (pmin, pmax, n_points) in problem.parameter_sweeps.items():
            if sweep_count >= max_sweeps:
                break

            attempt = self._run_sweep(
                sim_class, problem, param_name, pmin, pmax, n_points
            )
            result.attempts.append(attempt)
            result.n_simulations += len(attempt.parameter_values)

            if attempt.r_squared > result.best_r_squared.get(attempt.observable_name, 0):
                result.best_r_squared[attempt.observable_name] = attempt.r_squared
                result.best_equations[attempt.observable_name] = attempt.candidate_equation

            sweep_count += 1

        result.total_runtime = time.time() - start_time
        logger.info(f"Discovery complete: {result.summary()}")

        # Save results
        self._save_result(result)
        return result

    def _load_sim_class(self, class_path: str) -> type | None:
        """Dynamically load a simulation class."""
        try:
            parts = class_path.rsplit(".", 1)
            module = importlib.import_module(parts[0])
            return getattr(module, parts[1])
        except (ImportError, AttributeError) as e:
            logger.error(f"Could not load {class_path}: {e}")
            return None

    def _run_sweep(
        self,
        sim_class: type,
        problem: OpenProblem,
        param_name: str,
        pmin: float,
        pmax: float,
        n_points: int,
    ) -> DiscoveryAttempt:
        """Run a single parameter sweep and analyze results."""
        attempt = DiscoveryAttempt(
            problem_name=problem.name,
            parameter_name=param_name,
        )

        param_values = np.linspace(pmin, pmax, n_points)
        observables: dict[str, list[float]] = {}

        for p_val in param_values:
            params = {param_name: float(p_val)}
            # Add default parameters for other sweep params
            for other_name, (other_min, other_max, _) in problem.parameter_sweeps.items():
                if other_name != param_name and other_name not in params:
                    params[other_name] = (other_min + other_max) / 2

            try:
                config = SimulationConfig(
                    domain=Domain.CUSTOM,
                    dt=self.dt,
                    n_steps=self.n_steps,
                    parameters=params,
                )
                sim = sim_class(config)
                sim.reset()

                # Run simulation and collect observables
                states = []
                for _ in range(self.n_steps):
                    state = sim.step()
                    states.append(state.copy())

                states_arr = np.array(states)
                obs = self._extract_observables(sim, states_arr, problem)
                for obs_name, obs_val in obs.items():
                    if obs_name not in observables:
                        observables[obs_name] = []
                    observables[obs_name].append(obs_val)

                attempt.parameter_values.append(float(p_val))
            except Exception as e:
                logger.warning(f"Simulation failed at {param_name}={p_val}: {e}")
                continue

        # Analyze: fit simple power laws and polynomials
        if attempt.parameter_values:
            params_arr = np.array(attempt.parameter_values)
            best_r2 = -1.0
            best_obs = ""
            best_eq = ""

            for obs_name, obs_vals in observables.items():
                if len(obs_vals) != len(params_arr):
                    continue
                obs_arr = np.array(obs_vals)

                # Skip constant or invalid observables
                if np.std(obs_arr) < 1e-10 or not np.all(np.isfinite(obs_arr)):
                    continue

                eq, r2 = self._fit_relationship(params_arr, obs_arr, param_name, obs_name)
                if r2 > best_r2:
                    best_r2 = r2
                    best_obs = obs_name
                    best_eq = eq

            attempt.observable_name = best_obs
            attempt.observable_values = observables.get(best_obs, [])
            attempt.candidate_equation = best_eq
            attempt.r_squared = best_r2

        return attempt

    def _extract_observables(
        self,
        sim: Any,
        states: np.ndarray,
        problem: OpenProblem,
    ) -> dict[str, float]:
        """Extract observable quantities from simulation trajectory."""
        obs = {}

        # Generic observables
        if states.ndim == 2:
            obs["mean_state_0"] = float(np.mean(states[-self.n_steps // 2:, 0]))
            obs["std_state_0"] = float(np.std(states[-self.n_steps // 2:, 0]))

            if states.shape[1] >= 2:
                obs["mean_state_1"] = float(np.mean(states[-self.n_steps // 2:, 1]))

            # Amplitude (max - min of first variable in second half)
            half = states[self.n_steps // 2:]
            obs["amplitude_0"] = float(np.max(half[:, 0]) - np.min(half[:, 0]))

        # Domain-specific observables
        if hasattr(sim, "jacobi_constant"):
            try:
                obs["jacobi_constant"] = float(sim.jacobi_constant())
            except Exception:
                pass

        if hasattr(sim, "compute_lyapunov"):
            try:
                lyap = sim.compute_lyapunov(n_steps=min(2000, self.n_steps))
                obs["lyapunov"] = float(lyap)
            except Exception:
                pass

        # Boundedness check
        obs["max_displacement"] = float(np.max(np.abs(states)))
        obs["is_bounded"] = float(np.all(np.abs(states) < 1e6))

        return obs

    def _fit_relationship(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param_name: str,
        obs_name: str,
    ) -> tuple[str, float]:
        """Fit polynomial and power-law relationships to data."""
        best_r2 = -1.0
        best_eq = ""

        # Try polynomial fits (degree 1-4)
        for deg in range(1, 5):
            try:
                coeffs = np.polyfit(x, y, deg)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                if r2 > best_r2:
                    best_r2 = r2
                    terms = []
                    for i, c in enumerate(coeffs):
                        power = deg - i
                        if abs(c) > 1e-10:
                            if power == 0:
                                terms.append(f"{c:.4f}")
                            elif power == 1:
                                terms.append(f"{c:.4f}*{param_name}")
                            else:
                                terms.append(f"{c:.4f}*{param_name}^{power}")
                    best_eq = f"{obs_name} = {' + '.join(terms)}"
            except Exception:
                continue

        # Try power law: y = a * x^b (positive x and y only)
        pos_mask = (x > 0) & (y > 0)
        if np.sum(pos_mask) > 5:
            try:
                log_x = np.log(x[pos_mask])
                log_y = np.log(y[pos_mask])
                coeffs = np.polyfit(log_x, log_y, 1)
                b = coeffs[0]
                a = np.exp(coeffs[1])
                y_pred = a * x[pos_mask] ** b
                ss_res = np.sum((y[pos_mask] - y_pred) ** 2)
                ss_tot = np.sum((y[pos_mask] - np.mean(y[pos_mask])) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                if r2 > best_r2:
                    best_r2 = r2
                    best_eq = f"{obs_name} = {a:.4f} * {param_name}^{b:.4f}"
            except Exception:
                pass

        return best_eq, best_r2

    def _save_result(self, result: DiscoveryResult) -> None:
        """Save discovery result to output directory."""
        import json

        safe_name = result.problem.name.replace(" ", "_").lower()
        out_path = self.output_dir / f"{safe_name}.json"

        data = {
            "problem": result.problem.name,
            "domain": result.problem.domain,
            "n_simulations": result.n_simulations,
            "runtime_seconds": result.total_runtime,
            "best_equations": result.best_equations,
            "best_r_squared": result.best_r_squared,
            "attempts": [
                {
                    "parameter": a.parameter_name,
                    "observable": a.observable_name,
                    "equation": a.candidate_equation,
                    "r_squared": a.r_squared,
                    "n_points": len(a.parameter_values),
                }
                for a in result.attempts
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved results to {out_path}")
