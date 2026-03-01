"""Verify all 14 simulation domains produce deterministic, finite, physically valid results.

For each domain this script:
  1. Runs the simulation twice with identical seeds and checks exact output match.
  2. Verifies all state values are finite (no NaN/Inf).
  3. Validates state dimensions against expected values.
  4. Checks domain-specific invariants (SIR conservation, energy conservation, etc.).
  5. Integrates with domain_statistics.compute_all_stats() for summary metrics.

Outputs a structured JSON file and formatted text summary to stdout.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.analysis.domain_statistics import (
    DOMAIN_REGISTRY,
    DomainStats,
    compute_all_stats,
)
from simulating_anything.types.simulation import Domain, SimulationBackend, SimulationConfig

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/reproducibility")

# Gray-Scott is not in DOMAIN_REGISTRY but is one of the 14 core domains.
# We add it here so this script covers all 14.
GRAY_SCOTT_SPEC: dict[str, Any] = {
    "module": "simulating_anything.simulation.reaction_diffusion",
    "cls": "GrayScottSimulation",
    "domain": Domain.REACTION_DIFFUSION,
    "params": {"D_u": 0.16, "D_v": 0.08, "f": 0.035, "k": 0.065},
    "dt": 0.005,
    "n_steps": 100,
    "math_class": "PDE",
    "grid_resolution": (32, 32),
    "domain_size": (2.5, 2.5),
    "backend": SimulationBackend.JAX_FD,
}

# Expected observation shapes per domain (single timestep)
EXPECTED_OBS_SHAPES: dict[str, tuple[int, ...]] = {
    "projectile": (4,),
    "lotka_volterra": (2,),
    "gray_scott": (32, 32, 2),
    "sir_epidemic": (3,),
    "double_pendulum": (4,),
    "harmonic_oscillator": (2,),
    "lorenz": (3,),
    "navier_stokes": (1024,),  # Flattened 32x32 vorticity field
    "van_der_pol": (2,),
    "kuramoto": (20,),  # N=20 oscillator phases
    "brusselator": (2,),
    "fitzhugh_nagumo": (2,),
    "heat_equation": (64,),  # N=64 grid points
    "logistic_map": (1,),
}


@dataclass
class InvariantResult:
    """Result of a domain-specific invariant check."""

    name: str
    passed: bool
    expected: float
    actual: float
    tolerance: float
    message: str


@dataclass
class DomainVerification:
    """Full verification result for a single domain."""

    domain: str
    obs_shape: list[int]
    expected_shape: list[int]
    shape_match: bool
    n_steps: int
    all_finite: bool
    is_deterministic: bool
    max_diff: float
    invariants: list[dict[str, Any]]
    elapsed_s: float
    error: str | None = None


def _get_full_registry() -> dict[str, dict[str, Any]]:
    """Return the combined domain registry including Gray-Scott."""
    registry = dict(DOMAIN_REGISTRY)
    registry["gray_scott"] = GRAY_SCOTT_SPEC
    return registry


def _make_config(spec: dict[str, Any]) -> SimulationConfig:
    """Build a SimulationConfig from a domain spec dictionary."""
    kwargs: dict[str, Any] = {
        "domain": spec["domain"],
        "dt": spec["dt"],
        "n_steps": spec["n_steps"],
        "parameters": {k: float(v) for k, v in spec["params"].items()},
    }
    if spec["domain"] == Domain.REACTION_DIFFUSION:
        kwargs["backend"] = spec.get("backend", SimulationBackend.JAX_FD)
        kwargs["grid_resolution"] = spec.get("grid_resolution", (32, 32))
        kwargs["domain_size"] = spec.get("domain_size", (2.5, 2.5))
    return SimulationConfig(**kwargs)


def _make_sim(spec: dict[str, Any]) -> Any:
    """Instantiate a simulation from a domain spec."""
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])
    config = _make_config(spec)
    return cls(config)


def _check_sir_conservation(states: np.ndarray) -> InvariantResult:
    """Verify S + I + R = 1 at every timestep for SIR model."""
    sums = states.sum(axis=1)
    max_deviation = float(np.max(np.abs(sums - 1.0)))
    tol = 1e-8
    return InvariantResult(
        name="SIR conservation (S+I+R=1)",
        passed=max_deviation < tol,
        expected=1.0,
        actual=float(sums[-1]),
        tolerance=tol,
        message=f"Max deviation from 1.0: {max_deviation:.2e}",
    )


def _check_sir_non_negative(states: np.ndarray) -> InvariantResult:
    """Verify all SIR compartments remain non-negative."""
    min_val = float(np.min(states))
    return InvariantResult(
        name="SIR non-negativity",
        passed=min_val >= -1e-12,
        expected=0.0,
        actual=min_val,
        tolerance=1e-12,
        message=f"Minimum compartment value: {min_val:.2e}",
    )


def _check_pendulum_energy(spec: dict[str, Any], states: np.ndarray) -> InvariantResult:
    """Verify energy conservation for double pendulum over short trajectories."""
    from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation

    config = _make_config(spec)
    sim = DoublePendulumSimulation(config)
    sim.reset()

    E0 = sim.total_energy(states[0])
    E_final = sim.total_energy(states[-1])
    rel_drift = abs(E_final - E0) / abs(E0) if abs(E0) > 1e-12 else abs(E_final - E0)
    # RK4 with dt=0.001 and 500 steps should have very small drift
    tol = 1e-4
    return InvariantResult(
        name="Double pendulum energy conservation",
        passed=rel_drift < tol,
        expected=E0,
        actual=E_final,
        tolerance=tol,
        message=f"Relative energy drift: {rel_drift:.2e}",
    )


def _check_oscillator_energy_decay(
    spec: dict[str, Any], states: np.ndarray,
) -> InvariantResult:
    """Verify that damped harmonic oscillator energy decreases monotonically."""
    k = spec["params"]["k"]
    m = spec["params"]["m"]
    c = spec["params"]["c"]

    # Energy at each step: E = 0.5*k*x^2 + 0.5*m*v^2
    x_vals = states[:, 0]
    v_vals = states[:, 1]
    energies = 0.5 * k * x_vals**2 + 0.5 * m * v_vals**2

    # With damping c > 0, energy should be non-increasing
    # Allow tiny numerical noise per step
    energy_diffs = np.diff(energies)
    max_increase = float(np.max(energy_diffs))
    tol = 1e-8
    passed = max_increase < tol
    return InvariantResult(
        name="Harmonic oscillator energy decay (c > 0)",
        passed=passed,
        expected=0.0,
        actual=max_increase,
        tolerance=tol,
        message=f"Max energy increase per step: {max_increase:.2e} (c={c})",
    )


def _check_logistic_map_range(states: np.ndarray) -> InvariantResult:
    """Verify logistic map stays in [0, 1] for r in [0, 4]."""
    min_val = float(np.min(states))
    max_val = float(np.max(states))
    passed = min_val >= -1e-12 and max_val <= 1.0 + 1e-12
    return InvariantResult(
        name="Logistic map range [0, 1]",
        passed=passed,
        expected=1.0,
        actual=max_val,
        tolerance=1e-12,
        message=f"State range: [{min_val:.6f}, {max_val:.6f}]",
    )


def _check_lotka_volterra_positive(states: np.ndarray) -> InvariantResult:
    """Verify Lotka-Volterra populations remain non-negative."""
    min_val = float(np.min(states))
    return InvariantResult(
        name="Lotka-Volterra non-negativity",
        passed=min_val >= -1e-12,
        expected=0.0,
        actual=min_val,
        tolerance=1e-12,
        message=f"Minimum population value: {min_val:.2e}",
    )


def _check_heat_equation_conservation(states: np.ndarray) -> InvariantResult:
    """Verify total heat is conserved for periodic boundary conditions."""
    # With periodic BC and no source, total heat = sum(u) * dx is constant
    total_heat = states.sum(axis=1)
    max_drift = float(np.max(np.abs(total_heat - total_heat[0])))
    # Spectral solver should preserve total heat to machine precision
    tol = 1e-10
    return InvariantResult(
        name="Heat equation total heat conservation",
        passed=max_drift < tol,
        expected=float(total_heat[0]),
        actual=float(total_heat[-1]),
        tolerance=tol,
        message=f"Max total heat drift: {max_drift:.2e}",
    )


def _check_heat_equation_decay(states: np.ndarray) -> InvariantResult:
    """Verify max temperature decreases over time (diffusion smooths peaks)."""
    max_temps = np.max(states, axis=1)
    # After the first step, max temperature should be non-increasing
    # (with some tolerance for numerical precision)
    diffs = np.diff(max_temps[1:])  # Skip initial step
    max_increase = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    tol = 1e-10
    return InvariantResult(
        name="Heat equation peak decay",
        passed=max_increase < tol,
        expected=0.0,
        actual=max_increase,
        tolerance=tol,
        message=f"Max temperature increase between steps: {max_increase:.2e}",
    )


def check_invariants(
    domain_name: str, spec: dict[str, Any], states: np.ndarray,
) -> list[InvariantResult]:
    """Run domain-specific invariant checks.

    Args:
        domain_name: Key from the domain registry.
        spec: Domain specification dictionary.
        states: Full trajectory states array of shape (n_steps+1, *obs_shape).

    Returns:
        List of invariant check results.
    """
    results: list[InvariantResult] = []

    if domain_name == "sir_epidemic":
        results.append(_check_sir_conservation(states))
        results.append(_check_sir_non_negative(states))

    elif domain_name == "double_pendulum":
        results.append(_check_pendulum_energy(spec, states))

    elif domain_name == "harmonic_oscillator":
        results.append(_check_oscillator_energy_decay(spec, states))

    elif domain_name == "logistic_map":
        results.append(_check_logistic_map_range(states))

    elif domain_name == "lotka_volterra":
        results.append(_check_lotka_volterra_positive(states))

    elif domain_name == "heat_equation":
        results.append(_check_heat_equation_conservation(states))
        results.append(_check_heat_equation_decay(states))

    return results


def verify_domain(domain_name: str, spec: dict[str, Any]) -> DomainVerification:
    """Run full verification for a single domain.

    Checks determinism, finiteness, state shape, and domain-specific invariants.

    Args:
        domain_name: Key from the domain registry.
        spec: Domain specification dictionary.

    Returns:
        DomainVerification with all check results.
    """
    t0 = time.perf_counter()
    expected_shape = list(EXPECTED_OBS_SHAPES.get(domain_name, []))

    try:
        sim1 = _make_sim(spec)
        traj1 = sim1.run(n_steps=spec["n_steps"])
        states1 = traj1.states

        # Finiteness check
        all_finite = bool(np.all(np.isfinite(states1)))

        # Shape check
        actual_shape = list(states1.shape[1:])
        shape_match = actual_shape == expected_shape if expected_shape else True

        # Determinism: run a second time with same config
        # Skip for Kuramoto which uses random frequencies from seed
        if domain_name == "kuramoto":
            # Kuramoto IS deterministic with same seed via default_rng
            sim2 = _make_sim(spec)
            traj2 = sim2.run(n_steps=spec["n_steps"])
            states2 = traj2.states
        else:
            sim2 = _make_sim(spec)
            traj2 = sim2.run(n_steps=spec["n_steps"])
            states2 = traj2.states

        max_diff = float(np.max(np.abs(states1 - states2)))
        is_deterministic = bool(np.allclose(states1, states2, atol=0.0, rtol=0.0))

        # Domain-specific invariants
        invariant_results = check_invariants(domain_name, spec, states1)
        invariant_dicts = [
            {
                "name": r.name,
                "passed": r.passed,
                "expected": r.expected,
                "actual": r.actual,
                "tolerance": r.tolerance,
                "message": r.message,
            }
            for r in invariant_results
        ]

        elapsed = time.perf_counter() - t0

        return DomainVerification(
            domain=domain_name,
            obs_shape=actual_shape,
            expected_shape=expected_shape,
            shape_match=shape_match,
            n_steps=spec["n_steps"],
            all_finite=all_finite,
            is_deterministic=is_deterministic,
            max_diff=max_diff,
            invariants=invariant_dicts,
            elapsed_s=round(elapsed, 4),
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return DomainVerification(
            domain=domain_name,
            obs_shape=[],
            expected_shape=expected_shape,
            shape_match=False,
            n_steps=spec["n_steps"],
            all_finite=False,
            is_deterministic=False,
            max_diff=float("inf"),
            invariants=[],
            elapsed_s=round(elapsed, 4),
            error=str(e),
        )


def print_summary_table(results: list[DomainVerification]) -> str:
    """Print a formatted summary table to stdout and return as string.

    Args:
        results: List of verification results for all domains.

    Returns:
        Formatted table as a string.
    """
    header = (
        f"{'Domain':<24} {'Shape':<16} {'Match':>5} "
        f"{'Finite':>6} {'Determ':>6} {'max_diff':>12} "
        f"{'Invariants':>10} {'Time(s)':>8}"
    )
    separator = "-" * len(header)
    lines = ["", separator, header, separator]

    for r in results:
        shape_str = str(tuple(r.obs_shape)) if r.obs_shape else "(err)"
        match_str = "OK" if r.shape_match else "FAIL"
        fin_str = "OK" if r.all_finite else "FAIL"
        det_str = "OK" if r.is_deterministic else "FAIL"
        diff_str = f"{r.max_diff:.2e}" if r.max_diff < float("inf") else "inf"

        n_inv = len(r.invariants)
        n_pass = sum(1 for inv in r.invariants if inv["passed"])
        inv_str = f"{n_pass}/{n_inv}" if n_inv > 0 else "-"

        line = (
            f"{r.domain:<24} {shape_str:<16} {match_str:>5} "
            f"{fin_str:>6} {det_str:>6} {diff_str:>12} "
            f"{inv_str:>10} {r.elapsed_s:>8.3f}"
        )
        lines.append(line)

    lines.append(separator)

    # Summary counts
    n_total = len(results)
    n_finite = sum(1 for r in results if r.all_finite)
    n_det = sum(1 for r in results if r.is_deterministic)
    n_shape = sum(1 for r in results if r.shape_match)
    all_inv = [inv for r in results for inv in r.invariants]
    n_inv_pass = sum(1 for inv in all_inv if inv["passed"])
    n_inv_total = len(all_inv)
    total_time = sum(r.elapsed_s for r in results)

    lines.append("")
    lines.append(f"Finite values:    {n_finite}/{n_total}")
    lines.append(f"Deterministic:    {n_det}/{n_total}")
    lines.append(f"Shape match:      {n_shape}/{n_total}")
    lines.append(f"Invariants:       {n_inv_pass}/{n_inv_total}")
    lines.append(f"Total time:       {total_time:.2f}s")

    # Print invariant details
    if all_inv:
        lines.append("")
        lines.append("Invariant Details:")
        lines.append("-" * 70)
        for r in results:
            for inv in r.invariants:
                status = "PASS" if inv["passed"] else "FAIL"
                lines.append(f"  [{status}] {r.domain}: {inv['name']}")
                lines.append(f"         {inv['message']}")

    output = "\n".join(lines)
    print(output)
    return output


def build_summary_json(
    results: list[DomainVerification],
    domain_stats: list[DomainStats] | None = None,
) -> dict[str, Any]:
    """Build a structured JSON summary of all verification results.

    Args:
        results: List of verification results.
        domain_stats: Optional domain statistics from compute_all_stats().

    Returns:
        Dictionary suitable for JSON serialization.
    """
    n_total = len(results)
    all_inv = [inv for r in results for inv in r.invariants]

    summary: dict[str, Any] = {
        "description": "Reproducibility verification for all simulation domains",
        "n_domains": n_total,
        "all_finite": all(r.all_finite for r in results),
        "all_deterministic": all(r.is_deterministic for r in results),
        "all_shapes_match": all(r.shape_match for r in results),
        "all_invariants_pass": all(inv["passed"] for inv in all_inv) if all_inv else True,
        "total_time_s": round(sum(r.elapsed_s for r in results), 3),
        "results": [asdict(r) for r in results],
    }

    if domain_stats is not None:
        summary["domain_statistics"] = [asdict(s) for s in domain_stats]

    return summary


def save_results(summary: dict[str, Any]) -> Path:
    """Save the summary JSON to disk.

    Args:
        summary: Structured summary dictionary.

    Returns:
        Path to the written JSON file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "verification_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return output_path


def main() -> None:
    """Run reproducibility verification for all 14 domains."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("Simulating-Anything: Reproducibility Verification")
    print("=" * 70)
    print()
    print("Verifying all 14 simulation domains for:")
    print("  - Determinism (same seed produces identical output)")
    print("  - Finite values (no NaN or Inf)")
    print("  - State dimension correctness")
    print("  - Domain-specific physical invariants")
    print()

    registry = _get_full_registry()
    results: list[DomainVerification] = []

    for domain_name, spec in registry.items():
        print(f"  Checking {domain_name}...", end="", flush=True)
        result = verify_domain(domain_name, spec)

        if result.error:
            print(f" ERROR: {result.error}")
        else:
            checks = []
            if not result.all_finite:
                checks.append("non-finite")
            if not result.is_deterministic:
                checks.append("non-deterministic")
            if not result.shape_match:
                checks.append("shape-mismatch")
            failed_inv = [
                inv["name"] for inv in result.invariants if not inv["passed"]
            ]
            if failed_inv:
                checks.append(f"invariant({','.join(failed_inv)})")

            if checks:
                print(f" ISSUES: {'; '.join(checks)} ({result.elapsed_s:.3f}s)")
            else:
                print(f" OK ({result.elapsed_s:.3f}s)")

        results.append(result)

    # Print summary table
    print_summary_table(results)

    # Run domain statistics integration
    print()
    print("=" * 70)
    print("Domain Statistics (from compute_all_stats)")
    print("=" * 70)
    try:
        domain_stats = compute_all_stats(skip_kuramoto=False)
        from simulating_anything.analysis.domain_statistics import print_stats_table
        print()
        print(print_stats_table(domain_stats))
    except Exception as e:
        print(f"  Failed to compute domain statistics: {e}")
        domain_stats = None

    # Save combined results
    summary = build_summary_json(results, domain_stats)
    save_results(summary)

    # Exit with non-zero code if any check failed
    all_passed = (
        summary["all_finite"]
        and summary["all_deterministic"]
        and summary["all_shapes_match"]
        and summary["all_invariants_pass"]
    )
    if not all_passed:
        print("\nSome checks FAILED. See details above.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
