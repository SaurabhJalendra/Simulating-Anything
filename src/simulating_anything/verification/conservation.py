"""Conservation law checks for simulation validation."""

from __future__ import annotations

import numpy as np

from simulating_anything.types.discovery import CheckResult


def check_mass_conservation(
    states: np.ndarray, tolerance: float = 1e-6
) -> CheckResult:
    """Check that total mass (sum of field) is conserved across timesteps.

    Args:
        states: Array of shape (n_steps, ...) where each entry is a state snapshot.
        tolerance: Maximum allowed relative change in total mass.
    """
    if states.ndim < 2:
        return CheckResult(
            name="mass_conservation",
            passed=True,
            value=0.0,
            threshold=tolerance,
            message="Single state — trivially conserved.",
        )

    totals = np.array([np.sum(s) for s in states])
    if totals[0] == 0:
        max_drift = np.max(np.abs(totals))
        passed = max_drift < tolerance
    else:
        max_drift = np.max(np.abs(totals - totals[0]) / (np.abs(totals[0]) + 1e-30))
        passed = max_drift < tolerance

    return CheckResult(
        name="mass_conservation",
        passed=bool(passed),
        value=float(max_drift),
        threshold=tolerance,
        message=f"Max relative drift: {max_drift:.2e}",
    )


def check_energy_conservation(
    kinetic: np.ndarray, potential: np.ndarray, tolerance: float = 1e-4
) -> CheckResult:
    """Check that total energy (KE + PE) is conserved.

    Args:
        kinetic: Kinetic energy at each timestep.
        potential: Potential energy at each timestep.
        tolerance: Maximum allowed relative change in total energy.
    """
    total = kinetic + potential
    if total[0] == 0:
        max_drift = np.max(np.abs(total))
        passed = max_drift < tolerance
    else:
        max_drift = np.max(np.abs(total - total[0]) / (np.abs(total[0]) + 1e-30))
        passed = max_drift < tolerance

    return CheckResult(
        name="energy_conservation",
        passed=bool(passed),
        value=float(max_drift),
        threshold=tolerance,
        message=f"Max relative energy drift: {max_drift:.2e}",
    )


def check_positivity(states: np.ndarray, field_name: str = "state") -> CheckResult:
    """Check that all values remain non-negative."""
    min_val = float(np.min(states))
    passed = min_val >= 0.0

    return CheckResult(
        name=f"positivity_{field_name}",
        passed=bool(passed),
        value=min_val,
        threshold=0.0,
        message=f"Min {field_name} value: {min_val:.6e}",
    )


def check_boundedness(
    states: np.ndarray, lower: float, upper: float, field_name: str = "state"
) -> CheckResult:
    """Check that all values remain within [lower, upper]."""
    min_val = float(np.min(states))
    max_val = float(np.max(states))
    passed = min_val >= lower and max_val <= upper

    return CheckResult(
        name=f"boundedness_{field_name}",
        passed=bool(passed),
        value=max(abs(min_val - lower), abs(max_val - upper)),
        threshold=0.0,
        message=f"{field_name} range: [{min_val:.4e}, {max_val:.4e}], bounds: [{lower}, {upper}]",
    )
