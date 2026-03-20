"""Detect bifurcations from parameter sweeps with extracted observables.

Given parameter values and observables at each point, detects discontinuities,
classifies bifurcation types (fold, Hopf, period-doubling, chaos onset),
and returns critical parameter values with confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulating_anything.analysis.observable_extractor import TrajectoryObservables


@dataclass
class BifurcationPoint:
    """A detected bifurcation in parameter space."""
    parameter_name: str
    critical_value: float
    confidence_interval: tuple[float, float]
    bifurcation_type: str  # fold, hopf, period_doubling, chaos_onset, unknown
    observable_name: str
    before_classification: str
    after_classification: str
    gradient_magnitude: float
    evidence: dict = field(default_factory=dict)


@dataclass
class BifurcationResult:
    """Results of bifurcation analysis on a 1D parameter sweep."""
    parameter_name: str
    parameter_values: np.ndarray
    classifications: list[str]
    bifurcation_points: list[BifurcationPoint]
    observable_series: dict[str, np.ndarray]


def detect_bifurcations_1d(
    param_values: np.ndarray,
    observables: list[TrajectoryObservables],
    param_name: str,
    gradient_threshold: float = 3.0,
    min_gap: int = 3,
) -> BifurcationResult:
    """Detect bifurcations in a 1D parameter sweep.

    Args:
        param_values: Sorted parameter values.
        observables: Extracted observables at each parameter value.
        param_name: Name of the swept parameter.
        gradient_threshold: Z-score threshold for discontinuity detection.
        min_gap: Minimum points between detected bifurcations.
    """
    n = len(param_values)
    classifications = [o.classification for o in observables]

    # Extract observable series
    means_0 = np.array([o.mean[0] if len(o.mean) > 0 else 0 for o in observables])
    stds_0 = np.array([o.std[0] if len(o.std) > 0 else 0 for o in observables])
    amplitudes_0 = np.array([o.amplitude[0] if len(o.amplitude) > 0 else 0 for o in observables])
    lyapunovs = np.array([o.lyapunov_proxy for o in observables])
    periods = np.array([o.period if o.period is not None else 0 for o in observables])

    observable_series = {
        "mean_x0": means_0,
        "std_x0": stds_0,
        "amplitude_x0": amplitudes_0,
        "lyapunov": lyapunovs,
        "period": periods,
    }

    bifurcation_points = []

    # Method 1: Classification changes
    for i in range(1, n):
        if classifications[i] != classifications[i - 1]:
            crit = (param_values[i] + param_values[i - 1]) / 2
            ci = (float(param_values[i - 1]), float(param_values[i]))
            btype = _classify_transition(classifications[i - 1], classifications[i])

            bp = BifurcationPoint(
                parameter_name=param_name,
                critical_value=float(crit),
                confidence_interval=ci,
                bifurcation_type=btype,
                observable_name="classification",
                before_classification=classifications[i - 1],
                after_classification=classifications[i],
                gradient_magnitude=0.0,
                evidence={"method": "classification_change", "index": i},
            )
            bifurcation_points.append(bp)

    # Method 2: Gradient discontinuities in amplitude
    for obs_name, series in observable_series.items():
        if np.all(np.isnan(series)) or np.std(series) < 1e-12:
            continue

        grad = np.gradient(series, param_values)
        grad_clean = np.where(np.isnan(grad), 0, grad)

        if np.std(grad_clean) < 1e-12:
            continue

        z_scores = np.abs(grad_clean - np.mean(grad_clean)) / np.std(grad_clean)

        # Find peaks above threshold
        candidates = np.where(z_scores > gradient_threshold)[0]
        if len(candidates) == 0:
            continue

        # Merge nearby candidates
        merged = _merge_nearby(candidates, min_gap)

        for idx in merged:
            # Skip if too close to an existing bifurcation
            crit = float(param_values[idx])
            too_close = any(
                abs(crit - bp.critical_value) < (param_values[-1] - param_values[0]) / 20
                for bp in bifurcation_points
            )
            if too_close:
                continue

            ci_lo = float(param_values[max(0, idx - 1)])
            ci_hi = float(param_values[min(n - 1, idx + 1)])

            bp = BifurcationPoint(
                parameter_name=param_name,
                critical_value=crit,
                confidence_interval=(ci_lo, ci_hi),
                bifurcation_type="unknown",
                observable_name=obs_name,
                before_classification=classifications[max(0, idx - 2)],
                after_classification=classifications[min(n - 1, idx + 2)],
                gradient_magnitude=float(z_scores[idx]),
                evidence={"method": "gradient_discontinuity", "z_score": float(z_scores[idx])},
            )
            bifurcation_points.append(bp)

    # Sort by parameter value
    bifurcation_points.sort(key=lambda bp: bp.critical_value)

    return BifurcationResult(
        parameter_name=param_name,
        parameter_values=param_values,
        classifications=classifications,
        bifurcation_points=bifurcation_points,
        observable_series=observable_series,
    )


def _classify_transition(before: str, after: str) -> str:
    """Classify bifurcation type from before/after classifications."""
    transition = (before, after)
    if transition == ("steady", "oscillatory"):
        return "hopf"
    if transition == ("oscillatory", "steady"):
        return "inverse_hopf"
    if transition == ("oscillatory", "chaotic") or transition == ("oscillatory", "aperiodic"):
        return "chaos_onset"
    if transition == ("chaotic", "oscillatory"):
        return "chaos_exit"
    if transition == ("steady", "chaotic"):
        return "crisis"
    if transition in (("steady", "divergent"), ("oscillatory", "divergent")):
        return "blowup"
    return "unknown"


def _merge_nearby(indices: np.ndarray, min_gap: int) -> list[int]:
    """Merge nearby candidate indices, keeping the one with highest z-score."""
    if len(indices) == 0:
        return []
    merged = [indices[0]]
    for idx in indices[1:]:
        if idx - merged[-1] >= min_gap:
            merged.append(idx)
    return merged
