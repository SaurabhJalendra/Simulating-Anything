"""Extract physically meaningful observables from simulation trajectories.

Given a time series of states, extracts: mean, std, amplitude, period,
peak value, Lyapunov proxy, and classifies dynamics as steady/oscillatory/
chaotic/divergent. This is the foundation for phase diagrams and
bifurcation detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryObservables:
    """Extracted observables from a single simulation trajectory."""

    mean: np.ndarray                    # time-averaged state
    std: np.ndarray                     # standard deviation
    amplitude: np.ndarray               # max - min in analysis window
    period: float | None = None         # dominant oscillation period (FFT)
    peak_value: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_time: float | None = None      # time index of peak (first var)
    final_state: np.ndarray = field(default_factory=lambda: np.array([]))
    lyapunov_proxy: float = 0.0         # divergence rate estimate
    is_oscillatory: bool = False
    is_divergent: bool = False
    is_steady: bool = False
    is_chaotic: bool = False
    classification: str = "unknown"     # steady, oscillatory, chaotic, divergent


def extract_observables(
    states: np.ndarray,
    dt: float,
    warmup_fraction: float = 0.5,
    divergence_threshold: float = 1e8,
    oscillation_threshold: float = 0.01,
) -> TrajectoryObservables:
    """Extract observables from a simulation trajectory.

    Args:
        states: Array of shape (n_steps, n_vars).
        dt: Timestep.
        warmup_fraction: Fraction of trajectory to discard as transient.
        divergence_threshold: State magnitude above this = divergent.
        oscillation_threshold: Amplitude below this = steady.

    Returns:
        TrajectoryObservables with all extracted quantities.
    """
    n_steps, n_vars = states.shape
    warmup = int(n_steps * warmup_fraction)
    analysis = states[warmup:]

    if len(analysis) < 10:
        return TrajectoryObservables(
            mean=np.zeros(n_vars),
            std=np.zeros(n_vars),
            amplitude=np.zeros(n_vars),
            classification="insufficient_data",
        )

    # Check divergence
    if np.any(np.abs(analysis) > divergence_threshold) or np.any(np.isnan(analysis)):
        return TrajectoryObservables(
            mean=np.full(n_vars, np.nan),
            std=np.full(n_vars, np.nan),
            amplitude=np.full(n_vars, np.nan),
            is_divergent=True,
            classification="divergent",
        )

    # Basic statistics
    mean = np.mean(analysis, axis=0)
    std = np.std(analysis, axis=0)
    amplitude = np.max(analysis, axis=0) - np.min(analysis, axis=0)
    peak_value = np.max(analysis, axis=0)
    peak_idx = np.argmax(analysis[:, 0])
    peak_time = float((warmup + peak_idx) * dt)
    final_state = states[-1].copy()

    # Period detection via FFT (on first variable)
    period = _detect_period(analysis[:, 0], dt)

    # Lyapunov proxy (divergence rate of perturbed trajectory)
    lyap = _lyapunov_proxy(analysis)

    # Classification
    is_steady = bool(np.all(amplitude < oscillation_threshold * np.maximum(np.abs(mean), 1.0)))
    is_oscillatory = not is_steady and period is not None and period > 2 * dt
    is_chaotic = not is_steady and lyap > 0.01 and (period is None or amplitude[0] > 0.1)

    if is_steady:
        classification = "steady"
    elif is_chaotic:
        classification = "chaotic"
    elif is_oscillatory:
        classification = "oscillatory"
    else:
        classification = "aperiodic"

    return TrajectoryObservables(
        mean=mean,
        std=std,
        amplitude=amplitude,
        period=period,
        peak_value=peak_value,
        peak_time=peak_time,
        final_state=final_state,
        lyapunov_proxy=lyap,
        is_oscillatory=is_oscillatory,
        is_divergent=False,
        is_steady=is_steady,
        is_chaotic=is_chaotic,
        classification=classification,
    )


def _detect_period(signal: np.ndarray, dt: float) -> float | None:
    """Detect dominant oscillation period via FFT."""
    n = len(signal)
    if n < 8:
        return None

    # Remove mean
    centered = signal - np.mean(signal)

    # FFT
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(centered))

    # Ignore DC component
    spectrum[0] = 0

    if len(spectrum) < 2 or np.max(spectrum) < 1e-12:
        return None

    # Find peak frequency
    peak_idx = np.argmax(spectrum)
    if peak_idx == 0 or freqs[peak_idx] < 1e-10:
        return None

    period = 1.0 / freqs[peak_idx]

    # Check if peak is significant (>3x median)
    median_power = np.median(spectrum[1:])
    if spectrum[peak_idx] < 3 * median_power:
        return None

    return float(period)


def _lyapunov_proxy(states: np.ndarray) -> float:
    """Estimate Lyapunov exponent proxy from trajectory.

    Uses the rate of divergence between nearby points in the trajectory
    as a proxy for the maximal Lyapunov exponent.
    """
    n = len(states)
    if n < 20:
        return 0.0

    # Take pairs of nearby points and measure how they diverge
    step = max(1, n // 100)
    divergences = []

    for i in range(0, n - 10, step):
        # Find nearest neighbor (excluding immediate neighbors)
        dists = np.sum((states[i] - states[i + 5:min(i + 50, n)]) ** 2, axis=1)
        if len(dists) == 0:
            continue
        j = np.argmin(dists) + i + 5
        if j >= n - 5:
            continue

        d0 = np.sqrt(dists[j - i - 5])
        if d0 < 1e-12:
            continue

        # Measure divergence after 5 steps
        d5 = np.linalg.norm(states[min(i + 5, n - 1)] - states[min(j + 5, n - 1)])
        if d5 > 1e-12 and d0 > 1e-12:
            divergences.append(np.log(d5 / d0) / 5.0)

    if not divergences:
        return 0.0

    return float(np.median(divergences))


def classify_dynamics(obs: TrajectoryObservables) -> str:
    """Classify dynamics from observables."""
    return obs.classification
