"""Adversarial Dream Debate: two world models validate each other's predictions.

The core idea: train two RSSM world models on different subsets of the same
domain's trajectory data, then compare their dreamed futures. Regions of
disagreement indicate:
1. Insufficient training data (high uncertainty)
2. Chaotic/sensitive dynamics (genuine prediction limits)
3. Model failure modes (systematic bias)

This provides a model-free uncertainty estimate that doesn't require MC-dropout
or ensemble techniques -- just two independent world models.

Architecture:
  Model A: trained on trajectories 0..N/2
  Model B: trained on trajectories N/2..N
  -> Feed both the same context window
  -> Dream forward K steps independently
  -> Measure divergence: MSE, KL, correlation breakdown

Usage (WSL, requires JAX + trained checkpoints):
    from simulating_anything.analysis.dream_debate import DreamDebate
    debate = DreamDebate.from_checkpoints("path/to/model_a.eqx", "path/to/model_b.eqx")
    results = debate.run(context_observations, dream_steps=50)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """Result of a dream debate between two world models."""

    context_length: int
    dream_steps: int
    divergence_mse: np.ndarray  # (dream_steps,) -- MSE between dreams per step
    divergence_max: np.ndarray  # (dream_steps,) -- max absolute difference per step
    correlation: np.ndarray  # (dream_steps,) -- correlation between dreams per step
    agreement_horizon: int  # Steps until correlation drops below threshold
    dreams_a: np.ndarray | None = None  # (dream_steps, obs_dim) -- optional
    dreams_b: np.ndarray | None = None
    context_obs: np.ndarray | None = None

    def summary(self) -> dict:
        """Summary statistics for the debate."""
        return {
            "context_length": self.context_length,
            "dream_steps": self.dream_steps,
            "mean_divergence_mse": float(np.mean(self.divergence_mse)),
            "max_divergence_mse": float(np.max(self.divergence_mse)),
            "mean_correlation": float(np.mean(self.correlation)),
            "min_correlation": float(np.min(self.correlation)),
            "agreement_horizon": self.agreement_horizon,
            "divergence_growth_rate": self._growth_rate(),
        }

    def _growth_rate(self) -> float:
        """Estimate exponential growth rate of divergence (proxy for Lyapunov)."""
        mse = self.divergence_mse
        # Fit log(MSE) ~ slope * t in the linear growth region
        valid = mse > 1e-12
        if np.sum(valid) < 3:
            return 0.0
        log_mse = np.log(mse[valid])
        t = np.arange(len(log_mse))
        # Simple linear regression
        t_mean = np.mean(t)
        log_mean = np.mean(log_mse)
        slope = np.sum((t - t_mean) * (log_mse - log_mean)) / max(
            np.sum((t - t_mean) ** 2), 1e-12
        )
        return float(slope)


@dataclass
class DebateConfig:
    """Configuration for dream debate experiments."""

    context_steps: int = 20
    dream_steps: int = 50
    n_debates: int = 10  # Number of random context windows to test
    agreement_threshold: float = 0.9  # Correlation threshold for "agreement"
    save_dreams: bool = False


def compute_debate_metrics(
    dreams_a: np.ndarray,
    dreams_b: np.ndarray,
    threshold: float = 0.9,
) -> DebateResult:
    """Compute divergence metrics between two dreamed trajectories.

    Args:
        dreams_a: (n_steps, obs_dim) from model A.
        dreams_b: (n_steps, obs_dim) from model B.
        threshold: Correlation threshold for agreement horizon.

    Returns:
        DebateResult with all metrics.
    """
    n_steps = len(dreams_a)

    mse = np.mean((dreams_a - dreams_b) ** 2, axis=-1)
    max_diff = np.max(np.abs(dreams_a - dreams_b), axis=-1)

    # Per-step correlation
    corr = np.zeros(n_steps)
    for t in range(n_steps):
        a_t = dreams_a[t]
        b_t = dreams_b[t]
        if np.std(a_t) < 1e-12 or np.std(b_t) < 1e-12:
            corr[t] = 1.0 if np.allclose(a_t, b_t) else 0.0
        else:
            corr[t] = np.corrcoef(a_t, b_t)[0, 1]

    # Agreement horizon: first step where correlation drops below threshold
    horizon = n_steps
    for t in range(n_steps):
        if corr[t] < threshold:
            horizon = t
            break

    return DebateResult(
        context_length=0,  # Set by caller
        dream_steps=n_steps,
        divergence_mse=mse,
        divergence_max=max_diff,
        correlation=corr,
        agreement_horizon=horizon,
        dreams_a=dreams_a,
        dreams_b=dreams_b,
    )


def run_simulation_debate(
    sim_class,
    config_a: dict,
    config_b: dict,
    n_context: int = 20,
    n_dream: int = 50,
    n_trials: int = 10,
    agreement_threshold: float = 0.9,
) -> list[DebateResult]:
    """Run dream debate using two simulation configs as surrogate "world models".

    This is a CPU-friendly version that uses different simulation parameters
    to emulate the disagreement between two trained world models. Useful for
    testing the debate analysis pipeline without GPU.

    Args:
        sim_class: Simulation class (e.g., LorenzSimulation).
        config_a: Parameters for "model A" simulation.
        config_b: Parameters for "model B" (slightly perturbed).
        n_context: Context observation steps.
        n_dream: Dream-forward steps.
        n_trials: Number of random starting points to test.
        agreement_threshold: Correlation threshold.

    Returns:
        List of DebateResult for each trial.
    """
    from simulating_anything.types.simulation import SimulationConfig

    results = []
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        # Create simulations with slightly different parameters
        # (emulating two world models trained on different data)
        sim_a = sim_class(SimulationConfig(**config_a))
        sim_b = sim_class(SimulationConfig(**config_b))

        # Random initial perturbation
        seed = rng.integers(0, 10000)

        sim_a.reset(seed=seed)
        sim_b.reset(seed=seed)

        # Context phase: both see same observations
        for _ in range(n_context):
            sim_a.step()
            sim_b.step()

        # Dream phase: evolve independently
        dreams_a = []
        dreams_b = []
        for _ in range(n_dream):
            dreams_a.append(sim_a.step().copy())
            dreams_b.append(sim_b.step().copy())

        dreams_a = np.array(dreams_a)
        dreams_b = np.array(dreams_b)

        result = compute_debate_metrics(
            dreams_a, dreams_b, threshold=agreement_threshold
        )
        result.context_length = n_context
        results.append(result)

    return results


def run_lorenz_debate(
    n_trials: int = 10,
    rho_perturbation: float = 0.5,
    n_dream: int = 100,
) -> dict:
    """Run dream debate on Lorenz system with perturbed rho.

    Demonstrates that chaotic systems have short agreement horizons
    while stable systems maintain agreement longer.
    """
    from simulating_anything.simulation.lorenz import LorenzSimulation
    from simulating_anything.types.simulation import Domain

    logger.info("Running Lorenz dream debate...")

    # Chaotic regime (rho=28)
    config_a_chaotic = {
        "domain": Domain.LORENZ_ATTRACTOR,
        "dt": 0.01,
        "n_steps": 5000,
        "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    }
    config_b_chaotic = {
        "domain": Domain.LORENZ_ATTRACTOR,
        "dt": 0.01,
        "n_steps": 5000,
        "parameters": {
            "sigma": 10.0,
            "rho": 28.0 + rho_perturbation,
            "beta": 8.0 / 3.0,
        },
    }

    chaotic_results = run_simulation_debate(
        LorenzSimulation, config_a_chaotic, config_b_chaotic,
        n_context=100, n_dream=n_dream, n_trials=n_trials,
    )

    # Stable regime (rho=10)
    config_a_stable = {
        "domain": Domain.LORENZ_ATTRACTOR,
        "dt": 0.01,
        "n_steps": 5000,
        "parameters": {"sigma": 10.0, "rho": 10.0, "beta": 8.0 / 3.0},
    }
    config_b_stable = {
        "domain": Domain.LORENZ_ATTRACTOR,
        "dt": 0.01,
        "n_steps": 5000,
        "parameters": {
            "sigma": 10.0,
            "rho": 10.0 + rho_perturbation,
            "beta": 8.0 / 3.0,
        },
    }

    stable_results = run_simulation_debate(
        LorenzSimulation, config_a_stable, config_b_stable,
        n_context=100, n_dream=n_dream, n_trials=n_trials,
    )

    # Analyze
    chaotic_horizons = [r.agreement_horizon for r in chaotic_results]
    stable_horizons = [r.agreement_horizon for r in stable_results]
    chaotic_growth = [r._growth_rate() for r in chaotic_results]
    stable_growth = [r._growth_rate() for r in stable_results]

    summary = {
        "chaotic": {
            "mean_horizon": float(np.mean(chaotic_horizons)),
            "std_horizon": float(np.std(chaotic_horizons)),
            "mean_growth_rate": float(np.mean(chaotic_growth)),
            "all_horizons": [int(h) for h in chaotic_horizons],
        },
        "stable": {
            "mean_horizon": float(np.mean(stable_horizons)),
            "std_horizon": float(np.std(stable_horizons)),
            "mean_growth_rate": float(np.mean(stable_growth)),
            "all_horizons": [int(h) for h in stable_horizons],
        },
        "rho_perturbation": rho_perturbation,
        "n_dream": n_dream,
        "n_trials": n_trials,
    }

    logger.info(f"  Chaotic (rho=28): mean horizon = {np.mean(chaotic_horizons):.1f} steps")
    logger.info(f"  Stable (rho=10): mean horizon = {np.mean(stable_horizons):.1f} steps")
    logger.info(f"  Chaotic growth rate: {np.mean(chaotic_growth):.4f}")
    logger.info(f"  Stable growth rate: {np.mean(stable_growth):.4f}")

    return {
        "summary": summary,
        "chaotic_results": chaotic_results,
        "stable_results": stable_results,
    }


def run_multi_domain_debate(
    output_dir: str | Path = "output/dream_debate",
    n_trials: int = 10,
) -> dict:
    """Run dream debates across multiple domains and save results."""
    from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
    from simulating_anything.simulation.lorenz import LorenzSimulation
    from simulating_anything.types.simulation import Domain

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # --- Lorenz debate ---
    lorenz_data = run_lorenz_debate(n_trials=n_trials, n_dream=100)
    all_results["lorenz"] = lorenz_data["summary"]

    # --- Harmonic oscillator debate ---
    logger.info("Running harmonic oscillator dream debate...")
    config_a_ho = {
        "domain": Domain.HARMONIC_OSCILLATOR,
        "dt": 0.001,
        "n_steps": 5000,
        "parameters": {"k": 4.0, "m": 1.0, "c": 0.2, "x_0": 1.0, "v_0": 0.0},
    }
    config_b_ho = {
        "domain": Domain.HARMONIC_OSCILLATOR,
        "dt": 0.001,
        "n_steps": 5000,
        "parameters": {"k": 4.1, "m": 1.0, "c": 0.2, "x_0": 1.0, "v_0": 0.0},
    }
    ho_results = run_simulation_debate(
        DampedHarmonicOscillator, config_a_ho, config_b_ho,
        n_context=500, n_dream=2000, n_trials=n_trials,
    )
    ho_horizons = [r.agreement_horizon for r in ho_results]
    ho_growth = [r._growth_rate() for r in ho_results]
    all_results["harmonic_oscillator"] = {
        "mean_horizon": float(np.mean(ho_horizons)),
        "std_horizon": float(np.std(ho_horizons)),
        "mean_growth_rate": float(np.mean(ho_growth)),
        "all_horizons": [int(h) for h in ho_horizons],
    }
    logger.info(f"  Oscillator: mean horizon = {np.mean(ho_horizons):.1f} steps")

    # Save
    results_file = output_path / "debate_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return all_results
