"""Stochastic FitzHugh-Nagumo rediscovery.

Targets:
- Coherence resonance: CV of ISI has minimum at optimal noise sigma
- Noise-induced spiking from subthreshold regime (I < I_c)
- ISI statistics as function of noise intensity
- Comparison of subthreshold vs suprathreshold behavior under noise
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fhn_stochastic import FHNStochasticSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_noise_sweep_data(
    n_sigma: int = 20,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    current: float = 0.3,
    n_steps: int = 50000,
    dt: float = 0.01,
    n_trials: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sweep noise intensity sigma and measure spiking statistics.

    For each sigma value, runs multiple trials and averages the spike count,
    mean ISI, and CV of ISI.

    Args:
        n_sigma: number of sigma values in sweep.
        sigma_min: minimum noise intensity.
        sigma_max: maximum noise intensity.
        current: external current (default 0.3, subthreshold).
        n_steps: simulation steps per trial.
        dt: timestep.
        n_trials: number of independent trials per sigma.
        seed: base random seed.

    Returns:
        Dictionary with sigma, spike_count, mean_isi, cv arrays.
    """
    sigma_values = np.linspace(sigma_min, sigma_max, n_sigma)
    spike_counts = np.zeros(n_sigma)
    mean_isis = np.zeros(n_sigma)
    cvs = np.zeros(n_sigma)

    for i, sigma in enumerate(sigma_values):
        trial_spikes = []
        trial_isis = []
        trial_cvs = []

        for t in range(n_trials):
            config = SimulationConfig(
                domain=Domain.FHN_STOCHASTIC,
                dt=dt,
                n_steps=n_steps,
                seed=seed + i * n_trials + t,
                parameters={
                    "a": 0.7, "b": 0.8, "eps": 0.08,
                    "I": current, "sigma": sigma,
                    "v_0": -1.0, "w_0": -0.5,
                },
            )
            sim = FHNStochasticSimulation(config)
            traj = sim.run(n_steps=n_steps)
            v_trace = traj.states[:, 0]

            trial_spikes.append(sim.count_spikes(v_trace))
            trial_isis.append(sim.mean_interspike_interval(v_trace))
            trial_cvs.append(sim.coefficient_of_variation(v_trace))

        spike_counts[i] = np.mean(trial_spikes)
        mean_isis[i] = np.mean([x for x in trial_isis if x > 0] or [0.0])
        # Filter out inf CVs for averaging
        finite_cvs = [x for x in trial_cvs if np.isfinite(x)]
        cvs[i] = np.mean(finite_cvs) if finite_cvs else float("inf")

        if (i + 1) % 5 == 0:
            logger.info(
                f"  sigma={sigma:.3f}: spikes={spike_counts[i]:.1f}, "
                f"mean_ISI={mean_isis[i]:.2f}, CV={cvs[i]:.4f}"
            )

    return {
        "sigma": sigma_values,
        "spike_count": spike_counts,
        "mean_isi": mean_isis,
        "cv": cvs,
        "I": current,
        "n_steps": n_steps,
        "dt": dt,
        "n_trials": n_trials,
    }


def generate_subthreshold_vs_suprathreshold(
    n_steps: int = 30000,
    dt: float = 0.01,
    sigma: float = 0.3,
    seed: int = 100,
) -> dict[str, dict]:
    """Compare stochastic dynamics for subthreshold and suprathreshold I.

    Args:
        n_steps: simulation steps.
        dt: timestep.
        sigma: noise intensity.
        seed: random seed.

    Returns:
        Dictionary with 'subthreshold' and 'suprathreshold' sub-dictionaries.
    """
    results = {}

    for label, I_val in [("subthreshold", 0.3), ("suprathreshold", 0.5)]:
        config = SimulationConfig(
            domain=Domain.FHN_STOCHASTIC,
            dt=dt,
            n_steps=n_steps,
            seed=seed,
            parameters={
                "a": 0.7, "b": 0.8, "eps": 0.08,
                "I": I_val, "sigma": sigma,
                "v_0": -1.0, "w_0": -0.5,
            },
        )
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=n_steps)
        v_trace = traj.states[:, 0]

        n_spikes = sim.count_spikes(v_trace)
        mean_isi = sim.mean_interspike_interval(v_trace)
        cv = sim.coefficient_of_variation(v_trace)

        results[label] = {
            "I": float(I_val),
            "sigma": float(sigma),
            "n_spikes": int(n_spikes),
            "mean_isi": float(mean_isi),
            "cv": float(cv) if np.isfinite(cv) else None,
            "v_range": float(np.max(v_trace) - np.min(v_trace)),
        }
        logger.info(
            f"  {label} (I={I_val}): {n_spikes} spikes, "
            f"mean_ISI={mean_isi:.2f}, CV={cv:.4f}"
        )

    return results


def run_fhn_stochastic_rediscovery(
    output_dir: str | Path = "output/rediscovery/fhn_stochastic",
    n_sigma: int = 20,
    n_trials: int = 5,
) -> dict:
    """Run stochastic FHN rediscovery pipeline.

    Demonstrates coherence resonance -- an optimal noise level for the most
    regular (lowest CV) spiking in the subthreshold regime.

    Args:
        output_dir: directory for results output.
        n_sigma: number of noise intensity values in sweep.
        n_trials: number of independent trials per sigma.

    Returns:
        Results dictionary with noise sweep and comparison data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "fhn_stochastic",
        "targets": {
            "coherence_resonance": "CV of ISI has minimum at optimal sigma",
            "noise_induced_spiking": "Subthreshold I shows spikes under noise",
            "isi_statistics": "ISI mean and CV as function of noise",
        },
    }

    # --- Part 1: Noise sweep for coherence resonance ---
    logger.info("Part 1: Noise intensity sweep (coherence resonance)...")
    sweep_data = generate_noise_sweep_data(
        n_sigma=n_sigma,
        sigma_min=0.01,
        sigma_max=1.0,
        current=0.3,
        n_steps=50000,
        dt=0.01,
        n_trials=n_trials,
        seed=42,
    )

    # Find optimal sigma (minimum CV among finite values)
    finite_mask = np.isfinite(sweep_data["cv"])
    if np.any(finite_mask):
        finite_cvs = sweep_data["cv"][finite_mask]
        finite_sigmas = sweep_data["sigma"][finite_mask]
        optimal_idx = np.argmin(finite_cvs)
        optimal_sigma = float(finite_sigmas[optimal_idx])
        min_cv = float(finite_cvs[optimal_idx])

        results["coherence_resonance"] = {
            "optimal_sigma": optimal_sigma,
            "min_cv": min_cv,
            "n_finite_points": int(np.sum(finite_mask)),
        }
        logger.info(
            f"  Coherence resonance: optimal sigma={optimal_sigma:.3f}, "
            f"min CV={min_cv:.4f}"
        )
    else:
        results["coherence_resonance"] = {"note": "No finite CV values found"}

    # Spike rate as function of sigma
    spiking_mask = sweep_data["spike_count"] > 0
    results["noise_induced_spiking"] = {
        "n_spiking_sigma_values": int(np.sum(spiking_mask)),
        "max_spike_count": float(np.max(sweep_data["spike_count"])),
        "sigma_at_max_spikes": float(
            sweep_data["sigma"][np.argmax(sweep_data["spike_count"])]
        ),
    }

    # --- Part 2: Subthreshold vs suprathreshold comparison ---
    logger.info("Part 2: Subthreshold vs suprathreshold comparison...")
    comparison = generate_subthreshold_vs_suprathreshold(
        n_steps=30000, dt=0.01, sigma=0.3, seed=100,
    )
    results["comparison"] = comparison

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "noise_sweep.npz",
        sigma=sweep_data["sigma"],
        spike_count=sweep_data["spike_count"],
        mean_isi=sweep_data["mean_isi"],
        cv=sweep_data["cv"],
    )

    return results
