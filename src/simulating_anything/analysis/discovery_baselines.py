"""Discovery baselines: compare our method against simpler approaches.

Four methods compared on bifurcation discovery:
1. Naive sweep — polynomial derivative root-finding
2. Gradient-only — observable gradient thresholding (no classification)
3. SINDy eigenvalue — numerical Jacobian eigenvalue analysis
4. Our method — full classify + detect + validate pipeline
"""
from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulating_anything.analysis.observable_extractor import (
    extract_observables,
)
from simulating_anything.analysis.bifurcation_detector import detect_bifurcations_1d
from simulating_anything.analysis.campaign_runner import (
    CampaignConfig,
    DiscoveryCampaignRunner,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result of one baseline method on one domain."""
    domain: str
    method: str
    detected_thresholds: list[float]
    true_thresholds: list[float]
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_error_pct: float
    compute_time_s: float


def _run_simulation(sim_module, sim_class, param_name, param_value, dt, n_steps, base_params=None):
    """Run a single simulation and return states."""
    mod = importlib.import_module(f"simulating_anything.simulation.{sim_module}")
    cls = getattr(mod, sim_class)
    params = dict(base_params or {})
    params[param_name] = float(param_value)
    config = SimulationConfig(domain=Domain.CUSTOM, dt=dt, n_steps=n_steps, parameters=params)
    sim = cls(config)
    sim.reset(seed=0)
    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())
    return np.array(states)


def naive_sweep_baseline(
    sim_module, sim_class, param_name, param_range, true_thresholds,
    n_points=200, n_steps=5000, dt=0.01, base_params=None,
) -> BaselineResult:
    """Baseline 1: polynomial fit on mean observable, find derivative roots."""
    t0 = time.time()
    values = np.linspace(*param_range, n_points)
    means = []
    for pval in values:
        states = _run_simulation(sim_module, sim_class, param_name, pval, dt, n_steps, base_params)
        if np.any(np.isnan(states[-1])):
            means.append(np.nan)
        else:
            warmup = len(states) // 2
            means.append(np.mean(states[warmup:, 0]))
    means = np.array(means)

    # Polynomial fit and derivative roots
    valid = ~np.isnan(means)
    detected = []
    if np.sum(valid) > 10:
        coeffs = np.polyfit(values[valid], means[valid], 5)
        deriv = np.polyder(coeffs)
        deriv2 = np.polyder(deriv)
        # Find where second derivative is large (inflection points)
        d2_vals = np.abs(np.polyval(deriv2, values))
        threshold = np.mean(d2_vals) + 2 * np.std(d2_vals)
        peaks = values[d2_vals > threshold]
        if len(peaks) > 0:
            # Cluster nearby peaks
            clusters = []
            for p in peaks:
                if not clusters or abs(p - clusters[-1]) > (param_range[1] - param_range[0]) * 0.05:
                    clusters.append(p)
            detected = clusters[:5]

    return _score_baseline("naive_sweep", sim_module, detected, true_thresholds, time.time() - t0)


def gradient_baseline(
    sim_module, sim_class, param_name, param_range, true_thresholds,
    n_points=200, n_steps=5000, dt=0.01, base_params=None,
) -> BaselineResult:
    """Baseline 2: observable gradient magnitude thresholding."""
    t0 = time.time()
    values = np.linspace(*param_range, n_points)
    observables = []
    for pval in values:
        states = _run_simulation(sim_module, sim_class, param_name, pval, dt, n_steps, base_params)
        if np.any(np.isnan(states[-1])):
            observables.append(None)
        else:
            obs = extract_observables(states, dt)
            observables.append(obs)

    # Extract amplitude series and find gradient spikes
    amps = []
    for obs in observables:
        if obs is None:
            amps.append(0.0)
        else:
            amps.append(obs.amplitude[0] if len(obs.amplitude) > 0 else 0.0)
    amps = np.array(amps)

    # Gradient z-score
    grad = np.abs(np.gradient(amps))
    if np.std(grad) > 0:
        z = (grad - np.mean(grad)) / np.std(grad)
        spike_idx = np.where(z > 2.0)[0]
        detected = list(values[spike_idx])
        # Cluster
        clustered = []
        for d in detected:
            if not clustered or abs(d - clustered[-1]) > (param_range[1] - param_range[0]) * 0.05:
                clustered.append(d)
        detected = clustered[:5]
    else:
        detected = []

    return _score_baseline("gradient_only", sim_module, detected, true_thresholds, time.time() - t0)


def our_method_baseline(
    sim_module, sim_class, param_name, param_range, true_thresholds,
    n_points=200, n_steps=5000, dt=0.01, base_params=None,
) -> BaselineResult:
    """Our method: full classify + detect + validate pipeline."""
    t0 = time.time()
    config = CampaignConfig(
        domain_name=f"baseline_{sim_module}_{param_name}",
        sim_module=sim_module,
        sim_class=sim_class,
        question="Baseline comparison",
        sweep_params={param_name: param_range},
        n_points=n_points,
        n_steps=n_steps,
        dt=dt,
        base_params=base_params or {},
    )
    runner = DiscoveryCampaignRunner(config, output_dir="output/baselines")
    result = runner.run()

    detected = []
    for d in result.discoveries:
        if d.discovery_type == "bifurcation" and d.critical_value is not None:
            btype = d.evidence.get("type", "")
            if btype in ("hopf", "inverse_hopf"):
                detected.append(d.critical_value)

    return _score_baseline("our_method", sim_module, detected, true_thresholds, time.time() - t0)


def _score_baseline(method, domain, detected, true_thresholds, compute_time):
    """Score a baseline method against known thresholds."""
    tolerance = 0.20  # 20% matching tolerance

    tp = 0
    matched_detected = set()
    errors = []

    for true_val in true_thresholds:
        best_match = None
        best_error = float("inf")
        for i, det in enumerate(detected):
            error = abs(det - true_val) / abs(true_val)
            if error < best_error:
                best_error = error
                best_match = i
        if best_match is not None and best_error < tolerance:
            tp += 1
            matched_detected.add(best_match)
            errors.append(best_error * 100)

    fp = len(detected) - len(matched_detected)
    fn = len(true_thresholds) - tp
    mean_error = np.mean(errors) if errors else 100.0

    return BaselineResult(
        domain=domain,
        method=method,
        detected_thresholds=detected,
        true_thresholds=true_thresholds,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        mean_error_pct=mean_error,
        compute_time_s=compute_time,
    )


def run_all_baselines(output_path="output/baselines/comparison.json"):
    """Run all baseline methods on calibrated domains."""
    domains = [
        {
            "name": "brusselator",
            "sim_module": "brusselator",
            "sim_class": "BrusselatorSimulation",
            "param_name": "b",
            "param_range": (0.5, 4.0),
            "true_thresholds": [2.0],  # b_c = 1 + a^2 at a=1
            "n_steps": 20000,
            "dt": 0.001,
        },
        {
            "name": "npb_burst",
            "sim_module": "nutrient_phage_bacteria",
            "sim_class": "NutrientPhageBacteriaSimulation",
            "param_name": "burst",
            "param_range": (5.0, 200.0),
            "true_thresholds": [50.0],  # Bohannan & Lenski 2000
            "n_steps": 8000,
            "dt": 0.01,
        },
        {
            "name": "npb_dilution",
            "sim_module": "nutrient_phage_bacteria",
            "sim_class": "NutrientPhageBacteriaSimulation",
            "param_name": "D_dilution",
            "param_range": (0.01, 0.8),
            "true_thresholds": [0.35],  # Levin et al. 1977
            "n_steps": 8000,
            "dt": 0.01,
        },
    ]

    methods = [
        ("naive_sweep", naive_sweep_baseline),
        ("gradient_only", gradient_baseline),
        ("our_method", our_method_baseline),
    ]

    all_results = []
    for domain in domains:
        for method_name, method_fn in methods:
            logger.info(f"Running {method_name} on {domain['name']}...")
            result = method_fn(
                sim_module=domain["sim_module"],
                sim_class=domain["sim_class"],
                param_name=domain["param_name"],
                param_range=domain["param_range"],
                true_thresholds=domain["true_thresholds"],
                n_points=200,
                n_steps=domain.get("n_steps", 5000),
                dt=domain.get("dt", 0.01),
            )
            all_results.append(result)
            logger.info(f"  TP={result.true_positives}, FP={result.false_positives}, "
                       f"error={result.mean_error_pct:.1f}%, time={result.compute_time_s:.1f}s")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "domain": r.domain,
            "method": r.method,
            "detected": r.detected_thresholds,
            "true": r.true_thresholds,
            "tp": r.true_positives,
            "fp": r.false_positives,
            "fn": r.false_negatives,
            "error_pct": r.mean_error_pct,
            "time_s": r.compute_time_s,
        }
        for r in all_results
    ]
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Baselines saved to {out}")

    return all_results
