"""Single-factor ablation studies for causal verification."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from simulating_anything.types.discovery import AblationResult

logger = logging.getLogger(__name__)


def run_ablation(
    run_fn: Callable[[dict[str, float]], float],
    baseline_params: dict[str, float],
    factors: list[str] | None = None,
    ablation_fraction: float = 0.0,
    metric_name: str = "metric",
) -> list[AblationResult]:
    """Run single-factor ablation: zero out each parameter and measure effect.

    Args:
        run_fn: Function mapping parameters -> scalar metric value.
        baseline_params: Default parameter values.
        factors: Which parameters to ablate (default: all).
        ablation_fraction: Value to set ablated parameter to (default 0.0).
        metric_name: Name of the metric being measured.

    Returns:
        List of AblationResult, one per factor.
    """
    if factors is None:
        factors = list(baseline_params.keys())

    baseline_value = run_fn(baseline_params)
    results = []

    for factor in factors:
        if factor not in baseline_params:
            logger.warning(f"Factor '{factor}' not in baseline parameters — skipping")
            continue

        ablated_params = baseline_params.copy()
        ablated_params[factor] = ablation_fraction

        try:
            ablated_value = run_fn(ablated_params)
        except Exception as e:
            logger.warning(f"Ablation of '{factor}' failed: {e}")
            ablated_value = float("nan")

        if baseline_value != 0:
            effect_size = abs(ablated_value - baseline_value) / abs(baseline_value)
        else:
            effect_size = abs(ablated_value - baseline_value)

        results.append(
            AblationResult(
                factor_name=factor,
                original_value=float(baseline_value),
                ablated_value=float(ablated_value),
                effect_size=float(effect_size),
                is_essential=effect_size > 0.5,
                description=(
                    f"Setting {factor}={ablation_fraction}: "
                    f"{metric_name} changed from {baseline_value:.4g} to {ablated_value:.4g} "
                    f"(effect size {effect_size:.2%})"
                ),
            )
        )

    # Sort by effect size descending
    results.sort(key=lambda r: r.effect_size, reverse=True)
    return results
