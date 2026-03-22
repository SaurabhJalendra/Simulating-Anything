"""Literature calibration: compare discovered bifurcations to published thresholds.

Validates the discovery engine against known physics and experimental data.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulating_anything.analysis.campaign_runner import (
    CampaignConfig,
    CampaignDiscovery,
    DiscoveryCampaignRunner,
)

logger = logging.getLogger(__name__)


@dataclass
class LiteratureParameter:
    """A parameter value from published literature."""
    name: str
    value: float
    units: str
    source: str
    notes: str = ""


@dataclass
class LiteratureThreshold:
    """A known bifurcation threshold from literature or analytical result."""
    parameter: str
    critical_value: float
    threshold_type: str  # hopf, inverse_hopf, fold, extinction
    source: str
    exact: bool = False  # True if analytical (e.g., Brusselator b_c = 1+a^2)
    notes: str = ""


@dataclass
class ThresholdComparison:
    """Comparison between discovered and literature threshold."""
    parameter: str
    discovered_value: float
    literature_value: float
    error_pct: float
    threshold_type: str
    match: bool  # within acceptable error


@dataclass
class CalibrationResult:
    """Complete calibration results for one domain."""
    domain: str
    literature_params: list[LiteratureParameter]
    literature_thresholds: list[LiteratureThreshold]
    discovered_thresholds: list[CampaignDiscovery]
    comparisons: list[ThresholdComparison]
    overall_error_pct: float
    runtime_seconds: float


# ── Literature parameter databases ────────────────────────────────────

NPB_LITERATURE = {
    "params": [
        LiteratureParameter("mu_max", 0.7, "1/h", "Levin et al. 1977, J Bacteriol",
                           "E. coli max growth rate in glucose-limited chemostat"),
        LiteratureParameter("K_n", 0.05, "mg/mL", "Lenski 1988, Am Nat",
                           "Monod half-saturation for E. coli on glucose"),
        LiteratureParameter("phi", 0.01, "mL/(phage*h)", "Stent 1963 (scaled)",
                           "T4 adsorption rate, scaled from 1e-9 mL/min"),
        LiteratureParameter("burst", 100.0, "phages/cell", "Ellis & Delbruck 1939",
                           "T4 phage burst size on E. coli"),
        LiteratureParameter("D_dilution", 0.2, "1/h", "Bohannan & Lenski 2000",
                           "Standard chemostat dilution rate"),
        LiteratureParameter("eta", 0.5, "1/h", "Abedon 2011",
                           "Lysis rate ~40 min latent period for T4"),
        LiteratureParameter("Y_yield", 0.5, "cells/mg", "Levin et al. 1977",
                           "Bacterial yield on glucose"),
        LiteratureParameter("d_v", 0.1, "1/h", "Suttle 2005, Nature",
                           "Phage decay rate in aquatic environments"),
        LiteratureParameter("N_in", 5.0, "mg/mL", "Standard chemostat",
                           "Nutrient feed concentration"),
        LiteratureParameter("recycle", 0.3, "fraction", "Model assumption",
                           "Nutrient recycled from lysed cells"),
    ],
    "thresholds": [
        LiteratureThreshold("burst", 50.0, "hopf",
                           "Bohannan & Lenski 2000, Ecol Lett",
                           notes="Kill-the-winner oscillations onset at burst~50"),
        LiteratureThreshold("D_dilution", 0.35, "hopf",
                           "Levin et al. 1977, J Bacteriol",
                           notes="Oscillation onset at moderate dilution"),
    ],
}

TUMOR_IMMUNE_LITERATURE = {
    "params": [
        LiteratureParameter("a_t", 0.18, "1/day", "Kuznetsov et al. 1994, Bull Math Biol",
                           "Tumor doubling time ~4 days"),
        LiteratureParameter("K_t", 100.0, "scaled", "de Pillis et al. 2005",
                           "Carrying capacity (scaled model)"),
        LiteratureParameter("d_tn", 0.02, "1/day", "de Pillis et al. 2005",
                           "NK cell killing rate per tumor interaction"),
        LiteratureParameter("d_ti", 0.05, "1/day", "Kirschner & Panetta 1998",
                           "CD8+ T cell killing rate (higher than NK)"),
        LiteratureParameter("s_n", 0.3, "cells/day", "Kuznetsov et al. 1994",
                           "NK cell baseline production"),
        LiteratureParameter("d_n", 0.1, "1/day", "de Pillis et al. 2005",
                           "NK cell natural death rate"),
        LiteratureParameter("s_i", 0.1, "cells/day", "Kirschner & Panetta 1998",
                           "CD8+ baseline production"),
        LiteratureParameter("d_i", 0.05, "1/day", "de Pillis et al. 2005",
                           "CD8+ death rate"),
        LiteratureParameter("coupling_ct", 0.05, "unitless", "Model parameter",
                           "Cytokine immunosuppression coupling"),
    ],
    "thresholds": [
        LiteratureThreshold("a_t", 0.20, "hopf",
                           "Kuznetsov et al. 1994",
                           notes="Immune escape when growth exceeds killing capacity"),
    ],
}

BRUSSELATOR_LITERATURE = {
    "params": [
        LiteratureParameter("a", 1.0, "unitless", "Prigogine & Lefever 1968",
                           "Production rate parameter"),
    ],
    "thresholds": [
        LiteratureThreshold("b", 2.0, "hopf",
                           "Prigogine & Lefever 1968, J Chem Phys",
                           exact=True, notes="Exact: b_c = 1 + a^2 = 2.0 for a=1"),
        LiteratureThreshold("b", 3.25, "hopf",
                           "Analytical: b_c = 1 + a^2",
                           exact=True, notes="Exact: b_c = 1 + 1.5^2 = 3.25 for a=1.5"),
        LiteratureThreshold("b", 5.0, "hopf",
                           "Analytical: b_c = 1 + a^2",
                           exact=True, notes="Exact: b_c = 1 + 2^2 = 5.0 for a=2"),
    ],
}

DOMAIN_LITERATURE = {
    "nutrient_phage_bacteria": NPB_LITERATURE,
    "tumor_immune": TUMOR_IMMUNE_LITERATURE,
    "brusselator": BRUSSELATOR_LITERATURE,
}


def get_literature_base_params(domain: str) -> dict[str, float]:
    """Get literature-calibrated base parameters for a domain."""
    lit = DOMAIN_LITERATURE.get(domain, {})
    params = {}
    for lp in lit.get("params", []):
        params[lp.name] = lp.value
    return params


def run_calibrated_campaign(
    domain: str,
    sweep_param: str,
    sweep_range: tuple[float, float],
    sim_module: str,
    sim_class: str,
    question: str,
    n_points: int = 300,
    n_steps: int = 10000,
    dt: float = 0.005,
    output_dir: str = "output/calibration",
) -> CalibrationResult:
    """Run a discovery campaign with literature-calibrated parameters."""
    t0 = time.time()

    lit = DOMAIN_LITERATURE.get(domain, {})
    base_params = get_literature_base_params(domain)
    # Don't include the sweep param in base_params
    base_params.pop(sweep_param, None)

    config = CampaignConfig(
        domain_name=f"cal_{domain}_{sweep_param}",
        sim_module=sim_module,
        sim_class=sim_class,
        question=question,
        sweep_params={sweep_param: sweep_range},
        n_points=n_points,
        n_steps=n_steps,
        dt=dt,
        base_params=base_params,
    )

    runner = DiscoveryCampaignRunner(config, output_dir=output_dir)
    result = runner.run()

    # Extract discovered bifurcation thresholds
    discovered = []
    for d in result.discoveries:
        if d.discovery_type == "bifurcation" and d.critical_value is not None:
            discovered.append(d)

    # Compare to literature thresholds
    comparisons = []
    lit_thresholds = [t for t in lit.get("thresholds", []) if t.parameter == sweep_param]

    for lt in lit_thresholds:
        # Find closest discovered threshold
        best_match = None
        best_error = float("inf")
        for d in discovered:
            btype = d.evidence.get("type", "")
            error = abs(d.critical_value - lt.critical_value) / abs(lt.critical_value) * 100
            if error < best_error:
                best_error = error
                best_match = d

        if best_match is not None:
            comparisons.append(ThresholdComparison(
                parameter=sweep_param,
                discovered_value=best_match.critical_value,
                literature_value=lt.critical_value,
                error_pct=best_error,
                threshold_type=lt.threshold_type,
                match=best_error < 20.0,  # within 20%
            ))
        else:
            comparisons.append(ThresholdComparison(
                parameter=sweep_param,
                discovered_value=float("nan"),
                literature_value=lt.critical_value,
                error_pct=100.0,
                threshold_type=lt.threshold_type,
                match=False,
            ))

    overall_error = np.mean([c.error_pct for c in comparisons]) if comparisons else 100.0
    runtime = time.time() - t0

    cal_result = CalibrationResult(
        domain=domain,
        literature_params=[LiteratureParameter(**{
            "name": lp.name, "value": lp.value, "units": lp.units,
            "source": lp.source, "notes": lp.notes
        }) for lp in lit.get("params", [])],
        literature_thresholds=lit_thresholds,
        discovered_thresholds=discovered,
        comparisons=comparisons,
        overall_error_pct=overall_error,
        runtime_seconds=runtime,
    )

    # Save results
    out_path = Path(output_dir) / domain
    out_path.mkdir(parents=True, exist_ok=True)
    save_calibration_result(cal_result, out_path / f"calibration_{sweep_param}.json")

    return cal_result


def save_calibration_result(result: CalibrationResult, path: Path) -> None:
    """Save calibration result to JSON."""
    data = {
        "domain": result.domain,
        "overall_error_pct": result.overall_error_pct,
        "runtime_seconds": result.runtime_seconds,
        "comparisons": [
            {
                "parameter": c.parameter,
                "discovered": c.discovered_value,
                "literature": c.literature_value,
                "error_pct": c.error_pct,
                "type": c.threshold_type,
                "match": c.match,
            }
            for c in result.comparisons
        ],
        "literature_params": [
            {"name": p.name, "value": p.value, "units": p.units, "source": p.source}
            for p in result.literature_params
        ],
        "n_discovered": len(result.discovered_thresholds),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Calibration saved to {path}")
