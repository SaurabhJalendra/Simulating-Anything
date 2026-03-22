"""dt-invariance validation: verify bifurcations hold at different timesteps.

For each validated bifurcation, re-runs at dt/2 and dt*2 to confirm the
critical value does not shift by more than 5%.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulating_anything.analysis.observable_extractor import (
    TrajectoryObservables,
    extract_observables,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class DtInvarianceResult:
    """Result of dt-invariance test for one bifurcation."""
    domain: str
    parameter: str
    bif_type: str
    dt_original: float
    crit_original: float
    crit_half_dt: float | None
    crit_double_dt: float | None
    max_deviation_pct: float
    passed: bool


def find_bifurcation_at_dt(
    sim_module: str,
    sim_class: str,
    param_name: str,
    search_range: tuple[float, float],
    dt: float,
    n_points: int = 100,
    n_steps: int | None = None,
    base_params: dict | None = None,
) -> float | None:
    """Find the steady->oscillatory or oscillatory->steady transition at given dt."""
    mod = importlib.import_module(f"simulating_anything.simulation.{sim_module}")
    cls = getattr(mod, sim_class)

    if n_steps is None:
        n_steps = int(50.0 / dt)  # ~50 time units

    lo, hi = search_range
    values = np.linspace(lo, hi, n_points)
    classes = []

    for pval in values:
        params = dict(base_params or {})
        params[param_name] = float(pval)
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=dt, n_steps=n_steps, parameters=params,
        )
        try:
            sim = cls(config)
            sim.reset(seed=0)
            states = [sim.observe().copy()]
            for _ in range(n_steps):
                states.append(sim.step().copy())
            if np.any(np.isnan(states[-1])) or np.any(np.isinf(states[-1])):
                classes.append("divergent")
            else:
                obs = extract_observables(np.array(states), dt)
                classes.append(obs.classification)
        except Exception:
            classes.append("divergent")

    # Find first steady->oscillatory or oscillatory->steady transition
    for j in range(len(classes) - 1):
        if (classes[j] == "steady" and classes[j + 1] == "oscillatory") or \
           (classes[j] == "oscillatory" and classes[j + 1] == "steady"):
            return float((values[j] + values[j + 1]) / 2)

    return None


# Domain config mapping for validated bifurcations
DOMAIN_CONFIGS = {
    "npb": ("nutrient_phage_bacteria", "NutrientPhageBacteriaSimulation"),
    "laser": ("laser_absorber", "LaserAbsorberSimulation"),
    "gene_met": ("gene_metabolism", "GeneMetabolismSimulation"),
    "pdp": ("prey_disease_predator", "PreyDiseasePredatorSimulation"),
    "ppfear": ("predator_prey_fear", "PredatorPreyFearSimulation"),
    "ppmig": ("predator_prey_migration", "PredatorPreyMigrationSimulation"),
    "pppoll": ("predator_prey_pollution", "PredatorPreyPollutionSimulation"),
    "rcw": ("resource_consumer_waste", "ResourceConsumerWasteSimulation"),
    "battery": ("battery_thermal", "BatteryThermalSimulation"),
    "earthquake": ("earthquake_aftershock", "EarthquakeAftershockSimulation"),
    "circadian": ("circadian_metabolism", "CircadianMetabolismSimulation"),
    "lorenz_stommel": ("lorenz_stommel", "LorenzStommelSimulation"),
    "social": ("social_epidemic", "SocialEpidemicSimulation"),
    "Neuron": ("neuron_astrocyte", "NeuronAstrocyteSimulation"),
    "Ocean": ("ocean_carbon", "OceanCarbonSimulation"),
    "Social": ("social_epidemic", "SocialEpidemicSimulation"),
}


def resolve_domain_config(key: str) -> tuple[str, str] | None:
    """Resolve a validation key to (sim_module, sim_class)."""
    for prefix, config in DOMAIN_CONFIGS.items():
        if key.startswith(prefix):
            return config
    return None


def extract_param_from_key(key: str) -> tuple[str, float] | None:
    """Extract parameter name and critical value from validation key."""
    # Keys look like: npb_dilution_inverse_hopf_D_dilution_0.0722
    # or: battery_Iload_refined_hopf_I_load_1.7166
    for sep in ["_hopf_", "_inverse_hopf_"]:
        if sep in key:
            after = key.split(sep)[1]
            # Last part is the value, everything before is the param name
            parts = after.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    return parts[0], float(parts[1])
                except ValueError:
                    pass
    return None


def run_dt_invariance(
    validation_path: str = "output/discoveries/validation_results.json",
    dt_base: float = 0.01,
    output_path: str = "output/calibration/dt_invariance.json",
) -> list[DtInvarianceResult]:
    """Run dt-invariance on all validated bifurcations."""
    with open(validation_path) as f:
        validations = json.load(f)

    results = []
    tested = 0
    passed = 0

    for key, val in sorted(validations.items()):
        if not val.get("valid"):
            continue

        domain_config = resolve_domain_config(key)
        param_info = extract_param_from_key(key)

        if domain_config is None or param_info is None:
            logger.warning(f"Could not resolve: {key}")
            continue

        sim_module, sim_class = domain_config
        param_name, crit_value = param_info

        # Search range: +/- 50% around critical value
        search_lo = crit_value * 0.5
        search_hi = crit_value * 1.5

        tested += 1
        logger.info(f"  [{tested}] {key}: {param_name}={crit_value:.4f}")

        # Original dt
        crit_orig = find_bifurcation_at_dt(
            sim_module, sim_class, param_name, (search_lo, search_hi),
            dt=dt_base, n_points=80,
        )

        # dt/2
        crit_half = find_bifurcation_at_dt(
            sim_module, sim_class, param_name, (search_lo, search_hi),
            dt=dt_base / 2, n_points=80,
        )

        # dt*2
        crit_double = find_bifurcation_at_dt(
            sim_module, sim_class, param_name, (search_lo, search_hi),
            dt=dt_base * 2, n_points=80,
        )

        # Compute deviation
        deviations = []
        if crit_orig is not None and crit_half is not None:
            deviations.append(abs(crit_half - crit_orig) / abs(crit_orig) * 100)
        if crit_orig is not None and crit_double is not None:
            deviations.append(abs(crit_double - crit_orig) / abs(crit_orig) * 100)

        max_dev = max(deviations) if deviations else 100.0
        is_passed = max_dev < 5.0

        if is_passed:
            passed += 1

        result = DtInvarianceResult(
            domain=key,
            parameter=param_name,
            bif_type="hopf" if "_hopf_" in key else "inverse_hopf",
            dt_original=dt_base,
            crit_original=crit_value,
            crit_half_dt=crit_half,
            crit_double_dt=crit_double,
            max_deviation_pct=max_dev,
            passed=is_passed,
        )
        results.append(result)

        status = "PASS" if is_passed else "FAIL"
        logger.info(f"    {status}: orig={crit_orig}, half={crit_half}, "
                    f"double={crit_double}, dev={max_dev:.1f}%")

    # Save results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total_tested": tested,
        "total_passed": passed,
        "pass_rate": passed / tested * 100 if tested > 0 else 0,
        "results": [
            {
                "domain": r.domain,
                "parameter": r.parameter,
                "bif_type": r.bif_type,
                "crit_original": r.crit_original,
                "crit_half_dt": r.crit_half_dt,
                "crit_double_dt": r.crit_double_dt,
                "max_deviation_pct": r.max_deviation_pct,
                "passed": r.passed,
            }
            for r in results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"\ndt-invariance: {passed}/{tested} passed ({passed/tested*100:.0f}%)")
    return results
