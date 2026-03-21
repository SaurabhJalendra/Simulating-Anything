"""Batch discovery campaigns on unexplored domains and parameter axes.

Runs sequentially, auto-validates Hopf/InvHopf bifurcations, saves results.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.analysis.campaign_runner import (
    CampaignConfig,
    DiscoveryCampaignRunner,
)
from simulating_anything.analysis.observable_extractor import extract_observables
from simulating_anything.types.simulation import Domain, SimulationConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Campaign definitions ──────────────────────────────────────────────
CAMPAIGNS = [
    # === 3 UNEXPLORED DOMAINS ===
    {
        "name": "battery_thermal_alpha",
        "module": "battery_thermal",
        "cls": "BatteryThermalSimulation",
        "question": "What temperature coefficient triggers thermal runaway?",
        "sweep": {"alpha_R": (0.001, 0.05)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "battery_thermal_Iload",
        "module": "battery_thermal",
        "cls": "BatteryThermalSimulation",
        "question": "What discharge current causes thermal runaway?",
        "sweep": {"I_load": (0.5, 10.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "earthquake_sigma_c",
        "module": "earthquake_aftershock",
        "cls": "EarthquakeAftershockSimulation",
        "question": "What rupture threshold separates creep from stick-slip?",
        "sweep": {"sigma_c": (0.2, 3.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "earthquake_vplate",
        "module": "earthquake_aftershock",
        "cls": "EarthquakeAftershockSimulation",
        "question": "What plate velocity separates rare large from frequent small events?",
        "sweep": {"v_plate": (0.001, 0.1)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "npb_burst",
        "module": "nutrient_phage_bacteria",
        "cls": "NutrientPhageBacteriaSimulation",
        "question": "What phage burst size triggers kill-the-winner oscillations?",
        "sweep": {"burst": (5.0, 200.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "npb_phi",
        "module": "nutrient_phage_bacteria",
        "cls": "NutrientPhageBacteriaSimulation",
        "question": "What adsorption rate drives bacteria to collapse?",
        "sweep": {"phi": (0.001, 0.1)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    # === UNEXPLORED AXES ON EXISTING DOMAINS ===
    {
        "name": "ppc_coupling_TK",
        "module": "predator_prey_climate",
        "cls": "PredatorPreyClimateSimulation",
        "question": "What climate-ecology coupling triggers prey collapse?",
        "sweep": {"coupling_TK": (0.01, 1.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "ppc_d_pred",
        "module": "predator_prey_climate",
        "cls": "PredatorPreyClimateSimulation",
        "question": "What predator death rate destabilizes the ecosystem?",
        "sweep": {"d_pred": (0.05, 1.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "tumor_coupling_ct",
        "module": "tumor_immune",
        "cls": "TumorImmuneSimulation",
        "question": "What immunosuppression coupling allows tumor escape?",
        "sweep": {"coupling_ct": (0.001, 0.5)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "tumor_a_t",
        "module": "tumor_immune",
        "cls": "TumorImmuneSimulation",
        "question": "What tumor growth rate overwhelms immune control?",
        "sweep": {"a_t": (0.05, 1.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "social_epi_v_max",
        "module": "social_epidemic",
        "cls": "SocialEpidemicSimulation",
        "question": "What max vaccination rate prevents epidemic spread?",
        "sweep": {"v_max": (0.001, 0.2)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
    {
        "name": "social_epi_coupling_IS",
        "module": "social_epidemic",
        "cls": "SocialEpidemicSimulation",
        "question": "What infection-opinion coupling destabilizes vaccination?",
        "sweep": {"coupling_IS": (0.1, 5.0)},
        "n_points": 200, "n_steps": 5000, "dt": 0.01,
    },
]


def validate_bifurcation(module, cls_name, param_name, critical_value, dt, n_steps, base_params=None):
    """5-seed validation: check classification is unanimous on each side."""
    import importlib
    mod = importlib.import_module(f"simulating_anything.simulation.{module}")
    sim_cls = getattr(mod, cls_name)

    below_val = critical_value * 0.9
    above_val = critical_value * 1.1

    below_classes = []
    above_classes = []

    for seed in range(5):
        for label, pval, class_list in [("below", below_val, below_classes),
                                         ("above", above_val, above_classes)]:
            params = dict(base_params or {})
            params[param_name] = float(pval)
            config = SimulationConfig(
                domain=Domain.CUSTOM, dt=dt, n_steps=n_steps, parameters=params,
            )
            try:
                sim = sim_cls(config)
                sim.reset(seed=seed)
                states = [sim.observe().copy()]
                for _ in range(n_steps):
                    states.append(sim.step().copy())
                obs = extract_observables(np.array(states), dt)
                class_list.append(obs.classification)
            except Exception:
                class_list.append("divergent")

    below_unanimous = len(set(below_classes)) == 1
    above_unanimous = len(set(above_classes)) == 1
    different_sides = below_classes[0] != above_classes[0] if (below_unanimous and above_unanimous) else False

    valid = below_unanimous and above_unanimous and different_sides
    return {
        "valid": valid,
        "below": below_classes,
        "above": above_classes,
        "below_unanimous": below_unanimous,
        "above_unanimous": above_unanimous,
        "critical_value": critical_value,
        "param": param_name,
    }


def run_all():
    output_base = Path("output/discoveries")
    results_summary = []
    validation_results = {}

    # Load existing validation results
    val_path = output_base / "validation_results.json"
    if val_path.exists():
        with open(val_path) as f:
            validation_results = json.load(f)

    for i, camp in enumerate(CAMPAIGNS):
        name = camp["name"]
        print(f"\n{'='*60}")
        print(f"Campaign {i+1}/{len(CAMPAIGNS)}: {name}")
        print(f"Question: {camp['question']}")
        print(f"{'='*60}")

        config = CampaignConfig(
            domain_name=name,
            sim_module=camp["module"],
            sim_class=camp["cls"],
            question=camp["question"],
            sweep_params=camp["sweep"],
            n_points=camp.get("n_points", 200),
            n_steps=camp.get("n_steps", 5000),
            dt=camp.get("dt", 0.01),
            base_params=camp.get("base_params", {}),
        )

        try:
            runner = DiscoveryCampaignRunner(config)
            result = runner.run()

            n_bif = sum(len(b.bifurcation_points) for b in result.bifurcations)
            n_disc = len(result.discoveries)
            print(f"  Discoveries: {n_disc}, Bifurcations: {n_bif}")

            # Auto-validate any Hopf/InvHopf bifurcations
            for disc in result.discoveries:
                if disc.discovery_type == "bifurcation" and disc.critical_value is not None:
                    bif_type = disc.evidence.get("type", "")
                    if bif_type in ("hopf", "inverse_hopf"):
                        print(f"  Validating {bif_type} at {disc.parameter}={disc.critical_value:.4f}...")
                        vr = validate_bifurcation(
                            camp["module"], camp["cls"],
                            disc.parameter, disc.critical_value,
                            camp.get("dt", 0.01), camp.get("n_steps", 5000),
                            camp.get("base_params", {}),
                        )
                        key = f"{name}_{bif_type}_{disc.parameter}_{disc.critical_value:.4f}"
                        validation_results[key] = vr
                        status = "VALID" if vr["valid"] else "INVALID"
                        print(f"    {status}: below={vr['below']}, above={vr['above']}")

            results_summary.append({
                "name": name,
                "n_discoveries": n_disc,
                "n_bifurcations": n_bif,
                "runtime": result.runtime_seconds,
                "discoveries": [
                    {"type": d.discovery_type, "param": d.parameter,
                     "critical_value": d.critical_value, "description": d.description}
                    for d in result.discoveries
                ],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({"name": name, "error": str(e)})

    # Save validation results
    with open(val_path, "w") as f:
        json.dump(validation_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    total_disc = 0
    total_valid = sum(1 for v in validation_results.values() if v.get("valid"))
    for r in results_summary:
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            total_disc += r["n_discoveries"]
            print(f"  {r['name']}: {r['n_discoveries']} discoveries, {r['n_bifurcations']} bifurcations, {r['runtime']:.1f}s")

    print(f"\nTotal new discoveries: {total_disc}")
    print(f"Total validated (cumulative): {total_valid}")

    # Update master log
    master_path = output_base / "master_log.json"
    master = []
    if master_path.exists():
        with open(master_path) as f:
            master = json.load(f)
    master.extend(results_summary)
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2, default=str)

    print(f"\nMaster log updated: {len(master)} total campaign entries")


if __name__ == "__main__":
    run_all()
