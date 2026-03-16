"""Generate a comprehensive results dashboard from all discovery data.

Scans output/rediscovery/*/results.json and output/world_models/*/meta.json
to produce a console summary and a detailed JSON report at
output/results_dashboard.json.

Usage:
    python scripts/results_dashboard.py
    python scripts/results_dashboard.py --rediscovery-dir output/rediscovery
    python scripts/results_dashboard.py --output output/results_dashboard.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REDISCOVERY_DIR = PROJECT_ROOT / "output" / "rediscovery"
DEFAULT_WORLD_MODELS_DIR = PROJECT_ROOT / "output" / "world_models"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "results_dashboard.json"

# Novel discovery domains that deserve special highlighting.
NOVEL_DOMAINS = {
    "lorenz_stommel", "stochastic_resonance", "replicator_mutator",
    "climate_epidemic", "neural_cardiac",
    "predator_prey_climate", "epidemic_economy", "neural_ecosystem",
    "tumor_immune", "gene_metabolism", "plankton_ocean", "social_epidemic",
    "predator_prey_pollution", "circadian_metabolism", "prey_disease_predator",
    "vegetation_hydrology", "neuron_astrocyte", "infection_immunity",
    "resource_consumer_waste", "predator_prey_migration",
}

# ---------------------------------------------------------------------------
# Domain classification: maps domain key to mathematical class.
# Covers all 195 domains.  Domains not listed fall into "Other".
# ---------------------------------------------------------------------------
DOMAIN_CLASSIFICATION: dict[str, str] = {
    # --- ODE (linear / nonlinear / chaotic) ---
    "lotka_volterra": "ODE",
    "sir_epidemic": "ODE",
    "harmonic_oscillator": "ODE",
    "van_der_pol": "ODE",
    "brusselator": "ODE",
    "fitzhugh_nagumo": "ODE",
    "lorenz": "ODE",
    "double_pendulum": "ODE",
    "duffing": "ODE",
    "rossler": "ODE",
    "chua": "ODE",
    "chen": "ODE",
    "aizawa": "ODE",
    "halvorsen": "ODE",
    "burke_shaw": "ODE",
    "thomas": "ODE",
    "lorenz84": "ODE",
    "lorenz_84": "ODE",
    "lorenz96": "ODE",
    "lorenz_96": "ODE",
    "coupled_lorenz": "ODE",
    "stuart_landau": "ODE",
    "coupled_vdp": "ODE",
    "selkov": "ODE",
    "oregonator": "ODE",
    "rikitake": "ODE",
    "colpitts": "ODE",
    "cart_pole": "ODE",
    "three_species": "ODE",
    "elastic_pendulum": "ODE",
    "kepler": "ODE",
    "driven_pendulum": "ODE",
    "coupled_oscillators": "ODE",
    "wilberforce": "ODE",
    "swinging_atwood": "ODE",
    "hodgkin_huxley": "ODE",
    "hindmarsh_rose": "ODE",
    "morris_lecar": "ODE",
    "izhikevich": "ODE",
    "wilson_cowan": "ODE",
    "fitzhugh_rinzel": "ODE",
    "rosenzweig_macarthur": "ODE",
    "competitive_lv": "ODE",
    "allee_predator_prey": "ODE",
    "bazykin": "ODE",
    "may_leonard": "ODE",
    "eco_epidemic": "ODE",
    "predator_prey_mutualist": "ODE",
    "four_species_lv": "ODE",
    "predator_two_prey": "ODE",
    "chemostat": "ODE",
    "duffing_van_der_pol": "ODE",
    "delayed_predator_prey": "ODE",
    "mackey_glass": "ODE",
    "rabinovich_fabrikant": "ODE",
    "langford": "ODE",
    "laser_rate": "ODE",
    "sir_vaccination": "ODE",
    "lorenz_stenflo": "ODE",
    "seir": "ODE",
    "nose_hoover": "ODE",
    "lorenz_haken": "ODE",
    "sakarya": "ODE",
    "dadras": "ODE",
    "genesio_tesi": "ODE",
    "lu_chen": "ODE",
    "qi": "ODE",
    "windmi": "ODE",
    "finance": "ODE",
    "shimizu_morioka": "ODE",
    "newton_leipnik": "ODE",
    "wang": "ODE",
    "arneodo": "ODE",
    "rucklidge": "ODE",
    "liu": "ODE",
    "hadley": "ODE",
    "vallis": "ODE",
    "tigan": "ODE",
    "autocatalator": "ODE",
    "ueda": "ODE",
    "double_well": "ODE",
    "magnetic_pendulum": "ODE",
    "kapitza_pendulum": "ODE",
    "harvested_population": "ODE",
    "fhn_ring": "ODE",
    "rossler_hyperchaos": "ODE",
    "schwarzschild": "ODE",
    "three_body": "ODE",
    "lorenz_stommel": "ODE",
    "kuramoto": "ODE",
    "zombie_sir": "ODE",
    "network_sis": "ODE",
    # --- PDE ---
    "gray_scott": "PDE",
    "navier_stokes": "PDE",
    "heat_equation": "PDE",
    "kuramoto_sivashinsky": "PDE",
    "ginzburg_landau": "PDE",
    "cahn_hilliard": "PDE",
    "sine_gordon": "PDE",
    "shallow_water": "PDE",
    "schnakenberg": "PDE",
    "brusselator_diffusion": "PDE",
    "brusselator_2d": "PDE",
    "brusselator_1d": "PDE",
    "gray_scott_1d": "PDE",
    "bz_spiral": "PDE",
    "fhn_spatial": "PDE",
    "fhn_lattice": "PDE",
    "diffusive_lv": "PDE",
    "damped_wave": "PDE",
    "cable_equation": "PDE",
    "oregonator_1d": "PDE",
    "rayleigh_benard": "PDE",
    "turbulent_flow": "PDE",
    "advection_1d": "PDE",
    "burgers_1d": "PDE",
    "amari_neural_field": "PDE",
    # --- Map (discrete dynamics) ---
    "logistic_map": "Map",
    "henon_map": "Map",
    "standard_map": "Map",
    "ikeda_map": "Map",
    "coupled_map_lattice": "Map",
    "bouncing_ball": "Map",
    "ricker_map": "Map",
    "cubic_map": "Map",
    "tent_map": "Map",
    "lozi_map": "Map",
    "tinkerbell_map": "Map",
    "rulkov_map": "Map",
    "circle_map": "Map",
    # --- Stochastic / statistical mechanics ---
    "ising_model": "Stochastic",
    "bak_sneppen": "Stochastic",
    "boltzmann_gas": "Stochastic",
    "lennard_jones": "Stochastic",
    "hp_protein": "Stochastic",
    "stochastic_resonance": "Stochastic",
    "fhn_stochastic": "Stochastic",
    "toggle_switch_stochastic": "Stochastic",
    "sir_stochastic": "Stochastic",
    "sir_stochastic_test": "Stochastic",
    "fhn_stochastic_test": "Stochastic",
    "toggle_switch_stochastic_test": "Stochastic",
    # --- Agent-based / collective ---
    "vicsek": "Agent-based",
    "replicator_mutator": "Agent-based",
    # --- Algebraic / lattice ---
    "projectile": "Algebraic",
    "quantum_oscillator": "Algebraic",
    "spring_mass_chain": "Algebraic",
    "elastic_collision": "Algebraic",
    "fput": "Algebraic",
    "toda_lattice": "Algebraic",
    # --- Additional ODEs (extended domains) ---
    "age_structured": "ODE",
    "beddington_deangelis": "ODE",
    "bernoulli_ode": "ODE",
    "circadian_clock": "ODE",
    "fhn_coupled_pair": "ODE",
    "glucose_insulin": "ODE",
    "glycolytic_oscillator": "ODE",
    "goldbeter_glycolysis": "ODE",
    "gompertz": "ODE",
    "goodwin": "ODE",
    "group_defense": "ODE",
    "ivlev_predator_prey": "ODE",
    "jansen_rit": "ODE",
    "lienard": "ODE",
    "logistic_ode": "ODE",
    "lotka_volterra_delay": "ODE",
    "mapk_cascade": "ODE",
    "michaelis_menten": "ODE",
    "predator_prey_parasite": "ODE",
    "predator_prey_toxin": "ODE",
    "prey_refuge": "ODE",
    "repressilator": "ODE",
    "ring_oscillator": "ODE",
    "sei": "ODE",
    "seirs": "ODE",
    "sir_network_adaptive": "ODE",
    "sir_vital": "ODE",
    "sird": "ODE",
    "sirs": "ODE",
    "sis_endemic": "ODE",
    "sprott": "ODE",
    "stommel": "ODE",
    "theta_neuron": "ODE",
    "toggle_switch": "ODE",
    "tumor_growth": "ODE",
    "two_patch": "ODE",
    "winfree": "ODE",
    "coupled_rossler": "ODE",
    "seasonal_predator_prey": "ODE",
    "ratio_dependent": "ODE",
    "sir_metapopulation": "ODE",
    "sis_adaptive": "ODE",
    "allee_two": "ODE",
    "fhn_pulse": "ODE",
    # --- Additional PDEs ---
    "diffusion_2d": "PDE",
    "gierer_meinhardt": "PDE",
    "kdv": "PDE",
    "swift_hohenberg": "PDE",
    # --- Additional Maps ---
    "gauss_map": "Map",
}


# ---------------------------------------------------------------------------
# R-squared extraction
# ---------------------------------------------------------------------------

def _extract_r2_values(data: dict[str, Any]) -> list[float]:
    """Recursively extract all R-squared values from a results dict.

    Looks for keys containing 'r_squared', 'r2', or 'best_r2' at any
    nesting depth.  Returns a flat list of float values found.
    """
    values: list[float] = []
    _walk(data, values)
    return values


def _walk(obj: Any, acc: list[float]) -> None:
    """Depth-first walk collecting R-squared floats."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            key_lower = key.lower()
            # Direct R2 value stored as a scalar
            if isinstance(val, (int, float)) and _is_r2_key(key_lower):
                acc.append(float(val))
            else:
                _walk(val, acc)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, acc)


def _is_r2_key(key: str) -> bool:
    """Return True if *key* looks like an R-squared metric name."""
    r2_indicators = [
        "r_squared",
        "r2",
        "best_r2",
        "best_r_squared",
        "mean_r_squared",
        "min_r_squared",
        "onsager_correlation",
    ]
    for indicator in r2_indicators:
        if indicator in key:
            return True
    return False


def best_r2_for_domain(data: dict[str, Any]) -> float | None:
    """Return the single best (highest) R-squared for a domain.

    Only considers values in [0, 1].  Returns None when no valid R2 is found.
    """
    # Check explicit top-level best_r2 first
    if "best_r2" in data and isinstance(data["best_r2"], (int, float)):
        v = float(data["best_r2"])
        if 0.0 <= v <= 1.0:
            return v
    if "best_r_squared" in data and isinstance(data["best_r_squared"], (int, float)):
        v = float(data["best_r_squared"])
        if 0.0 <= v <= 1.0:
            return v

    all_r2 = _extract_r2_values(data)
    valid = [v for v in all_r2 if 0.0 <= v <= 1.0]
    if not valid:
        return None
    return max(valid)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_rediscovery_results(
    rediscovery_dir: Path,
) -> list[dict[str, Any]]:
    """Load all results.json files from *rediscovery_dir*/<domain>/.

    Returns a list of dicts, each augmented with ``_domain_key``.
    """
    results: list[dict[str, Any]] = []
    if not rediscovery_dir.is_dir():
        logger.warning("Rediscovery directory not found: %s", rediscovery_dir)
        return results

    for domain_dir in sorted(rediscovery_dir.iterdir()):
        rpath = domain_dir / "results.json"
        if not rpath.is_file():
            continue
        try:
            with open(rpath) as f:
                data = json.load(f)
            data["_domain_key"] = domain_dir.name
            results.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", rpath, exc)
    return results


def load_world_model_summaries(
    wm_dir: Path,
) -> list[dict[str, Any]]:
    """Load world model training metadata.

    Tries three sources in order:
    1. ``training_summary_full.json`` (entries with ``best_loss``)
    2. ``training_summary.json``
    3. Individual ``<domain>/meta.json`` files
    """
    models: list[dict[str, Any]] = []

    # Collect from full summary files first
    seen_domains: set[str] = set()
    for summary_name in (
        "training_summary_full.json",
        "training_summary.json",
        "training_summary_14domain.json",
    ):
        spath = wm_dir / summary_name
        if not spath.is_file():
            continue
        try:
            with open(spath) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for domain_key, info in summary.items():
            if not isinstance(info, dict):
                continue
            if domain_key in seen_domains:
                continue
            if "best_loss" not in info and info.get("status") == "already_trained":
                # Placeholder entry without metrics -- skip unless we find
                # a meta.json later.
                continue
            info.setdefault("domain", domain_key)
            models.append(info)
            seen_domains.add(domain_key)

    # Fill in from individual meta.json files
    if wm_dir.is_dir():
        for domain_dir in sorted(wm_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            if domain_dir.name in seen_domains:
                continue
            meta_path = domain_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta.setdefault("domain", domain_dir.name)
                models.append(meta)
                seen_domains.add(domain_dir.name)
            except (json.JSONDecodeError, OSError):
                pass

    return models


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_r2_statistics(
    r2_values: list[float],
) -> dict[str, float]:
    """Compute summary statistics for a list of R-squared values."""
    if not r2_values:
        return {}
    arr = np.array(r2_values)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "n_above_0.99": int(np.sum(arr >= 0.99)),
        "n_above_0.999": int(np.sum(arr >= 0.999)),
        "n_above_0.9999": int(np.sum(arr >= 0.9999)),
        "n_perfect_1.0": int(np.sum(arr >= 1.0 - 1e-12)),
    }


def classify_domain(domain_key: str) -> str:
    """Return the mathematical class for a domain key."""
    return DOMAIN_CLASSIFICATION.get(domain_key, "Other")


def build_class_groups(
    domain_r2: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Group domains by mathematical class and compute per-class stats."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for dk, r2 in domain_r2.items():
        cls = classify_domain(dk)
        groups.setdefault(cls, []).append((dk, r2))

    result: dict[str, dict[str, Any]] = {}
    for cls in sorted(groups):
        members = groups[cls]
        r2_vals = [r for _, r in members]
        result[cls] = {
            "count": len(members),
            "domains": sorted([d for d, _ in members]),
            "stats": compute_r2_statistics(r2_vals),
        }
    return result


def top_n_domains(
    domain_r2: dict[str, float],
    n: int = 10,
    highest: bool = True,
) -> list[tuple[str, float]]:
    """Return the top *n* domains by R-squared (highest or lowest)."""
    items = sorted(domain_r2.items(), key=lambda x: x[1], reverse=highest)
    return items[:n]


def build_world_model_summary(
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize world model training results."""
    if not models:
        return {"domains_trained": 0}

    losses = []
    training_times = []
    entries = []
    for m in models:
        entry: dict[str, Any] = {"domain": m.get("domain", "unknown")}
        if "best_loss" in m:
            bl = float(m["best_loss"])
            entry["best_loss"] = bl
            losses.append(bl)
        if "training_time_s" in m:
            tt = float(m["training_time_s"])
            entry["training_time_s"] = round(tt, 1)
            training_times.append(tt)
        if "n_epochs" in m:
            entry["n_epochs"] = m["n_epochs"]
        if "obs_shape" in m:
            entry["obs_shape"] = m["obs_shape"]
        dream = m.get("dream_results", {})
        if dream:
            entry["dream_mse"] = dream.get("mse_symlog")
            entry["error_growth"] = dream.get("error_growth_ratio")
        entries.append(entry)

    summary: dict[str, Any] = {
        "domains_trained": len(models),
        "models": sorted(entries, key=lambda e: e["domain"]),
    }
    if losses:
        summary["loss_stats"] = {
            "mean": float(np.mean(losses)),
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
            "median": float(np.median(losses)),
        }
    if training_times:
        summary["total_training_time_s"] = round(sum(training_times), 1)
        summary["mean_training_time_s"] = round(float(np.mean(training_times)), 1)
    return summary


def build_novel_highlights(
    all_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract highlight information for novel discovery domains."""
    highlights: dict[str, Any] = {}
    for data in all_results:
        dk = data.get("_domain_key", "")
        if dk not in NOVEL_DOMAINS:
            continue

        info: dict[str, Any] = {
            "domain": dk,
            "is_novel": bool(data.get("novel", False)),
        }

        r2 = best_r2_for_domain(data)
        if r2 is not None:
            info["best_r2"] = r2

        if dk == "lorenz_stommel":
            sindy = data.get("sindy_ode", {})
            info["sindy_r2"] = sindy.get("mean_r_squared")
            info["n_ode_equations"] = sindy.get("n_discoveries")
            lyap = data.get("lyapunov_sweep", {})
            info["coupling_effect"] = lyap.get("coupling_effect")
            info["lyapunov_range"] = [
                lyap.get("min_lyapunov"),
                lyap.get("max_lyapunov"),
            ]
            sync = data.get("synchronization", {})
            info["correlation_trend"] = sync.get("trend_direction")

        elif dk == "stochastic_resonance":
            snr = data.get("snr_sweep", {})
            info["D_opt"] = snr.get("D_opt")
            info["max_snr"] = snr.get("max_snr")
            kr = data.get("pysr_kramers", {})
            info["kramers_r2"] = kr.get("r2")

        elif dk == "replicator_mutator":
            hd = data.get("hawk_dove", {})
            hd_pysr = hd.get("pysr", {})
            info["hawk_dove_ess_r2"] = hd_pysr.get("best_r2")
            sindy = data.get("sindy_ode", {})
            info["rps_sindy_r2"] = sindy.get("best_r2")
            pd = data.get("prisoners_dilemma", {})
            info["cooperation_collapsed"] = pd.get("cooperation_collapsed")
            mut = data.get("mutation_sweep", {})
            info["mutation_stabilizes"] = mut.get("mutation_stabilizes")

        highlights[dk] = info
    return highlights


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 78


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_summary(dashboard: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    overview = dashboard.get("overview", {})
    r2_stats = dashboard.get("r2_statistics", {})
    classes = dashboard.get("domain_classes", {})
    top_high = dashboard.get("top_10_highest_r2", [])
    top_low = dashboard.get("top_10_lowest_r2", [])
    wm = dashboard.get("world_model_summary", {})
    novel = dashboard.get("novel_discoveries", {})

    # -- Overview --
    print_header("RESULTS DASHBOARD -- Simulating Anything")
    print(f"  Total domains analyzed:       {overview.get('total_domains', 0)}")
    print(f"  Domains with R2 values:       {overview.get('domains_with_r2', 0)}")
    print(f"  Domains without R2:           {overview.get('domains_without_r2', 0)}")
    print(f"  World models trained:         {wm.get('domains_trained', 0)}")
    print(f"  Novel discovery domains:      {len(novel)}")

    # -- R2 distribution --
    if r2_stats:
        print_header("R-SQUARED DISTRIBUTION")
        print(f"  Count:       {r2_stats.get('count', 0)}")
        print(f"  Mean:        {r2_stats.get('mean', 0):.6f}")
        print(f"  Median:      {r2_stats.get('median', 0):.6f}")
        print(f"  Std dev:     {r2_stats.get('std', 0):.6f}")
        print(f"  Min:         {r2_stats.get('min', 0):.6f}")
        print(f"  Max:         {r2_stats.get('max', 0):.6f}")
        print(f"  5th pctile:  {r2_stats.get('p5', 0):.6f}")
        print(f"  25th pctile: {r2_stats.get('p25', 0):.6f}")
        print(f"  75th pctile: {r2_stats.get('p75', 0):.6f}")
        print(f"  95th pctile: {r2_stats.get('p95', 0):.6f}")
        print()
        print(f"  R2 >= 0.99:    {r2_stats.get('n_above_0.99', 0)}")
        print(f"  R2 >= 0.999:   {r2_stats.get('n_above_0.999', 0)}")
        print(f"  R2 >= 0.9999:  {r2_stats.get('n_above_0.9999', 0)}")
        print(f"  R2 ~= 1.0:    {r2_stats.get('n_perfect_1.0', 0)}")

    # -- By mathematical class --
    if classes:
        print_header("DOMAINS BY MATHEMATICAL CLASS")
        for cls in sorted(classes, key=lambda c: classes[c]["count"], reverse=True):
            info = classes[cls]
            stats = info.get("stats", {})
            median = stats.get("median", 0)
            mean = stats.get("mean", 0)
            print(
                f"  {cls:<15s}  count={info['count']:>3d}"
                f"  median_R2={median:.4f}  mean_R2={mean:.4f}"
            )

    # -- Top 10 highest --
    if top_high:
        print_header("TOP 10 HIGHEST R-SQUARED")
        for i, entry in enumerate(top_high, 1):
            dk = entry["domain"]
            r2 = entry["r2"]
            cls = classify_domain(dk)
            print(f"  {i:>2d}. {dk:<35s} R2={r2:.10f}  [{cls}]")

    # -- Top 10 lowest --
    if top_low:
        print_header("TOP 10 LOWEST R-SQUARED")
        for i, entry in enumerate(top_low, 1):
            dk = entry["domain"]
            r2 = entry["r2"]
            cls = classify_domain(dk)
            print(f"  {i:>2d}. {dk:<35s} R2={r2:.10f}  [{cls}]")

    # -- World models --
    if wm.get("domains_trained", 0) > 0:
        print_header("WORLD MODEL TRAINING SUMMARY")
        print(f"  Models trained:  {wm['domains_trained']}")
        ls = wm.get("loss_stats", {})
        if ls:
            print(f"  Loss mean:       {ls.get('mean', 0):.4f}")
            print(f"  Loss min:        {ls.get('min', 0):.4f}")
            print(f"  Loss max:        {ls.get('max', 0):.4f}")
        if "total_training_time_s" in wm:
            total_min = wm["total_training_time_s"] / 60.0
            print(f"  Total time:      {total_min:.1f} min")
        if "mean_training_time_s" in wm:
            mean_min = wm["mean_training_time_s"] / 60.0
            print(f"  Mean time/model: {mean_min:.1f} min")
        print()
        wm_models = wm.get("models", [])
        # Print models with dream MSE
        dream_models = [m for m in wm_models if m.get("dream_mse") is not None]
        if dream_models:
            print("  Domain                  BestLoss   DreamMSE   ErrGrowth")
            print("  " + "-" * 60)
            for m in sorted(dream_models, key=lambda x: x.get("dream_mse", 999)):
                dm = m["domain"]
                bl = m.get("best_loss", 0)
                mse = m.get("dream_mse", 0)
                eg = m.get("error_growth", 0)
                print(f"  {dm:<24s} {bl:>8.3f}   {mse:>8.4f}   {eg:>8.3f}")

    # -- Novel discoveries --
    if novel:
        print_header("NOVEL DISCOVERY HIGHLIGHTS")
        for dk in sorted(novel):
            info = novel[dk]
            print(f"\n  --- {dk} ---")
            if info.get("is_novel"):
                print("  [NOVEL -- no known analytical results]")
            if "best_r2" in info:
                print(f"  Best R2: {info['best_r2']:.6f}")

            if dk == "lorenz_stommel":
                if info.get("sindy_r2") is not None:
                    print(f"  SINDy ODE R2:        {info['sindy_r2']:.4f}")
                print(f"  ODE equations found: {info.get('n_ode_equations', '?')}")
                print(f"  Coupling effect:     {info.get('coupling_effect', '?')}")
                lyap = info.get("lyapunov_range", [None, None])
                if lyap[0] is not None:
                    print(f"  Lyapunov range:      [{lyap[0]:.4f}, {lyap[1]:.4f}]")
                print(f"  Correlation trend:   {info.get('correlation_trend', '?')}")

            elif dk == "stochastic_resonance":
                if info.get("D_opt") is not None:
                    print(f"  Optimal noise D_opt: {info['D_opt']:.4f}")
                if info.get("max_snr") is not None:
                    print(f"  Max SNR:             {info['max_snr']:.1f}")
                if info.get("kramers_r2") is not None:
                    print(f"  Kramers rate R2:     {info['kramers_r2']:.4f}")

            elif dk == "replicator_mutator":
                if info.get("hawk_dove_ess_r2") is not None:
                    print(f"  Hawk-Dove ESS R2:    {info['hawk_dove_ess_r2']:.10f}")
                if info.get("rps_sindy_r2") is not None:
                    print(f"  RPS SINDy ODE R2:    {info['rps_sindy_r2']:.10f}")
                print(f"  PD cooperation collapsed: {info.get('cooperation_collapsed')}")
                print(f"  Mutation stabilizes:      {info.get('mutation_stabilizes')}")

    print(f"\n{SEPARATOR}")
    print("  Dashboard complete.")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dashboard(
    rediscovery_dir: Path,
    world_models_dir: Path,
) -> dict[str, Any]:
    """Build the complete dashboard data structure.

    Args:
        rediscovery_dir: Path to rediscovery results (domain subdirectories).
        world_models_dir: Path to world model training outputs.

    Returns:
        Dashboard dict suitable for JSON serialization.
    """
    logger.info("Loading rediscovery results from %s", rediscovery_dir)
    all_results = load_rediscovery_results(rediscovery_dir)
    logger.info("Loaded %d domain results", len(all_results))

    logger.info("Loading world model summaries from %s", world_models_dir)
    wm_models = load_world_model_summaries(world_models_dir)
    logger.info("Loaded %d world model entries", len(wm_models))

    # Compute best R2 per domain
    domain_r2: dict[str, float] = {}
    domains_no_r2: list[str] = []
    for data in all_results:
        dk = data["_domain_key"]
        r2 = best_r2_for_domain(data)
        if r2 is not None:
            domain_r2[dk] = r2
        else:
            domains_no_r2.append(dk)

    r2_list = list(domain_r2.values())

    # Build all sections
    r2_stats = compute_r2_statistics(r2_list)
    class_groups = build_class_groups(domain_r2)
    highest = top_n_domains(domain_r2, n=10, highest=True)
    lowest = top_n_domains(domain_r2, n=10, highest=False)
    wm_summary = build_world_model_summary(wm_models)
    novel = build_novel_highlights(all_results)

    dashboard: dict[str, Any] = {
        "overview": {
            "total_domains": len(all_results),
            "domains_with_r2": len(domain_r2),
            "domains_without_r2": len(domains_no_r2),
            "domains_without_r2_list": sorted(domains_no_r2),
        },
        "r2_statistics": r2_stats,
        "domain_classes": class_groups,
        "top_10_highest_r2": [
            {"domain": d, "r2": r, "class": classify_domain(d)} for d, r in highest
        ],
        "top_10_lowest_r2": [
            {"domain": d, "r2": r, "class": classify_domain(d)} for d, r in lowest
        ],
        "all_domain_r2": {
            k: v for k, v in sorted(domain_r2.items(), key=lambda x: x[1], reverse=True)
        },
        "world_model_summary": wm_summary,
        "novel_discoveries": novel,
    }
    return dashboard


def main() -> None:
    """Entry point: parse args, build dashboard, print and save."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive results dashboard from discovery data.",
    )
    parser.add_argument(
        "--rediscovery-dir",
        type=Path,
        default=DEFAULT_REDISCOVERY_DIR,
        help=(
            "Path to rediscovery results directory "
            f"(default: {DEFAULT_REDISCOVERY_DIR})"
        ),
    )
    parser.add_argument(
        "--world-models-dir",
        type=Path,
        default=DEFAULT_WORLD_MODELS_DIR,
        help=(
            "Path to world model outputs directory "
            f"(default: {DEFAULT_WORLD_MODELS_DIR})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dashboard = build_dashboard(
        rediscovery_dir=args.rediscovery_dir,
        world_models_dir=args.world_models_dir,
    )

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dashboard, f, indent=2)
    logger.info("Dashboard saved to %s", args.output)

    # Console output
    if not args.quiet:
        print_summary(dashboard)
        print(f"\n  Full results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
