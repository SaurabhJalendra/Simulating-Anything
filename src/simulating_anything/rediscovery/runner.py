"""Unified rediscovery runner -- runs all seven domain rediscoveries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_domain(name: str, run_fn, output_path: Path, **kwargs) -> tuple[dict, float]:
    """Run a single domain rediscovery with timing and error handling."""
    t0 = time.time()
    try:
        result = run_fn(output_dir=output_path / name, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"{name} completed in {elapsed:.1f}s")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"{name} rediscovery failed: {e}")
        return {"error": str(e)}, elapsed


def run_all_rediscoveries(
    output_dir: str | Path = "output/rediscovery",
    pysr_iterations: int = 40,
    domains: list[str] | None = None,
) -> dict:
    """Run domain rediscoveries and produce a unified report.

    Args:
        output_dir: Base output directory.
        pysr_iterations: Number of PySR iterations per run.
        domains: Which domains to run (default: all twelve).

    Returns:
        Combined results dict with all discoveries.
    """
    from simulating_anything.rediscovery.allee_predator_prey import (
        run_allee_predator_prey_rediscovery,
    )
    from simulating_anything.rediscovery.bak_sneppen import run_bak_sneppen_rediscovery
    from simulating_anything.rediscovery.boltzmann_gas import run_boltzmann_gas_rediscovery
    from simulating_anything.rediscovery.bouncing_ball import (
        run_bouncing_ball_rediscovery,
    )
    from simulating_anything.rediscovery.brusselator import run_brusselator_rediscovery
    from simulating_anything.rediscovery.brusselator_diffusion import (
        run_brusselator_diffusion_rediscovery,
    )
    from simulating_anything.rediscovery.bz_spiral import run_bz_spiral_rediscovery
    from simulating_anything.rediscovery.cable_equation import (
        run_cable_equation_rediscovery,
    )
    from simulating_anything.rediscovery.cart_pole import run_cart_pole_rediscovery
    from simulating_anything.rediscovery.chemostat import run_chemostat_rediscovery
    from simulating_anything.rediscovery.chua import run_chua_rediscovery
    from simulating_anything.rediscovery.competitive_lv import (
        run_competitive_lv_rediscovery,
    )
    from simulating_anything.rediscovery.coupled_lorenz import (
        run_coupled_lorenz_rediscovery,
    )
    from simulating_anything.rediscovery.coupled_oscillators import (
        run_coupled_oscillators_rediscovery,
    )
    from simulating_anything.rediscovery.damped_wave import run_damped_wave_rediscovery
    from simulating_anything.rediscovery.diffusive_lv import run_diffusive_lv_rediscovery
    from simulating_anything.rediscovery.double_pendulum import (
        run_double_pendulum_rediscovery,
    )
    from simulating_anything.rediscovery.driven_pendulum import (
        run_driven_pendulum_rediscovery,
    )
    from simulating_anything.rediscovery.duffing import run_duffing_rediscovery
    from simulating_anything.rediscovery.eco_epidemic import run_eco_epidemic_rediscovery
    from simulating_anything.rediscovery.elastic_pendulum import (
        run_elastic_pendulum_rediscovery,
    )
    from simulating_anything.rediscovery.fhn_spatial import run_fhn_spatial_rediscovery
    from simulating_anything.rediscovery.fitzhugh_nagumo import run_fitzhugh_nagumo_rediscovery
    from simulating_anything.rediscovery.ginzburg_landau import (
        run_ginzburg_landau_rediscovery,
    )
    from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
    from simulating_anything.rediscovery.harmonic_oscillator import (
        run_harmonic_oscillator_rediscovery,
    )
    from simulating_anything.rediscovery.heat_equation import run_heat_equation_rediscovery
    from simulating_anything.rediscovery.henon_map import run_henon_map_rediscovery
    from simulating_anything.rediscovery.hindmarsh_rose import run_hindmarsh_rose_rediscovery
    from simulating_anything.rediscovery.hodgkin_huxley import run_hodgkin_huxley_rediscovery
    from simulating_anything.rediscovery.ikeda_map import (
        run_ikeda_map_rediscovery,
    )
    from simulating_anything.rediscovery.ising_model import run_ising_model_rediscovery
    from simulating_anything.rediscovery.kepler import run_kepler_rediscovery
    from simulating_anything.rediscovery.kuramoto import run_kuramoto_rediscovery
    from simulating_anything.rediscovery.kuramoto_sivashinsky import (
        run_kuramoto_sivashinsky_rediscovery,
    )
    from simulating_anything.rediscovery.logistic_map import run_logistic_map_rediscovery
    from simulating_anything.rediscovery.lorenz import run_lorenz_rediscovery
    from simulating_anything.rediscovery.lorenz96 import run_lorenz96_rediscovery
    from simulating_anything.rediscovery.lotka_volterra import (
        run_lotka_volterra_rediscovery,
    )
    from simulating_anything.rediscovery.mackey_glass import run_mackey_glass_rediscovery
    from simulating_anything.rediscovery.magnetic_pendulum import (
        run_magnetic_pendulum_rediscovery,
    )
    from simulating_anything.rediscovery.may_leonard import run_may_leonard_rediscovery
    from simulating_anything.rediscovery.navier_stokes import run_navier_stokes_rediscovery
    from simulating_anything.rediscovery.oregonator import run_oregonator_rediscovery
    from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
    from simulating_anything.rediscovery.quantum_oscillator import (
        run_quantum_oscillator_rediscovery,
    )
    from simulating_anything.rediscovery.rayleigh_benard import (
        run_rayleigh_benard_rediscovery,
    )
    from simulating_anything.rediscovery.rosenzweig_macarthur import (
        run_rosenzweig_macarthur_rediscovery,
    )
    from simulating_anything.rediscovery.rossler import run_rossler_rediscovery
    from simulating_anything.rediscovery.schwarzschild import run_schwarzschild_rediscovery
    from simulating_anything.rediscovery.shallow_water import (
        run_shallow_water_rediscovery,
    )
    from simulating_anything.rediscovery.sine_gordon import (
        run_sine_gordon_rediscovery,
    )
    from simulating_anything.rediscovery.sir_epidemic import run_sir_rediscovery
    from simulating_anything.rediscovery.spring_mass_chain import (
        run_spring_mass_chain_rediscovery,
    )
    from simulating_anything.rediscovery.standard_map import run_standard_map_rediscovery
    from simulating_anything.rediscovery.swinging_atwood import (
        run_swinging_atwood_rediscovery,
    )
    from simulating_anything.rediscovery.thomas import run_thomas_rediscovery
    from simulating_anything.rediscovery.three_species import run_three_species_rediscovery
    from simulating_anything.rediscovery.van_der_pol import run_van_der_pol_rediscovery
    from simulating_anything.rediscovery.vicsek import run_vicsek_rediscovery
    from simulating_anything.rediscovery.wilberforce import run_wilberforce_rediscovery
    from simulating_anything.rediscovery.wilson_cowan import (
        run_wilson_cowan_rediscovery,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Registry of all domains
    domain_registry = {
        "projectile": {
            "label": "Projectile Range Equation",
            "fn": run_projectile_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "lotka_volterra": {
            "label": "Lotka-Volterra Equilibrium & ODE",
            "fn": run_lotka_volterra_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "gray_scott": {
            "label": "Gray-Scott Pattern Analysis",
            "fn": run_gray_scott_analysis,
            "kwargs": {},
        },
        "sir_epidemic": {
            "label": "SIR Epidemic R0 & ODE",
            "fn": run_sir_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "double_pendulum": {
            "label": "Double Pendulum Period & Energy",
            "fn": run_double_pendulum_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "harmonic_oscillator": {
            "label": "Harmonic Oscillator Frequency & Damping",
            "fn": run_harmonic_oscillator_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "lorenz": {
            "label": "Lorenz Attractor ODE & Chaos Transition",
            "fn": run_lorenz_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "navier_stokes": {
            "label": "Navier-Stokes 2D Viscous Decay",
            "fn": run_navier_stokes_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "van_der_pol": {
            "label": "Van der Pol Oscillator Limit Cycle",
            "fn": run_van_der_pol_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "kuramoto": {
            "label": "Kuramoto Synchronization Transition",
            "fn": run_kuramoto_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "brusselator": {
            "label": "Brusselator Hopf Bifurcation",
            "fn": run_brusselator_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "fitzhugh_nagumo": {
            "label": "FitzHugh-Nagumo f-I Curve",
            "fn": run_fitzhugh_nagumo_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "heat_equation": {
            "label": "Heat Equation 1D Mode Decay",
            "fn": run_heat_equation_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "logistic_map": {
            "label": "Logistic Map Feigenbaum & Chaos",
            "fn": run_logistic_map_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "boltzmann_gas": {
            "label": "Boltzmann Gas 2D Ideal Gas Law",
            "fn": run_boltzmann_gas_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "duffing": {
            "label": "Duffing Oscillator Chaos & ODE",
            "fn": run_duffing_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "quantum_oscillator": {
            "label": "Quantum Harmonic Oscillator Energy Spectrum",
            "fn": run_quantum_oscillator_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "schwarzschild": {
            "label": "Schwarzschild Geodesic ISCO & Precession",
            "fn": run_schwarzschild_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "spring_mass_chain": {
            "label": "Spring-Mass Chain Dispersion & Phonons",
            "fn": run_spring_mass_chain_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "kepler": {
            "label": "Kepler Orbit Third Law",
            "fn": run_kepler_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "driven_pendulum": {
            "label": "Driven Pendulum Period-Doubling & Chaos",
            "fn": run_driven_pendulum_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "coupled_oscillators": {
            "label": "Coupled Oscillators Normal Modes & Beats",
            "fn": run_coupled_oscillators_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "diffusive_lv": {
            "label": "Diffusive Lotka-Volterra Wave Speed & Patterns",
            "fn": run_diffusive_lv_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "damped_wave": {
            "label": "Damped Wave Equation Dispersion & Decay",
            "fn": run_damped_wave_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "ising_model": {
            "label": "2D Ising Model Phase Transition",
            "fn": run_ising_model_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "three_species": {
            "label": "Three-Species Food Chain ODE",
            "fn": run_three_species_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "cart_pole": {
            "label": "Cart-Pole Linearized Frequency",
            "fn": run_cart_pole_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "elastic_pendulum": {
            "label": "Elastic Pendulum Radial Frequency",
            "fn": run_elastic_pendulum_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "rossler": {
            "label": "Rossler Attractor ODE & Chaos",
            "fn": run_rossler_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "henon_map": {
            "label": "Henon Map Lyapunov & Chaos",
            "fn": run_henon_map_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "brusselator_diffusion": {
            "label": "Brusselator-Diffusion PDE Turing Patterns",
            "fn": run_brusselator_diffusion_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "rosenzweig_macarthur": {
            "label": "Rosenzweig-MacArthur Hopf Bifurcation",
            "fn": run_rosenzweig_macarthur_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "chua": {
            "label": "Chua Circuit Double-Scroll Chaos",
            "fn": run_chua_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "shallow_water": {
            "label": "Shallow Water Equations Wave Speed",
            "fn": run_shallow_water_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "bak_sneppen": {
            "label": "Bak-Sneppen Self-Organized Criticality",
            "fn": run_bak_sneppen_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "kuramoto_sivashinsky": {
            "label": "Kuramoto-Sivashinsky Spatiotemporal Chaos",
            "fn": run_kuramoto_sivashinsky_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "oregonator": {
            "label": "Oregonator BZ Reaction Oscillations",
            "fn": run_oregonator_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "ginzburg_landau": {
            "label": "Complex Ginzburg-Landau Benjamin-Feir Instability",
            "fn": run_ginzburg_landau_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "lorenz96": {
            "label": "Lorenz-96 Atmospheric Chaos",
            "fn": run_lorenz96_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "chemostat": {
            "label": "Chemostat Washout Bifurcation & Monod Kinetics",
            "fn": run_chemostat_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "fhn_spatial": {
            "label": "FHN Spatial PDE Excitable Medium Waves",
            "fn": run_fhn_spatial_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "wilberforce": {
            "label": "Wilberforce Pendulum Coupled Spring-Torsion",
            "fn": run_wilberforce_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "standard_map": {
            "label": "Chirikov Standard Map KAM Chaos",
            "fn": run_standard_map_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "hodgkin_huxley": {
            "label": "Hodgkin-Huxley Neuron Action Potential",
            "fn": run_hodgkin_huxley_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "eco_epidemic": {
            "label": "Eco-Epidemiological Predator-Prey with Disease",
            "fn": run_eco_epidemic_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "rayleigh_benard": {
            "label": "Rayleigh-Benard Convection Onset & Nusselt",
            "fn": run_rayleigh_benard_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "hindmarsh_rose": {
            "label": "Hindmarsh-Rose Neuron Bursting Dynamics",
            "fn": run_hindmarsh_rose_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "competitive_lv": {
            "label": "Competitive Lotka-Volterra 4-Species Exclusion",
            "fn": run_competitive_lv_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "vicsek": {
            "label": "Vicsek Model Flocking Transition",
            "fn": run_vicsek_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "magnetic_pendulum": {
            "label": "Magnetic Pendulum Fractal Basin Boundaries",
            "fn": run_magnetic_pendulum_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "bz_spiral": {
            "label": "BZ Spiral Wave 2D Oregonator PDE",
            "fn": run_bz_spiral_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "coupled_lorenz": {
            "label": "Coupled Lorenz Chaos Synchronization",
            "fn": run_coupled_lorenz_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "swinging_atwood": {
            "label": "Swinging Atwood Machine Lagrangian Chaos",
            "fn": run_swinging_atwood_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "allee_predator_prey": {
            "label": "Allee Predator-Prey Bistability",
            "fn": run_allee_predator_prey_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "mackey_glass": {
            "label": "Mackey-Glass Delay DDE Chaos",
            "fn": run_mackey_glass_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "bouncing_ball": {
            "label": "Bouncing Ball Impact Map Period-Doubling",
            "fn": run_bouncing_ball_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "wilson_cowan": {
            "label": "Wilson-Cowan Neural Population E-I Oscillation",
            "fn": run_wilson_cowan_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "cable_equation": {
            "label": "Cable Equation Passive Neurite Space Constant",
            "fn": run_cable_equation_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "may_leonard": {
            "label": "May-Leonard Cyclic Competition Heteroclinic Cycles",
            "fn": run_may_leonard_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "sine_gordon": {
            "label": "Sine-Gordon Topological Soliton Lorentz Contraction",
            "fn": run_sine_gordon_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "ikeda_map": {
            "label": "Ikeda Map Nonlinear Optical Chaos",
            "fn": run_ikeda_map_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "thomas": {
            "label": "Thomas Cyclically Symmetric Attractor",
            "fn": run_thomas_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
    }

    if domains is None:
        domains = list(domain_registry.keys())

    all_results = {}
    timings = {}

    for i, name in enumerate(domains, 1):
        if name not in domain_registry:
            logger.warning(f"Unknown domain '{name}' -- skipping")
            continue

        entry = domain_registry[name]
        logger.info("=" * 60)
        logger.info(f"REDISCOVERY {i}/{len(domains)}: {entry['label']}")
        logger.info("=" * 60)

        result, elapsed = _run_domain(
            name, entry["fn"], output_path, **entry["kwargs"]
        )
        all_results[name] = result
        timings[name] = elapsed

    # Build summary
    total_time = sum(timings.values())

    # Extract key metrics from each domain
    scorecard = {}
    for name, result in all_results.items():
        if "error" in result:
            scorecard[name] = {"status": "failed", "error": result["error"]}
            continue

        entry = {"status": "success"}

        # Try to extract best R² from various result keys
        for key in result:
            if isinstance(result[key], dict) and "best_r2" in result[key]:
                entry[f"{key}_r2"] = result[key]["best_r2"]
            if isinstance(result[key], dict) and "best" in result[key]:
                entry[f"{key}_expr"] = result[key]["best"]

        # Domain-specific metrics
        if "energy_conservation" in result:
            entry["energy_drift"] = result["energy_conservation"].get(
                "mean_final_drift", None
            )
        if "period_accuracy" in result:
            entry["period_error"] = result["period_accuracy"].get(
                "mean_relative_error", None
            )

        scorecard[name] = entry

    summary = {
        "n_domains": len(domains),
        "n_succeeded": sum(
            1 for s in scorecard.values() if s.get("status") == "success"
        ),
        "total_time_seconds": total_time,
        "timings": timings,
        "scorecard": scorecard,
        "results": all_results,
    }

    # Save combined results
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print scorecard
    logger.info("=" * 60)
    logger.info("REDISCOVERY SCORECARD")
    logger.info("=" * 60)
    for name, score in scorecard.items():
        status = score.get("status", "unknown")
        r2_keys = [k for k in score if k.endswith("_r2")]
        if r2_keys:
            best_r2 = max(score[k] for k in r2_keys)
            logger.info(f"  {name:20s}  {status:8s}  best R²={best_r2:.6f}")
        else:
            logger.info(f"  {name:20s}  {status:8s}")
    logger.info(f"\nTotal time: {total_time:.1f}s")
    logger.info(f"Summary saved to {summary_file}")
    logger.info("=" * 60)

    return summary
