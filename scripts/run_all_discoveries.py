"""Master discovery execution script.

Runs all rediscovery analyses across all domains using PySR + SINDy,
then trains world models on key domains.

Must run in WSL2 with GPU for JAX acceleration and Julia for PySR.

Usage:
    python scripts/run_all_discoveries.py --phase 1         # Core 14 domains
    python scripts/run_all_discoveries.py --phase 2         # Extended domains batch 1
    python scripts/run_all_discoveries.py --phase 3         # Extended domains batch 2
    python scripts/run_all_discoveries.py --phase 4         # World model training
    python scripts/run_all_discoveries.py --phase all       # Everything
    python scripts/run_all_discoveries.py --domain lorenz   # Single domain
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("output/discovery_run.log"),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/rediscovery")
WM_OUTPUT_DIR = Path("output/world_models")

# Phase 1: Core 14 domains (proven rediscoveries)
CORE_DOMAINS = [
    "projectile",
    "lotka_volterra",
    "gray_scott",
    "sir_epidemic",
    "double_pendulum",
    "harmonic_oscillator",
    "lorenz",
    "navier_stokes",
    "van_der_pol",
    "kuramoto",
    "brusselator",
    "fitzhugh_nagumo",
    "heat_equation",
    "logistic_map",
]

# Phase 2: Extended domains batch 1 -- ODEs, chaos, oscillators
EXTENDED_BATCH_1 = [
    "duffing",
    "schwarzschild",
    "quantum_oscillator",
    "boltzmann_gas",
    "spring_mass_chain",
    "kepler",
    "driven_pendulum",
    "coupled_oscillators",
    "rossler",
    "henon_map",
    "cart_pole",
    "three_species",
    "elastic_pendulum",
    "chua",
    "thomas",
    "standard_map",
    "van_der_pol",
    "stuart_landau",
    "coupled_vdp",
    "duffing_van_der_pol",
    "ueda",
    "colpitts",
    "rikitake",
    "selkov",
    "oregonator",
    "brusselator_diffusion",
    "lorenz96",
    "lorenz_84",
    "coupled_lorenz",
    "chen",
    "aizawa",
    "halvorsen",
    "burke_shaw",
    "nose_hoover",
    "sprott",
]

# Phase 3: Extended domains batch 2 -- ecology, epidemiology, PDEs, neuroscience
EXTENDED_BATCH_2 = [
    "rosenzweig_macarthur",
    "competitive_lv",
    "allee_predator_prey",
    "bazykin",
    "three_species",
    "predator_prey_mutualist",
    "predator_two_prey",
    "may_leonard",
    "eco_epidemic",
    "harvested_population",
    "sir_vaccination",
    "seir",
    "network_sis",
    "zombie_sir",
    "hodgkin_huxley",
    "hindmarsh_rose",
    "morris_lecar",
    "fitzhugh_rinzel",
    "izhikevich",
    "wilson_cowan",
    "cable_equation",
    "fhn_spatial",
    "fhn_ring",
    "fhn_lattice",
    "heat_equation",
    "damped_wave",
    "shallow_water",
    "kuramoto_sivashinsky",
    "ginzburg_landau",
    "cahn_hilliard",
    "sine_gordon",
    "bz_spiral",
    "schnakenberg",
    "brusselator_2d",
    "gray_scott_1d",
    "oregonator_1d",
    "diffusive_lv",
    "toda_lattice",
    "fput",
    "ising_model",
    "bak_sneppen",
    "vicsek",
    "bouncing_ball",
    "ikeda_map",
    "ricker_map",
    "coupled_map_lattice",
    "tent_map",
    "lozi_map",
    "cubic_map",
    "tinkerbell_map",
    "rulkov_map",
    "mackey_glass",
    "delayed_predator_prey",
    "kapitza_pendulum",
    "wilberforce",
    "swinging_atwood",
    "magnetic_pendulum",
    "elastic_collision",
    "chemostat",
    "laser_rate",
    "rayleigh_benard",
    "lennard_jones",
    "three_body",
    "turbulent_flow",
    "hp_protein",
]

# Phase 4: World model training domains
WM_DOMAINS = [
    "projectile",
    "lotka_volterra",
    "gray_scott",
    "sir_epidemic",
    "double_pendulum",
    "harmonic_oscillator",
    "lorenz",
    "navier_stokes",
    "van_der_pol",
    "kuramoto",
    "brusselator",
    "fitzhugh_nagumo",
    "heat_equation",
    "duffing",
    "rossler",
    "chua",
    "hodgkin_huxley",
    "three_species",
]


def run_rediscovery_phase(
    domain_list: list[str],
    phase_name: str,
    pysr_iterations: int = 40,
) -> dict:
    """Run rediscovery on a list of domains."""
    from simulating_anything.rediscovery.runner import run_all_rediscoveries

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE: {phase_name} -- {len(domain_list)} domains")
    logger.info(f"PySR iterations: {pysr_iterations}")
    logger.info(f"{'='*80}")

    t0 = time.time()
    results = run_all_rediscoveries(
        output_dir=str(OUTPUT_DIR),
        pysr_iterations=pysr_iterations,
        domains=domain_list,
    )
    total_time = time.time() - t0

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE {phase_name} COMPLETE in {total_time:.1f}s")
    logger.info(f"{'='*80}")

    n_success = 0
    n_error = 0
    for domain, result in results.get("domains", {}).items():
        if "error" in result:
            logger.warning(f"  FAILED: {domain}: {result['error']}")
            n_error += 1
        else:
            logger.info(f"  OK: {domain} ({result.get('elapsed', 0):.1f}s)")
            n_success += 1

    logger.info(f"  Success: {n_success}/{n_success + n_error}")

    # Save phase results
    phase_file = OUTPUT_DIR / f"phase_{phase_name.replace(' ', '_').lower()}_results.json"
    with open(phase_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results saved to {phase_file}")

    return results


def run_world_model_training(domains: list[str], n_epochs: int = 200) -> dict:
    """Train RSSM world models on specified domains."""
    import numpy as np

    WM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"WORLD MODEL TRAINING -- {len(domains)} domains, {n_epochs} epochs")
    logger.info(f"{'='*80}")

    # Import training infrastructure
    try:
        import jax
        logger.info(f"JAX devices: {jax.devices()}")
    except ImportError:
        logger.error("JAX not available -- must run in WSL2")
        return {"error": "JAX not available"}

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "train_world_models",
        Path(__file__).parent / "train_world_models.py",
    )
    _twm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_twm)
    generate_projectile_trajectories = _twm.generate_projectile_trajectories
    generate_lv_trajectories = _twm.generate_lv_trajectories
    train_domain = _twm.train_domain

    all_results = {}
    t0 = time.time()

    # Domain-specific trajectory generators
    trajectory_generators = {
        "projectile": lambda: generate_projectile_trajectories(100, 200),
        "lotka_volterra": lambda: generate_lv_trajectories(100, 200),
    }

    # Generic trajectory generator for ODE domains
    def generate_generic_trajectories(domain_name, n_traj=80, seq_len=200):
        """Generate trajectories from any simulation domain."""
        from simulating_anything.rediscovery.runner import run_all_rediscoveries

        # Use the simulation directly
        sim_module_map = {
            "sir_epidemic": ("simulating_anything.simulation.epidemiological", "SIRSimulation"),
            "harmonic_oscillator": ("simulating_anything.simulation.harmonic_oscillator", "DampedHarmonicOscillator"),
            "lorenz": ("simulating_anything.simulation.lorenz", "LorenzSimulation"),
            "van_der_pol": ("simulating_anything.simulation.van_der_pol", "VanDerPolSimulation"),
            "brusselator": ("simulating_anything.simulation.brusselator", "BrusselatorSimulation"),
            "fitzhugh_nagumo": ("simulating_anything.simulation.fitzhugh_nagumo", "FitzHughNagumoSimulation"),
            "duffing": ("simulating_anything.simulation.duffing", "DuffingOscillator"),
            "rossler": ("simulating_anything.simulation.rossler", "RosslerSimulation"),
            "chua": ("simulating_anything.simulation.chua", "ChuaCircuit"),
            "hodgkin_huxley": ("simulating_anything.simulation.hodgkin_huxley", "HodgkinHuxleySimulation"),
            "three_species": ("simulating_anything.simulation.three_species", "ThreeSpecies"),
            "kuramoto": ("simulating_anything.simulation.kuramoto", "KuramotoSimulation"),
            "double_pendulum": ("simulating_anything.simulation.chaotic_ode", "DoublePendulumSimulation"),
        }

        if domain_name not in sim_module_map:
            logger.warning(f"No trajectory generator for {domain_name}, skipping")
            return None

        mod_name, cls_name = sim_module_map[domain_name]
        import importlib
        mod = importlib.import_module(mod_name)
        SimClass = getattr(mod, cls_name)

        from simulating_anything.types.simulation import SimulationConfig, Domain
        rng = np.random.default_rng(42)
        trajectories = []

        for i in range(n_traj):
            config = SimulationConfig(
                domain=Domain.CHAOTIC_ODE, dt=0.01, n_steps=seq_len,
                parameters={},
            )
            try:
                sim = SimClass(config)
                sim.reset(seed=int(rng.integers(0, 10000)))
                states = [sim.observe().copy()]
                valid = True
                for _ in range(seq_len - 1):
                    s = sim.step()
                    if np.any(np.isnan(s)) or np.any(np.abs(s) > 1e10):
                        valid = False
                        break
                    states.append(s.copy())
                if valid and len(states) == seq_len:
                    trajectories.append(np.array(states))
            except Exception as e:
                logger.debug(f"Trajectory {i} failed for {domain_name}: {e}")
                continue

        if len(trajectories) < 10:
            logger.warning(f"Only {len(trajectories)} valid trajectories for {domain_name}")
            return None

        return np.array(trajectories)

    for domain in domains:
        logger.info(f"\n--- Training world model: {domain} ---")

        # Skip if already trained
        checkpoint_dir = WM_OUTPUT_DIR / domain
        if (checkpoint_dir / "model.eqx").exists():
            logger.info(f"  {domain}: checkpoint exists, skipping")
            all_results[domain] = {"status": "already_trained"}
            continue

        try:
            if domain in trajectory_generators:
                data = trajectory_generators[domain]()
            else:
                data = generate_generic_trajectories(domain)
        except Exception as e:
            logger.error(f"  {domain} trajectory generation failed: {e}")
            all_results[domain] = {"error": f"trajectory generation: {e}"}
            continue

        if data is None:
            logger.warning(f"Skipping {domain} -- no trajectory data")
            continue

        try:
            results = train_domain(domain, data, n_epochs=n_epochs, seq_len=50, lr=3e-4)
            all_results[domain] = {
                k: v for k, v in results.items() if k != "loss_history"
            }
            logger.info(f"  {domain}: best_loss={results['best_loss']:.4f}")
        except Exception as e:
            logger.error(f"  {domain} training failed: {e}")
            all_results[domain] = {"error": str(e)}

    total_time = time.time() - t0
    logger.info(f"\nWorld model training complete in {total_time:.1f}s")

    # Save summary
    summary_file = WM_OUTPUT_DIR / "training_summary_full.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run all scientific discoveries")
    parser.add_argument(
        "--phase", default="1",
        choices=["1", "2", "3", "4", "all"],
        help="Which phase to run (1=core 14, 2=extended batch 1, 3=extended batch 2, 4=world models, all=everything)"
    )
    parser.add_argument("--domain", type=str, default=None,
                        help="Run domain(s) by name (comma-separated for multiple)")
    parser.add_argument("--pysr-iterations", type=int, default=40,
                        help="Number of PySR iterations")
    parser.add_argument("--wm-epochs", type=int, default=200,
                        help="World model training epochs")
    args = parser.parse_args()

    Path("output").mkdir(exist_ok=True)

    if args.domain:
        domain_list = [d.strip() for d in args.domain.split(",")]
        label = f"custom_{len(domain_list)}_domains"
        run_rediscovery_phase(domain_list, label, args.pysr_iterations)
        return

    all_results = {}

    if args.phase in ("1", "all"):
        r = run_rediscovery_phase(CORE_DOMAINS, "Phase 1: Core 14", args.pysr_iterations)
        all_results["phase1"] = r

    if args.phase in ("2", "all"):
        r = run_rediscovery_phase(EXTENDED_BATCH_1, "Phase 2: Extended ODEs", args.pysr_iterations)
        all_results["phase2"] = r

    if args.phase in ("3", "all"):
        r = run_rediscovery_phase(EXTENDED_BATCH_2, "Phase 3: Extended PDEs+Eco+Neuro", args.pysr_iterations)
        all_results["phase3"] = r

    if args.phase in ("4", "all"):
        r = run_world_model_training(WM_DOMAINS, n_epochs=args.wm_epochs)
        all_results["phase4_world_models"] = r

    # Save master summary
    master_file = Path("output/master_discovery_results.json")
    with open(master_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nMaster results saved to {master_file}")


if __name__ == "__main__":
    main()
