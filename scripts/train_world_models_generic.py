"""Train RSSM world models on any simulation domain.

Generates training data from any SimulationEnvironment subclass,
trains an RSSM world model, evaluates dreaming quality, and saves checkpoints.

Must run in WSL2 with GPU for JAX acceleration.

Usage:
    python scripts/train_world_models_generic.py --domains lorenz,kepler,seir --epochs 200
    python scripts/train_world_models_generic.py --domains all_high_r2 --epochs 200
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/world_models")

# Map domain names to (module, class, default_params) tuples
DOMAIN_REGISTRY = {
    "kepler": ("kepler", "KeplerOrbit", {}),
    "seir": ("seir", "SEIRSimulation", {"beta": 0.5, "sigma": 0.2, "gamma": 0.1, "N": 1000.0}),
    "toda_lattice": ("toda_lattice", "TodaLattice", {}),
    "schwarzschild": ("schwarzschild", "SchwarzschildGeodesic", {}),
    "quantum_oscillator": ("quantum_oscillator", "QuantumHarmonicOscillator", {}),
    "sine_gordon": ("sine_gordon", "SineGordonSimulation", {}),
    "brusselator_2d": ("brusselator_2d", "Brusselator2DSimulation", {}),
    "schnakenberg": ("schnakenberg", "SchnakenbergSimulation", {}),
    "driven_pendulum": ("driven_pendulum", "DrivenPendulum", {}),
    "boltzmann_gas": ("boltzmann_gas", "BoltzmannGas2D", {}),
    "spring_mass_chain": ("spring_mass_chain", "SpringMassChain", {}),
    "elastic_pendulum": ("elastic_pendulum", "ElasticPendulum", {}),
    "cart_pole": ("cart_pole", "CartPole", {}),
    "coupled_oscillators": ("coupled_oscillators", "CoupledOscillators", {}),
    "henon_map": ("henon_map", "HenonMapSimulation", {}),
    "standard_map": ("standard_map", "StandardMapSimulation", {}),
    "selkov": ("selkov", "SelkovSimulation", {}),
    "wilberforce": ("wilberforce", "Wilberforce", {}),
    "rossler_hyperchaos": ("rossler_hyperchaos", "RosslerHyperchaosSimulation", {}),
    "kapitza_pendulum": ("kapitza_pendulum", "KapitzaPendulumSimulation", {}),
    "rikitake": ("rikitake", "RikitakeSimulation", {}),
    "stuart_landau": ("stuart_landau", "StuartLandauSimulation", {}),
    "stommel": ("stommel", "StommelSimulation", {}),
    "vallis": ("vallis", "VallisSimulation", {}),
    "kdv": ("kdv", "KdVSimulation", {}),
    "swift_hohenberg": ("swift_hohenberg", "SwiftHohenbergSimulation", {}),
    "repressilator": ("repressilator", "RepressilatorSimulation", {}),
    "mackey_glass": ("mackey_glass", "MackeyGlassSimulation", {}),
    "wilson_cowan": ("wilson_cowan", "WilsonCowanSimulation", {}),
    "langford": ("langford", "LangfordSimulation", {}),
    "finance": ("finance", "FinanceSimulation", {}),
    "stochastic_resonance": ("stochastic_resonance", "StochasticResonanceSimulation", {
        "gamma": 0.5, "signal_amplitude": 0.3, "signal_omega": 0.1, "noise_intensity": 0.3,
    }),
    "replicator_mutator": ("replicator_mutator", "ReplicatorMutatorSimulation", {
        "n_strategies": 3.0, "mutation_rate": 0.0,
    }),
    "lorenz_stommel": ("lorenz_stommel", "LorenzStommelSimulation", {
        "sigma": 10.0, "rho": 28.0, "beta": 2.667, "eta1": 3.0, "eta2": 1.0,
        "delta": 0.3, "coupling_strength": 0.1, "ocean_feedback": 0.01,
    }),
    "climate_epidemic": ("climate_epidemic", "ClimateEpidemicSimulation", {
        "coupling_TC": 0.3, "beta_0": 0.5, "gamma_epi": 0.1, "mu_pop": 0.01,
    }),
    "neural_cardiac": ("neural_cardiac", "NeuralCardiacSimulation", {
        "coupling_nc": 0.1, "coupling_cv": 0.05, "I_ext": 0.5, "mu_c": 1.0,
    }),
    "predator_prey_climate": ("predator_prey_climate", "PredatorPreyClimateSimulation", {
        "coupling_TK": 0.2, "coupling_Peta": 0.01,
    }),
    "epidemic_economy": ("epidemic_economy", "EpidemicEconomySimulation", {
        "coupling_uS": 0.3, "coupling_Iu": 0.5,
    }),
    "neural_ecosystem": ("neural_ecosystem", "NeuralEcosystemSimulation", {
        "coupling_EN": 0.2, "coupling_NE": 0.1,
    }),
    "tumor_immune": ("tumor_immune", "TumorImmuneSimulation", {
        "coupling_ct": 0.05,
    }),
    "gene_metabolism": ("gene_metabolism", "GeneMetabolismSimulation", {
        "coupling_gm": 0.5, "coupling_mg": 0.3,
    }),
    "plankton_ocean": ("plankton_ocean", "PlanktonOceanSimulation", {
        "w_mix": 0.05,
    }),
    "social_epidemic": ("social_epidemic", "SocialEpidemicSimulation", {
        "coupling_IS": 1.0,
    }),
    "predator_prey_pollution": ("predator_prey_pollution", "PredatorPreyPollutionSimulation", {
        "coupling_CK": 0.5, "coupling_Bd": 0.1,
    }),
    "circadian_metabolism": ("circadian_metabolism", "CircadianMetabolismSimulation", {
        "coupling_PE": 0.3, "coupling_SM": 0.2,
    }),
    "prey_disease_predator": ("prey_disease_predator", "PreyDiseasePredatorSimulation", {
        "a_i": 0.6, "a_s": 0.3, "beta_d": 0.3,
    }),
    "vegetation_hydrology": ("vegetation_hydrology", "VegetationHydrologySimulation", {
        "coupling_vw": 0.5,
    }),
    "neuron_astrocyte": ("neuron_astrocyte", "NeuronAstrocyteSimulation", {
        "coupling_na": 0.2, "coupling_an": 0.1,
    }),
    "infection_immunity": ("infection_immunity", "InfectionImmunitySimulation", {
        "coupling_MR": 2.0,
    }),
    "resource_consumer_waste": ("resource_consumer_waste", "ResourceConsumerWasteSimulation", {}),
    "predator_prey_migration": ("predator_prey_migration", "PredatorPreyMigrationSimulation", {}),
    "predator_prey_fear": ("predator_prey_fear", "PredatorPreyFearSimulation", {}),
    "nutrient_phage_bacteria": ("nutrient_phage_bacteria", "NutrientPhageBacteriaSimulation", {}),
    "laser_absorber": ("laser_absorber", "LaserAbsorberSimulation", {}),
    "atmosphere_vegetation": ("atmosphere_vegetation", "AtmosphereVegetationSimulation", {}),
    "battery_thermal": ("battery_thermal", "BatteryThermalSimulation", {}),
    "earthquake_aftershock": ("earthquake_aftershock", "EarthquakeAftershockSimulation", {}),
}

# Domains already trained
ALREADY_TRAINED = {
    "brusselator", "chua", "double_pendulum", "duffing", "fitzhugh_nagumo",
    "gray_scott", "harmonic_oscillator", "heat_equation", "hodgkin_huxley",
    "kuramoto", "logistic_map", "lorenz", "lotka_volterra", "navier_stokes",
    "projectile", "rossler", "sir_epidemic", "three_species", "van_der_pol",
}


def generate_trajectories(
    domain_name: str,
    n_trajectories: int = 50,
    seq_len: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Generate trajectories from any simulation domain.

    Uses the simulation's default parameters with small random variations.
    Returns array of shape (n_trajectories, seq_len, obs_dim).
    """
    module_name, class_name, default_params = DOMAIN_REGISTRY[domain_name]
    mod = importlib.import_module(f"simulating_anything.simulation.{module_name}")
    SimClass = getattr(mod, class_name)

    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(seed)
    trajectories = []

    for i in range(n_trajectories):
        # Add small parameter variations for diversity
        params = dict(default_params)

        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.01,
            n_steps=seq_len + 100,
            parameters=params,
        )

        try:
            sim = SimClass(config)
            sim.reset(seed=int(rng.integers(0, 10000)))

            states = []
            for step in range(seq_len):
                s = sim.step()
                obs = sim.observe()
                if obs.ndim == 0:
                    obs = np.array([float(obs)])
                # Flatten spatial data if needed (keep 1D for world model)
                if obs.ndim > 1:
                    obs = obs.flatten()
                states.append(obs.copy())
                # Check for NaN/Inf
                if not np.all(np.isfinite(obs)):
                    break

            if len(states) >= seq_len // 2:
                # Pad if needed
                while len(states) < seq_len:
                    states.append(states[-1])
                trajectories.append(np.array(states[:seq_len]))
        except Exception as e:
            if i == 0:
                logger.warning(f"  Trajectory {i} failed: {e}")

    if not trajectories:
        raise RuntimeError(f"No valid trajectories generated for {domain_name}")

    data = np.array(trajectories)
    logger.info(
        f"  Generated {len(trajectories)} trajectories, shape={data.shape}, "
        f"range=[{data.min():.3f}, {data.max():.3f}]"
    )
    return data


def train_domain(
    domain: str,
    data: np.ndarray,
    n_epochs: int = 200,
    seq_len: int = 50,
    lr: float = 3e-4,
) -> dict:
    """Train RSSM world model on a domain."""
    import jax
    import jax.numpy as jnp

    from simulating_anything.types.simulation import TrainingConfig
    from simulating_anything.world_model.trainer import WorldModelTrainer

    obs_shape = data.shape[2:]
    n_traj = data.shape[0]
    logger.info(f"Training {domain}: {n_traj} trajectories, obs_shape={obs_shape}")

    config = TrainingConfig(
        learning_rate=lr,
        batch_size=1,
        sequence_length=seq_len,
        n_epochs=n_epochs,
        warmup_steps=50,
        grad_clip_norm=100.0,
        kl_free_bits=1.0,
        seed=42,
    )

    key = jax.random.PRNGKey(42)
    trainer = WorldModelTrainer(obs_shape=obs_shape, action_size=0, config=config, key=key)

    rng = np.random.default_rng(42)
    loss_history = {"total": [], "recon": [], "kl": []}
    best_loss = float("inf")

    t0 = time.time()
    for epoch in range(n_epochs):
        traj_idx = rng.integers(0, n_traj)
        traj = data[traj_idx]
        max_start = max(0, len(traj) - seq_len)
        start = rng.integers(0, max_start + 1)
        obs_seq = jnp.array(traj[start:start + seq_len], dtype=jnp.float32)

        metrics = trainer.train_step(obs_seq, actions=None)

        loss_history["total"].append(metrics["loss"])
        loss_history["recon"].append(metrics["recon_loss"])
        loss_history["kl"].append(metrics["kl_loss"])

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  [{domain}] Epoch {epoch + 1}/{n_epochs}: "
                f"loss={metrics['loss']:.4f} [{elapsed:.1f}s]"
            )

    total_time = time.time() - t0
    logger.info(f"  [{domain}] Done in {total_time:.1f}s, best_loss={best_loss:.4f}")

    # Save checkpoint
    ckpt_dir = OUTPUT_DIR / domain
    trainer.save_checkpoint(ckpt_dir, model_id=domain)

    # Evaluate dreaming
    dream_results = evaluate_dreaming(trainer, data, domain, seq_len)

    results = {
        "domain": domain,
        "n_trajectories": n_traj,
        "obs_shape": list(obs_shape),
        "n_epochs": n_epochs,
        "best_loss": float(best_loss),
        "final_loss": float(loss_history["total"][-1]),
        "training_time_s": total_time,
        "dream_results": dream_results,
    }

    with open(ckpt_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        ckpt_dir / "loss_curves.npz",
        total=np.array(loss_history["total"]),
        recon=np.array(loss_history["recon"]),
        kl=np.array(loss_history["kl"]),
    )

    return results


def evaluate_dreaming(
    trainer, data: np.ndarray, domain: str, seq_len: int = 50, dream_steps: int = 30,
) -> dict:
    """Evaluate dreaming quality."""
    import jax
    import jax.numpy as jnp

    traj = jnp.array(data[0][:seq_len], dtype=jnp.float32)
    context_len = min(20, seq_len // 2)
    dream_len = min(dream_steps, seq_len - context_len)

    encoder, rssm, decoder = trainer.params
    key = jax.random.PRNGKey(0)

    state = rssm.initial_state()
    for t in range(context_len):
        obs = traj[t]
        obs_input = obs.reshape(-1) if obs.ndim > 1 else obs
        embed = encoder(obs_input)
        action = jnp.array(0.0)
        key, step_key = jax.random.split(key)
        state, _, _ = rssm.observe_step(state, action, embed, key=step_key)

    dreamed_obs = []
    for t in range(dream_len):
        key, step_key = jax.random.split(key)
        action = jnp.array(0.0)
        state, _ = rssm.imagine_step(state, action, key=step_key)
        features = rssm.get_features(state)
        pred = decoder(features)
        dreamed_obs.append(np.array(pred))

    dreamed = np.array(dreamed_obs)
    gt = np.array(traj[context_len:context_len + dream_len])
    gt_flat = gt.reshape(dream_len, -1)
    dreamed_flat = dreamed.reshape(dream_len, -1)

    gt_symlog = np.sign(gt_flat) * np.log1p(np.abs(gt_flat))
    mse_symlog = float(np.mean((dreamed_flat - gt_symlog) ** 2))

    step_errors = [float(np.mean((dreamed_flat[t] - gt_symlog[t]) ** 2)) for t in range(dream_len)]
    growth = step_errors[-1] / max(step_errors[0], 1e-10) if step_errors else 0

    save_dir = OUTPUT_DIR / domain
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "dream_comparison.npz",
        ground_truth=np.array(traj[:context_len + dream_len]),
        dreamed=dreamed,
        context_len=context_len,
    )

    logger.info(f"  [{domain}] Dream MSE: {mse_symlog:.4f}, growth: {growth:.2f}x")
    return {"mse_symlog": mse_symlog, "error_growth_ratio": float(growth), "dream_len": dream_len}


def main():
    parser = argparse.ArgumentParser(description="Train RSSM world models on any domain")
    parser.add_argument(
        "--domains", default="all_new",
        help="Comma-separated domains, 'all_new' for untrained, or 'all' for everything",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.domains == "all":
        domains = list(DOMAIN_REGISTRY.keys())
    elif args.domains == "all_new":
        domains = [d for d in DOMAIN_REGISTRY.keys() if d not in ALREADY_TRAINED]
    else:
        domains = [d.strip() for d in args.domains.split(",")]

    logger.info(f"Training {len(domains)} domains: {domains}")
    all_results = {}
    t0 = time.time()

    for i, domain in enumerate(domains):
        logger.info(f"\n{'='*60}")
        logger.info(f"DOMAIN {i + 1}/{len(domains)}: {domain}")
        logger.info(f"{'='*60}")

        if domain not in DOMAIN_REGISTRY:
            logger.warning(f"Skipping unknown domain: {domain}")
            continue

        try:
            data = generate_trajectories(domain, n_trajectories=args.n_traj, seq_len=args.seq_len)

            # Limit obs dim for very large states
            if data.shape[-1] > 1024:
                logger.info(f"  Truncating obs from {data.shape[-1]} to 1024 dims")
                data = data[..., :1024]

            results = train_domain(domain, data, n_epochs=args.epochs)
            all_results[domain] = {k: v for k, v in results.items() if k != "loss_history"}

        except Exception as e:
            logger.error(f"  {domain} failed: {e}")
            all_results[domain] = {"error": str(e)}

    total_time = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL TRAINING COMPLETE in {total_time:.1f}s")
    logger.info(f"{'='*60}")

    for domain, r in all_results.items():
        if "error" in r:
            logger.info(f"  {domain}: FAILED - {r['error']}")
        else:
            logger.info(
                f"  {domain}: loss={r['best_loss']:.4f}, "
                f"dream_MSE={r['dream_results']['mse_symlog']:.4f}, "
                f"time={r['training_time_s']:.1f}s"
            )

    with open(OUTPUT_DIR / "training_summary_extended.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'training_summary_extended.json'}")


if __name__ == "__main__":
    main()
