"""Cross-domain world model transfer experiments for V7 Phase 7A.

Loads a trained RSSM world model from one domain and evaluates how well it
predicts dynamics from a different but mathematically related domain. This
tests whether world models learn transferable latent representations of
dynamical systems.

Transfer pairs tested:
1. lorenz -> rossler (both 3D chaotic ODEs)
2. sir_epidemic -> seir (both compartmental epidemics)
3. harmonic_oscillator -> duffing (linear -> nonlinear oscillator)
4. lotka_volterra -> three_species (2-species -> 3-species ecology)
5. brusselator -> fitzhugh_nagumo (both Hopf bifurcation systems)
6. van_der_pol -> duffing (both nonlinear oscillators)

Must run in WSL2 with GPU for JAX acceleration.

Usage:
    python scripts/cross_domain_transfer.py
    python scripts/cross_domain_transfer.py --pairs lorenz:rossler
    python scripts/cross_domain_transfer.py --context-steps 30 --dream-steps 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("output/world_models")
OUTPUT_DIR = Path("output/cross_domain_transfer")

# Each entry: (source_domain, target_domain, description)
TRANSFER_PAIRS = [
    ("lorenz", "rossler", "3D chaotic ODEs"),
    ("sir_epidemic", "seir", "compartmental epidemics"),
    ("harmonic_oscillator", "duffing", "linear -> nonlinear oscillator"),
    ("lotka_volterra", "three_species", "2-species -> 3-species ecology"),
    ("brusselator", "fitzhugh_nagumo", "Hopf bifurcation systems"),
    ("van_der_pol", "duffing", "nonlinear oscillators"),
    ("chua", "lorenz", "dual 3D chaotic attractors"),
    ("selkov", "brusselator", "2D chemical oscillators"),
    ("stochastic_resonance", "van_der_pol", "2D nonlinear dynamics"),
    ("replicator_mutator", "lorenz", "3D nonlinear dynamics"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_world_model(domain: str):
    """Load a trained RSSM world model from its checkpoint.

    Rebuilds a WorldModelTrainer with the same architecture that was used
    during training, then deserialises the saved parameters into it.

    Args:
        domain: Name of the domain whose model to load.

    Returns:
        Tuple of ((encoder, rssm, decoder), metadata_dict).

    Raises:
        FileNotFoundError: If the checkpoint files do not exist.
    """
    import equinox as eqx
    import jax

    from simulating_anything.types.simulation import TrainingConfig
    from simulating_anything.world_model.trainer import WorldModelTrainer

    ckpt_dir = CHECKPOINT_DIR / domain
    meta_path = ckpt_dir / "meta.json"
    model_path = ckpt_dir / "model.eqx"

    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json found at {meta_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No model.eqx found at {model_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    obs_shape = tuple(meta["obs_shape"])
    action_size = meta["action_size"]

    logger.info(
        f"Loading {domain} world model: obs_shape={obs_shape}, "
        f"action_size={action_size}, steps_trained={meta['step_count']}"
    )

    # Rebuild trainer with identical architecture to create the parameter tree
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=1,
        sequence_length=50,
        n_epochs=100,
        warmup_steps=50,
        grad_clip_norm=100.0,
        kl_free_bits=1.0,
        seed=42,
    )
    key = jax.random.PRNGKey(42)
    trainer = WorldModelTrainer(
        obs_shape=obs_shape, action_size=action_size, config=config, key=key
    )

    # Deserialise saved weights into the parameter tree
    trainer.params = eqx.tree_deserialise_leaves(str(model_path), trainer.params)

    encoder, rssm, decoder = trainer.params
    logger.info(
        f"  Loaded: feature_size={rssm.feature_size}, "
        f"hidden={rssm.hidden_size}, stoch={rssm.stoch_vars}x{rssm.stoch_classes}"
    )
    return (encoder, rssm, decoder), meta


# ---------------------------------------------------------------------------
# Target domain trajectory generation
# ---------------------------------------------------------------------------

def generate_target_trajectories(
    domain: str,
    n_trajectories: int = 20,
    seq_len: int = 200,
) -> np.ndarray:
    """Generate test trajectories from a target domain simulation.

    Args:
        domain: Target domain name.
        n_trajectories: Number of trajectories to generate.
        seq_len: Length of each trajectory in timesteps.

    Returns:
        Array of shape (n_trajectories, seq_len, obs_dim).
    """
    rng = np.random.default_rng(123)
    trajectories = []

    for i in range(n_trajectories):
        config, sim = _create_simulation(domain, rng)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))

    return np.array(trajectories)


def _create_simulation(domain: str, rng: np.random.Generator):
    """Create a simulation instance for the given domain with varied params.

    Args:
        domain: Domain name string.
        rng: Numpy random generator for parameter variation.

    Returns:
        Tuple of (SimulationConfig, SimulationEnvironment).
    """
    from simulating_anything.types.simulation import Domain, SimulationConfig

    if domain == "rossler":
        from simulating_anything.simulation.rossler import RosslerSimulation

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "a": rng.uniform(0.15, 0.25),
                "b": rng.uniform(0.15, 0.25),
                "c": rng.uniform(4.0, 7.0),
                "x_0": rng.uniform(0.5, 2.0),
                "y_0": rng.uniform(0.5, 2.0),
                "z_0": rng.uniform(0.0, 1.0),
            },
        )
        return config, RosslerSimulation(config)

    if domain == "seir":
        from simulating_anything.simulation.seir import SEIRSimulation

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=300,
            parameters={
                "beta": rng.uniform(0.3, 0.8),
                "sigma": rng.uniform(0.1, 0.3),
                "gamma": rng.uniform(0.05, 0.2),
                "N": 1000.0,
                "S_0": rng.uniform(900.0, 995.0),
                "E_0": rng.uniform(1.0, 10.0),
                "I_0": rng.uniform(1.0, 10.0),
                "R_0_init": 0.0,
            },
        )
        return config, SEIRSimulation(config)

    if domain == "duffing":
        from simulating_anything.simulation.duffing import DuffingOscillator

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "alpha": rng.uniform(0.5, 2.0),
                "beta": rng.uniform(0.5, 2.0),
                "delta": rng.uniform(0.1, 0.4),
                "gamma_f": rng.uniform(0.1, 0.5),
                "omega": rng.uniform(0.8, 1.2),
                "x_0": rng.uniform(-1.0, 1.0),
                "v_0": rng.uniform(-0.5, 0.5),
            },
        )
        return config, DuffingOscillator(config)

    if domain == "three_species":
        from simulating_anything.simulation.three_species import ThreeSpecies

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "a1": rng.uniform(0.8, 1.5),
                "b1": rng.uniform(0.3, 0.7),
                "a2": rng.uniform(0.3, 0.7),
                "b2": rng.uniform(0.1, 0.3),
                "a3": rng.uniform(0.2, 0.5),
                "x0": rng.uniform(0.5, 2.0),
                "y0": rng.uniform(0.3, 1.0),
                "z0": rng.uniform(0.3, 1.0),
            },
        )
        return config, ThreeSpecies(config)

    if domain == "fitzhugh_nagumo":
        from simulating_anything.simulation.fitzhugh_nagumo import (
            FitzHughNagumoSimulation,
        )

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=300,
            parameters={
                "a": rng.uniform(0.5, 0.9),
                "b": rng.uniform(0.6, 1.0),
                "eps": rng.uniform(0.05, 0.12),
                "I": rng.uniform(0.3, 0.7),
                "v_0": rng.uniform(-1.5, 0.5),
                "w_0": rng.uniform(-1.0, 0.0),
            },
        )
        return config, FitzHughNagumoSimulation(config)

    if domain == "lorenz":
        from simulating_anything.simulation.lorenz import LorenzSimulation

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "sigma": rng.uniform(8.0, 12.0),
                "rho": rng.uniform(24.0, 32.0),
                "beta": rng.uniform(2.0, 3.5),
                "x_0": rng.uniform(0.5, 2.0),
                "y_0": rng.uniform(0.5, 2.0),
                "z_0": rng.uniform(0.5, 2.0),
            },
        )
        return config, LorenzSimulation(config)

    if domain == "sir_epidemic":
        from simulating_anything.simulation.epidemiological import SIRSimulation

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=300,
            parameters={
                "beta": rng.uniform(0.2, 0.5),
                "gamma": rng.uniform(0.05, 0.15),
                "S_0": rng.uniform(0.95, 0.99),
                "I_0": rng.uniform(0.01, 0.05),
                "R_0_init": 0.0,
            },
        )
        return config, SIRSimulation(config)

    if domain == "harmonic_oscillator":
        from simulating_anything.simulation.harmonic_oscillator import (
            DampedHarmonicOscillator,
        )

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "m": rng.uniform(0.5, 2.0),
                "k": rng.uniform(1.0, 5.0),
                "c": rng.uniform(0.0, 0.5),
                "x_0": rng.uniform(0.5, 2.0),
                "v_0": 0.0,
            },
        )
        return config, DampedHarmonicOscillator(config)

    if domain == "lotka_volterra":
        from simulating_anything.simulation.agent_based import (
            LotkaVolterraSimulation,
        )

        config = SimulationConfig(
            domain=Domain.AGENT_BASED, dt=0.01, n_steps=300,
            parameters={
                "alpha": rng.uniform(0.8, 1.5),
                "beta": rng.uniform(0.2, 0.6),
                "gamma": rng.uniform(0.2, 0.6),
                "delta": rng.uniform(0.05, 0.2),
                "prey_0": rng.uniform(20.0, 60.0),
                "predator_0": rng.uniform(5.0, 15.0),
            },
        )
        return config, LotkaVolterraSimulation(config)

    if domain == "brusselator":
        from simulating_anything.simulation.brusselator import (
            BrusselatorSimulation,
        )

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "a": rng.uniform(0.8, 1.5),
                "b": rng.uniform(2.5, 4.0),
                "u_0": rng.uniform(0.5, 1.5),
                "v_0": rng.uniform(0.5, 1.5),
            },
        )
        return config, BrusselatorSimulation(config)

    if domain == "van_der_pol":
        from simulating_anything.simulation.van_der_pol import (
            VanDerPolSimulation,
        )

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "mu": rng.uniform(0.5, 3.0),
                "x_0": rng.uniform(0.05, 0.5),
                "v_0": 0.0,
            },
        )
        return config, VanDerPolSimulation(config)

    if domain == "chua":
        from simulating_anything.simulation.chua import ChuaCircuit

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.001, n_steps=300,
            parameters={
                "alpha": rng.uniform(9.0, 11.0),
                "beta": rng.uniform(14.0, 16.0),
                "m0": rng.uniform(-1.3, -1.1),
                "m1": rng.uniform(-0.8, -0.6),
                "x_0": rng.uniform(-0.5, 0.5),
                "y_0": rng.uniform(-0.1, 0.1),
                "z_0": rng.uniform(-0.1, 0.1),
            },
        )
        return config, ChuaCircuit(config)

    if domain == "selkov":
        from simulating_anything.simulation.selkov import SelkovSimulation

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "a": rng.uniform(0.05, 0.15),
                "b": rng.uniform(0.5, 1.0),
                "x_0": rng.uniform(0.5, 2.0),
                "y_0": rng.uniform(0.5, 2.0),
            },
        )
        return config, SelkovSimulation(config)

    if domain == "stochastic_resonance":
        from simulating_anything.simulation.stochastic_resonance import (
            StochasticResonanceSimulation,
        )

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "gamma": rng.uniform(0.3, 0.7),
                "signal_amplitude": rng.uniform(0.2, 0.4),
                "signal_omega": rng.uniform(0.08, 0.12),
                "noise_intensity": rng.uniform(0.1, 0.5),
                "x_0": rng.uniform(-1.0, 1.0),
                "v_0": 0.0,
            },
        )
        return config, StochasticResonanceSimulation(config)

    if domain == "replicator_mutator":
        from simulating_anything.simulation.replicator_mutator import (
            ReplicatorMutatorSimulation,
        )

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=300,
            parameters={
                "n_strategies": 3.0,
                "mutation_rate": 0.0,
                "A_0_0": 0.0,
                "A_0_1": rng.uniform(-1.0, 1.0),
                "A_0_2": rng.uniform(-1.0, 1.0),
                "A_1_0": rng.uniform(-1.0, 1.0),
                "A_1_1": 0.0,
                "A_1_2": rng.uniform(-1.0, 1.0),
                "A_2_0": rng.uniform(-1.0, 1.0),
                "A_2_1": rng.uniform(-1.0, 1.0),
                "A_2_2": 0.0,
            },
        )
        return config, ReplicatorMutatorSimulation(config)

    raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Transfer evaluation
# ---------------------------------------------------------------------------

def check_dimension_compatibility(
    source_meta: dict, target_domain: str, target_data: np.ndarray
) -> bool:
    """Check whether source model and target data have matching obs dims.

    Args:
        source_meta: Metadata dict from the source model checkpoint.
        target_domain: Name of the target domain (for logging).
        target_data: Target trajectory array (n_traj, seq_len, obs_dim).

    Returns:
        True if dimensions match, False otherwise.
    """
    source_obs_shape = tuple(source_meta["obs_shape"])
    target_obs_dim = target_data.shape[2:]

    if source_obs_shape != target_obs_dim:
        logger.warning(
            f"  Dimension mismatch: source obs_shape={source_obs_shape}, "
            f"target obs_shape={target_obs_dim}. Skipping."
        )
        return False
    return True


def run_in_domain_baseline(
    source_domain: str,
    encoder,
    rssm,
    decoder,
    context_steps: int = 20,
    dream_steps: int = 30,
) -> dict:
    """Run in-domain dreaming as a baseline for comparison.

    Generates trajectories from the source domain and evaluates dream quality
    using the source model on its own data.

    Args:
        source_domain: Domain that the model was trained on.
        encoder: RSSM encoder module.
        rssm: RSSM core module.
        decoder: RSSM decoder module.
        context_steps: Number of ground-truth observation steps to feed.
        dream_steps: Number of imagination steps to run.

    Returns:
        Dict with in-domain evaluation metrics.
    """
    import jax.numpy as jnp

    logger.info(f"  Running in-domain baseline on {source_domain}...")
    data = generate_target_trajectories(source_domain, n_trajectories=5, seq_len=200)

    all_mse = []
    all_corr = []

    for traj_idx in range(min(5, len(data))):
        traj = jnp.array(data[traj_idx], dtype=jnp.float32)
        seq_len = len(traj)
        actual_context = min(context_steps, seq_len // 2)
        actual_dream = min(dream_steps, seq_len - actual_context)

        if actual_dream < 1:
            continue

        metrics = _dream_and_evaluate(
            encoder, rssm, decoder, traj, actual_context, actual_dream
        )
        all_mse.append(metrics["mse_symlog"])
        if metrics["correlation"] is not None:
            all_corr.append(metrics["correlation"])

    result = {
        "mean_mse_symlog": float(np.mean(all_mse)) if all_mse else float("nan"),
        "std_mse_symlog": float(np.std(all_mse)) if all_mse else float("nan"),
        "mean_correlation": float(np.mean(all_corr)) if all_corr else float("nan"),
        "n_trajectories": len(all_mse),
    }
    logger.info(
        f"  In-domain {source_domain}: MSE={result['mean_mse_symlog']:.4f} "
        f"(+/- {result['std_mse_symlog']:.4f}), "
        f"corr={result['mean_correlation']:.4f}"
    )
    return result


def run_cross_domain_evaluation(
    source_domain: str,
    target_domain: str,
    encoder,
    rssm,
    decoder,
    target_data: np.ndarray,
    context_steps: int = 20,
    dream_steps: int = 30,
) -> dict:
    """Evaluate how well the source model predicts target domain dynamics.

    Feeds target observations through the source model's encoder, then
    dreams forward and compares against actual target trajectories.

    Args:
        source_domain: Domain the model was trained on.
        target_domain: Domain whose trajectories are being predicted.
        encoder: RSSM encoder module.
        rssm: RSSM core module.
        decoder: RSSM decoder module.
        target_data: Target trajectories (n_traj, seq_len, obs_dim).
        context_steps: Number of observation steps to feed before dreaming.
        dream_steps: Number of imagination steps to predict.

    Returns:
        Dict with cross-domain evaluation metrics.
    """
    import jax.numpy as jnp

    logger.info(
        f"  Evaluating {source_domain} -> {target_domain} "
        f"(context={context_steps}, dream={dream_steps})..."
    )

    all_mse = []
    all_corr = []
    all_step_errors = []

    for traj_idx in range(min(10, len(target_data))):
        traj = jnp.array(target_data[traj_idx], dtype=jnp.float32)
        seq_len = len(traj)
        actual_context = min(context_steps, seq_len // 2)
        actual_dream = min(dream_steps, seq_len - actual_context)

        if actual_dream < 1:
            continue

        metrics = _dream_and_evaluate(
            encoder, rssm, decoder, traj, actual_context, actual_dream
        )
        all_mse.append(metrics["mse_symlog"])
        if metrics["correlation"] is not None:
            all_corr.append(metrics["correlation"])
        all_step_errors.append(metrics["step_errors"])

    # Compute mean step-wise error growth across trajectories
    max_steps = max(len(se) for se in all_step_errors) if all_step_errors else 0
    mean_step_errors = []
    for t in range(max_steps):
        step_vals = [se[t] for se in all_step_errors if t < len(se)]
        mean_step_errors.append(float(np.mean(step_vals)))

    error_growth = (
        mean_step_errors[-1] / max(mean_step_errors[0], 1e-10)
        if len(mean_step_errors) >= 2
        else float("nan")
    )

    result = {
        "mean_mse_symlog": float(np.mean(all_mse)) if all_mse else float("nan"),
        "std_mse_symlog": float(np.std(all_mse)) if all_mse else float("nan"),
        "mean_correlation": float(np.mean(all_corr)) if all_corr else float("nan"),
        "std_correlation": float(np.std(all_corr)) if all_corr else float("nan"),
        "error_growth_ratio": error_growth,
        "mean_step_errors": mean_step_errors,
        "n_trajectories": len(all_mse),
    }
    logger.info(
        f"  Cross-domain {source_domain}->{target_domain}: "
        f"MSE={result['mean_mse_symlog']:.4f} (+/- {result['std_mse_symlog']:.4f}), "
        f"corr={result['mean_correlation']:.4f}, "
        f"error_growth={error_growth:.2f}x"
    )
    return result


def _dream_and_evaluate(
    encoder,
    rssm,
    decoder,
    trajectory: np.ndarray,
    context_steps: int,
    dream_steps: int,
) -> dict:
    """Feed context observations then dream forward and compare.

    Args:
        encoder: RSSM encoder module.
        rssm: RSSM core module.
        decoder: RSSM decoder module.
        trajectory: Full trajectory as a JAX array (seq_len, obs_dim).
        context_steps: Number of steps to observe before dreaming.
        dream_steps: Number of steps to imagine.

    Returns:
        Dict with mse_symlog, correlation, and step_errors.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    action = jnp.float32(0)

    # Feed context observations to build up the latent state
    state = rssm.initial_state()
    for t in range(context_steps):
        obs = trajectory[t]
        obs_flat = obs.reshape(-1)
        embed = encoder(obs_flat)
        key, step_key = jax.random.split(key)
        state, _, _ = rssm.observe_step(state, action, embed, key=step_key)

    # Dream forward without observations
    dreamed_obs = []
    for t in range(dream_steps):
        key, step_key = jax.random.split(key)
        state, _ = rssm.imagine_step(state, action, key=step_key)
        features = rssm.get_features(state)
        pred = decoder(features)
        dreamed_obs.append(np.array(pred))

    dreamed = np.array(dreamed_obs)

    # Ground truth for the dreamed segment
    gt = np.array(trajectory[context_steps:context_steps + dream_steps])
    gt_flat = gt.reshape(dream_steps, -1)
    dreamed_flat = dreamed.reshape(dream_steps, -1)

    # Compute symlog of ground truth for fair comparison (decoder outputs symlog)
    gt_symlog = np.sign(gt_flat) * np.log1p(np.abs(gt_flat))

    # MSE in symlog space
    mse_symlog = float(np.mean((dreamed_flat - gt_symlog) ** 2))

    # Per-step error for error growth analysis
    step_errors = []
    for t in range(dream_steps):
        err = float(np.mean((dreamed_flat[t] - gt_symlog[t]) ** 2))
        step_errors.append(err)

    # Pearson correlation between dreamed and true dynamics
    correlation = _safe_correlation(dreamed_flat.flatten(), gt_symlog.flatten())

    return {
        "mse_symlog": mse_symlog,
        "correlation": correlation,
        "step_errors": step_errors,
    }


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Compute Pearson correlation, returning None if undefined.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Pearson correlation coefficient or None if variance is zero.
    """
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_transfer_experiment(
    source_domain: str,
    target_domain: str,
    description: str,
    context_steps: int = 20,
    dream_steps: int = 30,
) -> dict:
    """Run a single cross-domain transfer experiment.

    Loads the source model, generates target trajectories, checks dimension
    compatibility, runs both in-domain and cross-domain evaluation, and
    returns a combined results dict.

    Args:
        source_domain: Domain whose trained model to use.
        target_domain: Domain whose dynamics to predict.
        description: Human-readable description of the pair.
        context_steps: Observation context length before dreaming.
        dream_steps: Number of dreamed prediction steps.

    Returns:
        Dict with all experiment results and metadata.
    """
    t0 = time.time()
    result = {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "description": description,
        "context_steps": context_steps,
        "dream_steps": dream_steps,
        "status": "pending",
    }

    # Load source model
    try:
        (encoder, rssm, decoder), source_meta = load_world_model(source_domain)
    except FileNotFoundError as e:
        logger.error(f"  Cannot load {source_domain} model: {e}")
        result["status"] = "model_not_found"
        result["error"] = str(e)
        return result

    # Load target meta to check dimensions before generating trajectories
    target_meta_path = CHECKPOINT_DIR / target_domain / "meta.json"
    if target_meta_path.exists():
        with open(target_meta_path) as f:
            target_meta = json.load(f)
        target_obs_shape = tuple(target_meta["obs_shape"])
    else:
        # Infer from simulation by generating a single trajectory
        target_obs_shape = None

    source_obs_shape = tuple(source_meta["obs_shape"])

    # Check dimension compatibility early if we can
    if target_obs_shape is not None and source_obs_shape != target_obs_shape:
        logger.warning(
            f"  Dimension mismatch: {source_domain} obs={source_obs_shape} vs "
            f"{target_domain} obs={target_obs_shape}. "
            f"Direct transfer not possible."
        )
        result["status"] = "dimension_mismatch"
        result["source_obs_shape"] = list(source_obs_shape)
        result["target_obs_shape"] = list(target_obs_shape)
        result["elapsed_s"] = time.time() - t0
        return result

    # Generate target trajectories
    logger.info(f"  Generating {target_domain} test trajectories...")
    target_data = generate_target_trajectories(
        target_domain, n_trajectories=20, seq_len=200
    )
    logger.info(
        f"  Generated {target_data.shape[0]} trajectories, "
        f"shape={target_data.shape}"
    )

    # Final dimension check with actual data
    if not check_dimension_compatibility(source_meta, target_domain, target_data):
        result["status"] = "dimension_mismatch"
        result["source_obs_shape"] = list(source_obs_shape)
        result["target_obs_shape"] = list(target_data.shape[2:])
        result["elapsed_s"] = time.time() - t0
        return result

    # In-domain baseline
    in_domain = run_in_domain_baseline(
        source_domain, encoder, rssm, decoder,
        context_steps=context_steps, dream_steps=dream_steps,
    )

    # Cross-domain evaluation
    cross_domain = run_cross_domain_evaluation(
        source_domain, target_domain, encoder, rssm, decoder,
        target_data, context_steps=context_steps, dream_steps=dream_steps,
    )

    # Compute transfer ratio (cross-domain MSE / in-domain MSE)
    in_mse = in_domain["mean_mse_symlog"]
    cross_mse = cross_domain["mean_mse_symlog"]
    transfer_ratio = cross_mse / max(in_mse, 1e-10)

    # Strip step_errors from saved results to keep JSON compact
    cross_domain_summary = {
        k: v for k, v in cross_domain.items() if k != "mean_step_errors"
    }

    result.update({
        "status": "success",
        "source_obs_shape": list(source_obs_shape),
        "target_obs_shape": list(target_data.shape[2:]),
        "in_domain": in_domain,
        "cross_domain": cross_domain_summary,
        "transfer_ratio": transfer_ratio,
        "elapsed_s": time.time() - t0,
    })

    logger.info(
        f"  Transfer ratio ({source_domain}->{target_domain}): "
        f"{transfer_ratio:.2f}x "
        f"(in-domain MSE={in_mse:.4f}, cross-domain MSE={cross_mse:.4f})"
    )

    return result


def main() -> None:
    """Run all cross-domain transfer experiments and save results."""
    parser = argparse.ArgumentParser(
        description="Cross-domain world model transfer experiments"
    )
    parser.add_argument(
        "--pairs", type=str, default="all",
        help=(
            "Comma-separated list of transfer pairs as source:target, "
            "or 'all' for all pairs. Example: lorenz:rossler,brusselator:fitzhugh_nagumo"
        ),
    )
    parser.add_argument(
        "--context-steps", type=int, default=20,
        help="Number of ground-truth observation steps before dreaming (default: 20)",
    )
    parser.add_argument(
        "--dream-steps", type=int, default=30,
        help="Number of imagination steps to predict (default: 30)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which pairs to run
    if args.pairs == "all":
        pairs = TRANSFER_PAIRS
    else:
        pairs = []
        for pair_str in args.pairs.split(","):
            src, tgt = pair_str.strip().split(":")
            # Find description from predefined pairs, or use a generic one
            desc = "custom pair"
            for s, t, d in TRANSFER_PAIRS:
                if s == src and t == tgt:
                    desc = d
                    break
            pairs.append((src, tgt, desc))

    logger.info("=" * 70)
    logger.info("CROSS-DOMAIN WORLD MODEL TRANSFER EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Pairs to test: {len(pairs)}")
    logger.info(f"Context steps: {args.context_steps}, Dream steps: {args.dream_steps}")

    all_results = []
    t0 = time.time()

    for i, (source, target, desc) in enumerate(pairs):
        logger.info("")
        logger.info(f"{'='*70}")
        logger.info(f"[{i + 1}/{len(pairs)}] {source} -> {target} ({desc})")
        logger.info(f"{'='*70}")

        result = run_transfer_experiment(
            source, target, desc,
            context_steps=args.context_steps,
            dream_steps=args.dream_steps,
        )
        all_results.append(result)

    total_time = time.time() - t0

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"ALL EXPERIMENTS COMPLETE in {total_time:.1f}s")
    logger.info("=" * 70)

    successful = [r for r in all_results if r["status"] == "success"]
    mismatched = [r for r in all_results if r["status"] == "dimension_mismatch"]
    failed = [r for r in all_results if r["status"] not in ("success", "dimension_mismatch")]

    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Dimension mismatch (skipped): {len(mismatched)}")
    logger.info(f"  Failed: {len(failed)}")

    if successful:
        logger.info("")
        logger.info("Transfer Results:")
        logger.info(f"  {'Source':<20} {'Target':<20} {'In-domain MSE':>14} "
                     f"{'Cross MSE':>10} {'Ratio':>8} {'Corr':>8}")
        logger.info(f"  {'-'*18:<20} {'-'*18:<20} {'-'*14:>14} "
                     f"{'-'*10:>10} {'-'*8:>8} {'-'*8:>8}")
        for r in successful:
            logger.info(
                f"  {r['source_domain']:<20} {r['target_domain']:<20} "
                f"{r['in_domain']['mean_mse_symlog']:>14.4f} "
                f"{r['cross_domain']['mean_mse_symlog']:>10.4f} "
                f"{r['transfer_ratio']:>8.2f} "
                f"{r['cross_domain']['mean_correlation']:>8.4f}"
            )

    if mismatched:
        logger.info("")
        logger.info("Dimension Mismatches:")
        for r in mismatched:
            logger.info(
                f"  {r['source_domain']} ({r.get('source_obs_shape')}) -> "
                f"{r['target_domain']} ({r.get('target_obs_shape')})"
            )

    # Save results
    summary = {
        "experiment": "cross_domain_world_model_transfer",
        "context_steps": args.context_steps,
        "dream_steps": args.dream_steps,
        "total_time_s": total_time,
        "n_pairs": len(pairs),
        "n_successful": len(successful),
        "n_dimension_mismatch": len(mismatched),
        "n_failed": len(failed),
        "results": all_results,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
