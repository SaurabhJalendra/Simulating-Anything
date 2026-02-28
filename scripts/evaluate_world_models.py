"""Evaluate trained RSSM world models across domains.

Compares dreamed trajectories against ground-truth simulation data to
measure world model prediction quality. Key metrics:
- MSE between dreamed and actual observations
- Correlation between dreamed and actual time series
- Multi-step prediction accuracy (1-step, 5-step, 20-step)
- Coverage of phase space

Results go to output/world_models/evaluation_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Domain-to-simulation mapping
DOMAIN_CONFIGS = {
    "lorenz": {
        "module": "simulating_anything.simulation.lorenz",
        "class": "LorenzSimulation",
        "domain_enum": "CHAOTIC_ODE",
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3},
        "dt": 0.01,
        "n_steps": 500,
        "obs_dim": 3,
    },
    "van_der_pol": {
        "module": "simulating_anything.simulation.van_der_pol",
        "class": "VanDerPolSimulation",
        "domain_enum": "NONLINEAR_ODE",
        "params": {"mu": 1.0, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01,
        "n_steps": 1000,
        "obs_dim": 2,
    },
    "harmonic_oscillator": {
        "module": "simulating_anything.simulation.harmonic_oscillator",
        "class": "HarmonicOscillatorSimulation",
        "domain_enum": "LINEAR_ODE",
        "params": {"k": 4.0, "m": 1.0, "c": 0.4, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01,
        "n_steps": 1000,
        "obs_dim": 2,
    },
    "sir_epidemic": {
        "module": "simulating_anything.simulation.epidemiological",
        "class": "SIRSimulation",
        "domain_enum": "EPIDEMIOLOGICAL",
        "params": {"beta": 0.3, "gamma": 0.1, "S_0": 0.99, "I_0": 0.01},
        "dt": 0.1,
        "n_steps": 500,
        "obs_dim": 3,
    },
    "brusselator": {
        "module": "simulating_anything.simulation.brusselator",
        "class": "BrusselatorSimulation",
        "domain_enum": "NONLINEAR_ODE",
        "params": {"a": 1.0, "b": 3.0, "u_0": 1.0, "v_0": 1.0},
        "dt": 0.01,
        "n_steps": 1000,
        "obs_dim": 2,
    },
    "fitzhugh_nagumo": {
        "module": "simulating_anything.simulation.fitzhugh_nagumo",
        "class": "FitzHughNagumoSimulation",
        "domain_enum": "NONLINEAR_ODE",
        "params": {
            "a": 0.7, "b": 0.8, "eps": 0.08,
            "I_ext": 0.5, "v_0": -1.0, "w_0": -0.5,
        },
        "dt": 0.1,
        "n_steps": 500,
        "obs_dim": 2,
    },
    "double_pendulum": {
        "module": "simulating_anything.simulation.chaotic_ode",
        "class": "DoublePendulumSimulation",
        "domain_enum": "CHAOTIC_ODE",
        "params": {
            "m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
            "theta1_0": 2.0, "theta2_0": 2.5,
            "omega1_0": 0.0, "omega2_0": 0.0,
        },
        "dt": 0.001,
        "n_steps": 1000,
        "obs_dim": 4,
    },
    "projectile": {
        "module": "simulating_anything.simulation.rigid_body",
        "class": "ProjectileSimulation",
        "domain_enum": "RIGID_BODY",
        "params": {
            "initial_speed": 30.0, "launch_angle": 45.0,
            "gravity": 9.81, "drag_coefficient": 0.1, "mass": 1.0,
        },
        "dt": 0.01,
        "n_steps": 400,
        "obs_dim": 4,
    },
}


def generate_ground_truth(domain_name: str) -> np.ndarray:
    """Generate ground-truth trajectory for a domain."""
    import importlib

    cfg = DOMAIN_CONFIGS[domain_name]
    mod = importlib.import_module(cfg["module"])
    sim_class = getattr(mod, cfg["class"])

    # Build SimulationConfig
    from simulating_anything.types.simulation import Domain, SimulationConfig
    config = SimulationConfig(
        domain=getattr(Domain, cfg["domain_enum"]),
        dt=cfg["dt"],
        n_steps=cfg["n_steps"],
        parameters=cfg["params"],
    )
    sim = sim_class(config)
    traj = sim.run(n_steps=cfg["n_steps"])
    return traj.states


def evaluate_prediction_accuracy(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
) -> dict:
    """Compute prediction quality metrics between two trajectories."""
    # Align lengths
    min_len = min(len(true_traj), len(pred_traj))
    true = true_traj[:min_len]
    pred = pred_traj[:min_len]

    # MSE
    mse = np.mean((true - pred) ** 2)

    # Normalized MSE (relative to variance)
    var = np.var(true)
    nmse = mse / max(var, 1e-10)

    # Per-dimension correlation
    n_dims = true.shape[1] if true.ndim > 1 else 1
    correlations = []
    for d in range(n_dims):
        t_d = true[:, d] if true.ndim > 1 else true
        p_d = pred[:, d] if pred.ndim > 1 else pred
        if np.std(t_d) > 1e-10 and np.std(p_d) > 1e-10:
            corr = np.corrcoef(t_d, p_d)[0, 1]
            correlations.append(float(corr))
        else:
            correlations.append(0.0)

    # Multi-step accuracy: compute MSE at different horizons
    horizons = [1, 5, 20, 50, 100]
    horizon_mse = {}
    for h in horizons:
        if h < min_len:
            horizon_mse[str(h)] = float(np.mean((true[h] - pred[h]) ** 2))

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "mean_correlation": float(np.mean(correlations)),
        "per_dim_correlation": correlations,
        "horizon_mse": horizon_mse,
        "trajectory_length": min_len,
    }


def evaluate_domain(domain_name: str, model_dir: str | Path) -> dict | None:
    """Evaluate a trained world model on a single domain.

    Checks if a saved model exists, loads it, dreams trajectories,
    and compares against ground truth.
    """
    model_path = Path(model_dir) / domain_name / "best_model.eqx"
    if not model_path.exists():
        logger.info(f"  No saved model for {domain_name} at {model_path}")
        return None

    cfg = DOMAIN_CONFIGS[domain_name]
    logger.info(f"  Evaluating {domain_name} (obs_dim={cfg['obs_dim']})...")

    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx

        from simulating_anything.world_model.rssm import RSSM, RSSMState
        from simulating_anything.world_model.encoder import MLPEncoder
        from simulating_anything.world_model.decoder import MLPDecoder

        # Generate ground truth
        true_traj = generate_ground_truth(domain_name)
        obs_dim = cfg["obs_dim"]

        # Create and load model
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        rssm = RSSM(
            action_size=0,
            embed_size=256,
            hidden_size=512,
            stoch_vars=32,
            stoch_classes=32,
            key=k1,
        )
        encoder = MLPEncoder(obs_dim=obs_dim, embed_dim=256, key=k2)
        decoder = MLPDecoder(
            latent_dim=512 + 32 * 32, obs_dim=obs_dim, key=k3
        )

        # Load model weights
        rssm = eqx.tree_deserialise_leaves(str(model_path), rssm)

        # Run open-loop prediction: encode first observation, then dream
        state = rssm.initial_state()
        action = jnp.float32(0)

        # Observe first few steps to seed the model
        n_seed = min(10, len(true_traj) - 1)
        for t in range(n_seed):
            obs_jnp = jnp.array(true_traj[t], dtype=jnp.float32)
            embed = encoder(obs_jnp)
            state = rssm.observe(state, action, embed)

        # Dream the rest
        dream_steps = min(100, len(true_traj) - n_seed)
        dreamed = []
        for _ in range(dream_steps):
            state = rssm.imagine(state, action)
            latent = jnp.concatenate([state.deter, state.stoch])
            obs_pred = decoder(latent)
            dreamed.append(np.array(obs_pred))

        dreamed = np.array(dreamed)
        true_segment = true_traj[n_seed:n_seed + dream_steps]

        # Evaluate
        metrics = evaluate_prediction_accuracy(true_segment, dreamed)
        metrics["n_seed_steps"] = n_seed
        metrics["dream_steps"] = dream_steps
        return metrics

    except Exception as e:
        logger.warning(f"  Failed to evaluate {domain_name}: {e}")
        return {"error": str(e)}


def run_evaluation(
    model_dir: str = "output/world_models",
    output_dir: str = "output/world_models",
) -> dict:
    """Run world model evaluation across all domains with saved models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("WORLD MODEL EVALUATION")
    logger.info("=" * 60)

    results = {}
    for domain in DOMAIN_CONFIGS:
        result = evaluate_domain(domain, model_dir)
        if result is not None:
            results[domain] = result
            if "error" not in result:
                logger.info(
                    f"  {domain}: MSE={result['mse']:.6f}, "
                    f"corr={result['mean_correlation']:.4f}"
                )

    # Summary
    evaluated = {k: v for k, v in results.items() if "error" not in v}
    if evaluated:
        mean_corr = np.mean([v["mean_correlation"] for v in evaluated.values()])
        mean_nmse = np.mean([v["nmse"] for v in evaluated.values()])
        logger.info("")
        logger.info(f"Evaluated {len(evaluated)} domains")
        logger.info(f"Mean correlation: {mean_corr:.4f}")
        logger.info(f"Mean NMSE: {mean_nmse:.4f}")

    results["_summary"] = {
        "n_evaluated": len(evaluated),
        "n_total": len(DOMAIN_CONFIGS),
        "mean_correlation": float(mean_corr) if evaluated else 0,
        "mean_nmse": float(mean_nmse) if evaluated else 0,
    }

    # Save
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained world models")
    parser.add_argument(
        "--model-dir", default="output/world_models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output-dir", default="output/world_models",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--domain", default=None,
        help="Evaluate a single domain (default: all available)",
    )
    args = parser.parse_args()

    if args.domain:
        if args.domain not in DOMAIN_CONFIGS:
            logger.error(f"Unknown domain: {args.domain}")
            logger.error(f"Available: {list(DOMAIN_CONFIGS.keys())}")
            sys.exit(1)
        result = evaluate_domain(args.domain, args.model_dir)
        if result:
            print(json.dumps(result, indent=2))
    else:
        run_evaluation(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
