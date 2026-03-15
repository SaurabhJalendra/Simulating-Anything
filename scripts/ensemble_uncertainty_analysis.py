"""Ensemble uncertainty analysis: compare predictions across domain models.

Uses multiple trained RSSM models to generate predictions for the same
trajectories, measuring disagreement as epistemic uncertainty. This enables:
1. Identifying which dynamical regimes are hardest to predict
2. Estimating model confidence without ground-truth
3. Detecting out-of-distribution dynamics

This is a key scientific result: universal uncertainty quantification
across dynamical system classes without domain-specific calibration.

Usage (must run in WSL2 for GPU):
    python scripts/ensemble_uncertainty_analysis.py
"""
from __future__ import annotations

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

WM_DIR = Path("output/world_models")
OUTPUT_DIR = Path("output/ensemble_analysis")

# Group domains by obs dim for ensemble comparison
# Models with same obs dim can form an ensemble (shared input/output space)
DOMAIN_GROUPS = {
    "2D_oscillators": [
        "harmonic_oscillator", "van_der_pol", "brusselator",
        "fitzhugh_nagumo", "lotka_volterra", "selkov",
        "wilson_cowan", "stommel", "stochastic_resonance",
    ],
    "3D_chaotic": [
        "lorenz", "rossler", "chua", "sir_epidemic",
        "three_species", "replicator_mutator", "langford",
        "driven_pendulum",
    ],
    "4D_systems": [
        "double_pendulum", "seir", "hodgkin_huxley", "kepler",
        "elastic_pendulum", "coupled_oscillators", "neural_cardiac",
    ],
}


def load_model(domain: str):
    """Load a trained RSSM world model.

    Args:
        domain: Domain name.

    Returns:
        Tuple of (encoder, rssm, decoder, meta) or None on failure.
    """
    import equinox as eqx
    import jax

    from simulating_anything.types.simulation import TrainingConfig
    from simulating_anything.world_model.trainer import WorldModelTrainer

    meta_path = WM_DIR / domain / "meta.json"
    model_path = WM_DIR / domain / "model.eqx"

    if not meta_path.exists() or not model_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    obs_shape = tuple(meta["obs_shape"])
    if len(obs_shape) >= 2 and obs_shape[-1] > 4 and obs_shape[-2] > 4:
        return None

    try:
        key = jax.random.PRNGKey(42)
        tc = TrainingConfig(
            learning_rate=3e-4, batch_size=1, sequence_length=50,
            n_epochs=100, warmup_steps=50, grad_clip_norm=100.0,
            kl_free_bits=1.0, seed=42,
        )
        trainer = WorldModelTrainer(
            obs_shape=obs_shape, action_size=meta["action_size"],
            config=tc, key=key,
        )
        trainer.params = eqx.tree_deserialise_leaves(
            str(model_path), trainer.params,
        )
        encoder, rssm, decoder = trainer.params
        return encoder, rssm, decoder, meta
    except Exception as e:
        logger.warning(f"  Failed to load {domain}: {e}")
        return None


def dream_trajectory(encoder, rssm, decoder, obs_sequence, context_steps=20):
    """Dream forward from a context sequence.

    Args:
        encoder: RSSM encoder.
        rssm: RSSM core.
        decoder: RSSM decoder.
        obs_sequence: Ground truth observations (T, obs_dim).
        context_steps: Steps of context to feed.

    Returns:
        Dreamed observations array (dream_steps, obs_dim).
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    action = jnp.float32(0)

    # Feed context
    state = rssm.initial_state()
    for t in range(min(context_steps, len(obs_sequence))):
        obs = jnp.array(obs_sequence[t], dtype=jnp.float32)
        embed = encoder(obs.reshape(-1))
        key, step_key = jax.random.split(key)
        state, _, _ = rssm.observe_step(state, action, embed, key=step_key)

    # Dream forward
    dream_steps = len(obs_sequence) - context_steps
    dreamed = []
    for t in range(dream_steps):
        key, step_key = jax.random.split(key)
        state, _ = rssm.imagine_step(state, action, key=step_key)
        features = rssm.get_features(state)
        pred = decoder(features)
        dreamed.append(np.array(pred))

    return np.array(dreamed) if dreamed else np.array([])


def analyze_ensemble_group(group_name: str, domain_list: list[str]):
    """Analyze prediction disagreement across models in a group.

    Args:
        group_name: Name of the domain group.
        domain_list: List of domain names with matching obs dims.

    Returns:
        Dict with ensemble analysis results.
    """
    import jax.numpy as jnp

    # Load all available models in this group
    models = {}
    for domain in domain_list:
        loaded = load_model(domain)
        if loaded is not None:
            models[domain] = loaded

    if len(models) < 2:
        logger.warning(f"  Need >= 2 models for ensemble, got {len(models)}")
        return None

    logger.info(f"  Loaded {len(models)} models for group '{group_name}'")
    model_names = list(models.keys())

    # Use the first model's training data as test input
    first_domain = model_names[0]
    dream_path = WM_DIR / first_domain / "dream_comparison.npz"
    if not dream_path.exists():
        # Try to find a domain with dream data
        for d in model_names:
            dp = WM_DIR / d / "dream_comparison.npz"
            if dp.exists():
                dream_path = dp
                first_domain = d
                break

    if not dream_path.exists():
        logger.warning(f"  No dream data available for group {group_name}")
        return None

    data = np.load(dream_path)
    if "ground_truth" not in data:
        return None

    gt = data["ground_truth"]
    obs_dim = gt.shape[-1] if gt.ndim > 1 else 1
    if gt.ndim == 1:
        gt = gt.reshape(-1, obs_dim)

    # Dream from each model
    context_steps = 20
    predictions = {}
    for domain, (enc, rssm, dec, meta) in models.items():
        model_obs = int(np.prod(meta["obs_shape"]))
        if model_obs != obs_dim:
            continue
        try:
            pred = dream_trajectory(enc, rssm, dec, gt, context_steps)
            if len(pred) > 0:
                predictions[domain] = pred
        except Exception as e:
            logger.warning(f"    {domain} dreaming failed: {e}")

    if len(predictions) < 2:
        return None

    # Compute ensemble statistics
    pred_names = list(predictions.keys())
    pred_arrays = [predictions[n] for n in pred_names]
    min_len = min(len(p) for p in pred_arrays)
    pred_stack = np.array([p[:min_len] for p in pred_arrays])  # (n_models, T, obs_dim)

    # Ensemble mean and std
    ensemble_mean = np.mean(pred_stack, axis=0)
    ensemble_std = np.std(pred_stack, axis=0)

    # Per-step disagreement
    step_disagreement = np.mean(ensemble_std, axis=-1)  # (T,)

    # Pairwise prediction distances
    n_models = len(pred_names)
    pairwise_distances = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            mse = float(np.mean((pred_stack[i] - pred_stack[j]) ** 2))
            pair = f"{pred_names[i]}_vs_{pred_names[j]}"
            pairwise_distances[pair] = mse

    result = {
        "group": group_name,
        "test_domain": first_domain,
        "n_models": len(predictions),
        "model_names": pred_names,
        "dream_steps": min_len,
        "obs_dim": obs_dim,
        "mean_ensemble_std": float(np.mean(ensemble_std)),
        "max_ensemble_std": float(np.max(ensemble_std)),
        "mean_step_disagreement": step_disagreement.tolist()[:50],
        "disagreement_growth": (
            float(step_disagreement[-1] / max(step_disagreement[0], 1e-10))
            if len(step_disagreement) >= 2 else 1.0
        ),
        "pairwise_mse": pairwise_distances,
        "mean_pairwise_mse": float(np.mean(list(pairwise_distances.values()))),
    }

    logger.info(
        f"  Group '{group_name}': {len(predictions)} models, "
        f"mean_std={result['mean_ensemble_std']:.4f}, "
        f"mean_pairwise_MSE={result['mean_pairwise_mse']:.4f}"
    )
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ENSEMBLE UNCERTAINTY ANALYSIS")
    logger.info("=" * 60)

    t0 = time.time()
    results = []

    for group_name, domains in DOMAIN_GROUPS.items():
        logger.info(f"\n--- Group: {group_name} ---")
        r = analyze_ensemble_group(group_name, domains)
        if r is not None:
            results.append(r)

    elapsed = time.time() - t0

    output = {
        "n_groups": len(results),
        "elapsed_s": elapsed,
        "groups": results,
    }

    out_path = OUTPUT_DIR / "ensemble_uncertainty.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE UNCERTAINTY SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"  {r['group']:20s}: {r['n_models']} models, "
            f"std={r['mean_ensemble_std']:.4f}, "
            f"pairwise_MSE={r['mean_pairwise_mse']:.4f}"
        )

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path("paper/figures")

        if results:
            fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
            if len(results) == 1:
                axes = [axes]

            for ax, r in zip(axes, results):
                steps = r["mean_step_disagreement"]
                ax.plot(steps, color="steelblue", linewidth=2)
                ax.fill_between(range(len(steps)), 0, steps, alpha=0.2, color="steelblue")
                ax.set_xlabel("Dream Step")
                ax.set_ylabel("Ensemble Disagreement")
                ax.set_title(f"{r['group']}\n({r['n_models']} models)")

            plt.suptitle("Ensemble Uncertainty Across Domain Groups",
                        fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / "ensemble_uncertainty.pdf", dpi=150)
            plt.savefig(fig_dir / "ensemble_uncertainty.png", dpi=150)
            plt.close()
            logger.info("Ensemble uncertainty figure saved")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


if __name__ == "__main__":
    main()
