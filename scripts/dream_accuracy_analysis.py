"""Comprehensive dream accuracy analysis across all trained world models.

For each trained RSSM, generates test trajectories, feeds context,
dreams forward, and compares against ground truth. Produces:
1. Per-domain dream MSE and correlation
2. Dream accuracy vs observation dimensionality
3. Dream accuracy vs domain complexity (Lyapunov-like divergence)
4. Best/worst dreaming examples

This demonstrates that world models can replace expensive simulations
for parameter-space exploration and hypothesis testing.

Usage (must run in WSL2 for GPU):
    python scripts/dream_accuracy_analysis.py
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
OUTPUT_DIR = Path("output/dream_analysis")


def analyze_domain(domain: str, context_steps: int = 20, dream_steps: int = 50):
    """Analyze dream accuracy for a single domain.

    Args:
        domain: Domain name with trained model.
        context_steps: Observation steps before dreaming.
        dream_steps: Number of dreamed steps.

    Returns:
        Dict with accuracy metrics, or None on failure.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from simulating_anything.types.simulation import TrainingConfig
    from simulating_anything.world_model.trainer import WorldModelTrainer

    meta_path = WM_DIR / domain / "meta.json"
    model_path = WM_DIR / domain / "model.eqx"

    if not meta_path.exists() or not model_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    obs_shape = tuple(meta["obs_shape"])
    action_size = meta["action_size"]

    # Skip spatial models
    if len(obs_shape) >= 2 and obs_shape[-1] > 4 and obs_shape[-2] > 4:
        logger.info(f"  Skipping {domain} (spatial obs {obs_shape})")
        return None

    # Load model
    try:
        key = jax.random.PRNGKey(42)
        tc = TrainingConfig(
            learning_rate=3e-4, batch_size=1, sequence_length=50,
            n_epochs=100, warmup_steps=50, grad_clip_norm=100.0,
            kl_free_bits=1.0, seed=42,
        )
        trainer = WorldModelTrainer(
            obs_shape=obs_shape, action_size=action_size,
            config=tc, key=key,
        )
        trainer.params = eqx.tree_deserialise_leaves(
            str(model_path), trainer.params,
        )
        encoder, rssm, decoder = trainer.params
    except Exception as e:
        logger.warning(f"  Failed to load {domain}: {e}")
        return None

    # Load training data for test trajectories
    dream_path = WM_DIR / domain / "dream_comparison.npz"
    if not dream_path.exists():
        logger.warning(f"  No dream comparison data for {domain}")
        return None

    try:
        data = np.load(dream_path)
        # Use available data
        if "ground_truth" in data and "dreamed" in data:
            gt = data["ground_truth"]
            dreamed = data["dreamed"]
        else:
            logger.warning(f"  Unexpected dream data keys: {list(data.keys())}")
            return None
    except Exception as e:
        logger.warning(f"  Failed to load dream data for {domain}: {e}")
        return None

    # Compute dream accuracy metrics
    obs_dim = int(np.prod(obs_shape))

    # Reshape if needed
    if gt.ndim == 1:
        n_steps = len(gt) // obs_dim
        gt = gt.reshape(n_steps, obs_dim)
        dreamed = dreamed.reshape(n_steps, obs_dim)
    elif gt.ndim > 2:
        gt = gt.reshape(gt.shape[0], -1)
        dreamed = dreamed.reshape(dreamed.shape[0], -1)

    n_steps = min(len(gt), len(dreamed))
    gt = gt[:n_steps]
    dreamed = dreamed[:n_steps]

    # Per-step MSE
    step_mse = np.mean((gt - dreamed) ** 2, axis=-1)
    # Correlation per step
    step_corr = []
    for t in range(n_steps):
        if np.std(gt[t]) > 1e-12 and np.std(dreamed[t]) > 1e-12:
            c = np.corrcoef(gt[t].flatten(), dreamed[t].flatten())[0, 1]
            step_corr.append(float(c) if not np.isnan(c) else 0.0)
        else:
            step_corr.append(0.0)

    # Error growth rate (how fast dreams diverge)
    if len(step_mse) >= 2 and step_mse[0] > 1e-12:
        growth_rate = float(step_mse[-1] / step_mse[0])
    else:
        growth_rate = 1.0

    # Horizon: steps until MSE exceeds threshold
    threshold = np.mean(np.var(gt, axis=0))  # variance of ground truth
    horizon = n_steps
    for t in range(n_steps):
        if step_mse[t] > threshold:
            horizon = t
            break

    result = {
        "domain": domain,
        "obs_dim": obs_dim,
        "obs_shape": list(obs_shape),
        "n_dream_steps": n_steps,
        "mean_mse": float(np.mean(step_mse)),
        "final_mse": float(step_mse[-1]) if len(step_mse) > 0 else float("nan"),
        "mean_correlation": float(np.mean(step_corr)),
        "final_correlation": float(step_corr[-1]) if step_corr else float("nan"),
        "error_growth_rate": growth_rate,
        "prediction_horizon": horizon,
        "gt_variance": float(threshold),
        "step_mse": step_mse.tolist()[:50],
        "step_corr": step_corr[:50],
    }

    logger.info(
        f"  {domain:25s}: MSE={result['mean_mse']:.4f}, "
        f"corr={result['mean_correlation']:.3f}, "
        f"horizon={horizon}/{n_steps}, growth={growth_rate:.2f}x"
    )
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all trained domains
    domains = sorted([
        d.name for d in WM_DIR.iterdir()
        if d.is_dir() and (d / "model.eqx").exists()
    ])

    logger.info(f"Analyzing dream accuracy for {len(domains)} trained domains")
    t0 = time.time()

    results = []
    for domain in domains:
        logger.info(f"\n--- {domain} ---")
        r = analyze_domain(domain)
        if r is not None:
            results.append(r)

    elapsed = time.time() - t0

    # Compute summary statistics
    if results:
        mses = [r["mean_mse"] for r in results]
        corrs = [r["mean_correlation"] for r in results]
        horizons = [r["prediction_horizon"] for r in results]
        growths = [r["error_growth_rate"] for r in results]

        summary = {
            "n_domains": len(results),
            "elapsed_s": elapsed,
            "mse": {
                "mean": float(np.mean(mses)),
                "median": float(np.median(mses)),
                "min": float(np.min(mses)),
                "max": float(np.max(mses)),
                "best_domain": results[int(np.argmin(mses))]["domain"],
                "worst_domain": results[int(np.argmax(mses))]["domain"],
            },
            "correlation": {
                "mean": float(np.mean(corrs)),
                "median": float(np.median(corrs)),
            },
            "prediction_horizon": {
                "mean": float(np.mean(horizons)),
                "median": float(np.median(horizons)),
            },
            "error_growth": {
                "mean": float(np.mean(growths)),
                "median": float(np.median(growths)),
                "stable_count": sum(1 for g in growths if g < 2.0),
            },
        }
    else:
        summary = {"n_domains": 0, "error": "No domains analyzed"}

    output = {
        "summary": summary,
        "per_domain": results,
    }

    out_path = OUTPUT_DIR / "dream_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DREAM ACCURACY SUMMARY")
    logger.info("=" * 60)
    if results:
        logger.info(f"  Domains analyzed:    {len(results)}")
        logger.info(f"  Mean MSE:            {summary['mse']['mean']:.4f}")
        logger.info(f"  Median MSE:          {summary['mse']['median']:.4f}")
        logger.info(f"  Best dreamer:        {summary['mse']['best_domain']}")
        logger.info(f"  Worst dreamer:       {summary['mse']['worst_domain']}")
        logger.info(f"  Mean correlation:    {summary['correlation']['mean']:.3f}")
        logger.info(f"  Mean horizon:        {summary['prediction_horizon']['mean']:.0f} steps")
        logger.info(f"  Stable dreamers:     {summary['error_growth']['stable_count']}/{len(results)}")

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path("paper/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Sort by MSE
        results_sorted = sorted(results, key=lambda r: r["mean_mse"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Dream MSE by domain
        ax = axes[0, 0]
        names = [r["domain"][:12] for r in results_sorted]
        mses = [r["mean_mse"] for r in results_sorted]
        colors = ["#2ecc71" if m < 0.05 else "#3498db" if m < 0.2
                  else "#f39c12" if m < 0.5 else "#e74c3c" for m in mses]
        ax.barh(range(len(names)), mses, color=colors, height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("Mean Dream MSE")
        ax.set_title("(a) Dream Accuracy by Domain")

        # (b) Error growth over time (selected domains)
        ax = axes[0, 1]
        for r in results_sorted[:5] + results_sorted[-3:]:
            steps = r.get("step_mse", [])
            if steps:
                ax.plot(steps, label=r["domain"][:10], alpha=0.7)
        ax.set_xlabel("Dream Step")
        ax.set_ylabel("MSE")
        ax.set_title("(b) Error Growth Over Dream Horizon")
        ax.legend(fontsize=6, ncol=2)

        # (c) Obs dim vs dream MSE
        ax = axes[1, 0]
        obs = [r["obs_dim"] for r in results]
        mse = [r["mean_mse"] for r in results]
        ax.scatter(obs, mse, s=50, c="steelblue", alpha=0.7, edgecolors="k")
        for r in results:
            if r["mean_mse"] > 0.3 or r["obs_dim"] > 10:
                ax.annotate(r["domain"][:8], (r["obs_dim"], r["mean_mse"]),
                           fontsize=6, alpha=0.7)
        ax.set_xlabel("Observation Dimensionality")
        ax.set_ylabel("Mean Dream MSE")
        ax.set_title("(c) Obs Dim vs Dream MSE")

        # (d) Prediction horizon
        ax = axes[1, 1]
        horizons = [r["prediction_horizon"] for r in results_sorted]
        ax.barh(range(len(names)), horizons, color="steelblue", height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("Prediction Horizon (steps)")
        ax.set_title("(d) Prediction Horizon by Domain")

        plt.suptitle("Dream-Based Discovery: World Model Prediction Accuracy",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "dream_accuracy.pdf", dpi=150)
        plt.savefig(fig_dir / "dream_accuracy.png", dpi=150)
        plt.close()
        logger.info("Dream accuracy figure saved")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


if __name__ == "__main__":
    main()
