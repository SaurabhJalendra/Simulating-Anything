"""Train RSSM world models on all three V1 domains.

Generates training data from simulations, trains world models, evaluates
dreaming quality, and saves checkpoints + training curves.

Must run in WSL2 with GPU for JAX acceleration.

Usage:
    python scripts/train_world_models.py [--domain all|projectile|lotka_volterra|gray_scott]
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/world_models")


def generate_projectile_trajectories(
    n_trajectories: int = 100,
    seq_len: int = 200,
) -> np.ndarray:
    """Generate projectile trajectory sequences for world model training.

    Returns array of shape (n_trajectories, seq_len, 4) where state = [x, y, vx, vy].
    """
    from simulating_anything.simulation.rigid_body import ProjectileSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []

    for i in range(n_trajectories):
        v0 = rng.uniform(15, 45)
        angle = rng.uniform(20, 70)
        config = SimulationConfig(
            domain=Domain.RIGID_BODY, dt=0.01, n_steps=seq_len + 100,
            parameters={
                "gravity": 9.81, "drag_coefficient": rng.uniform(0, 0.05),
                "initial_speed": float(v0), "launch_angle": float(angle), "mass": 1.0,
            },
        )
        sim = ProjectileSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len + 50):
            s = sim.step()
            states.append(s.copy())
            if sim._landed:
                break

        traj = np.array(states[:seq_len])
        # Pad if landed early
        if len(traj) < seq_len:
            pad = np.tile(traj[-1:], (seq_len - len(traj), 1))
            traj = np.vstack([traj, pad])
        trajectories.append(traj)

    return np.array(trajectories)


def generate_lv_trajectories(
    n_trajectories: int = 100,
    seq_len: int = 200,
) -> np.ndarray:
    """Generate Lotka-Volterra trajectory sequences.

    Returns array of shape (n_trajectories, seq_len, 2) where state = [prey, predator].
    """
    from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []

    for i in range(n_trajectories):
        alpha = rng.uniform(0.8, 1.5)
        beta = rng.uniform(0.2, 0.6)
        gamma = rng.uniform(0.2, 0.6)
        delta = rng.uniform(0.05, 0.2)
        prey_0 = rng.uniform(20, 60)
        pred_0 = rng.uniform(5, 15)

        config = SimulationConfig(
            domain=Domain.AGENT_BASED, dt=0.01, n_steps=seq_len,
            parameters={
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                "prey_0": prey_0, "predator_0": pred_0,
            },
        )
        sim = LotkaVolterraSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))

    return np.array(trajectories)


def generate_gs_trajectories(
    n_trajectories: int = 20,
    seq_len: int = 100,
    grid_size: int = 64,
    snapshot_interval: int = 100,
) -> np.ndarray:
    """Generate Gray-Scott trajectory sequences.

    Returns array of shape (n_trajectories, seq_len, 2, grid_size, grid_size).
    Saves spatial snapshots every snapshot_interval steps.
    Uses smaller grid (64x64) for tractable training.
    """
    from simulating_anything.rediscovery.gray_scott import _run_gray_scott_jax

    rng = np.random.default_rng(42)
    trajectories = []

    # Sample from parameter space that produces patterns
    f_values = rng.uniform(0.02, 0.05, n_trajectories)
    k_values = rng.uniform(0.05, 0.068, n_trajectories)

    for i in range(n_trajectories):
        u = np.ones((grid_size, grid_size), dtype=np.float64)
        v = np.zeros((grid_size, grid_size), dtype=np.float64)
        cx, cy = grid_size // 2, grid_size // 2
        r = max(grid_size // 10, 2)
        u[cx - r:cx + r, cy - r:cy + r] = 0.50
        v[cx - r:cx + r, cy - r:cy + r] = 0.25
        u += 0.05 * rng.standard_normal(u.shape)
        v += 0.05 * rng.standard_normal(v.shape)
        v = np.clip(v, 0, 1)

        snapshots = []
        for t in range(seq_len):
            # Run snapshot_interval steps between snapshots
            u, v = _run_gray_scott_jax(
                u, v, 0.16, 0.08, float(f_values[i]), float(k_values[i]),
                1.0, 1.0, snapshot_interval
            )
            snapshots.append(np.stack([u, v], axis=0))

        trajectories.append(np.array(snapshots))
        if (i + 1) % 5 == 0:
            logger.info(f"  GS trajectory {i + 1}/{n_trajectories}")

    return np.array(trajectories)


def train_domain(
    domain: str,
    data: np.ndarray,
    n_epochs: int = 100,
    seq_len: int = 50,
    lr: float = 3e-4,
) -> dict:
    """Train RSSM world model on a domain.

    Args:
        domain: Domain name for logging
        data: Training trajectories (n_traj, T, *obs_shape)
        n_epochs: Number of training epochs
        seq_len: Sequence length per training batch
        lr: Learning rate

    Returns:
        Training results dict with loss curves and checkpoint path.
    """
    import jax
    import jax.numpy as jnp

    from simulating_anything.types.simulation import TrainingConfig
    from simulating_anything.world_model.trainer import WorldModelTrainer

    obs_shape = data.shape[2:]
    n_traj = data.shape[0]
    logger.info(f"Training {domain} world model: {n_traj} trajectories, obs_shape={obs_shape}")

    config = TrainingConfig(
        learning_rate=lr,
        batch_size=1,  # Single sequence per step (for simplicity)
        sequence_length=seq_len,
        n_epochs=n_epochs,
        warmup_steps=50,
        grad_clip_norm=100.0,
        kl_free_bits=1.0,
        seed=42,
    )

    key = jax.random.PRNGKey(42)
    trainer = WorldModelTrainer(obs_shape=obs_shape, action_size=0, config=config, key=key)

    # Training loop
    rng = np.random.default_rng(42)
    loss_history = {"total": [], "recon": [], "kl": []}
    best_loss = float("inf")

    logger.info(f"Starting {domain} training: {n_epochs} epochs, seq_len={seq_len}, lr={lr}")
    t0 = time.time()

    for epoch in range(n_epochs):
        # Sample a random trajectory and random subsequence
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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  [{domain}] Epoch {epoch + 1}/{n_epochs}: "
                f"loss={metrics['loss']:.4f} (recon={metrics['recon_loss']:.4f}, "
                f"kl={metrics['kl_loss']:.4f}) [{elapsed:.1f}s]"
            )

    total_time = time.time() - t0
    logger.info(f"  [{domain}] Training complete in {total_time:.1f}s, best_loss={best_loss:.4f}")

    # Save checkpoint
    ckpt_dir = OUTPUT_DIR / domain
    checkpoint = trainer.save_checkpoint(ckpt_dir, model_id=domain)
    logger.info(f"  [{domain}] Checkpoint saved to {ckpt_dir}")

    # Evaluate dreaming quality
    dream_results = evaluate_dreaming(trainer, data, domain, seq_len)

    results = {
        "domain": domain,
        "n_trajectories": n_traj,
        "obs_shape": list(obs_shape),
        "n_epochs": n_epochs,
        "best_loss": best_loss,
        "final_loss": loss_history["total"][-1],
        "final_recon": loss_history["recon"][-1],
        "final_kl": loss_history["kl"][-1],
        "training_time_s": total_time,
        "checkpoint_path": str(ckpt_dir),
        "loss_history": loss_history,
        "dream_results": dream_results,
    }

    # Save results
    results_file = ckpt_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "loss_history"}, f, indent=2)

    # Save loss curves separately
    np.savez(ckpt_dir / "loss_curves.npz",
             total=np.array(loss_history["total"]),
             recon=np.array(loss_history["recon"]),
             kl=np.array(loss_history["kl"]))

    return results


def evaluate_dreaming(
    trainer,
    data: np.ndarray,
    domain: str,
    seq_len: int = 50,
    dream_steps: int = 30,
) -> dict:
    """Evaluate world model dreaming quality.

    Feed a short context (ground truth observations), then dream forward
    and compare with actual future states.
    """
    import jax
    import jax.numpy as jnp

    logger.info(f"  [{domain}] Evaluating dreaming quality...")

    # Use first trajectory for evaluation
    traj = jnp.array(data[0][:seq_len], dtype=jnp.float32)
    context_len = min(20, seq_len // 2)
    dream_len = min(dream_steps, seq_len - context_len)

    encoder, rssm, decoder = trainer.params
    key = jax.random.PRNGKey(0)

    # Feed context
    state = rssm.initial_state()
    for t in range(context_len):
        obs = traj[t]
        if trainer.is_spatial and obs.ndim == 2:
            obs_input = obs[None, ...]
        else:
            obs_input = obs.reshape(-1) if obs.ndim > 1 and not trainer.is_spatial else obs
        embed = encoder(obs_input)
        action = jnp.array(0.0)
        key, step_key = jax.random.split(key)
        state, _, _ = rssm.observe_step(state, action, embed, key=step_key)

    # Dream forward
    dreamed_obs = []
    for t in range(dream_len):
        key, step_key = jax.random.split(key)
        action = jnp.array(0.0)
        state, _ = rssm.imagine_step(state, action, key=step_key)
        features = rssm.get_features(state)
        pred = decoder(features)
        dreamed_obs.append(np.array(pred))

    dreamed = np.array(dreamed_obs)

    # Compare with ground truth
    gt = np.array(traj[context_len:context_len + dream_len])
    if not trainer.is_spatial:
        gt_flat = gt.reshape(dream_len, -1)
    else:
        gt_flat = gt.reshape(dream_len, -1)
    dreamed_flat = dreamed.reshape(dream_len, -1)

    # Compute metrics
    from simulating_anything.world_model.decoder import symlog
    gt_symlog = np.array(jnp.sign(jnp.array(gt_flat)) * jnp.log1p(jnp.abs(jnp.array(gt_flat))))
    mse_symlog = float(np.mean((dreamed_flat - gt_symlog) ** 2))

    # Per-step error growth
    step_errors = []
    for t in range(dream_len):
        err = float(np.mean((dreamed_flat[t] - gt_symlog[t]) ** 2))
        step_errors.append(err)

    results = {
        "context_len": context_len,
        "dream_len": dream_len,
        "mse_symlog": mse_symlog,
        "step_errors": step_errors,
        "error_growth_ratio": step_errors[-1] / max(step_errors[0], 1e-10) if step_errors else 0,
    }

    logger.info(
        f"  [{domain}] Dream MSE (symlog): {mse_symlog:.4f}, "
        f"error growth: {results['error_growth_ratio']:.2f}x over {dream_len} steps"
    )

    # Save dreamed + ground truth for visualization
    save_dir = OUTPUT_DIR / domain
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "dream_comparison.npz",
        ground_truth=np.array(traj[:context_len + dream_len]),
        dreamed=dreamed,
        context_len=context_len,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train RSSM world models")
    parser.add_argument("--domain", default="all",
                        choices=["all", "projectile", "lotka_volterra", "gray_scott"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    t0 = time.time()

    domains = (["projectile", "lotka_volterra", "gray_scott"]
               if args.domain == "all" else [args.domain])

    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"DOMAIN: {domain}")
        logger.info(f"{'='*60}")

        if domain == "projectile":
            logger.info("Generating projectile trajectories...")
            data = generate_projectile_trajectories(n_trajectories=80, seq_len=150)
            results = train_domain(domain, data, n_epochs=args.epochs, seq_len=50, lr=3e-4)

        elif domain == "lotka_volterra":
            logger.info("Generating Lotka-Volterra trajectories...")
            data = generate_lv_trajectories(n_trajectories=80, seq_len=200)
            results = train_domain(domain, data, n_epochs=args.epochs, seq_len=50, lr=3e-4)

        elif domain == "gray_scott":
            logger.info("Generating Gray-Scott trajectories (JAX accelerated)...")
            data = generate_gs_trajectories(
                n_trajectories=16, seq_len=80, grid_size=64, snapshot_interval=100
            )
            results = train_domain(domain, data, n_epochs=args.epochs, seq_len=30, lr=1e-4)

        all_results[domain] = {
            k: v for k, v in results.items() if k != "loss_history"
        }

    total_time = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL TRAINING COMPLETE in {total_time:.1f}s")
    logger.info(f"{'='*60}")

    for domain, r in all_results.items():
        logger.info(
            f"  {domain}: best_loss={r['best_loss']:.4f}, "
            f"dream_MSE={r['dream_results']['mse_symlog']:.4f}, "
            f"time={r['training_time_s']:.1f}s"
        )

    # Save summary
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
