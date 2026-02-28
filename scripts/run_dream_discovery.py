"""Dream-based scientific discovery: train/load RSSM world models, dream
trajectories, and run symbolic regression on the dreamed data.

Compares discoveries from dreamed data vs ground truth simulation data to
evaluate whether world models preserve the underlying governing equations.

Must run in WSL2 with GPU for JAX acceleration. Requires Julia + PySR.

Usage:
    python scripts/run_dream_discovery.py --domain projectile --dream-steps 200
    python scripts/run_dream_discovery.py --domain lotka_volterra --dream-steps 200
    python scripts/run_dream_discovery.py --domain all --dream-steps 200
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

CHECKPOINT_DIR = Path("output/world_models")
OUTPUT_DIR = Path("output/dream_discovery")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_world_model(domain: str):
    """Load a trained RSSM world model from checkpoint.

    Rebuilds a WorldModelTrainer with the same architecture that was used
    during training, then deserialises the saved parameters into it.

    Returns:
        (encoder, rssm, decoder) parameter tuple and the metadata dict.
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
        f"  Model loaded: feature_size={rssm.feature_size}, "
        f"hidden={rssm.hidden_size}, stoch={rssm.stoch_vars}x{rssm.stoch_classes}"
    )
    return (encoder, rssm, decoder), meta


# ---------------------------------------------------------------------------
# Ground truth data generation
# ---------------------------------------------------------------------------

def generate_projectile_ground_truth(
    n_trajectories: int = 50,
    seq_len: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """Generate projectile trajectories with varied parameters.

    Returns:
        trajectories: (n_trajectories, seq_len, 4) array of [x, y, vx, vy]
        params_list: list of dicts with v0, angle, gravity for each trajectory
    """
    from simulating_anything.simulation.rigid_body import ProjectileSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(123)
    trajectories = []
    params_list = []

    for _ in range(n_trajectories):
        v0 = rng.uniform(15, 45)
        angle = rng.uniform(20, 70)
        config = SimulationConfig(
            domain=Domain.RIGID_BODY, dt=0.01, n_steps=seq_len + 100,
            parameters={
                "gravity": 9.81, "drag_coefficient": 0.0,
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
        if len(traj) < seq_len:
            pad = np.tile(traj[-1:], (seq_len - len(traj), 1))
            traj = np.vstack([traj, pad])
        trajectories.append(traj)
        params_list.append({
            "v0": v0, "angle_deg": angle,
            "theta_rad": np.radians(angle), "gravity": 9.81,
        })

    return np.array(trajectories), params_list


def generate_lv_ground_truth(
    n_trajectories: int = 50,
    seq_len: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """Generate Lotka-Volterra trajectories with varied parameters.

    Returns:
        trajectories: (n_trajectories, seq_len, 2) array of [prey, predator]
        params_list: list of dicts with alpha, beta, gamma, delta for each
    """
    from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(123)
    trajectories = []
    params_list = []

    for _ in range(n_trajectories):
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
        params_list.append({
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
            "prey_0": prey_0, "predator_0": pred_0,
        })

    return np.array(trajectories), params_list


# ---------------------------------------------------------------------------
# Dreaming
# ---------------------------------------------------------------------------

def dream_trajectories(
    encoder,
    rssm,
    decoder,
    context_obs: np.ndarray,
    dream_steps: int,
    is_spatial: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Dream forward from a set of context observations.

    Args:
        encoder: Encoder module (MLPEncoder or CNNEncoder).
        rssm: RSSM module.
        decoder: Decoder module (MLPDecoder or CNNDecoder).
        context_obs: (context_len, *obs_shape) ground truth observations to
            feed before dreaming.
        dream_steps: Number of imagination steps after the context phase.
        is_spatial: Whether the observations are spatial (CNN) or vector (MLP).
        seed: PRNG seed.

    Returns:
        dreamed: (dream_steps, obs_dim) array of decoded dreamed observations
            in real (symexp-decoded) space.
    """
    import jax
    import jax.numpy as jnp

    from simulating_anything.world_model.decoder import symexp

    key = jax.random.PRNGKey(seed)
    action = jnp.array(0.0)  # scalar action for action_size=0

    # Context phase: feed real observations to warm up the RSSM
    state = rssm.initial_state()
    for t in range(len(context_obs)):
        obs = jnp.array(context_obs[t], dtype=jnp.float32)
        if is_spatial and obs.ndim == 2:
            obs_input = obs[None, ...]
        else:
            obs_input = obs.reshape(-1) if obs.ndim > 1 and not is_spatial else obs
        embed = encoder(obs_input)
        key, step_key = jax.random.split(key)
        state, _, _ = rssm.observe_step(state, action, embed, key=step_key)

    # Dream phase: imagine forward without observations
    dreamed = []
    for _ in range(dream_steps):
        key, step_key = jax.random.split(key)
        state, _ = rssm.imagine_step(state, action, key=step_key)
        features = rssm.get_features(state)
        pred_symlog = decoder(features)
        # Decoder outputs in symlog space; invert to real space
        pred_real = symexp(pred_symlog)
        dreamed.append(np.array(pred_real))

    return np.array(dreamed)


def dream_multi_trajectory(
    encoder,
    rssm,
    decoder,
    trajectories: np.ndarray,
    context_len: int,
    dream_steps: int,
    is_spatial: bool = False,
    n_dream: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dream from multiple trajectories, collecting dreamed and ground truth data.

    Args:
        trajectories: (N, T, *obs_shape) batch of trajectories.
        context_len: Number of real observations to feed before dreaming.
        dream_steps: Number of steps to dream after context.
        n_dream: Number of trajectories to dream from (defaults to all).

    Returns:
        dreamed_all: (n_dream * dream_steps, obs_dim)
        gt_all: (n_dream * dream_steps, obs_dim) corresponding ground truth
    """
    n_traj = trajectories.shape[0]
    if n_dream is None:
        n_dream = n_traj
    n_dream = min(n_dream, n_traj)

    seq_len = trajectories.shape[1]
    effective_dream = min(dream_steps, seq_len - context_len)

    if effective_dream <= 0:
        raise ValueError(
            f"Not enough trajectory length ({seq_len}) for context_len={context_len} "
            f"+ dream_steps={dream_steps}"
        )

    dreamed_all = []
    gt_all = []

    for i in range(n_dream):
        context = trajectories[i, :context_len]
        dreamed = dream_trajectories(
            encoder, rssm, decoder,
            context_obs=context,
            dream_steps=effective_dream,
            is_spatial=is_spatial,
            seed=i,
        )
        gt = trajectories[i, context_len:context_len + effective_dream]

        # Flatten spatial dims if needed
        gt_flat = gt.reshape(effective_dream, -1)
        dreamed_flat = dreamed.reshape(effective_dream, -1)

        dreamed_all.append(dreamed_flat)
        gt_all.append(gt_flat)

        if (i + 1) % 10 == 0:
            logger.info(f"  Dreamed trajectory {i + 1}/{n_dream}")

    return np.concatenate(dreamed_all, axis=0), np.concatenate(gt_all, axis=0)


# ---------------------------------------------------------------------------
# Symbolic regression on dreamed data
# ---------------------------------------------------------------------------

def run_projectile_discovery(
    dreamed_data: np.ndarray,
    gt_data: np.ndarray,
    n_iterations: int = 40,
) -> dict:
    """Run symbolic regression on projectile dreamed vs ground truth data.

    For projectile, obs = [x, y, vx, vy]. We look at relationships between
    kinematic variables: e.g., can PySR recover y = f(x, vx, vy)?

    We focus on the energy relationship: 0.5*(vx^2+vy^2) + g*y = const,
    which is the conservation of energy (no drag). From dreamed data, PySR
    should find that vy^2 ~ -2*g*y + const or similar.
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    results = {"domain": "projectile"}

    # Extract kinematic variables: x, y, vx, vy
    # Target: predict vy^2 from y (energy conservation: vy^2 = vy0^2 - 2*g*y)
    # This tests whether the world model preserves the quadratic energy relationship.

    # -- Ground truth regression --
    gt_y = gt_data[:, 1]      # y position
    gt_vx = gt_data[:, 2]     # vx
    gt_vy = gt_data[:, 3]     # vy

    # Filter out padded/zero rows where the projectile has landed
    valid_gt = gt_y > 0.01
    if np.sum(valid_gt) < 20:
        # Fall back to all data if too few valid points
        valid_gt = np.ones(len(gt_y), dtype=bool)

    X_gt = np.column_stack([gt_y[valid_gt], gt_vx[valid_gt]])
    y_gt_target = gt_vy[valid_gt] ** 2

    logger.info(f"Ground truth: {np.sum(valid_gt)} valid points for regression")
    logger.info("Running PySR on ground truth data (target: vy^2 = f(y, vx))...")
    gt_discoveries = run_symbolic_regression(
        X_gt, y_gt_target,
        variable_names=["y_pos", "vx"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        max_complexity=15,
        populations=15,
        population_size=33,
    )

    results["ground_truth"] = _format_discoveries(gt_discoveries, "ground_truth")

    # -- Dreamed data regression --
    dr_y = dreamed_data[:, 1]
    dr_vx = dreamed_data[:, 2]
    dr_vy = dreamed_data[:, 3]

    valid_dr = np.isfinite(dr_y) & np.isfinite(dr_vx) & np.isfinite(dr_vy)
    # Also filter extreme outliers from dreaming artifacts
    dr_speed = np.sqrt(dr_vx**2 + dr_vy**2)
    valid_dr &= dr_speed < 200  # reasonable speed bound

    if np.sum(valid_dr) < 20:
        logger.warning("Too few valid dreamed points for regression")
        results["dreamed"] = {"error": "Too few valid points", "n_valid": int(np.sum(valid_dr))}
        return results

    X_dr = np.column_stack([dr_y[valid_dr], dr_vx[valid_dr]])
    y_dr_target = dr_vy[valid_dr] ** 2

    logger.info(f"Dreamed data: {np.sum(valid_dr)} valid points for regression")
    logger.info("Running PySR on dreamed data (target: vy^2 = f(y, vx))...")
    dr_discoveries = run_symbolic_regression(
        X_dr, y_dr_target,
        variable_names=["y_pos", "vx"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        max_complexity=15,
        populations=15,
        population_size=33,
    )

    results["dreamed"] = _format_discoveries(dr_discoveries, "dreamed")

    # -- Cross-evaluation: evaluate dreamed equation on GT data and vice versa --
    results["comparison"] = _cross_evaluate(
        gt_discoveries, dr_discoveries, X_gt, y_gt_target, X_dr, y_dr_target
    )

    return results


def run_lv_discovery(
    dreamed_data: np.ndarray,
    gt_data: np.ndarray,
    n_iterations: int = 40,
) -> dict:
    """Run symbolic regression on Lotka-Volterra dreamed vs ground truth data.

    For LV, obs = [prey, predator]. We compute finite-difference derivatives
    d(prey)/dt and d(pred)/dt, then use PySR to find the governing terms.

    Target: d(prey)/dt ~ alpha*prey - beta*prey*pred
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    results = {"domain": "lotka_volterra"}
    dt = 0.01  # timestep used in simulation

    # -- Ground truth regression --
    gt_prey = gt_data[:, 0]
    gt_pred = gt_data[:, 1]

    # Compute derivatives via central differences (skip first/last)
    gt_dprey = np.gradient(gt_prey, dt)
    gt_dpred = np.gradient(gt_pred, dt)

    # Filter out points with very small populations (numerical issues)
    valid_gt = (gt_prey > 0.5) & (gt_pred > 0.5) & np.isfinite(gt_dprey)
    if np.sum(valid_gt) < 20:
        valid_gt = np.ones(len(gt_prey), dtype=bool)

    X_gt = np.column_stack([gt_prey[valid_gt], gt_pred[valid_gt]])

    logger.info(f"Ground truth: {np.sum(valid_gt)} valid points for regression")

    # PySR for d(prey)/dt
    logger.info("Running PySR on ground truth d(prey)/dt...")
    gt_prey_disc = run_symbolic_regression(
        X_gt, gt_dprey[valid_gt],
        variable_names=["prey", "pred"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        max_complexity=12,
        populations=15,
        population_size=33,
    )

    # PySR for d(pred)/dt
    logger.info("Running PySR on ground truth d(pred)/dt...")
    gt_pred_disc = run_symbolic_regression(
        X_gt, gt_dpred[valid_gt],
        variable_names=["prey", "pred"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        max_complexity=12,
        populations=15,
        population_size=33,
    )

    results["ground_truth"] = {
        "dprey_dt": _format_discoveries(gt_prey_disc, "gt_dprey"),
        "dpred_dt": _format_discoveries(gt_pred_disc, "gt_dpred"),
    }

    # -- Dreamed data regression --
    dr_prey = dreamed_data[:, 0]
    dr_pred = dreamed_data[:, 1]

    dr_dprey = np.gradient(dr_prey, dt)
    dr_dpred = np.gradient(dr_pred, dt)

    valid_dr = (
        (dr_prey > 0.5) & (dr_pred > 0.5)
        & np.isfinite(dr_dprey) & np.isfinite(dr_dpred)
        & (np.abs(dr_dprey) < 1e4) & (np.abs(dr_dpred) < 1e4)
    )

    if np.sum(valid_dr) < 20:
        logger.warning("Too few valid dreamed points for LV regression")
        results["dreamed"] = {"error": "Too few valid points", "n_valid": int(np.sum(valid_dr))}
        return results

    X_dr = np.column_stack([dr_prey[valid_dr], dr_pred[valid_dr]])

    logger.info(f"Dreamed data: {np.sum(valid_dr)} valid points for regression")

    logger.info("Running PySR on dreamed d(prey)/dt...")
    dr_prey_disc = run_symbolic_regression(
        X_dr, dr_dprey[valid_dr],
        variable_names=["prey", "pred"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        max_complexity=12,
        populations=15,
        population_size=33,
    )

    logger.info("Running PySR on dreamed d(pred)/dt...")
    dr_pred_disc = run_symbolic_regression(
        X_dr, dr_dpred[valid_dr],
        variable_names=["prey", "pred"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        max_complexity=12,
        populations=15,
        population_size=33,
    )

    results["dreamed"] = {
        "dprey_dt": _format_discoveries(dr_prey_disc, "dr_dprey"),
        "dpred_dt": _format_discoveries(dr_pred_disc, "dr_dpred"),
    }

    # Cross-evaluation for prey derivative
    results["comparison_dprey"] = _cross_evaluate(
        gt_prey_disc, dr_prey_disc,
        X_gt, gt_dprey[valid_gt],
        X_dr, dr_dprey[valid_dr],
    )
    results["comparison_dpred"] = _cross_evaluate(
        gt_pred_disc, dr_pred_disc,
        X_gt, gt_dpred[valid_gt],
        X_dr, dr_dpred[valid_dr],
    )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_discoveries(discoveries: list, label: str) -> dict:
    """Format a list of Discovery objects into a JSON-serializable dict."""
    result = {
        "label": label,
        "n_discoveries": len(discoveries),
        "discoveries": [],
    }
    for d in discoveries[:10]:
        result["discoveries"].append({
            "expression": d.expression,
            "confidence": d.confidence,
            "r_squared": d.evidence.fit_r_squared,
            "description": d.description,
        })
    if discoveries:
        best = discoveries[0]
        result["best_equation"] = best.expression
        result["best_r_squared"] = best.evidence.fit_r_squared
    else:
        result["best_equation"] = "None found"
        result["best_r_squared"] = 0.0
    return result


def _cross_evaluate(
    gt_discoveries: list,
    dr_discoveries: list,
    X_gt: np.ndarray,
    y_gt: np.ndarray,
    X_dr: np.ndarray,
    y_dr: np.ndarray,
) -> dict:
    """Cross-evaluate: score ground truth equation on dreamed data and vice versa.

    Uses sympy to evaluate the best discovered expressions on both datasets.
    Falls back gracefully if parsing fails.
    """
    comparison = {}

    # R-squared helper
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-15:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    try:
        import sympy
    except ImportError:
        comparison["error"] = "sympy not available for cross-evaluation"
        return comparison

    for source_label, discoveries, X_self, y_self, X_other, y_other, other_label in [
        ("ground_truth", gt_discoveries, X_gt, y_gt, X_dr, y_dr, "dreamed"),
        ("dreamed", dr_discoveries, X_dr, y_dr, X_gt, y_gt, "ground_truth"),
    ]:
        if not discoveries:
            comparison[f"{source_label}_on_{other_label}"] = "No equation to evaluate"
            continue

        best = discoveries[0]
        expr_str = best.expression
        comparison[f"{source_label}_best_expr"] = expr_str
        comparison[f"{source_label}_r2_on_self"] = best.evidence.fit_r_squared

        try:
            # Parse the expression and evaluate on the other dataset
            expr = sympy.sympify(expr_str)
            free_syms = sorted(expr.free_symbols, key=lambda s: s.name)

            if len(free_syms) <= X_other.shape[1]:
                # Create a numpy-compatible function
                func = sympy.lambdify(free_syms, expr, modules="numpy")
                # Pass columns in order of symbol names
                cols = [X_other[:, j] for j in range(len(free_syms))]
                y_pred_other = func(*cols)
                r2_cross = r_squared(y_other, y_pred_other)
                comparison[f"{source_label}_r2_on_{other_label}"] = r2_cross
            else:
                comparison[f"{source_label}_r2_on_{other_label}"] = "Symbol count mismatch"
        except Exception as e:
            comparison[f"{source_label}_r2_on_{other_label}"] = f"Eval error: {e}"

    return comparison


def compute_dream_quality_metrics(
    dreamed: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Compute metrics comparing dreamed vs ground truth trajectories."""
    mse = float(np.mean((dreamed - ground_truth) ** 2))
    mae = float(np.mean(np.abs(dreamed - ground_truth)))

    # Per-feature correlation
    correlations = []
    n_features = dreamed.shape[1]
    for f in range(n_features):
        if np.std(ground_truth[:, f]) > 1e-10 and np.std(dreamed[:, f]) > 1e-10:
            corr = float(np.corrcoef(ground_truth[:, f], dreamed[:, f])[0, 1])
        else:
            corr = 0.0
        correlations.append(corr)

    # Distribution similarity: compare means and stds
    gt_mean = np.mean(ground_truth, axis=0)
    dr_mean = np.mean(dreamed, axis=0)
    gt_std = np.std(ground_truth, axis=0)
    dr_std = np.std(dreamed, axis=0)

    return {
        "mse": mse,
        "mae": mae,
        "per_feature_correlation": correlations,
        "mean_correlation": float(np.mean(correlations)),
        "gt_feature_means": gt_mean.tolist(),
        "dr_feature_means": dr_mean.tolist(),
        "gt_feature_stds": gt_std.tolist(),
        "dr_feature_stds": dr_std.tolist(),
    }


# ---------------------------------------------------------------------------
# Main orchestration per domain
# ---------------------------------------------------------------------------

def run_domain(domain: str, dream_steps: int, n_iterations: int) -> dict:
    """Run dream discovery for a single domain.

    Steps:
        1. Load trained world model checkpoint
        2. Generate ground truth trajectories
        3. Dream forward from context observations
        4. Compute dream quality metrics
        5. Run symbolic regression on both datasets
        6. Compare discovered equations
    """
    t0 = time.time()
    logger.info(f"{'=' * 60}")
    logger.info(f"DREAM DISCOVERY: {domain}")
    logger.info(f"{'=' * 60}")

    # 1. Load world model
    (encoder, rssm, decoder), meta = load_world_model(domain)
    is_spatial = meta.get("is_spatial", False)

    # 2. Generate ground truth data
    context_len = 20
    total_len = context_len + dream_steps

    if domain == "projectile":
        logger.info(f"Generating {total_len}-step projectile trajectories...")
        trajectories, params_list = generate_projectile_ground_truth(
            n_trajectories=50, seq_len=total_len,
        )
    elif domain == "lotka_volterra":
        logger.info(f"Generating {total_len}-step Lotka-Volterra trajectories...")
        trajectories, params_list = generate_lv_ground_truth(
            n_trajectories=50, seq_len=total_len,
        )
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    logger.info(f"  Generated {len(trajectories)} trajectories, shape={trajectories.shape}")

    # 3. Dream from each trajectory
    logger.info(
        f"Dreaming {dream_steps} steps from {len(trajectories)} trajectories "
        f"(context={context_len})..."
    )
    dreamed_all, gt_all = dream_multi_trajectory(
        encoder, rssm, decoder,
        trajectories=trajectories,
        context_len=context_len,
        dream_steps=dream_steps,
        is_spatial=is_spatial,
        n_dream=50,
    )
    logger.info(f"  Dreamed data shape: {dreamed_all.shape}")
    logger.info(f"  Ground truth shape: {gt_all.shape}")

    # 4. Dream quality metrics
    logger.info("Computing dream quality metrics...")
    quality = compute_dream_quality_metrics(dreamed_all, gt_all)
    logger.info(
        f"  MSE={quality['mse']:.4f}, MAE={quality['mae']:.4f}, "
        f"mean_corr={quality['mean_correlation']:.4f}"
    )

    # 5. Symbolic regression comparison
    logger.info("Running symbolic regression comparison...")
    if domain == "projectile":
        sr_results = run_projectile_discovery(dreamed_all, gt_all, n_iterations=n_iterations)
    elif domain == "lotka_volterra":
        sr_results = run_lv_discovery(dreamed_all, gt_all, n_iterations=n_iterations)
    else:
        sr_results = {}

    elapsed = time.time() - t0

    # 6. Assemble final results
    results = {
        "domain": domain,
        "dream_steps": dream_steps,
        "context_len": context_len,
        "n_trajectories": len(trajectories),
        "trajectory_shape": list(trajectories.shape),
        "n_dreamed_points": len(dreamed_all),
        "dream_quality": quality,
        "symbolic_regression": sr_results,
        "elapsed_time_s": elapsed,
    }

    # Save
    save_dir = OUTPUT_DIR / domain
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    np.savez(
        save_dir / "dreamed_vs_gt.npz",
        dreamed=dreamed_all,
        ground_truth=gt_all,
    )

    logger.info(f"Results saved to {save_dir}")
    logger.info(f"Domain {domain} complete in {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dream-based scientific discovery using trained RSSM world models",
    )
    parser.add_argument(
        "--domain", default="all",
        choices=["all", "projectile", "lotka_volterra"],
        help="Domain to run dream discovery for (default: all)",
    )
    parser.add_argument(
        "--dream-steps", type=int, default=200,
        help="Number of imagination steps after context (default: 200)",
    )
    parser.add_argument(
        "--context-len", type=int, default=20,
        help="Number of real observations to feed before dreaming (default: 20)",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=40,
        help="Number of PySR evolution iterations (default: 40)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    domains = (
        ["projectile", "lotka_volterra"]
        if args.domain == "all"
        else [args.domain]
    )

    all_results = {}
    t0 = time.time()

    for domain in domains:
        try:
            results = run_domain(domain, args.dream_steps, args.n_iterations)
            all_results[domain] = {
                "dream_quality_mse": results["dream_quality"]["mse"],
                "dream_quality_corr": results["dream_quality"]["mean_correlation"],
                "elapsed_s": results["elapsed_time_s"],
            }

            # Log symbolic regression summary
            sr = results["symbolic_regression"]
            if "ground_truth" in sr and isinstance(sr["ground_truth"], dict):
                gt_info = sr["ground_truth"]
                if "best_equation" in gt_info:
                    gt_eq = gt_info["best_equation"]
                    gt_r2 = gt_info.get("best_r_squared", "N/A")
                    logger.info(f"  GT best: {gt_eq} (R2={gt_r2})")
            if "dreamed" in sr and isinstance(sr["dreamed"], dict):
                dr_info = sr["dreamed"]
                if "best_equation" in dr_info:
                    dr_eq = dr_info["best_equation"]
                    dr_r2 = dr_info.get("best_r_squared", "N/A")
                    logger.info(f"  Dream best: {dr_eq} (R2={dr_r2})")

        except FileNotFoundError as e:
            logger.error(f"Skipping {domain}: {e}")
            logger.error("Train the world model first with: python scripts/train_world_models.py")
            all_results[domain] = {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in {domain}: {e}", exc_info=True)
            all_results[domain] = {"error": str(e)}

    total_time = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"DREAM DISCOVERY COMPLETE in {total_time:.1f}s")
    logger.info(f"{'=' * 60}")

    for domain, summary in all_results.items():
        if "error" in summary:
            logger.info(f"  {domain}: ERROR - {summary['error']}")
        else:
            logger.info(
                f"  {domain}: MSE={summary['dream_quality_mse']:.4f}, "
                f"corr={summary['dream_quality_corr']:.4f}, "
                f"time={summary['elapsed_s']:.1f}s"
            )

    # Save overall summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
