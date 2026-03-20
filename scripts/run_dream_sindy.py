"""Dream-based equation discovery: run SINDy on RSSM dreamed trajectories.

For each domain with a trained world model:
1. Generate ground truth trajectories with varied parameters
2. Feed context to RSSM, dream forward
3. Run SINDy on BOTH ground truth and dreamed data
4. Compare: do dream-discovered equations match simulation-discovered equations?

This proves the world model learned the underlying physics, not just correlations.

Usage (must run in WSL2 for GPU):
    python scripts/run_dream_sindy.py
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/dream_sindy")
WM_DIR = Path("output/world_models")

# Domains to test (must have trained world models)
DOMAINS = {
    "lorenz": {
        "module": "lorenz", "class": "LorenzSimulation",
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
        "feature_names": ["x", "y", "z"],
        "dt": 0.01, "n_steps": 300,
    },
    "lotka_volterra": {
        "module": "agent_based", "class": "LotkaVolterraSimulation",
        "params": {"alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1},
        "feature_names": ["prey", "pred"],
        "dt": 0.01, "n_steps": 300,
    },
    "harmonic_oscillator": {
        "module": "harmonic_oscillator", "class": "DampedHarmonicOscillator",
        "params": {"k": 4.0, "m": 1.0, "c": 0.4},
        "feature_names": ["x", "v"],
        "dt": 0.01, "n_steps": 300,
    },
    "van_der_pol": {
        "module": "van_der_pol", "class": "VanDerPolSimulation",
        "params": {"mu": 1.0},
        "feature_names": ["x", "v"],
        "dt": 0.01, "n_steps": 300,
    },
    "brusselator": {
        "module": "brusselator", "class": "BrusselatorSimulation",
        "params": {"a": 1.0, "b": 3.0},
        "feature_names": ["u", "v"],
        "dt": 0.01, "n_steps": 300,
    },
    "sir_epidemic": {
        "module": "epidemiological", "class": "SIRSimulation",
        "params": {"beta": 0.4, "gamma": 0.1},
        "feature_names": ["S", "I", "R"],
        "dt": 0.01, "n_steps": 300,
    },
}

CONTEXT_STEPS = 20
DREAM_STEPS = 200


def load_world_model(domain: str):
    """Load trained RSSM world model."""
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
    if len(obs_shape) >= 2 and obs_shape[-1] > 4:
        return None

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
    trainer.params = eqx.tree_deserialise_leaves(str(model_path), trainer.params)
    encoder, rssm, decoder = trainer.params
    return encoder, rssm, decoder, meta


def generate_trajectories(domain_config: dict, n_traj: int = 20) -> np.ndarray:
    """Generate ground truth trajectories."""
    from simulating_anything.types.simulation import Domain, SimulationConfig

    mod = importlib.import_module(
        f"simulating_anything.simulation.{domain_config['module']}"
    )
    cls = getattr(mod, domain_config["class"])

    trajectories = []
    for seed in range(n_traj):
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=domain_config["dt"],
            n_steps=CONTEXT_STEPS + DREAM_STEPS,
            parameters=domain_config["params"],
        )
        sim = cls(config)
        sim.reset(seed=seed)
        states = [sim.observe().copy()]
        for _ in range(CONTEXT_STEPS + DREAM_STEPS):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))

    return np.array(trajectories, dtype=np.float32)


def dream_from_context(encoder, rssm, decoder, trajectories):
    """Feed context then dream forward."""
    import jax
    import jax.numpy as jnp

    action = jnp.float32(0)
    all_dreamed = []

    for traj_idx in range(len(trajectories)):
        traj = jnp.array(trajectories[traj_idx])
        key = jax.random.PRNGKey(traj_idx)

        # Feed context
        state = rssm.initial_state()
        for t in range(CONTEXT_STEPS):
            obs = traj[t]
            embed = encoder(obs.reshape(-1))
            key, step_key = jax.random.split(key)
            state, _, _ = rssm.observe_step(state, action, embed, key=step_key)

        # Dream forward
        dreamed = []
        for t in range(DREAM_STEPS):
            key, step_key = jax.random.split(key)
            state, _ = rssm.imagine_step(state, action, key=step_key)
            features = rssm.get_features(state)
            pred = decoder(features)
            dreamed.append(np.array(pred))

        all_dreamed.append(np.array(dreamed))

    return np.array(all_dreamed)


def run_sindy_on_data(data: np.ndarray, dt: float, feature_names: list[str]):
    """Run SINDy on trajectory data."""
    import pysindy as ps

    dXdt = np.gradient(data, dt, axis=0)

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.01),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    model.fit(data, t=dt, x_dot=dXdt, feature_names=feature_names)

    equations = []
    r2_scores = []
    pred = model.predict(data)
    for i in range(data.shape[1]):
        eq = model.equations(precision=4)[i]
        equations.append(f"d({feature_names[i]})/dt = {eq}")
        ss_res = np.sum((dXdt[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
        r2_scores.append(float(1.0 - ss_res / max(ss_tot, 1e-10)))

    return equations, r2_scores, model.coefficients()


def compare_coefficients(gt_coeffs, dream_coeffs):
    """Compare SINDy coefficient matrices."""
    if gt_coeffs.shape != dream_coeffs.shape:
        return {"match": False, "reason": "shape_mismatch"}

    # Relative error for non-zero coefficients
    gt_flat = gt_coeffs.flatten()
    dr_flat = dream_coeffs.flatten()
    nonzero = np.abs(gt_flat) > 0.01

    if not np.any(nonzero):
        return {"match": True, "relative_error": 0.0, "n_matching_terms": 0}

    rel_errors = np.abs(gt_flat[nonzero] - dr_flat[nonzero]) / np.abs(gt_flat[nonzero])
    mean_rel_error = float(np.mean(rel_errors))

    # Count matching terms (same sign and within 50% magnitude)
    matching = np.sum(
        (np.sign(gt_flat[nonzero]) == np.sign(dr_flat[nonzero]))
        & (rel_errors < 0.5)
    )

    return {
        "match": mean_rel_error < 0.3,
        "mean_relative_error": mean_rel_error,
        "n_matching_terms": int(matching),
        "n_nonzero_gt": int(np.sum(nonzero)),
        "match_fraction": float(matching / np.sum(nonzero)) if np.sum(nonzero) > 0 else 0.0,
    }


def run_domain(domain_name: str, domain_config: dict) -> dict:
    """Run dream-based SINDy for one domain."""
    logger.info(f"\n{'='*60}")
    logger.info(f"DOMAIN: {domain_name}")
    logger.info(f"{'='*60}")

    # Load world model
    loaded = load_world_model(domain_name)
    if loaded is None:
        logger.warning(f"  No trained model for {domain_name}")
        return {"status": "no_model"}

    encoder, rssm, decoder, meta = loaded
    logger.info(f"  Loaded model: obs_shape={meta['obs_shape']}")

    # Generate ground truth
    trajectories = generate_trajectories(domain_config, n_traj=10)
    logger.info(f"  Generated {len(trajectories)} trajectories, shape={trajectories.shape}")

    # Dream
    dreamed = dream_from_context(encoder, rssm, decoder, trajectories)
    logger.info(f"  Dreamed shape: {dreamed.shape}")

    # Ground truth for SINDy (the part after context)
    gt_after_context = trajectories[:, CONTEXT_STEPS:CONTEXT_STEPS + DREAM_STEPS, :]

    # Inverse symlog for dreamed data (decoder outputs symlog)
    dreamed_inv = np.sign(dreamed) * (np.exp(np.abs(dreamed)) - 1)

    # Use first trajectory for SINDy (concatenate all for better statistics)
    gt_concat = gt_after_context.reshape(-1, gt_after_context.shape[-1])
    dr_concat = dreamed_inv.reshape(-1, dreamed_inv.shape[-1])

    # Filter NaN/Inf
    gt_valid = gt_concat[~np.any(np.isnan(gt_concat) | np.isinf(gt_concat), axis=1)]
    dr_valid = dr_concat[~np.any(np.isnan(dr_concat) | np.isinf(dr_concat), axis=1)]

    if len(gt_valid) < 50 or len(dr_valid) < 50:
        logger.warning(f"  Not enough valid data for SINDy")
        return {"status": "insufficient_data"}

    dt = domain_config["dt"]
    names = domain_config["feature_names"]

    # Run SINDy on ground truth
    try:
        gt_eqs, gt_r2, gt_coeffs = run_sindy_on_data(gt_valid, dt, names)
        logger.info(f"  GT SINDy R²: {np.mean(gt_r2):.4f}")
        for eq in gt_eqs:
            logger.info(f"    {eq}")
    except Exception as e:
        logger.warning(f"  GT SINDy failed: {e}")
        return {"status": "gt_sindy_failed", "error": str(e)}

    # Run SINDy on dreamed data
    try:
        dr_eqs, dr_r2, dr_coeffs = run_sindy_on_data(dr_valid, dt, names)
        logger.info(f"  Dream SINDy R²: {np.mean(dr_r2):.4f}")
        for eq in dr_eqs:
            logger.info(f"    {eq}")
    except Exception as e:
        logger.warning(f"  Dream SINDy failed: {e}")
        return {"status": "dream_sindy_failed", "error": str(e)}

    # Compare
    comparison = compare_coefficients(gt_coeffs, dr_coeffs)
    logger.info(f"  Coefficient match: {comparison.get('match_fraction', 0):.1%}")

    # Dream quality
    min_len = min(len(gt_valid), len(dr_valid))
    mse = float(np.mean((gt_valid[:min_len] - dr_valid[:min_len]) ** 2))

    return {
        "status": "success",
        "gt_equations": gt_eqs,
        "gt_r2": gt_r2,
        "gt_mean_r2": float(np.mean(gt_r2)),
        "dream_equations": dr_eqs,
        "dream_r2": dr_r2,
        "dream_mean_r2": float(np.mean(dr_r2)),
        "coefficient_comparison": comparison,
        "dream_mse": mse,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for domain_name, config in DOMAINS.items():
        result = run_domain(domain_name, config)
        results[domain_name] = result

    # Save
    out_path = OUTPUT_DIR / "dream_sindy_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DREAM-BASED EQUATION DISCOVERY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Domain':<25} {'GT R²':>8} {'Dream R²':>10} {'Coeff Match':>12}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*12}")
    for name, res in results.items():
        if res["status"] == "success":
            gt = f"{res['gt_mean_r2']:.4f}"
            dr = f"{res['dream_mean_r2']:.4f}"
            match = f"{res['coefficient_comparison'].get('match_fraction', 0):.0%}"
        else:
            gt = dr = match = res["status"]
        logger.info(f"  {name:<25} {gt:>8} {dr:>10} {match:>12}")


if __name__ == "__main__":
    main()
