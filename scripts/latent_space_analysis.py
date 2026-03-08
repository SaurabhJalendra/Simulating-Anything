"""Universal latent space analysis across trained RSSM world models.

Analyzes what trained world models learn by:
1. Encoding trajectories from each domain into latent space
2. Computing latent space statistics (dimensionality, structure)
3. Comparing latent representations across domains
4. Identifying shared latent structures (e.g., similar attractors)

This implements V7 Phase 7A: Universal latent space analysis.

Usage (must run in WSL2 for GPU):
    python scripts/latent_space_analysis.py
    python scripts/latent_space_analysis.py --domains lorenz,rossler,chua
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

# Domains with trained models and their simulation classes
TRAINED_DOMAINS = {
    "lorenz": ("lorenz", "LorenzSimulation", 3),
    "rossler": ("rossler", "RosslerSimulation", 3),
    "double_pendulum": ("chaotic_ode", "DoublePendulum", 4),
    "duffing": ("duffing", "DuffingOscillator", 2),
    "harmonic_oscillator": (
        "harmonic_oscillator", "DampedHarmonicOscillator", 2
    ),
    "van_der_pol": ("van_der_pol", "VanDerPolSimulation", 2),
    "brusselator": ("brusselator", "BrusselatorSimulation", 2),
    "fitzhugh_nagumo": ("fitzhugh_nagumo", "FitzHughNagumoSimulation", 2),
    "lotka_volterra": ("agent_based", "LotkaVolterraSimulation", 2),
    "sir_epidemic": ("epidemiological", "SIRSimulation", 3),
    "seir": ("seir", "SEIRSimulation", 4),
    "three_species": ("three_species", "ThreeSpecies", 3),
    "kuramoto": ("kuramoto", "KuramotoSimulation", None),
    "chua": ("chua", "ChuaCircuit", 3),
    "hodgkin_huxley": ("hodgkin_huxley", "HodgkinHuxleySimulation", 4),
    # V7 novel domains
    "stochastic_resonance": (
        "stochastic_resonance", "StochasticResonanceSimulation", 2
    ),
    "replicator_mutator": (
        "replicator_mutator", "ReplicatorMutatorSimulation", 3
    ),
    "lorenz_stommel": ("lorenz_stommel", "LorenzStommelSimulation", 5),
    # Phase 7A additional domains
    "kepler": ("kepler", "KeplerSimulation", 4),
    "toda_lattice": ("toda_lattice", "TodaLattice", None),
    "driven_pendulum": ("driven_pendulum", "DrivenPendulumSimulation", 2),
    "elastic_pendulum": ("elastic_pendulum", "ElasticPendulumSimulation", 4),
    "coupled_oscillators": (
        "coupled_oscillators", "CoupledOscillatorSimulation", 4
    ),
    "selkov": ("selkov", "SelkovSimulation", 2),
    "mackey_glass": ("mackey_glass", "MackeyGlassSimulation", None),
    "wilson_cowan": ("wilson_cowan", "WilsonCowanSimulation", 2),
    "langford": ("langford", "LangfordSimulation", 3),
}

WM_DIR = Path("output/world_models")


def load_model_and_encode(
    domain: str,
    n_trajectories: int = 20,
    seq_len: int = 200,
) -> dict | None:
    """Load a trained RSSM and encode trajectories into latent space.

    Args:
        domain: Domain name (must have trained model).
        n_trajectories: Number of test trajectories to encode.
        seq_len: Trajectory length.

    Returns:
        Dict with latent states, or None if loading fails.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from simulating_anything.world_model.rssm import RSSM

    model_path = WM_DIR / domain / "model.eqx"
    if not model_path.exists():
        logger.warning(f"No model for {domain} at {model_path}")
        return None

    if domain not in TRAINED_DOMAINS:
        logger.warning(f"Unknown domain {domain}")
        return None

    module_name, class_name, obs_dim = TRAINED_DOMAINS[domain]
    if obs_dim is None:
        logger.info(f"Skipping {domain} (variable obs dim)")
        return None

    # Generate trajectories
    try:
        import importlib
        mod = importlib.import_module(
            f"simulating_anything.simulation.{module_name}"
        )
        sim_class = getattr(mod, class_name)

        from simulating_anything.types.simulation import Domain, SimulationConfig
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.01,
            n_steps=seq_len,
            parameters={},
        )
        sim = sim_class(config)

        trajectories = []
        for seed in range(n_trajectories):
            sim.reset(seed=seed)
            states = [sim.observe().copy()]
            for _ in range(seq_len - 1):
                sim.step()
                states.append(sim.observe().copy())
            trajectories.append(np.array(states))

        obs_data = np.array(trajectories, dtype=np.float32)
        actual_obs_dim = obs_data.shape[-1]
    except Exception as e:
        logger.warning(f"Failed to generate trajectories for {domain}: {e}")
        return None

    # Load RSSM model
    try:
        key = jax.random.PRNGKey(0)
        model = RSSM(
            obs_shape=(actual_obs_dim,),
            action_size=0,
            deterministic_size=512,
            stochastic_size=32,
            num_categories=32,
            key=key,
        )
        model = eqx.tree_deserialise_leaves(str(model_path), model)
    except Exception as e:
        logger.warning(f"Failed to load model for {domain}: {e}")
        return None

    # Encode trajectories
    try:
        all_deterministic = []
        all_stochastic = []

        for traj_idx in range(min(n_trajectories, obs_data.shape[0])):
            obs_seq = jnp.array(obs_data[traj_idx])
            action = jnp.float32(0)

            # Initialize RSSM state
            h = jnp.zeros(512)
            z = jnp.zeros(32 * 32)

            det_states = []
            sto_states = []

            for t in range(obs_seq.shape[0]):
                obs_t = obs_seq[t]
                # Observe step: encode observation and update state
                h, z = model.observe_step(h, z, action, obs_t)
                det_states.append(np.array(h))
                sto_states.append(np.array(z))

            all_deterministic.append(np.array(det_states))
            all_stochastic.append(np.array(sto_states))

        det_array = np.array(all_deterministic)
        sto_array = np.array(all_stochastic)

        return {
            "domain": domain,
            "obs_dim": actual_obs_dim,
            "n_trajectories": det_array.shape[0],
            "seq_len": det_array.shape[1],
            "deterministic": det_array,
            "stochastic": sto_array,
        }
    except Exception as e:
        logger.warning(f"Failed to encode {domain}: {e}")
        return None


def compute_latent_statistics(encoded: dict) -> dict:
    """Compute statistics of the latent space representations.

    Args:
        encoded: Output from load_model_and_encode.

    Returns:
        Dict with statistics (mean, std, PCA variance, etc.).
    """
    det = encoded["deterministic"]
    # Flatten to (n_samples, latent_dim)
    flat_det = det.reshape(-1, det.shape[-1])

    # Basic statistics
    mean = np.mean(flat_det, axis=0)
    std = np.std(flat_det, axis=0)

    # PCA-based dimensionality analysis
    centered = flat_det - mean
    # Use SVD for PCA (more numerically stable)
    n_components = min(50, flat_det.shape[0], flat_det.shape[1])
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained_variance = (S ** 2) / (len(flat_det) - 1)
        total_var = np.sum(explained_variance)
        cumulative = np.cumsum(explained_variance) / total_var

        # Effective dimensionality: number of components for 95% variance
        dim_95 = int(np.searchsorted(cumulative, 0.95)) + 1
        dim_99 = int(np.searchsorted(cumulative, 0.99)) + 1
    except Exception:
        dim_95 = det.shape[-1]
        dim_99 = det.shape[-1]
        explained_variance = np.ones(n_components)
        cumulative = np.ones(n_components)

    # Activation sparsity
    active_dims = int(np.sum(std > 0.01))

    return {
        "domain": encoded["domain"],
        "obs_dim": encoded["obs_dim"],
        "latent_dim": det.shape[-1],
        "n_samples": len(flat_det),
        "mean_activation": float(np.mean(np.abs(mean))),
        "mean_std": float(np.mean(std)),
        "active_dimensions": active_dims,
        "effective_dim_95": dim_95,
        "effective_dim_99": dim_99,
        "top_5_variance": explained_variance[:5].tolist(),
        "cumulative_90pct": int(np.searchsorted(cumulative, 0.90)) + 1,
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representations.

    CKA measures similarity between representations independent of
    orthogonal transforms and isotropic scaling. A value of 1.0 means
    the representations capture identical structure; 0.0 means no
    shared structure.

    Reference: Kornblith et al., "Similarity of Neural Network
    Representations Revisited" (ICML 2019).

    Args:
        X: Representation matrix (n_samples, d1).
        Y: Representation matrix (n_samples, d2).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center the representations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # HSIC estimators
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_cka_matrix(
    encoded_list: list[dict],
    n_samples: int = 500,
) -> tuple[list[str], np.ndarray]:
    """Compute pairwise CKA similarity between all domain latent spaces.

    Subsamples each domain's deterministic latent states to n_samples
    for fair comparison, then computes the full pairwise CKA matrix.

    Args:
        encoded_list: List of encoded dicts from load_model_and_encode.
        n_samples: Number of samples to use per domain.

    Returns:
        Tuple of (domain_names, cka_matrix) where cka_matrix[i,j]
        is the CKA similarity between domains i and j.
    """
    domains = [e["domain"] for e in encoded_list]
    n = len(domains)

    # Flatten and subsample deterministic states
    flat_states = []
    for e in encoded_list:
        det = e["deterministic"]
        flat = det.reshape(-1, det.shape[-1])
        if len(flat) > n_samples:
            idx = np.random.default_rng(42).choice(
                len(flat), n_samples, replace=False
            )
            flat = flat[idx]
        flat_states.append(flat)

    cka_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            # Use the minimum sample count for fair comparison
            min_n = min(len(flat_states[i]), len(flat_states[j]))
            Xi = flat_states[i][:min_n]
            Yj = flat_states[j][:min_n]
            cka = linear_cka(Xi, Yj)
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka

    return domains, cka_matrix


def compare_latent_spaces(
    stats_list: list[dict],
    encoded_list: list[dict] | None = None,
) -> dict:
    """Compare latent space structure across domains.

    Args:
        stats_list: List of per-domain statistics.
        encoded_list: Optional list of encoded dicts for CKA analysis.

    Returns:
        Cross-domain comparison results.
    """
    if len(stats_list) < 2:
        return {"error": "need at least 2 domains"}

    domains = [s["domain"] for s in stats_list]
    eff_dims = [s["effective_dim_95"] for s in stats_list]
    active_dims = [s["active_dimensions"] for s in stats_list]

    result = {
        "n_domains": len(domains),
        "domains": domains,
        "effective_dim_95": {
            "mean": float(np.mean(eff_dims)),
            "std": float(np.std(eff_dims)),
            "min": int(np.min(eff_dims)),
            "max": int(np.max(eff_dims)),
            "per_domain": dict(zip(domains, eff_dims)),
        },
        "active_dimensions": {
            "mean": float(np.mean(active_dims)),
            "std": float(np.std(active_dims)),
            "per_domain": dict(zip(domains, active_dims)),
        },
    }

    # Compute CKA similarity matrix if encoded data is available
    if encoded_list is not None and len(encoded_list) >= 2:
        logger.info("Computing CKA similarity matrix...")
        cka_domains, cka_matrix = compute_cka_matrix(encoded_list)
        result["cka_similarity"] = {
            "domains": cka_domains,
            "matrix": cka_matrix.tolist(),
        }

        # Find most similar pairs (excluding self-similarity)
        pairs = []
        n = len(cka_domains)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((cka_domains[i], cka_domains[j], cka_matrix[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        result["cka_similarity"]["most_similar_pairs"] = [
            {"domain_a": a, "domain_b": b, "cka": float(c)}
            for a, b, c in pairs[:10]
        ]
        result["cka_similarity"]["mean_cka"] = float(
            np.mean([c for _, _, c in pairs])
        )

    result["observation"] = (
        "Domains with similar effective dimensionality and high CKA "
        "similarity share latent structure, suggesting transferable "
        "representations across dynamical system classes."
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Universal latent space analysis"
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain list (default: all trained)",
    )
    parser.add_argument(
        "--n-traj", type=int, default=20,
        help="Trajectories per domain",
    )
    parser.add_argument(
        "--seq-len", type=int, default=200,
        help="Trajectory length",
    )
    args = parser.parse_args()

    if args.domains:
        domains = args.domains.split(",")
    else:
        domains = [
            d for d in TRAINED_DOMAINS
            if (WM_DIR / d / "model.eqx").exists()
        ]

    logger.info(f"Analyzing {len(domains)} domains: {domains}")

    output_dir = Path("output/latent_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    all_encoded = []
    t0 = time.time()

    for domain in domains:
        logger.info(f"\n--- {domain} ---")
        encoded = load_model_and_encode(
            domain, n_trajectories=args.n_traj, seq_len=args.seq_len,
        )
        if encoded is None:
            continue

        stats = compute_latent_statistics(encoded)
        all_stats.append(stats)
        all_encoded.append(encoded)
        logger.info(
            f"  obs_dim={stats['obs_dim']}, "
            f"eff_dim_95={stats['effective_dim_95']}, "
            f"active={stats['active_dimensions']}/{stats['latent_dim']}"
        )

    # Cross-domain comparison with CKA similarity
    comparison = compare_latent_spaces(all_stats, all_encoded)

    results = {
        "n_domains_analyzed": len(all_stats),
        "per_domain_stats": all_stats,
        "cross_domain_comparison": comparison,
        "elapsed_seconds": time.time() - t0,
    }

    results_file = output_dir / "latent_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LATENT SPACE ANALYSIS SUMMARY")
    logger.info("=" * 60)
    for s in all_stats:
        logger.info(
            f"  {s['domain']:25s}: obs={s['obs_dim']}, "
            f"eff_dim={s['effective_dim_95']}, "
            f"active={s['active_dimensions']}"
        )
    if "effective_dim_95" in comparison:
        ed = comparison["effective_dim_95"]
        logger.info(
            f"\n  Mean effective dim: {ed['mean']:.1f} +/- {ed['std']:.1f}"
        )

    if "cka_similarity" in comparison:
        cka = comparison["cka_similarity"]
        logger.info(f"\n  Mean CKA similarity: {cka['mean_cka']:.4f}")
        logger.info("  Most similar domain pairs:")
        for pair in cka["most_similar_pairs"][:5]:
            logger.info(
                f"    {pair['domain_a']:20s} <-> {pair['domain_b']:20s}: "
                f"CKA={pair['cka']:.4f}"
            )


if __name__ == "__main__":
    main()
