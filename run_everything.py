"""One-command reproduction: generates all results from scratch.

Runs all phases of the Simulating-Anything discovery pipeline:
1. Core 14 domain rediscoveries (PySR + SINDy)
2. Extended 178 domain rediscoveries
3. V7 novel discovery domains (5 new)
4. World model training (RSSM on RTX 5090)
5. Meta-discovery cross-domain analysis
6. Latent space analysis (PCA + CKA)
7. Cross-domain world model transfer

Must run in WSL2 with GPU + Julia for PySR.

Usage:
    python run_everything.py                    # All phases
    python run_everything.py --skip-training    # Skip world model training
    python run_everything.py --phase meta       # Meta-discovery only
    python run_everything.py --phase novel      # V7 novel domains only
    python run_everything.py --phase latent     # Latent space analysis
    python run_everything.py --phase transfer   # Cross-domain transfer
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_all_rediscoveries(pysr_iterations: int = 40):
    """Run all 195 domain rediscoveries."""
    from simulating_anything.rediscovery.runner import run_all_rediscoveries

    logger.info("=" * 70)
    logger.info("PHASE 1-3: Running all 195 domain rediscoveries")
    logger.info("=" * 70)
    t0 = time.time()
    results = run_all_rediscoveries(pysr_iterations=pysr_iterations)
    elapsed = time.time() - t0
    logger.info(f"All rediscoveries complete in {elapsed:.1f}s")
    return results


def run_novel_discoveries(pysr_iterations: int = 40):
    """Run V7 novel discovery domains only."""
    from simulating_anything.rediscovery.climate_epidemic import (
        run_climate_epidemic_rediscovery,
    )
    from simulating_anything.rediscovery.lorenz_stommel import (
        run_lorenz_stommel_rediscovery,
    )
    from simulating_anything.rediscovery.neural_cardiac import (
        run_neural_cardiac_rediscovery,
    )
    from simulating_anything.rediscovery.replicator_mutator import (
        run_replicator_mutator_rediscovery,
    )
    from simulating_anything.rediscovery.stochastic_resonance import (
        run_stochastic_resonance_rediscovery,
    )

    logger.info("=" * 70)
    logger.info("V7: Running novel discovery domains")
    logger.info("=" * 70)
    results = {}
    t0 = time.time()

    for name, fn in [
        ("stochastic_resonance", run_stochastic_resonance_rediscovery),
        ("replicator_mutator", run_replicator_mutator_rediscovery),
        ("lorenz_stommel", run_lorenz_stommel_rediscovery),
        ("climate_epidemic", run_climate_epidemic_rediscovery),
        ("neural_cardiac", run_neural_cardiac_rediscovery),
    ]:
        logger.info(f"\n--- {name} ---")
        try:
            r = fn(n_iterations=pysr_iterations)
            results[name] = r
            logger.info(f"  {name}: complete")
        except Exception as e:
            logger.error(f"  {name}: failed - {e}")
            results[name] = {"error": str(e)}

    elapsed = time.time() - t0
    logger.info(f"Novel discoveries complete in {elapsed:.1f}s")
    return results


def run_world_model_training(domains: str = "all_new", epochs: int = 200):
    """Train RSSM world models on specified domains."""
    import subprocess

    logger.info("=" * 70)
    logger.info(f"PHASE 4: Training world models ({domains}, {epochs} epochs)")
    logger.info("=" * 70)
    t0 = time.time()

    cmd = [
        sys.executable,
        "scripts/train_world_models_generic.py",
        f"--domains={domains}",
        f"--epochs={epochs}",
    ]
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    logger.info(f"World model training complete in {elapsed:.1f}s")
    return result.returncode == 0


def run_meta_discovery():
    """Run meta-discovery analysis across all domains."""
    from simulating_anything.analysis.meta_discovery import run_meta_discovery

    logger.info("=" * 70)
    logger.info("PHASE 5: Meta-discovery cross-domain analysis")
    logger.info("=" * 70)
    t0 = time.time()
    results = run_meta_discovery()
    elapsed = time.time() - t0

    if "overall_statistics" in results:
        stats = results["overall_statistics"]
        logger.info(f"  Domains analyzed: {stats['n_domains']}")
        logger.info(f"  Median R2: {stats['median_r2']:.4f}")
        logger.info(f"  R2 >= 0.99: {stats['n_above_099']}/{stats['n_domains']}")
    logger.info(f"Meta-discovery complete in {elapsed:.1f}s")
    return results


def run_latent_space_analysis():
    """Run CKA latent space analysis across trained world models."""
    import subprocess

    logger.info("=" * 70)
    logger.info("PHASE 6: Latent space analysis (PCA + CKA)")
    logger.info("=" * 70)
    t0 = time.time()

    cmd = [sys.executable, "scripts/latent_space_analysis.py"]
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    logger.info(f"Latent space analysis complete in {elapsed:.1f}s")
    return result.returncode == 0


def run_cross_domain_transfer():
    """Run cross-domain world model transfer experiments."""
    import subprocess

    logger.info("=" * 70)
    logger.info("PHASE 7: Cross-domain transfer experiments")
    logger.info("=" * 70)
    t0 = time.time()

    cmd = [sys.executable, "scripts/cross_domain_transfer.py"]
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    logger.info(f"Cross-domain transfer complete in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all Simulating-Anything analyses")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "rediscovery", "novel", "training", "meta", "latent", "transfer"],
        help="Which phase to run (default: all)",
    )
    parser.add_argument("--skip-training", action="store_true", help="Skip world model training")
    parser.add_argument("--pysr-iterations", type=int, default=40, help="PySR iterations")
    parser.add_argument("--epochs", type=int, default=200, help="World model training epochs")
    args = parser.parse_args()

    Path("output").mkdir(exist_ok=True)
    Path("output/rediscovery").mkdir(exist_ok=True)

    t_total = time.time()

    if args.phase in ("all", "rediscovery"):
        run_all_rediscoveries(pysr_iterations=args.pysr_iterations)

    if args.phase in ("all", "novel"):
        run_novel_discoveries(pysr_iterations=args.pysr_iterations)

    if args.phase in ("all", "training") and not args.skip_training:
        run_world_model_training(epochs=args.epochs)

    if args.phase in ("all", "meta"):
        run_meta_discovery()

    if args.phase in ("all", "latent"):
        run_latent_space_analysis()

    if args.phase in ("all", "transfer"):
        run_cross_domain_transfer()

    total_time = time.time() - t_total
    logger.info("=" * 70)
    logger.info(f"ALL PHASES COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
