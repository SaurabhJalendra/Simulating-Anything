"""Generate ablation study datasets measuring how different factors affect discovery quality.

This script generates data for systematic ablation studies on the projectile and
Lotka-Volterra domains. It varies one factor at a time while holding others constant,
producing datasets that can then be fed to PySR (in WSL) to measure R-squared.

Ablation factors:
  1. Training data amount -- how many samples are needed for discovery?
  2. Simulation timestep (dt) -- how does integration quality affect discovery?
  3. Observation noise -- how much noise can be tolerated?

The script is CPU-only and runs on Windows. PySR evaluation requires WSL + Julia;
see the --run-pysr flag and WSL instructions in comments.

Usage:
    python scripts/run_ablation_studies.py --domain projectile --output-dir output/ablation
    python scripts/run_ablation_studies.py --domain lotka_volterra --output-dir output/ablation
    python scripts/run_ablation_studies.py --domain all --output-dir output/ablation

Output structure:
    output/ablation/
      projectile/
        sample_size/
          n010.npz, n025.npz, ..., n225.npz
          summary.json
        timestep/
          dt0.100.npz, dt0.010.npz, dt0.001.npz
          summary.json
        noise/
          noise0.00.npz, noise0.01.npz, ..., noise0.20.npz
          summary.json
      lotka_volterra/
        sample_size/
          n010.npz, ..., n200.npz
          summary.json
        timestep/
          ...
        noise/
          ...
      ablation_manifest.json   -- full summary of all conditions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path so we can import without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.simulation.rigid_body import ProjectileSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECTILE_SAMPLE_SIZES = [10, 25, 50, 100, 225]
PROJECTILE_DT_VALUES = [0.1, 0.01, 0.001]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.20]

LV_SAMPLE_SIZES = [10, 25, 50, 100, 200]
LV_DT_VALUES = [0.1, 0.01, 0.001]

# PySR iteration counts for the iteration-count ablation
PYSR_ITERATION_COUNTS = [5, 10, 20, 40, 80]

# Fixed parameters for projectile (no drag, standard gravity)
PROJECTILE_GRAVITY = 9.81
PROJECTILE_DRAG = 0.0
PROJECTILE_SPEED_RANGE = (10.0, 50.0)
PROJECTILE_ANGLE_RANGE = (10.0, 80.0)

# Fixed parameters for Lotka-Volterra
LV_N_STEPS = 10000
LV_DEFAULT_DT = 0.01


# ---------------------------------------------------------------------------
# Projectile data generation
# ---------------------------------------------------------------------------

def _simulate_projectile_range(
    v0: float,
    angle_deg: float,
    gravity: float = PROJECTILE_GRAVITY,
    drag: float = PROJECTILE_DRAG,
    dt: float = 0.001,
) -> float:
    """Run a single projectile simulation and return the landing range."""
    config = SimulationConfig(
        domain=Domain.RIGID_BODY,
        dt=dt,
        n_steps=50000,
        parameters={
            "gravity": gravity,
            "drag_coefficient": drag,
            "initial_speed": float(v0),
            "launch_angle": float(angle_deg),
            "mass": 1.0,
        },
    )
    sim = ProjectileSimulation(config)
    sim.reset()

    for _ in range(config.n_steps):
        state = sim.step()
        if sim._landed:
            break

    return float(state[0])


def generate_projectile_dataset(
    n_samples: int,
    dt: float = 0.001,
    noise_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate projectile range data for a given number of samples.

    Uses a grid layout when n_samples is a perfect square, otherwise
    uses a quasi-random Sobol-like spread across the speed/angle space.

    Args:
        n_samples: Total number of (v0, angle) pairs to simulate.
        dt: Simulation timestep.
        noise_fraction: Fraction of range magnitude to add as Gaussian noise.
        rng: Random generator for noise and non-grid sampling.

    Returns:
        Dict with keys v0, theta, g, range, range_theory, each shape (n_samples,).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Determine grid dimensions or fall back to random sampling
    sqrt_n = int(np.round(np.sqrt(n_samples)))
    if sqrt_n * sqrt_n == n_samples:
        n_speeds = sqrt_n
        n_angles = sqrt_n
        speeds = np.linspace(PROJECTILE_SPEED_RANGE[0], PROJECTILE_SPEED_RANGE[1], n_speeds)
        angles_deg = np.linspace(PROJECTILE_ANGLE_RANGE[0], PROJECTILE_ANGLE_RANGE[1], n_angles)
        pairs = [(v, a) for v in speeds for a in angles_deg]
    else:
        # Use stratified random sampling for non-square counts
        speeds = np.linspace(
            PROJECTILE_SPEED_RANGE[0], PROJECTILE_SPEED_RANGE[1], n_samples
        )
        angles_deg = np.linspace(
            PROJECTILE_ANGLE_RANGE[0], PROJECTILE_ANGLE_RANGE[1], n_samples
        )
        rng.shuffle(angles_deg)
        pairs = list(zip(speeds, angles_deg))

    all_v0 = []
    all_theta = []
    all_g = []
    all_range = []
    all_range_theory = []

    for v0, angle_deg in pairs:
        r = _simulate_projectile_range(v0, angle_deg, dt=dt)
        theta_rad = np.radians(angle_deg)
        r_theory = v0**2 * np.sin(2 * theta_rad) / PROJECTILE_GRAVITY

        all_v0.append(v0)
        all_theta.append(theta_rad)
        all_g.append(PROJECTILE_GRAVITY)
        all_range.append(r)
        all_range_theory.append(r_theory)

    range_arr = np.array(all_range)

    # Add observation noise proportional to range magnitude
    if noise_fraction > 0:
        noise_scale = noise_fraction * np.abs(range_arr)
        noise = rng.normal(0, noise_scale)
        range_arr = range_arr + noise

    return {
        "v0": np.array(all_v0),
        "theta": np.array(all_theta),
        "g": np.array(all_g),
        "range": range_arr,
        "range_theory": np.array(all_range_theory),
    }


# ---------------------------------------------------------------------------
# Lotka-Volterra data generation
# ---------------------------------------------------------------------------

def _simulate_lv_equilibrium(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    prey_0: float,
    pred_0: float,
    dt: float = LV_DEFAULT_DT,
    n_steps: int = LV_N_STEPS,
) -> tuple[float, float]:
    """Run a single LV simulation and return time-averaged populations.

    Skips the first 20% of the trajectory to discard transients.
    """
    config = SimulationConfig(
        domain=Domain.AGENT_BASED,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "prey_0": prey_0,
            "predator_0": pred_0,
        },
    )
    sim = LotkaVolterraSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    trajectory = np.array(states)
    skip = n_steps // 5
    prey_avg = float(np.mean(trajectory[skip:, 0]))
    pred_avg = float(np.mean(trajectory[skip:, 1]))
    return prey_avg, pred_avg


def generate_lv_dataset(
    n_samples: int,
    dt: float = LV_DEFAULT_DT,
    noise_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate Lotka-Volterra equilibrium data for varied parameters.

    Args:
        n_samples: Number of random parameter sets to simulate.
        dt: Simulation timestep.
        noise_fraction: Fraction of population magnitude to add as noise.
        rng: Random generator for parameter sampling and noise.

    Returns:
        Dict with keys a_, b_, g_, d_, prey_avg, pred_avg, prey_theory, pred_theory.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_alpha = []
    all_beta = []
    all_gamma = []
    all_delta = []
    all_prey_avg = []
    all_pred_avg = []
    all_prey_theory = []
    all_pred_theory = []

    for i in range(n_samples):
        alpha = rng.uniform(0.5, 2.0)
        beta = rng.uniform(0.2, 0.8)
        gamma = rng.uniform(0.2, 0.8)
        delta = rng.uniform(0.05, 0.3)

        # Start near equilibrium to reduce transient effects
        prey_eq = gamma / delta
        pred_eq = alpha / beta
        prey_0 = prey_eq * rng.uniform(0.8, 1.2)
        pred_0 = pred_eq * rng.uniform(0.8, 1.2)

        prey_avg, pred_avg = _simulate_lv_equilibrium(
            alpha, beta, gamma, delta, prey_0, pred_0, dt=dt
        )

        all_alpha.append(alpha)
        all_beta.append(beta)
        all_gamma.append(gamma)
        all_delta.append(delta)
        all_prey_avg.append(prey_avg)
        all_pred_avg.append(pred_avg)
        all_prey_theory.append(prey_eq)
        all_pred_theory.append(pred_eq)

        if (i + 1) % 25 == 0:
            logger.info("  LV: generated %d/%d trajectories", i + 1, n_samples)

    prey_arr = np.array(all_prey_avg)
    pred_arr = np.array(all_pred_avg)

    # Add observation noise
    if noise_fraction > 0:
        prey_arr = prey_arr + rng.normal(0, noise_fraction * np.abs(prey_arr))
        pred_arr = pred_arr + rng.normal(0, noise_fraction * np.abs(pred_arr))

    return {
        "a_": np.array(all_alpha),
        "b_": np.array(all_beta),
        "g_": np.array(all_gamma),
        "d_": np.array(all_delta),
        "prey_avg": prey_arr,
        "pred_avg": pred_arr,
        "prey_theory": np.array(all_prey_theory),
        "pred_theory": np.array(all_pred_theory),
    }


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------

def run_sample_size_ablation(
    domain: str,
    output_dir: Path,
) -> list[dict]:
    """Generate datasets at different sample sizes."""
    sizes = PROJECTILE_SAMPLE_SIZES if domain == "projectile" else LV_SAMPLE_SIZES
    out_path = output_dir / domain / "sample_size"
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = []
    for n in sizes:
        label = f"n{n:03d}"
        logger.info("[%s] sample_size: generating %d samples...", domain, n)
        t0 = time.time()

        if domain == "projectile":
            data = generate_projectile_dataset(n_samples=n)
        else:
            data = generate_lv_dataset(n_samples=n)

        elapsed = time.time() - t0

        # Save dataset
        npz_path = out_path / f"{label}.npz"
        np.savez(npz_path, **data)

        # Compute simulation-vs-theory error
        if domain == "projectile":
            rel_err = np.mean(np.abs(data["range"] - data["range_theory"])
                              / np.maximum(np.abs(data["range_theory"]), 1e-10))
        else:
            rel_err_prey = np.mean(np.abs(data["prey_avg"] - data["prey_theory"])
                                   / np.maximum(np.abs(data["prey_theory"]), 1e-10))
            rel_err_pred = np.mean(np.abs(data["pred_avg"] - data["pred_theory"])
                                   / np.maximum(np.abs(data["pred_theory"]), 1e-10))
            rel_err = (rel_err_prey + rel_err_pred) / 2

        condition = {
            "factor": "sample_size",
            "value": n,
            "label": label,
            "domain": domain,
            "file": str(npz_path),
            "sim_vs_theory_error": float(rel_err),
            "generation_time_s": round(elapsed, 2),
        }
        conditions.append(condition)
        logger.info("  -> saved %s (%.1fs, sim error=%.4f%%)",
                    npz_path.name, elapsed, rel_err * 100)

    # Save summary
    summary = {
        "factor": "sample_size",
        "domain": domain,
        "values": sizes,
        "description": "Vary number of training samples while holding other factors constant.",
        "conditions": conditions,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return conditions


def run_timestep_ablation(
    domain: str,
    output_dir: Path,
) -> list[dict]:
    """Generate datasets at different simulation timesteps."""
    dt_values = PROJECTILE_DT_VALUES if domain == "projectile" else LV_DT_VALUES
    out_path = output_dir / domain / "timestep"
    out_path.mkdir(parents=True, exist_ok=True)

    # Use a moderate sample size to isolate the dt effect
    n_samples = 100 if domain == "projectile" else 50

    conditions = []
    for dt in dt_values:
        label = f"dt{dt:.3f}"
        logger.info("[%s] timestep: generating with dt=%g (n=%d)...", domain, dt, n_samples)
        t0 = time.time()

        if domain == "projectile":
            data = generate_projectile_dataset(n_samples=n_samples, dt=dt)
        else:
            data = generate_lv_dataset(n_samples=n_samples, dt=dt)

        elapsed = time.time() - t0

        npz_path = out_path / f"{label}.npz"
        np.savez(npz_path, **data)

        if domain == "projectile":
            rel_err = np.mean(np.abs(data["range"] - data["range_theory"])
                              / np.maximum(np.abs(data["range_theory"]), 1e-10))
        else:
            rel_err_prey = np.mean(np.abs(data["prey_avg"] - data["prey_theory"])
                                   / np.maximum(np.abs(data["prey_theory"]), 1e-10))
            rel_err_pred = np.mean(np.abs(data["pred_avg"] - data["pred_theory"])
                                   / np.maximum(np.abs(data["pred_theory"]), 1e-10))
            rel_err = (rel_err_prey + rel_err_pred) / 2

        condition = {
            "factor": "timestep",
            "value": dt,
            "label": label,
            "domain": domain,
            "file": str(npz_path),
            "n_samples": n_samples,
            "sim_vs_theory_error": float(rel_err),
            "generation_time_s": round(elapsed, 2),
        }
        conditions.append(condition)
        logger.info("  -> saved %s (%.1fs, sim error=%.4f%%)",
                    npz_path.name, elapsed, rel_err * 100)

    summary = {
        "factor": "timestep",
        "domain": domain,
        "values": dt_values,
        "n_samples": n_samples,
        "description": (
            "Vary simulation timestep (dt) to measure how integration "
            "accuracy affects downstream discovery quality."
        ),
        "conditions": conditions,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return conditions


def run_noise_ablation(
    domain: str,
    output_dir: Path,
) -> list[dict]:
    """Generate datasets with different levels of observation noise."""
    out_path = output_dir / domain / "noise"
    out_path.mkdir(parents=True, exist_ok=True)

    # Use a moderate-to-large sample size so noise effects are not confounded
    # with small-sample variance
    n_samples = 100 if domain == "projectile" else 100

    conditions = []
    for noise in NOISE_LEVELS:
        label = f"noise{noise:.2f}"
        logger.info("[%s] noise: generating with %.0f%% noise (n=%d)...",
                    domain, noise * 100, n_samples)
        t0 = time.time()

        # Use a fixed seed per noise level so the underlying data is identical
        # and only the noise realization differs
        rng = np.random.default_rng(42)

        if domain == "projectile":
            data = generate_projectile_dataset(
                n_samples=n_samples, noise_fraction=noise, rng=rng
            )
        else:
            data = generate_lv_dataset(
                n_samples=n_samples, noise_fraction=noise, rng=rng
            )

        elapsed = time.time() - t0

        npz_path = out_path / f"{label}.npz"
        np.savez(npz_path, **data)

        # Error here includes both simulation error and noise
        if domain == "projectile":
            rel_err = np.mean(np.abs(data["range"] - data["range_theory"])
                              / np.maximum(np.abs(data["range_theory"]), 1e-10))
        else:
            rel_err_prey = np.mean(np.abs(data["prey_avg"] - data["prey_theory"])
                                   / np.maximum(np.abs(data["prey_theory"]), 1e-10))
            rel_err_pred = np.mean(np.abs(data["pred_avg"] - data["pred_theory"])
                                   / np.maximum(np.abs(data["pred_theory"]), 1e-10))
            rel_err = (rel_err_prey + rel_err_pred) / 2

        condition = {
            "factor": "noise",
            "value": noise,
            "label": label,
            "domain": domain,
            "file": str(npz_path),
            "n_samples": n_samples,
            "noise_fraction": noise,
            "observed_vs_theory_error": float(rel_err),
            "generation_time_s": round(elapsed, 2),
        }
        conditions.append(condition)
        logger.info("  -> saved %s (%.1fs, obs error=%.4f%%)",
                    npz_path.name, elapsed, rel_err * 100)

    summary = {
        "factor": "noise",
        "domain": domain,
        "values": NOISE_LEVELS,
        "n_samples": n_samples,
        "description": (
            "Add Gaussian observation noise (as a fraction of signal magnitude) "
            "to measure noise robustness of symbolic regression."
        ),
        "conditions": conditions,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return conditions


def generate_pysr_iterations_manifest(
    domain: str,
    output_dir: Path,
) -> dict:
    """Generate a manifest for the PySR iteration-count ablation.

    This ablation reuses the full-size, clean dataset but varies the number of
    PySR iterations. Data generation is not needed -- only PySR needs to be
    run multiple times with different iteration counts. This function writes
    a manifest describing the conditions.
    """
    out_path = output_dir / domain / "pysr_iterations"
    out_path.mkdir(parents=True, exist_ok=True)

    # Point to the full-size clean dataset
    if domain == "projectile":
        reference_file = str(output_dir / domain / "sample_size" / "n225.npz")
        n_samples = 225
    else:
        reference_file = str(output_dir / domain / "sample_size" / "n200.npz")
        n_samples = 200

    conditions = []
    for n_iter in PYSR_ITERATION_COUNTS:
        conditions.append({
            "factor": "pysr_iterations",
            "value": n_iter,
            "label": f"iter{n_iter:03d}",
            "domain": domain,
            "reference_data_file": reference_file,
            "n_samples": n_samples,
            "note": "Run PySR with niterations={} on the reference dataset.".format(n_iter),
        })

    summary = {
        "factor": "pysr_iterations",
        "domain": domain,
        "values": PYSR_ITERATION_COUNTS,
        "reference_data_file": reference_file,
        "description": (
            "Vary PySR iteration count while using the full clean dataset. "
            "This ablation requires WSL + Julia + PySR to execute."
        ),
        "conditions": conditions,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("[%s] pysr_iterations: manifest written with %d conditions",
                domain, len(PYSR_ITERATION_COUNTS))
    return summary


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_conditions: list[dict]) -> None:
    """Print a human-readable summary of all ablation conditions."""
    print("\n" + "=" * 90)
    print("ABLATION STUDY SUMMARY")
    print("=" * 90)

    # Group by domain and factor
    by_domain: dict[str, dict[str, list[dict]]] = {}
    for c in all_conditions:
        domain = c["domain"]
        factor = c["factor"]
        by_domain.setdefault(domain, {}).setdefault(factor, []).append(c)

    for domain, factors in sorted(by_domain.items()):
        print(f"\n  Domain: {domain}")
        print("  " + "-" * 86)
        for factor, conds in sorted(factors.items()):
            values = [c["value"] for c in conds]
            n_files = len(conds)
            print(f"    {factor:20s}  {n_files} conditions: {values}")

    total = len(all_conditions)
    print(f"\n  Total conditions: {total}")
    print("=" * 90)

    # Print the PySR execution instructions
    print("\nTo run PySR on these datasets (requires WSL + Julia + PySR):")
    print("  wsl.exe -d Ubuntu -- bash -lc \"cd '/mnt/d/Git Repos/Simulating-Anything' && \\")
    print("    source .venv/bin/activate && \\")
    print("    python -c 'from scripts.run_ablation_studies import run_pysr_evaluation; "
          "run_pysr_evaluation()'\"")
    print()


# ---------------------------------------------------------------------------
# PySR evaluation (requires WSL + Julia)
# ---------------------------------------------------------------------------

def run_pysr_on_dataset(
    npz_path: str,
    domain: str,
    n_iterations: int = 40,
) -> dict[str, float]:
    """Run PySR on a saved dataset and return R-squared values.

    This function requires PySR (Julia backend) and should be run in WSL.

    Args:
        npz_path: Path to the .npz dataset file.
        domain: Either 'projectile' or 'lotka_volterra'.
        n_iterations: Number of PySR iterations.

    Returns:
        Dict with r_squared (and r_squared_prey, r_squared_pred for LV).
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    data = dict(np.load(npz_path))

    if domain == "projectile":
        X = np.column_stack([data["v0"], data["theta"], data["g"]])
        y = data["range"]
        variable_names = ["v0", "theta", "g"]

        discoveries = run_symbolic_regression(
            X, y,
            variable_names=variable_names,
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "square"],
            max_complexity=20,
            populations=20,
            population_size=40,
        )

        best_r2 = discoveries[0].evidence.fit_r_squared if discoveries else 0.0
        best_expr = discoveries[0].expression if discoveries else "none"
        return {
            "r_squared": best_r2,
            "best_expression": best_expr,
            "n_equations_found": len(discoveries),
        }

    else:
        # Lotka-Volterra: run PySR for both prey and predator equilibrium
        X = np.column_stack([data["a_"], data["b_"], data["g_"], data["d_"]])
        var_names = ["a_", "b_", "g_", "d_"]

        prey_disc = run_symbolic_regression(
            X, data["prey_avg"],
            variable_names=var_names,
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            max_complexity=15,
            populations=20,
            population_size=40,
        )
        pred_disc = run_symbolic_regression(
            X, data["pred_avg"],
            variable_names=var_names,
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        prey_r2 = prey_disc[0].evidence.fit_r_squared if prey_disc else 0.0
        pred_r2 = pred_disc[0].evidence.fit_r_squared if pred_disc else 0.0
        prey_expr = prey_disc[0].expression if prey_disc else "none"
        pred_expr = pred_disc[0].expression if pred_disc else "none"

        return {
            "r_squared": (prey_r2 + pred_r2) / 2,
            "r_squared_prey": prey_r2,
            "r_squared_pred": pred_r2,
            "best_expression_prey": prey_expr,
            "best_expression_pred": pred_expr,
        }


def run_pysr_evaluation(output_dir: str = "output/ablation") -> None:
    """Run PySR on all generated ablation datasets and save results.

    This function is meant to be called from WSL where PySR/Julia are available.
    It reads the manifest, runs PySR on each dataset, and appends R-squared
    results to the summary JSON files.
    """
    base = Path(output_dir)
    manifest_path = base / "ablation_manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found at %s. Run data generation first.", manifest_path)
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    results_all = []

    for condition in manifest["conditions"]:
        factor = condition["factor"]
        domain = condition["domain"]

        # For pysr_iterations, use the reference file with varied iteration count
        if factor == "pysr_iterations":
            npz_path = condition["reference_data_file"]
            n_iter = condition["value"]
        else:
            npz_path = condition["file"]
            n_iter = 40  # Default PySR iterations

        if not Path(npz_path).exists():
            logger.warning("Dataset not found: %s -- skipping", npz_path)
            continue

        logger.info("PySR: %s / %s = %s (n_iter=%d)...",
                    domain, factor, condition.get("label", ""), n_iter)
        try:
            pysr_result = run_pysr_on_dataset(npz_path, domain, n_iterations=n_iter)
            condition.update(pysr_result)
            results_all.append(condition)
            logger.info("  -> R2 = %.6f", pysr_result["r_squared"])
        except Exception as e:
            logger.error("  -> FAILED: %s", e)
            condition["r_squared"] = None
            condition["error"] = str(e)
            results_all.append(condition)

    # Save full results
    results_path = base / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump({"conditions": results_all}, f, indent=2)
    logger.info("PySR results saved to %s", results_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation study datasets for discovery quality analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_ablation_studies.py --domain projectile\n"
            "  python scripts/run_ablation_studies.py --domain all --output-dir output/ablation\n"
            "\n"
            "After generating data, run PySR evaluation in WSL:\n"
            "  wsl.exe -d Ubuntu -- bash -lc \"\n"
            "    cd '/mnt/d/Git Repos/Simulating-Anything' && \n"
            "    source .venv/bin/activate && \n"
            "    python -c 'from scripts.run_ablation_studies import "
            "run_pysr_evaluation; run_pysr_evaluation()'\"\n"
        ),
    )
    parser.add_argument(
        "--domain",
        choices=["projectile", "lotka_volterra", "all"],
        default="all",
        help="Which domain to generate ablation data for (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/ablation",
        help="Base output directory (default: output/ablation).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = ["projectile", "lotka_volterra"] if args.domain == "all" else [args.domain]

    logger.info("Ablation study data generation")
    logger.info("  Domains: %s", ", ".join(domains))
    logger.info("  Output: %s", output_dir.resolve())

    all_conditions = []
    t_start = time.time()

    for domain in domains:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Domain: %s", domain)
        logger.info("=" * 60)

        # Factor 1: Sample size
        conditions = run_sample_size_ablation(domain, output_dir)
        all_conditions.extend(conditions)

        # Factor 2: Simulation timestep
        conditions = run_timestep_ablation(domain, output_dir)
        all_conditions.extend(conditions)

        # Factor 3: Observation noise
        conditions = run_noise_ablation(domain, output_dir)
        all_conditions.extend(conditions)

        # Factor 4: PySR iterations (manifest only, no data generation needed)
        generate_pysr_iterations_manifest(domain, output_dir)

    total_time = time.time() - t_start

    # Write global manifest
    manifest = {
        "description": "Ablation study datasets for discovery quality analysis",
        "domains": domains,
        "factors": ["sample_size", "timestep", "noise", "pysr_iterations"],
        "total_conditions": len(all_conditions),
        "total_generation_time_s": round(total_time, 1),
        "conditions": all_conditions,
        "pysr_note": (
            "PySR evaluation requires WSL + Julia. Run run_pysr_evaluation() "
            "after generating datasets. The pysr_iterations factor reuses the "
            "largest clean dataset with varied iteration counts."
        ),
    }
    manifest_path = output_dir / "ablation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("")
    logger.info("Manifest saved to %s", manifest_path)

    print_summary_table(all_conditions)
    logger.info("Total data generation time: %.1fs", total_time)


if __name__ == "__main__":
    main()
