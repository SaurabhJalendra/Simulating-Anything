"""Complex Ginzburg-Landau equation rediscovery.

Targets:
- Benjamin-Feir instability threshold: 1 + c1*c2 < 0
- Amplitude statistics vs parameters (c1, c2)
- Phase defect density in the chaotic regime
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ginzburg_landau import GinzburgLandau
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_amplitude_data(
    n_c2: int = 20,
    c1: float = 2.0,
    n_steps: int = 2000,
    dt: float = 0.05,
    N: int = 128,
) -> dict[str, np.ndarray]:
    """Generate amplitude statistics vs c2 for fixed c1.

    For each c2, run the CGLE and measure the mean amplitude and its
    spatial standard deviation after transient decay.

    The Benjamin-Feir threshold is at c2 = -1/c1.
    """
    c2_values = np.linspace(-2.0, 0.5, n_c2)
    bf_threshold = -1.0 / c1 if c1 != 0 else np.inf

    all_c2 = []
    all_mean_amp = []
    all_std_amp = []
    all_bf_param = []

    for i, c2 in enumerate(c2_values):
        config = SimulationConfig(
            domain=Domain.GINZBURG_LANDAU,
            dt=dt,
            n_steps=n_steps,
            parameters={"c1": c1, "c2": c2, "L": 50.0, "N": float(N)},
        )
        sim = GinzburgLandau(config)
        sim.reset()

        # Run past transient
        for _ in range(n_steps):
            sim.step()

        # Measure statistics over a window
        amp_samples = []
        std_samples = []
        for _ in range(200):
            sim.step()
            amp_samples.append(sim.amplitude)
            std_samples.append(sim.spatial_std())

        mean_amp = float(np.mean(amp_samples))
        mean_std = float(np.mean(std_samples))
        bf_param = 1.0 + c1 * c2

        all_c2.append(c2)
        all_mean_amp.append(mean_amp)
        all_std_amp.append(mean_std)
        all_bf_param.append(bf_param)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  c2={c2:.3f}: <|A|>={mean_amp:.4f}, "
                f"std(|A|)={mean_std:.4f}, 1+c1*c2={bf_param:.3f}"
            )

    return {
        "c1": c1,
        "c2": np.array(all_c2),
        "mean_amplitude": np.array(all_mean_amp),
        "std_amplitude": np.array(all_std_amp),
        "bf_parameter": np.array(all_bf_param),
        "bf_threshold_c2": bf_threshold,
    }


def generate_benjamin_feir_data(
    n_c1: int = 10,
    n_c2: int = 10,
    n_steps: int = 2000,
    dt: float = 0.05,
    N: int = 64,
) -> dict[str, np.ndarray]:
    """Map the Benjamin-Feir instability boundary in (c1, c2) space.

    For each (c1, c2) pair, measure spatial non-uniformity (std of |A|).
    The boundary is at 1 + c1*c2 = 0, i.e., c2 = -1/c1.
    """
    c1_values = np.linspace(0.5, 3.0, n_c1)
    c2_values = np.linspace(-2.5, 0.5, n_c2)

    # Grid of results
    uniformity = np.zeros((n_c1, n_c2))
    bf_param = np.zeros((n_c1, n_c2))

    for i, c1 in enumerate(c1_values):
        for j, c2 in enumerate(c2_values):
            config = SimulationConfig(
                domain=Domain.GINZBURG_LANDAU,
                dt=dt,
                n_steps=n_steps,
                parameters={"c1": c1, "c2": c2, "L": 50.0, "N": float(N)},
            )
            sim = GinzburgLandau(config)
            sim.reset()

            # Run past transient
            for _ in range(n_steps):
                sim.step()

            # Measure spatial non-uniformity
            std_vals = []
            for _ in range(100):
                sim.step()
                std_vals.append(sim.spatial_std())

            uniformity[i, j] = float(np.mean(std_vals))
            bf_param[i, j] = 1.0 + c1 * c2

        logger.info(f"  Benjamin-Feir sweep: c1={c1:.2f} done")

    return {
        "c1": c1_values,
        "c2": c2_values,
        "uniformity": uniformity,
        "bf_parameter": bf_param,
    }


def generate_defect_data(
    c1: float = 2.0,
    c2: float = -1.2,
    n_steps: int = 3000,
    dt: float = 0.05,
    N: int = 256,
) -> dict[str, np.ndarray]:
    """Count phase defects over time for a specific (c1, c2) in the chaotic regime.

    Phase defects are topological objects where the amplitude goes through zero
    and the phase winds by 2*pi. Their density is a measure of spatiotemporal
    chaos.
    """
    config = SimulationConfig(
        domain=Domain.GINZBURG_LANDAU,
        dt=dt,
        n_steps=n_steps,
        parameters={"c1": c1, "c2": c2, "L": 100.0, "N": float(N)},
    )
    sim = GinzburgLandau(config)
    sim.reset()

    # Run past transient
    for _ in range(n_steps):
        sim.step()

    # Measure defects over a window
    times = []
    defect_counts = []
    amplitudes = []
    for step in range(500):
        sim.step()
        times.append(step * dt)
        defect_counts.append(sim.count_phase_defects())
        amplitudes.append(sim.amplitude)

    return {
        "c1": c1,
        "c2": c2,
        "bf_parameter": 1.0 + c1 * c2,
        "time": np.array(times),
        "defect_counts": np.array(defect_counts),
        "mean_amplitude": np.array(amplitudes),
    }


def run_ginzburg_landau_rediscovery(
    output_dir: str | Path = "output/rediscovery/ginzburg_landau",
    n_iterations: int = 40,
) -> dict:
    """Run the full CGLE rediscovery.

    1. Generate amplitude vs c2 data (fixed c1)
    2. Run PySR to find amplitude relationships
    3. Map Benjamin-Feir boundary
    4. Count phase defects in chaotic regime

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "ginzburg_landau",
        "targets": {
            "benjamin_feir": "Instability when 1 + c1*c2 < 0",
            "amplitude": "Mean |A| ~ 1 in uniform regime",
            "defects": "Phase defect density in chaotic regime",
        },
    }

    # --- Part 1: Amplitude vs c2 ---
    logger.info("Part 1: Generating amplitude vs c2 data...")
    amp_data = generate_amplitude_data(n_c2=20, c1=2.0, n_steps=2000, dt=0.05, N=128)

    bf_thresh_c2 = amp_data["bf_threshold_c2"]
    logger.info(f"  Theoretical BF threshold c2 = {bf_thresh_c2:.4f}")

    # Find empirical threshold: where spatial std increases sharply
    stable = amp_data["bf_parameter"] > 0
    unstable = amp_data["bf_parameter"] < 0
    if np.any(stable) and np.any(unstable):
        mean_std_stable = float(np.mean(amp_data["std_amplitude"][stable]))
        mean_std_unstable = float(np.mean(amp_data["std_amplitude"][unstable]))
        results["amplitude_data"] = {
            "n_samples": len(amp_data["c2"]),
            "bf_threshold_theory": bf_thresh_c2,
            "mean_std_stable": mean_std_stable,
            "mean_std_unstable": mean_std_unstable,
            "instability_detected": mean_std_unstable > 2 * mean_std_stable,
        }
        logger.info(
            f"  Stable std(|A|)={mean_std_stable:.4f}, "
            f"unstable std(|A|)={mean_std_unstable:.4f}"
        )

    # PySR: try to find relationship between bf_parameter and std_amplitude
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        valid = np.isfinite(amp_data["std_amplitude"])
        X = amp_data["bf_parameter"][valid].reshape(-1, 1)
        y = amp_data["std_amplitude"][valid]

        logger.info("  Running PySR: std(|A|) = f(1+c1*c2)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["bf"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "abs"],
            max_complexity=12,
            populations=15,
            population_size=30,
        )
        results["amplitude_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["amplitude_pysr"]["best"] = best.expression
            results["amplitude_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # --- Part 2: Benjamin-Feir boundary ---
    logger.info("Part 2: Mapping Benjamin-Feir boundary...")
    bf_data = generate_benjamin_feir_data(
        n_c1=8, n_c2=8, n_steps=1500, dt=0.05, N=64
    )

    # Classify each point: uniform (std < threshold) or non-uniform
    threshold = 0.05
    n_uniform = int(np.sum(bf_data["uniformity"] < threshold))
    n_nonuniform = int(np.sum(bf_data["uniformity"] >= threshold))
    # Check correlation between bf_parameter sign and uniformity
    bf_negative = bf_data["bf_parameter"] < 0
    correctly_unstable = np.sum(bf_negative & (bf_data["uniformity"] >= threshold))
    correctly_stable = np.sum(~bf_negative & (bf_data["uniformity"] < threshold))
    total = bf_data["uniformity"].size
    accuracy = float(correctly_unstable + correctly_stable) / total if total > 0 else 0

    results["benjamin_feir_boundary"] = {
        "n_c1": len(bf_data["c1"]),
        "n_c2": len(bf_data["c2"]),
        "n_uniform": n_uniform,
        "n_nonuniform": n_nonuniform,
        "classification_accuracy": accuracy,
    }
    logger.info(
        f"  BF boundary: {n_uniform} uniform, {n_nonuniform} non-uniform, "
        f"accuracy={accuracy:.2%}"
    )

    # --- Part 3: Phase defects ---
    logger.info("Part 3: Counting phase defects in chaotic regime...")
    defect_data = generate_defect_data(c1=2.0, c2=-1.2, n_steps=3000, dt=0.05, N=256)

    mean_defects = float(np.mean(defect_data["defect_counts"]))
    max_defects = int(np.max(defect_data["defect_counts"]))
    results["defect_data"] = {
        "c1": defect_data["c1"],
        "c2": defect_data["c2"],
        "bf_parameter": defect_data["bf_parameter"],
        "mean_defect_count": mean_defects,
        "max_defect_count": max_defects,
        "mean_amplitude": float(np.mean(defect_data["mean_amplitude"])),
    }
    logger.info(
        f"  Mean defects: {mean_defects:.1f}, max: {max_defects}, "
        f"<|A|>={np.mean(defect_data['mean_amplitude']):.4f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
