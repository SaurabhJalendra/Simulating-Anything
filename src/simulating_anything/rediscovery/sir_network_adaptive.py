"""SIR Network Adaptive rediscovery.

Targets:
- Rewiring suppresses epidemic: final size decreases with w
- Network clustering increases with rewiring
- Comparison of final size with vs without rewiring
- Epidemic threshold shift due to adaptive rewiring
- Optional PySR: final_size = f(w, beta, gamma)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sir_network_adaptive import (
    SIRNetworkAdaptiveSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.SIR_NETWORK_ADAPTIVE


def _make_config(
    N_nodes: int = 200,
    k_avg: float = 6.0,
    beta: float = 0.1,
    gamma: float = 0.05,
    w: float = 0.3,
    I_0: int = 5,
    dt: float = 1.0,
    n_steps: int = 200,
) -> SimulationConfig:
    """Create a SimulationConfig for SIR network adaptive."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N_nodes": float(N_nodes),
            "k_avg": k_avg,
            "beta": beta,
            "gamma": gamma,
            "w": w,
            "I_0": float(I_0),
        },
    )


def generate_rewiring_sweep_data(
    n_w: int = 20,
    n_trials: int = 10,
    N_nodes: int = 200,
    k_avg: float = 6.0,
    beta: float = 0.1,
    gamma: float = 0.05,
    n_steps: int = 300,
) -> dict[str, np.ndarray | float]:
    """Sweep rewiring rate w and measure epidemic final size.

    For each w value, run multiple trials and record the mean final size
    (fraction recovered + infected at end). This shows epidemic suppression
    as rewiring increases.
    """
    w_values = np.linspace(0.0, 1.0, n_w)
    mean_final_sizes = []
    std_final_sizes = []
    mean_clustering = []

    for i, w in enumerate(w_values):
        trial_sizes = []
        trial_clustering = []
        for trial in range(n_trials):
            config = _make_config(
                N_nodes=N_nodes, k_avg=k_avg, beta=beta, gamma=gamma,
                w=w, n_steps=n_steps,
            )
            sim = SIRNetworkAdaptiveSimulation(config)
            sim.reset(seed=42 + trial)

            for _ in range(n_steps):
                sim.step()

            trial_sizes.append(sim.final_size())
            trial_clustering.append(sim.get_clustering_coefficient())

        mean_final_sizes.append(float(np.mean(trial_sizes)))
        std_final_sizes.append(float(np.std(trial_sizes)))
        mean_clustering.append(float(np.mean(trial_clustering)))

        if (i + 1) % 5 == 0:
            logger.info(
                f"  w={w:.2f}: final_size={mean_final_sizes[-1]:.4f} "
                f"+/- {std_final_sizes[-1]:.4f}, "
                f"clustering={mean_clustering[-1]:.4f}"
            )

    return {
        "w": w_values,
        "mean_final_size": np.array(mean_final_sizes),
        "std_final_size": np.array(std_final_sizes),
        "mean_clustering": np.array(mean_clustering),
        "N_nodes": N_nodes,
        "k_avg": k_avg,
        "beta": beta,
        "gamma": gamma,
        "n_trials": n_trials,
    }


def generate_comparison_data(
    n_trials: int = 20,
    N_nodes: int = 200,
    k_avg: float = 6.0,
    beta: float = 0.1,
    gamma: float = 0.05,
    n_steps: int = 300,
) -> dict[str, np.ndarray | float]:
    """Compare epidemic outcomes with and without rewiring.

    Runs matched pairs of simulations (same seed) with w=0 and w=0.5 to
    directly measure the protective effect of rewiring.
    """
    no_rewire_sizes = []
    rewire_sizes = []
    no_rewire_peak = []
    rewire_peak = []

    for trial in range(n_trials):
        seed = 100 + trial

        # Without rewiring
        config_0 = _make_config(
            N_nodes=N_nodes, k_avg=k_avg, beta=beta, gamma=gamma,
            w=0.0, n_steps=n_steps,
        )
        sim_0 = SIRNetworkAdaptiveSimulation(config_0)
        sim_0.reset(seed=seed)
        peak_0 = 0.0
        for _ in range(n_steps):
            state = sim_0.step()
            if state[1] > peak_0:
                peak_0 = state[1]
        no_rewire_sizes.append(sim_0.final_size())
        no_rewire_peak.append(peak_0)

        # With rewiring
        config_w = _make_config(
            N_nodes=N_nodes, k_avg=k_avg, beta=beta, gamma=gamma,
            w=0.5, n_steps=n_steps,
        )
        sim_w = SIRNetworkAdaptiveSimulation(config_w)
        sim_w.reset(seed=seed)
        peak_w = 0.0
        for _ in range(n_steps):
            state = sim_w.step()
            if state[1] > peak_w:
                peak_w = state[1]
        rewire_sizes.append(sim_w.final_size())
        rewire_peak.append(peak_w)

    return {
        "no_rewire_final_size": np.array(no_rewire_sizes),
        "rewire_final_size": np.array(rewire_sizes),
        "no_rewire_peak": np.array(no_rewire_peak),
        "rewire_peak": np.array(rewire_peak),
        "n_trials": n_trials,
        "reduction_mean": float(
            np.mean(no_rewire_sizes) - np.mean(rewire_sizes)
        ),
        "reduction_frac": float(
            1.0 - np.mean(rewire_sizes) / max(np.mean(no_rewire_sizes), 1e-12)
        ),
    }


def generate_clustering_timeseries(
    N_nodes: int = 200,
    k_avg: float = 6.0,
    beta: float = 0.1,
    gamma: float = 0.05,
    w: float = 0.5,
    n_steps: int = 200,
    measure_interval: int = 10,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Track clustering coefficient and S-S edge fraction over time.

    Measures how rewiring restructures the network during the epidemic.
    """
    config = _make_config(
        N_nodes=N_nodes, k_avg=k_avg, beta=beta, gamma=gamma,
        w=w, n_steps=n_steps,
    )
    sim = SIRNetworkAdaptiveSimulation(config)
    sim.reset(seed=seed)

    times = [0]
    clustering = [sim.get_clustering_coefficient()]
    ss_fraction = [sim.get_ss_edge_fraction()]
    s_frac = [sim.observe()[0]]
    i_frac = [sim.observe()[1]]

    for step in range(1, n_steps + 1):
        sim.step()
        if step % measure_interval == 0:
            times.append(step)
            clustering.append(sim.get_clustering_coefficient())
            ss_fraction.append(sim.get_ss_edge_fraction())
            s_frac.append(sim.observe()[0])
            i_frac.append(sim.observe()[1])

    return {
        "time": np.array(times),
        "clustering": np.array(clustering),
        "ss_fraction": np.array(ss_fraction),
        "S_frac": np.array(s_frac),
        "I_frac": np.array(i_frac),
    }


def generate_beta_sweep_data(
    n_beta: int = 15,
    n_trials: int = 10,
    N_nodes: int = 200,
    k_avg: float = 6.0,
    gamma: float = 0.05,
    w: float = 0.3,
    n_steps: int = 300,
) -> dict[str, np.ndarray]:
    """Sweep beta for fixed w to show shifted epidemic threshold."""
    beta_values = np.linspace(0.01, 0.3, n_beta)
    mean_sizes = []

    for beta in beta_values:
        trial_sizes = []
        for trial in range(n_trials):
            config = _make_config(
                N_nodes=N_nodes, k_avg=k_avg, beta=beta, gamma=gamma,
                w=w, n_steps=n_steps,
            )
            sim = SIRNetworkAdaptiveSimulation(config)
            sim.reset(seed=42 + trial)
            for _ in range(n_steps):
                sim.step()
            trial_sizes.append(sim.final_size())
        mean_sizes.append(float(np.mean(trial_sizes)))

    return {
        "beta": beta_values,
        "mean_final_size": np.array(mean_sizes),
        "w": w,
        "gamma": gamma,
    }


def run_sir_network_adaptive_rediscovery(
    output_dir: str | Path = "output/rediscovery/sir_network_adaptive",
    n_iterations: int = 50,
) -> dict:
    """Run the SIR Network Adaptive rediscovery pipeline.

    Discovers:
    1. Rewiring suppresses epidemic final size
    2. Final size comparison with vs without rewiring
    3. Clustering coefficient evolution during epidemic
    4. Optional PySR: final_size = f(w, beta, gamma)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sir_network_adaptive",
        "targets": {
            "epidemic_suppression": "final size decreases with w",
            "clustering_increase": "clustering grows during rewiring",
            "threshold_shift": "effective R0 threshold shifts with w",
        },
    }

    # --- Part 1: Rewiring sweep ---
    logger.info("Part 1: Sweeping rewiring rate w...")
    sweep_data = generate_rewiring_sweep_data(
        n_w=20, n_trials=10, N_nodes=200,
        beta=0.1, gamma=0.05, n_steps=300,
    )

    # Quantify suppression: ratio of final size at w=1 vs w=0
    w_arr = sweep_data["w"]
    fs_arr = sweep_data["mean_final_size"]
    fs_no_rewire = fs_arr[0]  # w=0
    fs_max_rewire = fs_arr[-1]  # w=1
    suppression_ratio = (
        1.0 - fs_max_rewire / max(fs_no_rewire, 1e-12)
    )

    results["rewiring_sweep"] = {
        "n_points": len(w_arr),
        "final_size_w0": float(fs_no_rewire),
        "final_size_w1": float(fs_max_rewire),
        "suppression_ratio": float(suppression_ratio),
        "clustering_w0": float(sweep_data["mean_clustering"][0]),
        "clustering_w1": float(sweep_data["mean_clustering"][-1]),
    }
    logger.info(
        f"  w=0 final_size={fs_no_rewire:.4f}, "
        f"w=1 final_size={fs_max_rewire:.4f}, "
        f"suppression={suppression_ratio:.2%}"
    )

    # --- Part 2: With vs without rewiring comparison ---
    logger.info("Part 2: Comparing epidemics with vs without rewiring...")
    comp_data = generate_comparison_data(
        n_trials=20, N_nodes=200,
        beta=0.1, gamma=0.05, n_steps=300,
    )

    results["comparison"] = {
        "no_rewire_mean": float(np.mean(comp_data["no_rewire_final_size"])),
        "rewire_mean": float(np.mean(comp_data["rewire_final_size"])),
        "reduction_mean": comp_data["reduction_mean"],
        "reduction_frac": comp_data["reduction_frac"],
        "no_rewire_peak_mean": float(np.mean(comp_data["no_rewire_peak"])),
        "rewire_peak_mean": float(np.mean(comp_data["rewire_peak"])),
    }
    logger.info(
        f"  No rewire: {results['comparison']['no_rewire_mean']:.4f}, "
        f"With rewire: {results['comparison']['rewire_mean']:.4f}, "
        f"Reduction: {comp_data['reduction_frac']:.2%}"
    )

    # --- Part 3: Clustering time series ---
    logger.info("Part 3: Tracking clustering coefficient over time...")
    cluster_data = generate_clustering_timeseries(
        N_nodes=200, beta=0.1, gamma=0.05, w=0.5,
        n_steps=200, measure_interval=10,
    )

    # Check if clustering increased
    c_initial = cluster_data["clustering"][0]
    c_final = cluster_data["clustering"][-1]
    c_max = float(np.max(cluster_data["clustering"]))

    results["clustering_evolution"] = {
        "initial": float(c_initial),
        "final": float(c_final),
        "max": c_max,
        "increased": bool(c_max > c_initial),
        "n_timepoints": len(cluster_data["time"]),
    }
    logger.info(
        f"  Clustering: initial={c_initial:.4f}, "
        f"max={c_max:.4f}, final={c_final:.4f}"
    )

    # --- Part 4: Beta sweep with rewiring ---
    logger.info("Part 4: Beta sweep to detect threshold shift...")
    beta_data = generate_beta_sweep_data(
        n_beta=15, n_trials=10, N_nodes=200,
        gamma=0.05, w=0.3, n_steps=300,
    )

    # Also run without rewiring for comparison
    beta_data_no_w = generate_beta_sweep_data(
        n_beta=15, n_trials=10, N_nodes=200,
        gamma=0.05, w=0.0, n_steps=300,
    )

    # Find approximate threshold: first beta where final_size > 0.1
    threshold_w = _find_threshold(beta_data["beta"], beta_data["mean_final_size"])
    threshold_no_w = _find_threshold(
        beta_data_no_w["beta"], beta_data_no_w["mean_final_size"],
    )

    results["beta_sweep"] = {
        "threshold_with_rewiring": threshold_w,
        "threshold_without_rewiring": threshold_no_w,
        "threshold_shift": threshold_w - threshold_no_w,
    }
    logger.info(
        f"  Threshold (w=0): beta~{threshold_no_w:.4f}, "
        f"(w=0.3): beta~{threshold_w:.4f}"
    )

    # --- Part 5: PySR symbolic regression (optional) ---
    logger.info("Part 5: PySR symbolic regression...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use rewiring sweep: predict final size from w
        mask = fs_arr > 0.01
        if np.sum(mask) > 5:
            X = w_arr[mask].reshape(-1, 1)
            y = fs_arr[mask]
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["w_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "sqrt"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["pysr"]["best"] = best.expression
                results["pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        else:
            results["pysr"] = {"error": "Not enough data above threshold"}

    except ImportError:
        logger.warning("PySR not available -- skipping symbolic regression")
        results["pysr"] = {"error": "PySR not installed"}
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sweep_data.npz",
        w=w_arr,
        mean_final_size=fs_arr,
    )

    return results


def _find_threshold(
    beta_arr: np.ndarray,
    size_arr: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """Find approximate beta value where final size first exceeds threshold."""
    mask = size_arr > threshold
    if np.any(mask):
        idx = np.argmax(mask)
        if idx > 0:
            # Linear interpolation between adjacent points
            b0, b1 = float(beta_arr[idx - 1]), float(beta_arr[idx])
            s0, s1 = float(size_arr[idx - 1]), float(size_arr[idx])
            if s1 - s0 > 1e-12:
                return b0 + (threshold - s0) / (s1 - s0) * (b1 - b0)
        return float(beta_arr[idx])
    return float(beta_arr[-1])
