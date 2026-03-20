"""Noise robustness: how does SINDy equation recovery degrade with measurement noise?

For 10 key domains, adds Gaussian noise at increasing levels and measures
whether SINDy can still recover the correct equation structure.

Usage (requires WSL for PySINDy):
    python scripts/run_noise_robustness.py
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

OUTPUT_DIR = Path("output/noise_robustness")

DOMAINS = {
    "lorenz": ("lorenz", "LorenzSimulation", {"sigma": 10.0, "rho": 28.0, "beta": 2.667}, 0.01, 5000),
    "harmonic_oscillator": ("harmonic_oscillator", "DampedHarmonicOscillator", {"k": 4.0, "m": 1.0, "c": 0.4}, 0.01, 5000),
    "sir_epidemic": ("epidemiological", "SIRSimulation", {"beta": 0.4, "gamma": 0.1}, 0.1, 3000),
    "van_der_pol": ("van_der_pol", "VanDerPolSimulation", {"mu": 1.0}, 0.01, 5000),
    "brusselator": ("brusselator", "BrusselatorSimulation", {"a": 1.0, "b": 3.0}, 0.01, 5000),
    "lotka_volterra": ("agent_based", "LotkaVolterraSimulation", {"alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1}, 0.01, 5000),
    "fitzhugh_nagumo": ("fitzhugh_nagumo", "FitzHughNagumoSimulation", {"a": 0.7, "b": 0.8, "eps": 0.08, "I": 0.5}, 0.1, 5000),
    "rossler": ("rossler", "RosslerSimulation", {"a": 0.2, "b": 0.2, "c": 5.7}, 0.01, 5000),
    "predator_prey_climate": ("predator_prey_climate", "PredatorPreyClimateSimulation", {"coupling_TK": 0.2}, 0.01, 5000),
    "tumor_immune": ("tumor_immune", "TumorImmuneSimulation", {"coupling_ct": 0.05}, 0.1, 5000),
}

NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]


def generate_clean_data(module_name, class_name, params, dt, n_steps):
    """Generate clean simulation trajectory."""
    from simulating_anything.types.simulation import Domain, SimulationConfig

    mod = importlib.import_module(f"simulating_anything.simulation.{module_name}")
    cls = getattr(mod, class_name)
    config = SimulationConfig(domain=Domain.CUSTOM, dt=dt, n_steps=n_steps, parameters=params)
    sim = cls(config)
    sim.reset(seed=0)

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return np.array(states)


def fit_sindy_noisy(data, noise_frac, dt):
    """Add noise and fit SINDy."""
    import pysindy as ps

    rng = np.random.default_rng(42)
    signal_std = np.std(data, axis=0)
    signal_std = np.where(signal_std > 1e-10, signal_std, 1.0)

    noisy = data + noise_frac * signal_std * rng.normal(size=data.shape)
    dXdt = np.gradient(noisy, dt, axis=0)

    n_vars = data.shape[1]
    names = [f"x{i}" for i in range(n_vars)]

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.01),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    model.fit(noisy, t=dt, x_dot=dXdt, feature_names=names)

    pred = model.predict(noisy)
    r2s = []
    for i in range(n_vars):
        ss_res = np.sum((dXdt[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
        r2s.append(float(1.0 - ss_res / max(ss_tot, 1e-10)))

    n_active = sum(1 for c in model.coefficients().flatten() if abs(c) > 1e-10)

    return {
        "mean_r2": float(np.mean(r2s)),
        "r2_per_var": r2s,
        "n_active_terms": n_active,
        "equations": [model.equations(precision=4)[i] for i in range(n_vars)],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for domain_name, (mod, cls, params, dt, n_steps) in DOMAINS.items():
        logger.info(f"\n--- {domain_name} ---")

        try:
            clean_data = generate_clean_data(mod, cls, params, dt, n_steps)
        except Exception as e:
            logger.warning(f"  Failed to generate data: {e}")
            continue

        if np.any(np.isnan(clean_data)) or np.any(np.isinf(clean_data)):
            logger.warning(f"  Data contains NaN/Inf")
            continue

        domain_results = {"noise_levels": {}}
        for noise in NOISE_LEVELS:
            try:
                res = fit_sindy_noisy(clean_data, noise, dt)
                domain_results["noise_levels"][str(noise)] = res
                logger.info(f"  noise={noise:.1%}: R²={res['mean_r2']:.4f}, terms={res['n_active_terms']}")
            except Exception as e:
                domain_results["noise_levels"][str(noise)] = {"mean_r2": 0.0, "error": str(e)}
                logger.warning(f"  noise={noise:.1%}: FAILED - {e}")

        results[domain_name] = domain_results

    # Save
    out_path = OUTPUT_DIR / "noise_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("NOISE ROBUSTNESS SUMMARY")
    logger.info("=" * 70)
    header = f"  {'Domain':<25}" + "".join(f"{n:>8.1%}" for n in NOISE_LEVELS)
    logger.info(header)
    logger.info("  " + "-" * 73)
    for name, res in results.items():
        row = f"  {name:<25}"
        for n in NOISE_LEVELS:
            r2 = res.get("noise_levels", {}).get(str(n), {}).get("mean_r2", 0)
            row += f"{r2:>8.4f}"
        logger.info(row)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path("paper/figures")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, res in results.items():
            noises = []
            r2s = []
            for n in NOISE_LEVELS:
                r2 = res.get("noise_levels", {}).get(str(n), {}).get("mean_r2")
                if r2 is not None:
                    noises.append(n * 100)
                    r2s.append(r2)
            if noises:
                ax.plot(noises, r2s, "o-", label=name, linewidth=2, markersize=5)

        ax.set_xlabel("Noise Level (%)", fontsize=12)
        ax.set_ylabel("SINDy R²", fontsize=12)
        ax.set_title("Noise Robustness: Equation Recovery Under Measurement Noise", fontsize=13)
        ax.axhline(y=0.99, color="k", linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-0.1, 1.05)
        plt.tight_layout()
        plt.savefig(fig_dir / "noise_robustness.pdf", dpi=150)
        plt.savefig(fig_dir / "noise_robustness.png", dpi=150)
        plt.close()
        logger.info("Saved noise robustness figure")
    except Exception as e:
        logger.warning(f"Figure failed: {e}")


if __name__ == "__main__":
    main()
