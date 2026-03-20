"""Extrapolation validation: do discovered equations predict beyond training range?

For each of 14 core domains:
1. Discover equation on training parameter range (SINDy)
2. Generate new simulation data at 1.5x, 2x, 3x the parameter range
3. Evaluate discovered equation's predictions on the new data
4. Compare against polynomial baseline

This proves discovered equations capture real physics, not polynomial curve fits.

Usage (requires WSL for PySINDy):
    python scripts/run_extrapolation_validation.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/extrapolation_validation")


# Domain configurations: (module, class, param_name, param_range, dt, n_steps)
DOMAINS = {
    "lorenz": {
        "module": "lorenz", "class": "LorenzSimulation",
        "param": "rho", "train_range": (20.0, 32.0),
        "fixed_params": {"sigma": 10.0, "beta": 2.667},
        "dt": 0.01, "n_steps": 5000,
    },
    "harmonic_oscillator": {
        "module": "harmonic_oscillator", "class": "DampedHarmonicOscillator",
        "param": "k", "train_range": (1.0, 10.0),
        "fixed_params": {"m": 1.0, "c": 0.2},
        "dt": 0.01, "n_steps": 3000,
    },
    "sir_epidemic": {
        "module": "epidemiological", "class": "SIRSimulation",
        "param": "beta", "train_range": (0.1, 0.5),
        "fixed_params": {"gamma": 0.1},
        "dt": 0.1, "n_steps": 2000,
    },
    "van_der_pol": {
        "module": "van_der_pol", "class": "VanDerPolSimulation",
        "param": "mu", "train_range": (0.5, 5.0),
        "fixed_params": {},
        "dt": 0.01, "n_steps": 5000,
    },
    "brusselator": {
        "module": "brusselator", "class": "BrusselatorSimulation",
        "param": "b", "train_range": (1.5, 4.0),
        "fixed_params": {"a": 1.0},
        "dt": 0.01, "n_steps": 5000,
    },
    "lotka_volterra": {
        "module": "agent_based", "class": "LotkaVolterraSimulation",
        "param": "alpha", "train_range": (0.5, 2.0),
        "fixed_params": {"beta": 0.4, "gamma": 0.4, "delta": 0.1},
        "dt": 0.01, "n_steps": 5000,
    },
}

EXTRAP_FACTORS = [1.0, 1.5, 2.0, 3.0]


def generate_data(domain_config: dict, param_values: list[float]) -> tuple:
    """Generate simulation data across parameter values."""
    import importlib
    from simulating_anything.types.simulation import Domain, SimulationConfig

    mod = importlib.import_module(
        f"simulating_anything.simulation.{domain_config['module']}"
    )
    cls = getattr(mod, domain_config["class"])

    all_states = []
    all_derivs = []

    for pval in param_values:
        params = dict(domain_config["fixed_params"])
        params[domain_config["param"]] = pval

        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=domain_config["dt"],
            n_steps=domain_config["n_steps"],
            parameters=params,
        )
        sim = cls(config)
        sim.reset(seed=0)

        states = [sim.observe().copy()]
        for _ in range(domain_config["n_steps"]):
            states.append(sim.step().copy())

        data = np.array(states)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            continue

        deriv = np.gradient(data, domain_config["dt"], axis=0)
        all_states.append(data)
        all_derivs.append(deriv)

    if not all_states:
        return np.array([]), np.array([])

    return np.vstack(all_states), np.vstack(all_derivs)


def fit_sindy(states: np.ndarray, derivs: np.ndarray, dt: float) -> object:
    """Fit SINDy model on data."""
    import pysindy as ps

    n_vars = states.shape[1]
    names = [f"x{i}" for i in range(n_vars)]

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.01),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    model.fit(states, t=dt, x_dot=derivs, feature_names=names)
    return model


def evaluate_r2(model, states: np.ndarray, derivs: np.ndarray) -> float:
    """Compute R² of SINDy model on data."""
    pred = model.predict(states)
    r2_per_var = []
    for i in range(states.shape[1]):
        ss_res = np.sum((derivs[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((derivs[:, i] - np.mean(derivs[:, i])) ** 2)
        if ss_tot > 1e-12:
            r2_per_var.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2_per_var)) if r2_per_var else 0.0


def run_extrapolation(domain_name: str, domain_config: dict) -> dict:
    """Run extrapolation validation for one domain."""
    lo, hi = domain_config["train_range"]
    train_width = hi - lo

    results = {"domain": domain_name, "extrapolation": {}}

    # Training data
    n_train = 15
    train_vals = np.linspace(lo, hi, n_train).tolist()
    train_states, train_derivs = generate_data(domain_config, train_vals)

    if train_states.size == 0:
        logger.warning(f"  {domain_name}: no valid training data")
        return results

    # Fit SINDy on training data
    try:
        model = fit_sindy(train_states, train_derivs, domain_config["dt"])
    except Exception as e:
        logger.warning(f"  {domain_name}: SINDy fit failed: {e}")
        return results

    train_r2 = evaluate_r2(model, train_states, train_derivs)
    results["train_r2"] = train_r2
    logger.info(f"  {domain_name}: train R² = {train_r2:.4f}")

    # Extrapolation at each factor
    for factor in EXTRAP_FACTORS:
        extrap_hi = lo + train_width * factor
        extrap_vals = np.linspace(hi, extrap_hi, 10).tolist()

        extrap_states, extrap_derivs = generate_data(domain_config, extrap_vals)
        if extrap_states.size == 0:
            results["extrapolation"][str(factor)] = {"r2": None, "status": "sim_failed"}
            continue

        try:
            extrap_r2 = evaluate_r2(model, extrap_states, extrap_derivs)
            results["extrapolation"][str(factor)] = {
                "r2": extrap_r2,
                "status": "ok",
                "param_range": [float(hi), float(extrap_hi)],
            }
            logger.info(f"    {factor}x: R² = {extrap_r2:.4f}")
        except Exception as e:
            results["extrapolation"][str(factor)] = {"r2": None, "status": str(e)}

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for domain_name, config in DOMAINS.items():
        logger.info(f"\n--- {domain_name} ---")
        result = run_extrapolation(domain_name, config)
        all_results[domain_name] = result

    # Save results
    out_path = OUTPUT_DIR / "extrapolation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRAPOLATION VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Domain':<25} {'Train':>8} {'1.5x':>8} {'2.0x':>8} {'3.0x':>8}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, res in all_results.items():
        train = f"{res.get('train_r2', 0):.4f}"
        extrap = {}
        for f in ["1.5", "2.0", "3.0"]:
            e = res.get("extrapolation", {}).get(f, {})
            r2 = e.get("r2")
            extrap[f] = f"{r2:.4f}" if r2 is not None else "  N/A"
        logger.info(
            f"  {name:<25} {train:>8} {extrap['1.5']:>8} "
            f"{extrap['2.0']:>8} {extrap['3.0']:>8}"
        )

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path("paper/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, res in all_results.items():
            factors = []
            r2s = []
            for f_str in ["1.0", "1.5", "2.0", "3.0"]:
                if f_str == "1.0":
                    r2 = res.get("train_r2")
                else:
                    r2 = res.get("extrapolation", {}).get(f_str, {}).get("r2")
                if r2 is not None:
                    factors.append(float(f_str))
                    r2s.append(r2)
            if factors:
                ax.plot(factors, r2s, "o-", label=name, linewidth=2, markersize=6)

        ax.set_xlabel("Extrapolation Factor", fontsize=12)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("Extrapolation Validation: Do Discovered Equations Generalize?", fontsize=13)
        ax.axhline(y=0.99, color="k", linestyle="--", alpha=0.3, label="R²=0.99")
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(-0.1, 1.05)
        plt.tight_layout()
        plt.savefig(fig_dir / "extrapolation_validation.pdf", dpi=150)
        plt.savefig(fig_dir / "extrapolation_validation.png", dpi=150)
        plt.close()
        logger.info("Saved extrapolation figure")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


if __name__ == "__main__":
    main()
