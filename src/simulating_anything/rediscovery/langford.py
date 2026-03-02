"""Langford system (Hopf-Hopf bifurcation) rediscovery.

Targets:
- SINDy recovery of Langford ODEs
- Quasiperiodic vs chaotic behavior detection (power spectrum analysis)
- Two incommensurate frequencies on a torus
- Lyapunov exponent estimation
- Parameter sweep for Hopf-Hopf bifurcation (varying b or a)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.langford import LangfordSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.LANGFORD


def _make_config(
    dt: float = 0.01,
    n_steps: int = 10000,
    **params: float,
) -> SimulationConfig:
    """Create a SimulationConfig for the Langford system."""
    defaults = {
        "a": 0.95, "b": 0.7, "c": 0.6,
        "d": 3.5, "e": 0.25, "f": 0.1,
        "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
    }
    defaults.update(params)
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single Langford trajectory for SINDy ODE recovery.

    Uses default parameters that produce torus / quasiperiodic dynamics.
    """
    config = _make_config(
        dt=dt, n_steps=n_steps,
        a=a, b=b, c=c, d=d, e=e, f=f,
        x_0=0.1, y_0=0.0, z_0=0.0,
    )
    sim = LangfordSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
    }


def generate_frequency_data(
    n_steps: int = 8192,
    dt: float = 0.01,
    **params: float,
) -> dict[str, np.ndarray]:
    """Compute the power spectrum of the Langford x-component.

    For quasiperiodic motion on a torus, the spectrum should show two
    dominant incommensurate frequencies. For chaos, it becomes broadband.
    """
    config = _make_config(dt=dt, n_steps=n_steps + 5000, **params)
    sim = LangfordSimulation(config)
    freqs, power = sim.compute_frequency_spectrum(
        n_steps=n_steps, n_transient=5000
    )

    # Find dominant peaks
    peak_indices = []
    for i in range(1, len(power) - 1):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            peak_indices.append(i)

    # Sort by power, take top peaks
    if peak_indices:
        peak_indices = sorted(peak_indices, key=lambda j: power[j], reverse=True)
    top_freqs = [float(freqs[j]) for j in peak_indices[:10]]
    top_powers = [float(power[j]) for j in peak_indices[:10]]

    return {
        "frequencies": freqs,
        "power": power,
        "peak_frequencies": np.array(top_freqs),
        "peak_powers": np.array(top_powers),
        "n_peaks": len(peak_indices),
    }


def generate_bifurcation_sweep_data(
    param_name: str = "b",
    n_values: int = 30,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep a bifurcation parameter and compute Lyapunov exponents.

    Default sweeps b (Hopf parameter) to map the Hopf-Hopf transition.
    """
    if param_name == "b":
        param_values = np.linspace(0.3, 1.2, n_values)
    elif param_name == "a":
        param_values = np.linspace(0.5, 1.5, n_values)
    else:
        param_values = np.linspace(0.0, 2.0, n_values)

    lyapunov_exps = []
    max_radii = []
    attractor_types = []

    for i, val in enumerate(param_values):
        params = {param_name: val}
        config = _make_config(dt=dt, n_steps=n_steps, **params)
        sim = LangfordSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Run more steps to measure radius
        sim.reset()
        for _ in range(5000):
            sim.step()
        r_vals = []
        for _ in range(5000):
            state = sim.step()
            r_vals.append(np.sqrt(state[0] ** 2 + state[1] ** 2))
        max_radii.append(np.max(r_vals))

        # Classify attractor type
        if lam > 0.01:
            atype = "chaotic"
        elif lam > -0.01:
            atype = "quasiperiodic_or_marginal"
        else:
            atype = "periodic_or_fixed"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  {param_name}={val:.3f}: Lyapunov={lam:.4f}, type={atype}"
            )

    return {
        "param_name": param_name,
        "param_values": param_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_radius": np.array(max_radii),
        "attractor_type": np.array(attractor_types),
    }


def detect_torus(
    freqs: np.ndarray,
    power: np.ndarray,
    threshold_ratio: float = 0.01,
) -> dict:
    """Detect quasiperiodic (torus) dynamics from a power spectrum.

    Quasiperiodic motion has two dominant incommensurate frequencies f1, f2
    with the ratio f1/f2 being irrational. In practice, we check that the
    ratio is not close to a simple rational number.

    Args:
        freqs: Frequency array from FFT.
        power: Power spectrum array.
        threshold_ratio: Minimum relative power for a peak.

    Returns:
        Dict with torus detection results.
    """
    # Skip DC component
    if len(freqs) < 3:
        return {"is_torus": False, "reason": "insufficient_data"}

    power_no_dc = power[1:].copy()
    freqs_no_dc = freqs[1:]

    if np.max(power_no_dc) == 0:
        return {"is_torus": False, "reason": "zero_power"}

    # Find peaks
    peak_indices = []
    for i in range(1, len(power_no_dc) - 1):
        if (power_no_dc[i] > power_no_dc[i - 1]
                and power_no_dc[i] > power_no_dc[i + 1]):
            peak_indices.append(i)

    if len(peak_indices) < 2:
        return {"is_torus": False, "reason": "fewer_than_2_peaks"}

    # Sort by power
    peak_indices = sorted(
        peak_indices, key=lambda j: power_no_dc[j], reverse=True
    )

    f1 = float(freqs_no_dc[peak_indices[0]])
    f2 = float(freqs_no_dc[peak_indices[1]])
    p1 = float(power_no_dc[peak_indices[0]])
    p2 = float(power_no_dc[peak_indices[1]])

    # Check that second peak is significant
    if p2 < threshold_ratio * p1:
        return {
            "is_torus": False,
            "reason": "second_peak_too_small",
            "f1": f1,
            "f2": f2,
        }

    # Check that ratio is not close to a simple rational p/q (q <= 8)
    if f1 == 0:
        return {"is_torus": False, "reason": "zero_frequency"}

    ratio = f2 / f1 if f1 < f2 else f1 / f2
    is_rational = False
    for q in range(1, 9):
        for p in range(1, q * 3):
            if abs(ratio - p / q) < 0.02:
                is_rational = True
                break
        if is_rational:
            break

    return {
        "is_torus": not is_rational,
        "f1": f1,
        "f2": f2,
        "frequency_ratio": float(ratio),
        "power_ratio": float(p2 / p1),
        "is_rational": is_rational,
        "n_significant_peaks": sum(
            1 for j in peak_indices
            if power_no_dc[j] > threshold_ratio * p1
        ),
    }


def run_langford_rediscovery(
    output_dir: str | Path = "output/rediscovery/langford",
    n_iterations: int = 40,
) -> dict:
    """Run the full Langford system rediscovery.

    1. Generate trajectory for SINDy ODE recovery
    2. Frequency analysis for quasiperiodic / torus detection
    3. Parameter sweep (b) for Hopf-Hopf bifurcation mapping
    4. Lyapunov exponent at default parameters
    5. Torus detection from power spectrum

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "langford",
        "targets": {
            "ode_x": "dx/dt = (z - b)*x - d*y",
            "ode_y": "dy/dt = d*x + (z - b)*y",
            "ode_z": "dz/dt = c + a*z - z^3/3 - (x^2+y^2)*(1+e*z) + f*z*x^3",
            "dynamics": "Hopf-Hopf bifurcation, torus, quasiperiodicity",
            "symmetry": "rotational symmetry in (x, y) plane",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Langford trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=float(ode_data["dt"]),
            feature_names=["x", "y", "z"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
            "true_a": float(ode_data["a"]),
            "true_b": float(ode_data["b"]),
            "true_c": float(ode_data["c"]),
            "true_d": float(ode_data["d"]),
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Frequency analysis ---
    logger.info("Part 2: Frequency analysis for torus detection...")
    freq_data = generate_frequency_data(n_steps=8192, dt=0.01)

    torus_result = detect_torus(
        freq_data["frequencies"], freq_data["power"]
    )
    results["frequency_analysis"] = {
        "n_peaks": int(freq_data["n_peaks"]),
        "top_frequencies": freq_data["peak_frequencies"][:5].tolist(),
        "top_powers": freq_data["peak_powers"][:5].tolist(),
        "torus_detection": {
            k: v for k, v in torus_result.items()
            if not isinstance(v, np.ndarray)
        },
    }
    logger.info(
        f"  Torus detected: {torus_result.get('is_torus', False)}"
    )
    if "f1" in torus_result and "f2" in torus_result:
        logger.info(
            f"  Frequencies: f1={torus_result['f1']:.4f}, "
            f"f2={torus_result['f2']:.4f}"
        )

    # --- Part 3: Bifurcation sweep (b parameter) ---
    logger.info("Part 3: Mapping Hopf-Hopf bifurcation (b sweep)...")
    bif_data = generate_bifurcation_sweep_data(
        param_name="b", n_values=30, n_steps=20000, dt=0.01
    )

    lam = bif_data["lyapunov_exponent"]
    b_vals = bif_data["param_values"]
    n_chaotic = int(np.sum(bif_data["attractor_type"] == "chaotic"))
    n_periodic = int(
        np.sum(bif_data["attractor_type"] == "periodic_or_fixed")
    )
    n_quasi = int(
        np.sum(bif_data["attractor_type"] == "quasiperiodic_or_marginal")
    )

    results["bifurcation_sweep"] = {
        "param_name": "b",
        "n_values": len(b_vals),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "n_quasiperiodic": n_quasi,
        "b_range": [float(b_vals[0]), float(b_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }

    # Find approximate transition points
    transitions = []
    for j in range(len(lam) - 1):
        if (lam[j] > 0 and lam[j + 1] <= 0) or (lam[j] <= 0 and lam[j + 1] > 0):
            frac = abs(lam[j]) / (abs(lam[j]) + abs(lam[j + 1]))
            b_cross = float(b_vals[j] + frac * (b_vals[j + 1] - b_vals[j]))
            transitions.append(b_cross)
    results["bifurcation_sweep"]["transitions"] = transitions
    if transitions:
        logger.info(f"  Lyapunov transitions at b = {transitions}")

    # --- Part 4: Lyapunov at default parameters ---
    logger.info("Part 4: Lyapunov exponent at default parameters...")
    config_default = _make_config(dt=0.01, n_steps=50000)
    sim_default = LangfordSimulation(config_default)
    sim_default.reset()
    for _ in range(10000):
        sim_default.step()
    lam_default = sim_default.estimate_lyapunov(n_steps=40000, dt=0.01)

    results["default_parameters"] = {
        "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1,
        "lyapunov_exponent": float(lam_default),
        "positive": bool(lam_default > 0),
    }
    logger.info(f"  Default Langford Lyapunov: {lam_default:.4f}")

    # --- Part 5: Trajectory statistics ---
    logger.info("Part 5: Trajectory statistics...")
    config_stats = _make_config(dt=0.01, n_steps=20000)
    sim_stats = LangfordSimulation(config_stats)
    traj_stats = sim_stats.compute_trajectory_statistics(
        n_steps=15000, n_transient=5000
    )
    results["trajectory_statistics"] = traj_stats

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )
    np.savez(
        output_path / "frequency_data.npz",
        frequencies=freq_data["frequencies"],
        power=freq_data["power"],
    )
    np.savez(
        output_path / "bifurcation_sweep.npz",
        b=bif_data["param_values"],
        lyapunov_exponent=bif_data["lyapunov_exponent"],
        max_radius=bif_data["max_radius"],
    )

    return results
