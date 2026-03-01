"""Schwarzschild geodesic rediscovery.

Targets:
- ISCO radius: r_isco = 6M
- Effective potential: V_eff(r) = -M/r + L^2/(2r^2) - ML^2/r^3
- Energy conservation along geodesics
- Orbital precession (non-Keplerian, GR correction)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.schwarzschild import SchwarzschildGeodesic
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_sim(
    M: float = 1.0,
    L: float = 4.0,
    r_0: float = 10.0,
    pr_0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 10000,
) -> SchwarzschildGeodesic:
    """Helper to create a SchwarzschildGeodesic with given parameters."""
    config = SimulationConfig(
        domain=Domain.SCHWARZSCHILD,
        dt=dt,
        n_steps=n_steps,
        parameters={"M": M, "L": L, "r_0": r_0, "pr_0": pr_0},
    )
    return SchwarzschildGeodesic(config)


def generate_orbit_data(
    n_orbits: int = 20,
    n_steps: int = 50000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate orbits with varying L and measure orbital properties.

    For each angular momentum value, we place the particle at the stable
    circular orbit radius and give it a small radial perturbation, then
    measure the orbital period and energy.

    Returns:
        Dictionary with arrays: L_vals, M_val, r_circ, E_orbit,
        T_orbit (measured), r_min, r_max.
    """
    rng = np.random.default_rng(42)
    M = 1.0

    all_L = []
    all_r_circ = []
    all_E = []
    all_T = []
    all_r_min = []
    all_r_max = []

    # L must be > 2*sqrt(3)*M for stable circular orbits to exist
    L_min = 2.0 * np.sqrt(3.0) * M * 1.05  # ~3.64
    L_max = 8.0
    L_vals = np.linspace(L_min, L_max, n_orbits)

    for i, L in enumerate(L_vals):
        sim = _make_sim(M=M, L=L, r_0=20.0, dt=dt, n_steps=1)
        r_circ = sim.find_circular_orbit_radius(L)
        if r_circ is None or r_circ <= sim.schwarzschild_radius:
            continue

        # Start slightly off circular orbit for a precessing ellipse
        pr_perturb = rng.uniform(0.0, 0.02)
        sim = _make_sim(M=M, L=L, r_0=r_circ, pr_0=pr_perturb, dt=dt, n_steps=n_steps)
        sim.reset()

        E_initial = sim.energy

        # Collect trajectory
        r_vals = [sim.observe()[0]]
        phi_vals = [sim.observe()[1]]
        for _ in range(n_steps):
            state = sim.step()
            if sim._captured:
                break
            r_vals.append(state[0])
            phi_vals.append(state[1])

        if sim._captured:
            continue

        r_arr = np.array(r_vals)

        # Measure radial period from radial turning points (periapsis crossings)
        # Find periapsis passages: local minima in r
        peri_times = []
        for j in range(1, len(r_arr) - 1):
            if r_arr[j] < r_arr[j - 1] and r_arr[j] < r_arr[j + 1]:
                peri_times.append(j * dt)

        if len(peri_times) >= 2:
            radial_periods = np.diff(peri_times)
            T_measured = float(np.median(radial_periods))
        else:
            T_measured = float("nan")

        all_L.append(L)
        all_r_circ.append(r_circ)
        all_E.append(E_initial)
        all_T.append(T_measured)
        all_r_min.append(float(np.min(r_arr)))
        all_r_max.append(float(np.max(r_arr)))

        if (i + 1) % 5 == 0:
            logger.info(f"  Orbit {i + 1}/{n_orbits}: L={L:.3f}, r_circ={r_circ:.3f}")

    return {
        "L": np.array(all_L),
        "M": np.full(len(all_L), M),
        "r_circ": np.array(all_r_circ),
        "E_orbit": np.array(all_E),
        "T_orbit": np.array(all_T),
        "r_min": np.array(all_r_min),
        "r_max": np.array(all_r_max),
    }


def generate_isco_data(
    n_points: int = 30,
) -> dict[str, np.ndarray]:
    """Sweep angular momentum L and find minimum stable circular orbit radius.

    For each L, compute the circular orbit radius from dV_eff/dr = 0 and
    check stability via d^2V_eff/dr^2 > 0.  The minimum stable r across
    all L should converge to 6M.

    Returns:
        Dictionary with arrays: L_vals, r_circ_vals, V_eff_at_circ, M_val,
        r_isco_theoretical.
    """
    M = 1.0

    L_min = 2.0 * np.sqrt(3.0) * M * 1.001  # Just above critical L
    L_max = 10.0
    L_vals = np.linspace(L_min, L_max, n_points)

    r_circ_vals = []
    V_eff_vals = []
    L_valid = []

    for L in L_vals:
        sim = _make_sim(M=M, L=L, r_0=20.0)
        r_circ = sim.find_circular_orbit_radius(L)
        if r_circ is not None and r_circ > sim.schwarzschild_radius:
            # Check stability: d^2V_eff/dr^2 > 0 at r_circ
            # d^2V/dr^2 = -2M/r^3 + 3L^2/r^4 - 12ML^2/r^5
            d2V = (
                -2.0 * M / r_circ**3
                + 3.0 * L**2 / r_circ**4
                - 12.0 * M * L**2 / r_circ**5
            )
            if d2V > 0:
                L_valid.append(L)
                r_circ_vals.append(r_circ)
                V_eff_vals.append(sim.effective_potential(r_circ))

    return {
        "L": np.array(L_valid),
        "r_circ": np.array(r_circ_vals),
        "V_eff_at_circ": np.array(V_eff_vals),
        "M": np.full(len(L_valid), M),
        "r_isco_theoretical": np.full(len(L_valid), 6.0 * M),
    }


def generate_precession_data(
    n_orbits: int = 15,
    n_steps: int = 100000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Measure orbital precession for near-circular orbits.

    The GR precession per orbit for a nearly circular orbit at radius r is:
        delta_phi = 6*pi*M / (r * (1 - 2M/r))

    We measure this by tracking the azimuthal angle accumulated between
    successive periapsis passages.

    Returns:
        Dictionary with arrays: L_vals, r_mean, precession_measured,
        precession_theory.
    """
    M = 1.0

    # Use L values giving orbits well outside ISCO
    L_vals = np.linspace(4.0, 7.0, n_orbits)

    all_L = []
    all_r_mean = []
    all_prec_measured = []
    all_prec_theory = []

    for i, L in enumerate(L_vals):
        sim = _make_sim(M=M, L=L, r_0=20.0, dt=dt, n_steps=1)
        r_circ = sim.find_circular_orbit_radius(L)
        if r_circ is None:
            continue

        # Small radial perturbation for a precessing ellipse
        pr_perturb = 0.01
        sim = _make_sim(
            M=M, L=L, r_0=r_circ, pr_0=pr_perturb,
            dt=dt, n_steps=n_steps,
        )
        sim.reset()

        r_vals = [sim.observe()[0]]
        phi_vals = [sim.observe()[1]]
        for _ in range(n_steps):
            state = sim.step()
            if sim._captured:
                break
            r_vals.append(state[0])
            phi_vals.append(state[1])

        if sim._captured or len(r_vals) < 100:
            continue

        r_arr = np.array(r_vals)
        phi_arr = np.array(phi_vals)

        # Find periapsis passages (local minima in r)
        peri_indices = []
        for j in range(1, len(r_arr) - 1):
            if r_arr[j] < r_arr[j - 1] and r_arr[j] < r_arr[j + 1]:
                peri_indices.append(j)

        if len(peri_indices) < 3:
            continue

        # Precession per orbit = delta_phi - 2*pi
        phi_per_orbit = []
        for j in range(1, len(peri_indices)):
            dphi = phi_arr[peri_indices[j]] - phi_arr[peri_indices[j - 1]]
            phi_per_orbit.append(dphi)

        if not phi_per_orbit:
            continue

        mean_dphi = float(np.mean(phi_per_orbit))
        precession = mean_dphi - 2.0 * np.pi
        r_mean = float(np.mean(r_arr))

        # Theoretical precession for nearly circular orbit at r_mean:
        # delta_phi_GR = 6*pi*M / (r * (1 - 2M/r))
        # For nearly circular orbit, approximate r ~ r_circ
        prec_theory = 6.0 * np.pi * M / (r_circ * (1.0 - 2.0 * M / r_circ))

        all_L.append(L)
        all_r_mean.append(r_mean)
        all_prec_measured.append(precession)
        all_prec_theory.append(prec_theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  Precession {i + 1}/{n_orbits}: L={L:.3f}, "
                f"prec={precession:.5f} rad (theory {prec_theory:.5f})"
            )

    return {
        "L": np.array(all_L),
        "r_mean": np.array(all_r_mean),
        "precession_measured": np.array(all_prec_measured),
        "precession_theory": np.array(all_prec_theory),
    }


def _generate_veff_data(
    n_L: int = 10,
    n_r: int = 50,
) -> dict[str, np.ndarray]:
    """Generate effective potential evaluations for PySR fitting.

    Sample V_eff(r, L) at many (r, L) pairs with M=1.

    Returns:
        Dictionary with arrays: r_vals, L_vals, M_vals, V_eff_vals.
    """
    M = 1.0
    L_range = np.linspace(3.5, 8.0, n_L)
    r_range = np.linspace(3.0, 30.0, n_r)

    all_r = []
    all_L = []
    all_V = []

    for L in L_range:
        sim = _make_sim(M=M, L=L)
        for r in r_range:
            V = sim.effective_potential(r)
            all_r.append(r)
            all_L.append(L)
            all_V.append(V)

    return {
        "r": np.array(all_r),
        "L": np.array(all_L),
        "M": np.full(len(all_r), M),
        "V_eff": np.array(all_V),
    }


def run_schwarzschild_rediscovery(
    output_dir: str | Path = "output/rediscovery/schwarzschild",
    n_iterations: int = 40,
) -> dict:
    """Run the full Schwarzschild geodesic rediscovery.

    1. ISCO = 6M relationship via PySR on circular orbit radii
    2. Effective potential V_eff(r, L) via PySR
    3. Energy conservation verification along geodesics
    4. Precession measurement and comparison to theory

    Args:
        output_dir: Directory for saving results.
        n_iterations: PySR iteration count.

    Returns:
        Results dictionary with all discoveries.
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "schwarzschild",
        "targets": {
            "isco": "r_isco = 6*M",
            "effective_potential": "V_eff = -M/r + L^2/(2r^2) - ML^2/r^3",
            "energy_conservation": "E = const along geodesic",
            "precession": "delta_phi = 6*pi*M / (r*(1 - 2M/r))",
        },
    }

    # --- Part 1: ISCO rediscovery ---
    logger.info("Part 1: ISCO radius = 6M...")
    isco_data = generate_isco_data(n_points=30)

    # The minimum r_circ across all L should approach 6M
    r_isco_measured = float(np.min(isco_data["r_circ"]))
    r_isco_theory = 6.0  # 6M with M=1
    isco_error = abs(r_isco_measured - r_isco_theory) / r_isco_theory

    results["isco"] = {
        "r_isco_measured": r_isco_measured,
        "r_isco_theory": r_isco_theory,
        "relative_error": isco_error,
        "n_points": len(isco_data["L"]),
    }
    logger.info(
        f"  ISCO: measured={r_isco_measured:.4f}, theory={r_isco_theory:.1f}, "
        f"error={isco_error:.4%}"
    )

    # PySR: r_circ = f(L, M) -- we expect r_circ ~ L^2/(2M) for large L
    # but at the ISCO limit, r_circ -> 6M
    logger.info(f"  Running PySR for r_circ = f(L, M) with {n_iterations} iterations...")
    X_isco = np.column_stack([isco_data["L"], isco_data["M"]])
    y_isco = isco_data["r_circ"]

    isco_discoveries = run_symbolic_regression(
        X_isco,
        y_isco,
        variable_names=["L_", "M_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square"],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    results["isco_pysr"] = {
        "n_discoveries": len(isco_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in isco_discoveries[:5]
        ],
    }
    if isco_discoveries:
        best = isco_discoveries[0]
        results["isco_pysr"]["best"] = best.expression
        results["isco_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # --- Part 2: Effective potential rediscovery ---
    logger.info("Part 2: Effective potential V_eff(r, L)...")
    veff_data = _generate_veff_data(n_L=10, n_r=50)

    logger.info(f"  Running PySR for V_eff = f(r, L) with {n_iterations} iterations...")
    X_veff = np.column_stack([veff_data["r"], veff_data["L"]])
    y_veff = veff_data["V_eff"]

    veff_discoveries = run_symbolic_regression(
        X_veff,
        y_veff,
        variable_names=["r_", "L_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square", "cube"],
        max_complexity=25,
        populations=30,
        population_size=50,
    )

    results["veff_pysr"] = {
        "n_discoveries": len(veff_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in veff_discoveries[:5]
        ],
    }
    if veff_discoveries:
        best = veff_discoveries[0]
        results["veff_pysr"]["best"] = best.expression
        results["veff_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # --- Part 3: Energy conservation ---
    logger.info("Part 3: Energy conservation along geodesics...")
    n_test_orbits = 10
    energy_drifts = []

    for j in range(n_test_orbits):
        L = 3.8 + j * 0.4
        sim = _make_sim(M=1.0, L=L, r_0=12.0, pr_0=0.0, dt=0.005, n_steps=20000)
        sim.reset()
        E0 = sim.energy
        E_values = [E0]

        for _ in range(20000):
            sim.step()
            if sim._captured:
                break
            E_values.append(sim.energy)

        if not sim._captured and len(E_values) > 100:
            E_arr = np.array(E_values)
            drift = float(np.max(np.abs(E_arr - E0)) / max(abs(E0), 1e-15))
            energy_drifts.append(drift)

    results["energy_conservation"] = {
        "n_orbits_tested": len(energy_drifts),
        "mean_max_drift": float(np.mean(energy_drifts)) if energy_drifts else None,
        "worst_drift": float(np.max(energy_drifts)) if energy_drifts else None,
        "all_drifts": [float(d) for d in energy_drifts],
    }
    if energy_drifts:
        logger.info(
            f"  Energy conservation: mean max drift = {np.mean(energy_drifts):.2e}, "
            f"worst = {np.max(energy_drifts):.2e}"
        )

    # --- Part 4: Precession ---
    logger.info("Part 4: Orbital precession...")
    prec_data = generate_precession_data(n_orbits=15, n_steps=100000, dt=0.005)

    if len(prec_data["precession_measured"]) > 0:
        prec_corr = float(np.corrcoef(
            prec_data["precession_measured"],
            prec_data["precession_theory"],
        )[0, 1])
        prec_rel_error = np.abs(
            prec_data["precession_measured"] - prec_data["precession_theory"]
        ) / np.maximum(np.abs(prec_data["precession_theory"]), 1e-10)
        results["precession"] = {
            "n_orbits": len(prec_data["L"]),
            "correlation_with_theory": prec_corr,
            "mean_relative_error": float(np.mean(prec_rel_error)),
            "precession_range_rad": [
                float(np.min(prec_data["precession_measured"])),
                float(np.max(prec_data["precession_measured"])),
            ],
        }
        logger.info(
            f"  Precession: {len(prec_data['L'])} orbits, "
            f"correlation={prec_corr:.4f}, "
            f"mean rel error={np.mean(prec_rel_error):.4%}"
        )
    else:
        results["precession"] = {"error": "No valid precession data"}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "orbit_data.npz",
        **{k: v for k, v in generate_orbit_data(n_orbits=20).items()},
    )
    np.savez(
        output_path / "isco_data.npz",
        **{k: v for k, v in isco_data.items()},
    )
    np.savez(
        output_path / "veff_data.npz",
        **{k: v for k, v in veff_data.items()},
    )
    if len(prec_data["L"]) > 0:
        np.savez(
            output_path / "precession_data.npz",
            **{k: v for k, v in prec_data.items()},
        )

    return results
