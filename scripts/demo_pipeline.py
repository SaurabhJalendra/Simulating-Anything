"""Self-contained demo of the Simulating Anything pipeline.

Showcases three domains (projectile, Lotka-Volterra, harmonic oscillator)
with data generation, lightweight equation fitting, and publication-quality
figures -- all on CPU in under 30 seconds, no Julia/PySR/WSL required.

Usage:
    python scripts/demo_pipeline.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402, I001
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from simulating_anything.simulation.agent_based import (  # noqa: E402
    LotkaVolterraSimulation,
)
from simulating_anything.simulation.harmonic_oscillator import (  # noqa: E402
    DampedHarmonicOscillator,
)
from simulating_anything.simulation.rigid_body import (  # noqa: E402
    ProjectileSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output/demo")
SUMMARY_RESULTS: list[dict] = []


def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parabolic_peak(magnitude: np.ndarray, freqs: np.ndarray, idx: int) -> float:
    """Refine FFT peak frequency with parabolic (3-point) interpolation.

    Given the magnitude spectrum and the index of the discrete peak, fits a
    parabola through the peak and its two neighbors to estimate the true
    peak frequency with sub-bin accuracy.
    """
    if idx <= 0 or idx >= len(magnitude) - 1:
        return freqs[idx]
    alpha = np.log(magnitude[idx - 1] + 1e-30)
    beta = np.log(magnitude[idx] + 1e-30)
    gamma = np.log(magnitude[idx + 1] + 1e-30)
    delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    bin_spacing = freqs[1] - freqs[0]
    return freqs[idx] + delta * bin_spacing


# ===========================================================================
# Domain 1 -- Projectile Motion (algebraic physics)
# ===========================================================================

def run_projectile_demo() -> dict:
    """Sweep launch speed and angle, fit the range equation R = v^2 sin(2t)/g.

    Uses linear regression on the feature v0^2 * sin(2*theta) as a lightweight
    proxy for symbolic regression.  Returns a results dict with R-squared and
    discovered coefficient.
    """
    print("\n" + "=" * 70)
    print("  DOMAIN 1: Projectile Motion  (algebraic)")
    print("=" * 70)

    g = 9.81
    drag = 0.0  # drag-free for clean equation recovery
    dt = 0.002  # 2 ms timestep for accuracy

    # -- Single showcase trajectory -------------------------------------------
    config_single = SimulationConfig(
        domain=Domain.RIGID_BODY,
        dt=dt,
        n_steps=5000,
        parameters={
            "initial_speed": 30.0,
            "launch_angle": 45.0,
            "gravity": g,
            "drag_coefficient": drag,
        },
    )
    sim = ProjectileSimulation(config_single)
    traj = sim.run(n_steps=5000)
    states = traj.states  # shape (N, 4): [x, y, vx, vy]
    in_air = states[:, 1] >= 0
    x_traj = states[in_air, 0]
    y_traj = states[in_air, 1]

    # -- Parameter sweep: 15 speeds x 15 angles --------------------------------
    speeds = np.linspace(10, 50, 15)
    angles_deg = np.linspace(10, 80, 15)
    records = []

    for v0 in speeds:
        for theta_deg in angles_deg:
            # Estimate flight time: T ~ 2*v0*sin(theta)/g, then add margin
            flight_time = 2.0 * v0 * np.sin(np.radians(theta_deg)) / g
            n = max(int(flight_time / dt) + 500, 1000)

            cfg = SimulationConfig(
                domain=Domain.RIGID_BODY,
                dt=dt,
                n_steps=n,
                parameters={
                    "initial_speed": float(v0),
                    "launch_angle": float(theta_deg),
                    "gravity": g,
                    "drag_coefficient": drag,
                },
            )
            s = ProjectileSimulation(cfg)
            t = s.run(n_steps=n)

            # The simulation interpolates to the exact landing point (y=0, v=0)
            # so the last state with y >= 0 has the correct range.
            airborne = t.states[:, 1] >= 0
            landed_x = t.states[airborne, 0][-1]
            records.append((v0, np.radians(theta_deg), landed_x))

    data = np.array(records)  # columns: v0, theta_rad, range
    v0_arr = data[:, 0]
    theta_arr = data[:, 1]
    range_arr = data[:, 2]

    # -- Fit: R = coeff * v0^2 * sin(2*theta) ----------------------------------
    feature = v0_arr ** 2 * np.sin(2 * theta_arr)
    coeff, _, _, _ = np.linalg.lstsq(feature[:, None], range_arr, rcond=None)
    coeff = coeff[0]
    predicted = coeff * feature
    ss_res = np.sum((range_arr - predicted) ** 2)
    ss_tot = np.sum((range_arr - np.mean(range_arr)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    theoretical_coeff = 1.0 / g
    coeff_error_pct = abs(coeff - theoretical_coeff) / theoretical_coeff * 100

    equation_str = f"R = {coeff:.5f} * v0^2 * sin(2*theta)"
    print(f"  Discovered equation:  {equation_str}")
    print(f"  Coefficient: {coeff:.6f}  (theory 1/g = {theoretical_coeff:.6f})")
    print(f"  Coefficient error: {coeff_error_pct:.4f}%")
    print(f"  R-squared: {r_squared:.8f}")
    print(f"  Data points: {len(records)} (15 speeds x 15 angles)")

    # -- Figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: single trajectory
    ax = axes[0]
    ax.plot(x_traj, y_traj, "b-", linewidth=2)
    ax.set_xlabel("Horizontal distance (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title("Projectile Trajectory (v=30 m/s, 45 deg)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel B: range vs angle for several speeds
    ax = axes[1]
    for v0_plot in [15, 25, 35, 50]:
        mask = np.isclose(v0_arr, v0_plot, atol=0.5)
        if mask.any():
            order = np.argsort(theta_arr[mask])
            ax.plot(
                np.degrees(theta_arr[mask][order]),
                range_arr[mask][order],
                "o-",
                markersize=4,
                label=f"v0={v0_plot:.0f} m/s",
            )
            # Overlay fit
            theta_fit = np.linspace(
                theta_arr[mask].min(), theta_arr[mask].max(), 50
            )
            range_fit = coeff * v0_plot ** 2 * np.sin(2 * theta_fit)
            ax.plot(np.degrees(theta_fit), range_fit, "--", alpha=0.5)

    ax.set_xlabel("Launch angle (degrees)", fontsize=12)
    ax.set_ylabel("Range (m)", fontsize=12)
    ax.set_title(f"Range Equation Fit  (R$^2$ = {r_squared:.6f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "projectile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {OUTPUT_DIR / 'projectile.png'}")

    result = {
        "domain": "Projectile Motion",
        "method": "Linear regression on v0^2*sin(2*theta)",
        "r_squared": round(r_squared, 8),
        "discovery": equation_str,
        "theory": f"R = (1/g) * v0^2 * sin(2*theta)  [1/g = {theoretical_coeff:.6f}]",
        "coeff_error_pct": round(coeff_error_pct, 4),
        "n_data_points": len(records),
    }
    SUMMARY_RESULTS.append(result)
    return result


# ===========================================================================
# Domain 2 -- Lotka-Volterra (nonlinear ODE)
# ===========================================================================

def run_lotka_volterra_demo() -> dict:
    """Simulate predator-prey dynamics, extract equilibrium, compare to theory.

    Theoretical equilibrium: prey* = gamma/delta, predator* = alpha/beta.
    The time-average of a periodic orbit in a conservative Lotka-Volterra
    system equals the fixed point, so averaging over many cycles recovers it.
    """
    print("\n" + "=" * 70)
    print("  DOMAIN 2: Lotka-Volterra  (nonlinear ODE)")
    print("=" * 70)

    alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1
    prey_0, predator_0 = 40.0, 9.0
    dt = 0.01
    n_steps = 20000  # 200 time units, ~13 full oscillation cycles

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
            "predator_0": predator_0,
        },
    )
    sim = LotkaVolterraSimulation(config)
    traj = sim.run(n_steps=n_steps)
    states = traj.states  # shape (n_steps+1, 2): [prey, predator]
    timestamps = traj.timestamps

    prey = states[:, 0]
    predator = states[:, 1]

    # -- Time-averaged equilibrium (skip initial transient) --------------------
    skip = n_steps // 10  # skip first 10% for transient
    prey_eq_measured = np.mean(prey[skip:])
    pred_eq_measured = np.mean(predator[skip:])

    prey_eq_theory = gamma / delta
    pred_eq_theory = alpha / beta

    prey_err = abs(prey_eq_measured - prey_eq_theory) / prey_eq_theory * 100
    pred_err = abs(pred_eq_measured - pred_eq_theory) / pred_eq_theory * 100

    print(f"  Measured equilibrium:  prey = {prey_eq_measured:.4f},  "
          f"predator = {pred_eq_measured:.4f}")
    print(f"  Theoretical:          prey = {prey_eq_theory:.4f},  "
          f"predator = {pred_eq_theory:.4f}")
    print(f"  Error:                prey = {prey_err:.4f}%,  "
          f"predator = {pred_err:.4f}%")

    # -- Detect oscillation period from prey peaks -----------------------------
    peaks, _ = find_peaks(prey[skip:], distance=50)
    if len(peaks) >= 2:
        peak_times = timestamps[skip:][peaks]
        period = np.mean(np.diff(peak_times))
        n_cycles = len(peaks) - 1
        print(f"  Oscillation period: {period:.3f} time units ({n_cycles} cycles)")
    else:
        period = None
        print("  (Could not detect oscillation period)")

    # -- R-squared: how well does the time-average predict equilibrium? --------
    # Use 1 - relative_error^2 as a proxy for quality
    r_sq_prey = 1.0 - (prey_eq_measured - prey_eq_theory) ** 2 / prey_eq_theory ** 2
    r_sq_pred = 1.0 - (pred_eq_measured - pred_eq_theory) ** 2 / pred_eq_theory ** 2
    r_squared = min(r_sq_prey, r_sq_pred)

    # -- Figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: time series (show first 100 time units for clarity)
    t_show = min(10000, n_steps)
    ax = axes[0]
    ax.plot(timestamps[:t_show], prey[:t_show], "b-", linewidth=1.2, label="Prey")
    ax.plot(timestamps[:t_show], predator[:t_show], "r-", linewidth=1.2,
            label="Predator")
    ax.axhline(prey_eq_theory, color="b", linestyle="--", alpha=0.4,
               label=f"Prey eq = {prey_eq_theory:.1f}")
    ax.axhline(pred_eq_theory, color="r", linestyle="--", alpha=0.4,
               label=f"Pred eq = {pred_eq_theory:.2f}")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.set_title("Lotka-Volterra Time Series", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel B: phase portrait
    ax = axes[1]
    ax.plot(prey, predator, "purple", linewidth=0.6, alpha=0.6)
    ax.plot(prey[0], predator[0], "go", markersize=8, zorder=5, label="Start")
    ax.plot(prey_eq_theory, pred_eq_theory, "k*", markersize=14, zorder=5,
            label=f"Equilibrium ({prey_eq_theory:.1f}, {pred_eq_theory:.2f})")
    ax.set_xlabel("Prey", fontsize=12)
    ax.set_ylabel("Predator", fontsize=12)
    ax.set_title("Phase Portrait", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lotka_volterra.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {OUTPUT_DIR / 'lotka_volterra.png'}")

    result = {
        "domain": "Lotka-Volterra",
        "method": "Time-averaged equilibrium extraction",
        "r_squared": round(r_squared, 6),
        "discovery": (
            f"Equilibrium: prey* = gamma/delta = {prey_eq_measured:.3f}, "
            f"pred* = alpha/beta = {pred_eq_measured:.3f}"
        ),
        "theory": (
            f"prey* = gamma/delta = {prey_eq_theory:.1f}, "
            f"pred* = alpha/beta = {pred_eq_theory:.2f}"
        ),
        "prey_error_pct": round(prey_err, 4),
        "pred_error_pct": round(pred_err, 4),
        "oscillation_period": round(period, 4) if period else None,
        "n_steps": n_steps,
    }
    SUMMARY_RESULTS.append(result)
    return result


# ===========================================================================
# Domain 3 -- Harmonic Oscillator (linear ODE)
# ===========================================================================

def run_harmonic_oscillator_demo() -> dict:
    """Simulate a damped oscillator, extract frequency via FFT, compare to theory.

    Theory: omega_0 = sqrt(k/m), damped frequency omega_d = omega_0*sqrt(1-zeta^2).
    Uses parabolic interpolation on FFT peak for sub-bin frequency accuracy.
    """
    print("\n" + "=" * 70)
    print("  DOMAIN 3: Harmonic Oscillator  (linear ODE)")
    print("=" * 70)

    k = 4.0      # spring constant
    m = 1.0      # mass
    c = 0.4      # damping
    x_0 = 1.0    # initial displacement
    v_0 = 0.0    # initial velocity
    dt = 0.005   # 5 ms timestep
    n_steps = 5000  # 25 seconds, ~12 full cycles

    config = SimulationConfig(
        domain=Domain.HARMONIC_OSCILLATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "k": k,
            "m": m,
            "c": c,
            "x_0": x_0,
            "v_0": v_0,
        },
    )
    sim = DampedHarmonicOscillator(config)
    traj = sim.run(n_steps=n_steps)
    states = traj.states  # shape (n_steps+1, 2): [x, v]
    timestamps = traj.timestamps

    x = states[:, 0]
    v = states[:, 1]

    # -- Theoretical values ----------------------------------------------------
    omega_0_theory = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    omega_d_theory = omega_0_theory * np.sqrt(1 - zeta ** 2)
    damping_rate_theory = c / (2 * m)

    print(f"  Parameters: k={k}, m={m}, c={c}")
    print(f"  Theory: omega_0 = sqrt({k}/{m}) = {omega_0_theory:.6f}")
    print(f"  Theory: zeta = {zeta:.6f}")
    print(f"  Theory: omega_d = {omega_d_theory:.6f}")

    # -- FFT frequency extraction with parabolic interpolation ------------------
    x_centered = x - np.mean(x)
    window = np.hanning(len(x_centered))
    x_windowed = x_centered * window

    fft_vals = np.fft.rfft(x_windowed)
    fft_freqs = np.fft.rfftfreq(len(x_windowed), d=dt)
    fft_magnitude = np.abs(fft_vals)

    # Find peak frequency (skip DC at index 0)
    peak_idx = np.argmax(fft_magnitude[1:]) + 1
    peak_freq_hz = _parabolic_peak(fft_magnitude, fft_freqs, peak_idx)
    omega_measured = 2 * np.pi * peak_freq_hz

    freq_error = abs(omega_measured - omega_d_theory) / omega_d_theory * 100

    print(f"  FFT peak frequency: {peak_freq_hz:.6f} Hz")
    print(f"  Measured omega_d: {omega_measured:.6f}")
    print(f"  Frequency error: {freq_error:.4f}%")

    # -- Damping rate from envelope peak decay ----------------------------------
    peaks, _ = find_peaks(x, distance=10)
    if len(peaks) >= 3:
        peak_times = timestamps[peaks]
        peak_amps = np.abs(x[peaks])
        # Fit log(amplitude) = -damping_rate * t + const
        log_amps = np.log(peak_amps + 1e-30)
        fit = np.polyfit(peak_times, log_amps, 1)
        damping_rate_measured = -fit[0]
        damping_err = (
            abs(damping_rate_measured - damping_rate_theory)
            / damping_rate_theory * 100
        )
        print(f"  Measured damping rate: {damping_rate_measured:.6f}  "
              f"(theory: {damping_rate_theory:.6f}, error: {damping_err:.4f}%)")
    else:
        damping_rate_measured = None
        damping_err = None
        print("  (Not enough peaks to extract damping rate)")

    # -- R-squared for frequency recovery --------------------------------------
    r_squared = 1.0 - (omega_measured - omega_d_theory) ** 2 / omega_d_theory ** 2

    # -- Analytical comparison -------------------------------------------------
    t_analytical = timestamps
    x_analytical = np.array([
        sim.analytical_solution(t)[0] for t in t_analytical
    ])
    sim_vs_theory_err = np.max(np.abs(x - x_analytical))
    print(f"  Max |x_sim - x_analytical|: {sim_vs_theory_err:.2e}")

    # -- Figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: time series with analytical overlay (show first 10s for detail)
    t_show = min(2000, len(timestamps))
    ax = axes[0]
    ax.plot(timestamps[:t_show], x[:t_show], "b-", linewidth=1.5,
            label="Simulation", alpha=0.8)
    ax.plot(t_analytical[:t_show], x_analytical[:t_show], "r--", linewidth=1.2,
            label="Analytical", alpha=0.7)
    # Envelope
    envelope = x_0 * np.exp(-damping_rate_theory * t_analytical[:t_show])
    ax.plot(t_analytical[:t_show], envelope, "k:", linewidth=1, alpha=0.5,
            label="Envelope")
    ax.plot(t_analytical[:t_show], -envelope, "k:", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement x", fontsize=12)
    ax.set_title(
        f"Damped Oscillator  "
        f"(omega_d = {omega_measured:.4f}, theory = {omega_d_theory:.4f})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: phase portrait
    ax = axes[1]
    scatter = ax.scatter(x, v, c=timestamps, cmap="viridis", s=2, alpha=0.7)
    ax.plot(x[0], v[0], "go", markersize=8, label="Start")
    ax.plot(0, 0, "r*", markersize=12, label="Equilibrium")
    ax.set_xlabel("Displacement x", fontsize=12)
    ax.set_ylabel("Velocity v", fontsize=12)
    ax.set_title("Phase Portrait (color = time)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Time (s)", fontsize=10)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "harmonic_oscillator.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  Figure saved: {OUTPUT_DIR / 'harmonic_oscillator.png'}")

    result = {
        "domain": "Harmonic Oscillator",
        "method": "FFT frequency extraction + envelope damping fit",
        "r_squared": round(r_squared, 8),
        "discovery": (
            f"omega_d = {omega_measured:.4f} (FFT), "
            f"damping_rate = {damping_rate_measured:.4f} (envelope fit)"
            if damping_rate_measured is not None
            else f"omega_d = {omega_measured:.4f} (FFT)"
        ),
        "theory": (
            f"omega_0 = sqrt(k/m) = {omega_0_theory:.4f}, "
            f"omega_d = {omega_d_theory:.4f}, "
            f"damping_rate = c/(2m) = {damping_rate_theory:.4f}"
        ),
        "frequency_error_pct": round(freq_error, 4),
        "damping_error_pct": round(damping_err, 4) if damping_err is not None else None,
        "max_analytical_error": float(f"{sim_vs_theory_err:.2e}"),
        "n_steps": n_steps,
    }
    SUMMARY_RESULTS.append(result)
    return result


# ===========================================================================
# Summary and output
# ===========================================================================

def print_summary_table() -> None:
    """Print a formatted summary table of all demo results."""
    print("\n" + "=" * 70)
    print("  SUMMARY -- Simulating Anything Pipeline Demo")
    print("=" * 70)
    print(f"  {'Domain':<25} {'R-squared':>12}  {'Key Discovery'}")
    print("  " + "-" * 66)
    for r in SUMMARY_RESULTS:
        disc = r["discovery"]
        if len(disc) > 48:
            disc = disc[:45] + "..."
        print(f"  {r['domain']:<25} {r['r_squared']:>12.6f}  {disc}")
    print("  " + "-" * 66)
    print()


def save_results() -> None:
    """Save demo results to JSON."""
    output_path = OUTPUT_DIR / "demo_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "title": "Simulating Anything -- Pipeline Demo",
                "description": (
                    "Three-domain demonstration: projectile motion (algebraic), "
                    "Lotka-Volterra (nonlinear ODE), and harmonic oscillator "
                    "(linear ODE). All equations recovered from simulation data "
                    "using lightweight numerical methods on CPU."
                ),
                "domains": SUMMARY_RESULTS,
            },
            f,
            indent=2,
        )
    print(f"  Results saved: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the three-domain demo pipeline."""
    print()
    print("  ============================================================")
    print("  Simulating Anything -- Pipeline Demo")
    print("  ============================================================")
    print("  Three domains, zero external dependencies (no Julia/PySR/WSL)")
    print("  Running on CPU with numpy + scipy + matplotlib")
    print()

    t0 = time.time()
    ensure_output_dir()

    run_projectile_demo()
    run_lotka_volterra_demo()
    run_harmonic_oscillator_demo()

    elapsed = time.time() - t0

    print_summary_table()

    print(f"  Total runtime: {elapsed:.1f} seconds")
    print(f"  Figures saved to: {OUTPUT_DIR.resolve()}")

    save_results()

    print()
    print("  The full pipeline uses PySR symbolic regression, SINDy ODE")
    print("  recovery, RSSM world models, and uncertainty-driven exploration")
    print("  to rediscover governing equations across 14 scientific domains.")
    print("  See CLAUDE.md for complete rediscovery results.")
    print()


if __name__ == "__main__":
    main()
