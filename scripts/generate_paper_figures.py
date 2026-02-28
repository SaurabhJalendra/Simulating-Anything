"""Generate comprehensive paper figures for all 7 domains + cross-domain analysis.

Creates publication-quality matplotlib figures. Runs on CPU (no WSL needed).
Reads saved results from output/rediscovery/ and generates simulation data
on the fly where needed.

Usage:
    python scripts/generate_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path("output/paper_figures")
DATA_DIR = Path("output/rediscovery")


def setup_style() -> None:
    """Configure publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ---------------------------------------------------------------------------
# Figure 1: Lorenz strange attractor (3D projection + time series)
# ---------------------------------------------------------------------------
def fig_lorenz_attractor() -> None:
    """3D Lorenz attractor + x(t) time series + Lyapunov vs rho."""
    print("\n=== Lorenz Attractor ===")
    from simulating_anything.simulation.lorenz import LorenzSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    config = SimulationConfig(
        domain=Domain.LORENZ_ATTRACTOR, dt=0.01, n_steps=10000,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    )
    sim = LorenzSimulation(config)
    sim.reset()
    states = []
    for _ in range(10000):
        states.append(sim.step().copy())
    states = np.array(states)

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])

    # 3D attractor
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax1.plot(states[:, 0], states[:, 1], states[:, 2],
             linewidth=0.3, color="#1976D2", alpha=0.8)
    fps = sim.fixed_points
    for fp in fps:
        ax1.scatter(*fp, color="red", s=30, zorder=5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Lorenz Strange Attractor")
    ax1.view_init(elev=25, azim=130)

    # Time series
    ax2 = fig.add_subplot(gs[1])
    t = np.arange(len(states)) * 0.01
    ax2.plot(t[:3000], states[:3000, 0], linewidth=0.5, color="#1976D2")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("x(t)")
    ax2.set_title("Chaotic Time Series")

    # Lyapunov vs rho (from saved data or compute)
    ax3 = fig.add_subplot(gs[2])
    lyap_file = DATA_DIR / "lorenz" / "lyapunov_fine.npz"
    if lyap_file.exists():
        ld = dict(np.load(lyap_file))
        ax3.plot(ld["rho"], ld["lyapunov_exponent"], "o-",
                 markersize=3, color="#D32F2F")
        ax3.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax3.axvline(24.74, color="gray", linewidth=0.5, linestyle=":",
                    label=r"$\rho_c$ = 24.74")
        ax3.set_xlabel(r"$\rho$")
        ax3.set_ylabel("Largest Lyapunov Exponent")
        ax3.set_title("Chaos Transition")
        ax3.legend()
    else:
        # Quick computation
        rho_vals = np.linspace(20, 30, 15)
        lyaps = []
        for rho in rho_vals:
            c = SimulationConfig(
                domain=Domain.LORENZ_ATTRACTOR, dt=0.01, n_steps=5000,
                parameters={"sigma": 10.0, "rho": rho, "beta": 8.0 / 3.0},
            )
            s = LorenzSimulation(c)
            s.reset()
            for _ in range(2000):
                s.step()
            lyaps.append(s.estimate_lyapunov(n_steps=10000, dt=0.01))
        ax3.plot(rho_vals, lyaps, "o-", markersize=3, color="#D32F2F")
        ax3.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax3.set_xlabel(r"$\rho$")
        ax3.set_ylabel("Lyapunov Exponent")
        ax3.set_title("Chaos Transition")

    fig.tight_layout()
    save(fig, "lorenz_attractor")


# ---------------------------------------------------------------------------
# Figure 2: SINDy ODE recovery comparison (Lorenz)
# ---------------------------------------------------------------------------
def fig_lorenz_sindy() -> None:
    """Compare SINDy-recovered ODEs with true Lorenz dynamics."""
    print("\n=== Lorenz SINDy Recovery ===")
    from simulating_anything.simulation.lorenz import LorenzSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    config = SimulationConfig(
        domain=Domain.LORENZ_ATTRACTOR, dt=0.01, n_steps=2000,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    )
    sim = LorenzSimulation(config)
    sim.reset()
    states_true = [sim.observe().copy()]
    for _ in range(2000):
        states_true.append(sim.step().copy())
    states_true = np.array(states_true)

    # SINDy-recovered coefficients (from results)
    sigma_r, rho_r, beta_r = 9.977, 27.804, 2.659
    # Simulate with recovered params
    config_r = SimulationConfig(
        domain=Domain.LORENZ_ATTRACTOR, dt=0.01, n_steps=2000,
        parameters={"sigma": sigma_r, "rho": rho_r, "beta": beta_r},
    )
    sim_r = LorenzSimulation(config_r)
    sim_r.reset()
    states_rec = [sim_r.observe().copy()]
    for _ in range(2000):
        states_rec.append(sim_r.step().copy())
    states_rec = np.array(states_rec)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.arange(len(states_true)) * 0.01
    labels = ["x", "y", "z"]
    true_params = [10.0, 28.0, 8.0 / 3.0]
    rec_params = [sigma_r, rho_r, beta_r]
    names = [r"$\sigma$", r"$\rho$", r"$\beta$"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t[:500], states_true[:500, i], "b-", linewidth=1, label="True", alpha=0.8)
        ax.plot(t[:500], states_rec[:500, i], "r--", linewidth=1, label="SINDy", alpha=0.8)
        ax.set_ylabel(f"{label}(t)")
        # Note recovered vs true param in legend
        coeff_str = f"{names[i]}: true={true_params[i]:.3f}, recovered={rec_params[i]:.3f}"
        ax.legend(loc="upper right", title=coeff_str)
    axes[-1].set_xlabel("Time")
    axes[0].set_title("Lorenz ODE Recovery via SINDy")
    fig.tight_layout()
    save(fig, "lorenz_sindy_recovery")


# ---------------------------------------------------------------------------
# Figure 3: Harmonic oscillator (undamped + damped + phase portrait)
# ---------------------------------------------------------------------------
def fig_harmonic_oscillator() -> None:
    """Harmonic oscillator dynamics and rediscovery."""
    print("\n=== Harmonic Oscillator ===")
    from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
    from simulating_anything.types.simulation import Domain, SimulationConfig

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Undamped oscillation
    config = SimulationConfig(
        domain=Domain.HARMONIC_OSCILLATOR, dt=0.001, n_steps=10000,
        parameters={"k": 4.0, "m": 1.0, "c": 0.0, "x_0": 1.0, "v_0": 0.0},
    )
    sim = DampedHarmonicOscillator(config)
    sim.reset()
    states = [sim.observe().copy()]
    for _ in range(10000):
        states.append(sim.step().copy())
    states = np.array(states)
    t = np.arange(len(states)) * 0.001

    axes[0, 0].plot(t, states[:, 0], "b-", linewidth=1)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x(t)")
    axes[0, 0].set_title(r"Undamped: $\omega_0 = \sqrt{k/m} = 2.0$ rad/s")

    # (b) Damped oscillation
    config_d = SimulationConfig(
        domain=Domain.HARMONIC_OSCILLATOR, dt=0.001, n_steps=10000,
        parameters={"k": 4.0, "m": 1.0, "c": 0.4, "x_0": 1.0, "v_0": 0.0},
    )
    sim_d = DampedHarmonicOscillator(config_d)
    sim_d.reset()
    states_d = [sim_d.observe().copy()]
    for _ in range(10000):
        states_d.append(sim_d.step().copy())
    states_d = np.array(states_d)

    axes[0, 1].plot(t, states_d[:, 0], "b-", linewidth=1)
    # Analytical envelope
    gamma = 0.4 / (2 * 1.0)
    env = np.exp(-gamma * t)
    axes[0, 1].plot(t, env, "r--", linewidth=1, label="Envelope $e^{-ct/2m}$")
    axes[0, 1].plot(t, -env, "r--", linewidth=1)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("x(t)")
    axes[0, 1].set_title(r"Underdamped: $\zeta = 0.1$")
    axes[0, 1].legend()

    # (c) Phase portrait (multiple damping levels)
    colors = ["#1976D2", "#388E3C", "#F57C00", "#D32F2F"]
    damping_vals = [0.0, 0.4, 2.0, 8.0]
    labels = [r"$\zeta=0$", r"$\zeta=0.1$", r"$\zeta=0.5$", r"$\zeta=2$"]
    for c_val, color, label in zip(damping_vals, colors, labels):
        c_conf = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR, dt=0.001, n_steps=10000,
            parameters={"k": 4.0, "m": 1.0, "c": c_val, "x_0": 1.0, "v_0": 0.0},
        )
        s = DampedHarmonicOscillator(c_conf)
        s.reset()
        st = [s.observe().copy()]
        for _ in range(10000):
            st.append(s.step().copy())
        st = np.array(st)
        axes[1, 0].plot(st[:, 0], st[:, 1], color=color, linewidth=1, label=label)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("v")
    axes[1, 0].set_title("Phase Portraits")
    axes[1, 0].legend(fontsize=8)

    # (d) Frequency vs sqrt(k/m) verification
    k_vals = np.linspace(1, 16, 20)
    omega_theory = np.sqrt(k_vals / 1.0)
    omega_meas = []
    for k in k_vals:
        c_conf = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR, dt=0.001, n_steps=15000,
            parameters={"k": k, "m": 1.0, "c": 0.0, "x_0": 1.0, "v_0": 0.0},
        )
        s = DampedHarmonicOscillator(c_conf)
        s.reset()
        # Find zero crossings
        prev_x = s.observe()[0]
        crossings = []
        for step in range(15000):
            state = s.step()
            if prev_x < 0 and state[0] >= 0:
                frac = -prev_x / (state[0] - prev_x)
                crossings.append((step + frac) * 0.001)
            prev_x = state[0]
        if len(crossings) >= 2:
            T = np.median(np.diff(crossings))
            omega_meas.append(2 * np.pi / T)
        else:
            omega_meas.append(np.nan)

    axes[1, 1].plot(omega_theory, omega_meas, "o", markersize=4, color="#1976D2")
    lim = [0, max(omega_theory) * 1.1]
    axes[1, 1].plot(lim, lim, "k--", linewidth=0.5, label="Perfect agreement")
    axes[1, 1].set_xlabel(r"$\omega_{theory} = \sqrt{k/m}$ (rad/s)")
    axes[1, 1].set_ylabel(r"$\omega_{measured}$ (rad/s)")
    axes[1, 1].set_title(r"Frequency Rediscovery: $\omega_0 = \sqrt{k/m}$")
    axes[1, 1].legend()

    fig.suptitle("Damped Harmonic Oscillator", fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "harmonic_oscillator")


# ---------------------------------------------------------------------------
# Figure 4: SIR Epidemic dynamics
# ---------------------------------------------------------------------------
def fig_sir_epidemic() -> None:
    """SIR epidemic R0 relationship and dynamics."""
    print("\n=== SIR Epidemic ===")
    from simulating_anything.simulation.epidemiological import SIRSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Classic SIR dynamics
    config = SimulationConfig(
        domain=Domain.EPIDEMIOLOGICAL, dt=0.1, n_steps=3000,
        parameters={"beta": 0.3, "gamma": 0.1, "S_0": 0.99, "I_0": 0.01, "R_0_init": 0.0},
    )
    sim = SIRSimulation(config)
    sim.reset()
    states = [sim.observe().copy()]
    for _ in range(3000):
        states.append(sim.step().copy())
    states = np.array(states)
    t = np.arange(len(states)) * 0.1

    axes[0].plot(t, states[:, 0], "b-", linewidth=1.5, label="S")
    axes[0].plot(t, states[:, 1], "r-", linewidth=1.5, label="I")
    axes[0].plot(t, states[:, 2], "g-", linewidth=1.5, label="R")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Fraction")
    axes[0].set_title(r"SIR Dynamics ($R_0 = \beta/\gamma = 3.0$)")
    axes[0].legend()

    # (b) Final size vs R0
    rng = np.random.default_rng(42)
    r0_vals = []
    final_sizes = []
    for _ in range(100):
        beta = rng.uniform(0.1, 0.8)
        gamma = rng.uniform(0.02, 0.3)
        r0 = beta / gamma
        c = SimulationConfig(
            domain=Domain.EPIDEMIOLOGICAL, dt=0.1, n_steps=5000,
            parameters={"beta": beta, "gamma": gamma, "S_0": 0.99, "I_0": 0.01, "R_0_init": 0.0},
        )
        s = SIRSimulation(c)
        s.reset()
        for _ in range(5000):
            state = s.step()
            if state[1] < 1e-6 and _ > 100:
                break
        r0_vals.append(r0)
        final_sizes.append(state[2])

    axes[1].scatter(r0_vals, final_sizes, s=10, alpha=0.6, color="#D32F2F")
    axes[1].axvline(1.0, color="k", linestyle="--", linewidth=0.5, label=r"$R_0 = 1$")
    axes[1].set_xlabel(r"$R_0 = \beta/\gamma$")
    axes[1].set_ylabel("Final Epidemic Size")
    axes[1].set_title("Epidemic Threshold")
    axes[1].legend()

    # (c) R0 = beta/gamma verification (scatter)
    betas = np.array([r0_vals[i] for i in range(len(r0_vals))])
    # Show beta vs gamma colored by R0
    b_arr = rng.uniform(0.1, 0.8, 200)
    g_arr = rng.uniform(0.02, 0.3, 200)
    r0_arr = b_arr / g_arr
    sc = axes[2].scatter(g_arr, b_arr, c=r0_arr, s=15, cmap="RdYlGn_r",
                         vmin=0, vmax=10, alpha=0.7)
    # R0=1 line: beta = gamma
    g_line = np.linspace(0.02, 0.3, 100)
    axes[2].plot(g_line, g_line, "k--", linewidth=1, label=r"$R_0 = 1$")
    axes[2].set_xlabel(r"$\gamma$ (recovery rate)")
    axes[2].set_ylabel(r"$\beta$ (transmission rate)")
    axes[2].set_title(r"Parameter Space: $R_0 = \beta/\gamma$")
    axes[2].legend()
    plt.colorbar(sc, ax=axes[2], label=r"$R_0$")

    fig.tight_layout()
    save(fig, "sir_epidemic")


# ---------------------------------------------------------------------------
# Figure 5: Double pendulum chaos
# ---------------------------------------------------------------------------
def fig_double_pendulum() -> None:
    """Double pendulum energy conservation and chaos."""
    print("\n=== Double Pendulum ===")
    from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Chaotic trajectory
    config = SimulationConfig(
        domain=Domain.CHAOTIC_ODE, dt=0.001, n_steps=20000,
        parameters={"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
                     "theta1_0": 2.5, "theta2_0": 2.5, "omega1_0": 0.0, "omega2_0": 0.0},
    )
    sim = DoublePendulumSimulation(config)
    sim.reset()
    states = [sim.observe().copy()]
    for _ in range(20000):
        states.append(sim.step().copy())
    states = np.array(states)

    # Convert to Cartesian
    L1, L2 = 1.0, 1.0
    x1 = L1 * np.sin(states[:, 0])
    y1 = -L1 * np.cos(states[:, 0])
    x2 = x1 + L2 * np.sin(states[:, 1])
    y2 = y1 - L2 * np.cos(states[:, 1])

    axes[0].plot(x2[::5], y2[::5], linewidth=0.2, color="#1976D2", alpha=0.5)
    axes[0].plot(x2[-1], y2[-1], "ro", markersize=5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Chaotic Trajectory (tip of pendulum 2)")
    axes[0].set_aspect("equal")

    # (b) Sensitive dependence on initial conditions
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    eps_vals = [0, 1e-6, 1e-4, 1e-2, 0.1]
    for eps, color in zip(eps_vals, colors):
        c = SimulationConfig(
            domain=Domain.CHAOTIC_ODE, dt=0.001, n_steps=10000,
            parameters={"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
                         "theta1_0": 2.5 + eps, "theta2_0": 2.5, "omega1_0": 0.0, "omega2_0": 0.0},
        )
        s = DoublePendulumSimulation(c)
        s.reset()
        th1 = [s.observe()[0]]
        for _ in range(10000):
            th1.append(s.step()[0])
        t = np.arange(len(th1)) * 0.001
        axes[1].plot(t[:5000], th1[:5000], linewidth=0.5, color=color,
                     label=f"$\\Delta\\theta = {eps}$")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$\theta_1$ (rad)")
    axes[1].set_title("Sensitive Dependence on Initial Conditions")
    axes[1].legend(fontsize=7)

    # (c) Energy conservation
    c_e = SimulationConfig(
        domain=Domain.CHAOTIC_ODE, dt=0.001, n_steps=10000,
        parameters={"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
                     "theta1_0": 0.5, "theta2_0": 0.5, "omega1_0": 0.0, "omega2_0": 0.0},
    )
    s_e = DoublePendulumSimulation(c_e)
    s_e.reset()
    energies = [s_e.total_energy()]
    for _ in range(10000):
        s_e.step()
        energies.append(s_e.total_energy())
    energies = np.array(energies)
    drift = (energies - energies[0]) / abs(energies[0])
    t_e = np.arange(len(energies)) * 0.001

    axes[2].plot(t_e, drift * 1e8, linewidth=0.5, color="#388E3C")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(r"$\Delta E / E_0 \times 10^{-8}$")
    axes[2].set_title("Energy Conservation (RK4)")

    fig.tight_layout()
    save(fig, "double_pendulum")


# ---------------------------------------------------------------------------
# Figure 6: Cross-domain analogy matrix
# ---------------------------------------------------------------------------
def fig_cross_domain_matrix() -> None:
    """Cross-domain similarity matrix heatmap."""
    print("\n=== Cross-Domain Analogy Matrix ===")
    from simulating_anything.analysis.cross_domain import (
        build_domain_signatures,
        detect_structural_analogies,
        detect_dimensional_analogies,
        detect_topological_analogies,
    )

    sigs = build_domain_signatures()
    all_analogies = (
        detect_structural_analogies(sigs)
        + detect_dimensional_analogies(sigs)
        + detect_topological_analogies(sigs)
    )

    names = [s.name for s in sigs]
    n = len(names)
    matrix = np.zeros((n, n))

    for a in all_analogies:
        i = names.index(a.domain_a)
        j = names.index(a.domain_b)
        matrix[i, j] = max(matrix[i, j], a.strength)
        matrix[j, i] = max(matrix[j, i], a.strength)
    np.fill_diagonal(matrix, 1.0)

    display_names = [
        "Projectile", "Lotka-\nVolterra", "Gray-\nScott",
        "SIR", "Double\nPendulum", "Harmonic\nOscillator", "Lorenz"
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, fontsize=8)
    ax.set_yticklabels(display_names, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0 and i != j:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Analogy Strength", shrink=0.8)
    ax.set_title("Cross-Domain Mathematical Analogy Matrix\n(7 domains, 9 isomorphisms)")
    fig.tight_layout()
    save(fig, "cross_domain_matrix")


# ---------------------------------------------------------------------------
# Figure 7: Comprehensive rediscovery summary (all 7 domains)
# ---------------------------------------------------------------------------
def fig_rediscovery_summary() -> None:
    """Bar chart of R² values across all 7 domains + methods."""
    print("\n=== Rediscovery Summary ===")

    # Consolidated results from all rediscovery runs
    discoveries = [
        ("Projectile\nR=v^2sin(2t)/g", "PySR", 0.9999),
        ("LV Equil.\nprey=g/d", "PySR", 0.9999),
        ("LV Equil.\npred=a/b", "PySR", 0.9999),
        ("LV ODE\n(SINDy)", "SINDy", 1.0),
        ("GS Wavelength\nscaling", "PySR", 0.985),
        ("SIR R0\n=b/g", "PySR", 1.0),
        ("Pendulum\nT=2pi*sqrt(L/g)", "PySR", 0.999993),
        ("Oscillator\nomega=sqrt(k/m)", "PySR", 1.0),
        ("Oscillator\ndamping=c/2m", "PySR", 1.0),
        ("Lorenz ODE\n(SINDy)", "SINDy", 0.99999),
    ]

    labels = [d[0] for d in discoveries]
    methods = [d[1] for d in discoveries]
    r2_vals = [d[2] for d in discoveries]

    colors = ["#1976D2" if m == "PySR" else "#D32F2F" for m in methods]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(labels)), r2_vals, color=colors, alpha=0.8, edgecolor="k",
                  linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("R^2 (coefficient of determination)")
    ax.set_title("Seven-Domain Rediscovery Results")
    ax.set_ylim(0.97, 1.001)
    ax.axhline(1.0, color="k", linewidth=0.3, linestyle="--", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#1976D2", label="PySR (symbolic regression)"),
                       Patch(facecolor="#D32F2F", label="SINDy (sparse identification)")]
    ax.legend(handles=legend_elements, loc="lower left")

    # Annotate R2 values
    for i, (bar, r2) in enumerate(zip(bars, r2_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{r2:.4f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save(fig, "rediscovery_summary_7domain")


# ---------------------------------------------------------------------------
# Figure 8: Domain taxonomy
# ---------------------------------------------------------------------------
def fig_domain_taxonomy() -> None:
    """Visual taxonomy of all 7 domains by mathematical class."""
    print("\n=== Domain Taxonomy ===")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Math classes and their domains
    classes = {
        "Algebraic": [("Projectile", "R=v^2sin(2t)/g")],
        "Linear ODE": [("Harmonic Osc.", "x''+cx'+kx=0")],
        "Nonlinear ODE": [("Lotka-Volterra", "x'=ax-bxy"), ("SIR", "S'=-bSI")],
        "Chaotic ODE": [("Double Pendulum", "Lagrangian"), ("Lorenz", "x'=s(y-x)")],
        "PDE": [("Gray-Scott", "u_t=Du*Lap(u)-uv^2")],
    }

    y_positions = {}
    y = 0.9
    colors = {"Algebraic": "#64B5F6", "Linear ODE": "#81C784",
              "Nonlinear ODE": "#FFB74D", "Chaotic ODE": "#E57373", "PDE": "#BA68C8"}

    for cls, domains in classes.items():
        color = colors[cls]
        # Class box
        ax.add_patch(plt.Rectangle((0.02, y - 0.06), 0.15, 0.1,
                                    facecolor=color, alpha=0.3, edgecolor=color,
                                    linewidth=2, transform=ax.transAxes))
        ax.text(0.095, y - 0.01, cls, ha="center", va="center",
                fontsize=10, fontweight="bold", transform=ax.transAxes)

        # Domain boxes
        for j, (name, eq) in enumerate(domains):
            x_pos = 0.3 + j * 0.3
            ax.add_patch(plt.Rectangle((x_pos - 0.1, y - 0.06), 0.2, 0.1,
                                        facecolor=color, alpha=0.15, edgecolor=color,
                                        linewidth=1, transform=ax.transAxes))
            ax.text(x_pos, y - 0.005, name, ha="center", va="center",
                    fontsize=9, fontweight="bold", transform=ax.transAxes)
            ax.text(x_pos, y - 0.04, eq, ha="center", va="center",
                    fontsize=7, style="italic", transform=ax.transAxes)

            # Connection line
            ax.annotate("", xy=(x_pos - 0.1, y - 0.01), xytext=(0.17, y - 0.01),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        transform=ax.transAxes)

        y -= 0.18

    ax.set_title("Seven-Domain Taxonomy: 5 Mathematical Classes", fontsize=14, pad=20)
    fig.tight_layout()
    save(fig, "domain_taxonomy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    print("Generating comprehensive paper figures for all 7 domains...")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    fig_lorenz_attractor()
    fig_lorenz_sindy()
    fig_harmonic_oscillator()
    fig_sir_epidemic()
    fig_double_pendulum()
    fig_cross_domain_matrix()
    fig_rediscovery_summary()
    fig_domain_taxonomy()

    n_png = len(list(OUTPUT_DIR.glob("*.png")))
    n_pdf = len(list(OUTPUT_DIR.glob("*.pdf")))
    print(f"\nDone! Generated {n_png} PNG + {n_pdf} PDF in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
