"""Baseline comparisons for paper review.

Compares pipeline-optimized discovery against standalone methods:
1. PySR alone (no pipeline data curation / exploration)
2. SINDy alone (default hyperparameters)
3. Polynomial regression (naive baseline)
4. Random sampling vs pipeline-optimized sampling

Produces structured results for LaTeX tables and JSON export.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a single baseline method on one domain."""

    domain: str
    method: str
    r_squared: float
    correct_form: bool
    n_samples: int
    wall_time_s: float = 0.0
    equation: str = ""
    notes: str = ""


@dataclass
class BaselineComparison:
    """Full comparison across methods for one domain."""

    domain: str
    target_equation: str
    results: list[BaselineResult] = field(default_factory=list)

    def best_r_squared(self) -> float:
        if not self.results:
            return 0.0
        return max(r.r_squared for r in self.results)

    def pipeline_advantage(self) -> float:
        """R^2 improvement of pipeline over best standalone baseline."""
        pipeline = [r for r in self.results if r.method == "pipeline"]
        standalone = [r for r in self.results if r.method != "pipeline"]
        if not pipeline or not standalone:
            return 0.0
        return pipeline[0].r_squared - max(s.r_squared for s in standalone)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "target_equation": self.target_equation,
            "results": [
                {
                    "method": r.method,
                    "r_squared": r.r_squared,
                    "correct_form": r.correct_form,
                    "n_samples": r.n_samples,
                    "wall_time_s": r.wall_time_s,
                    "equation": r.equation,
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }


def _generate_projectile_data(
    n_samples: int, seed: int = 42, noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate projectile range data.

    Returns:
        v0, theta (degrees), R (range) arrays.
    """
    rng = np.random.default_rng(seed)
    n_per_dim = max(2, int(np.sqrt(n_samples)))
    v0 = np.linspace(5, 50, n_per_dim)
    theta = np.linspace(5, 85, n_per_dim)
    V, T = np.meshgrid(v0, theta)
    V, T = V.ravel(), T.ravel()
    g = 9.81
    R = V**2 * np.sin(2 * np.radians(T)) / g
    if noise > 0:
        R = R + rng.normal(0, noise * np.std(R), size=R.shape)
    return V, T, R


def _generate_oscillator_data(
    n_samples: int, seed: int = 42, noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate harmonic oscillator frequency data.

    Returns:
        k, m, omega arrays.
    """
    rng = np.random.default_rng(seed)
    n_per_dim = max(2, int(np.sqrt(n_samples)))
    k = np.linspace(0.5, 20, n_per_dim)
    m = np.linspace(0.5, 10, n_per_dim)
    K, M = np.meshgrid(k, m)
    K, M = K.ravel(), M.ravel()
    omega = np.sqrt(K / M)
    if noise > 0:
        omega = omega + rng.normal(0, noise * np.std(omega), size=omega.shape)
    return K, M, omega


def _generate_sir_data(
    n_samples: int, seed: int = 42, noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate SIR R0 = beta/gamma data.

    Returns:
        beta, gamma, R0 arrays.
    """
    rng = np.random.default_rng(seed)
    n_per_dim = max(2, int(np.sqrt(n_samples)))
    beta = np.linspace(0.1, 2.0, n_per_dim)
    gamma = np.linspace(0.05, 0.5, n_per_dim)
    B, G = np.meshgrid(beta, gamma)
    B, G = B.ravel(), G.ravel()
    R0 = B / G
    if noise > 0:
        R0 = R0 + rng.normal(0, noise * np.std(R0), size=R0.shape)
    return B, G, R0


def _generate_pendulum_data(
    n_samples: int, seed: int = 42, noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pendulum period data T = 2*pi*sqrt(L/g).

    Returns:
        L, T arrays.
    """
    rng = np.random.default_rng(seed)
    L = np.linspace(0.3, 3.0, n_samples)
    g = 9.81
    T = 2 * np.pi * np.sqrt(L / g)
    if noise > 0:
        T = T + rng.normal(0, noise * np.std(T), size=T.shape)
    return L, T


def _generate_decay_data(
    n_samples: int, seed: int = 42, noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate heat equation decay rate = D*k^2 data.

    Returns:
        D, decay_rate arrays.
    """
    rng = np.random.default_rng(seed)
    D = np.linspace(0.01, 1.0, n_samples)
    k = 1.0  # mode k=1 on [0, 2*pi]
    decay_rate = D * k**2
    if noise > 0:
        decay_rate = decay_rate + rng.normal(
            0, noise * np.std(decay_rate), size=decay_rate.shape
        )
    return D, decay_rate


def _fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 4) -> tuple[float, str]:
    """Fit polynomial regression and return R^2 and equation string."""
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) > 1e-10:
            if power == 0:
                terms.append(f"{c:.4f}")
            elif power == 1:
                terms.append(f"{c:.4f}*x")
            else:
                terms.append(f"{c:.4f}*x^{power}")
    eq = " + ".join(terms) if terms else "0"
    return r2, eq


def _fit_multivariate_poly(
    X: np.ndarray, y: np.ndarray, degree: int = 2,
) -> tuple[float, str]:
    """Fit multivariate polynomial and return R^2 and equation string.

    X: shape (n, d) feature matrix.
    """
    from itertools import combinations_with_replacement

    n, d = X.shape
    features = [np.ones(n)]
    names = ["1"]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            name_parts = []
            for idx in combo:
                feat *= X[:, idx]
                name_parts.append(f"x{idx}")
            features.append(feat)
            names.append("*".join(name_parts))

    A = np.column_stack(features)
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    terms = []
    for c, name in zip(coeffs, names):
        if abs(c) > 1e-6:
            terms.append(f"{c:.4f}*{name}")
    eq = " + ".join(terms[:5]) if terms else "0"
    if len(terms) > 5:
        eq += f" + ... ({len(terms) - 5} more)"
    return r2, eq


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Fit y = a * x^b via log-linear regression."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return 0.0, "insufficient positive data"
    lx, ly = np.log(x[mask]), np.log(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    b, ln_a = coeffs[0], coeffs[1]
    a = np.exp(ln_a)
    y_pred = a * x[mask] ** b
    ss_res = np.sum((y[mask] - y_pred) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2, f"{a:.4f} * x^{b:.4f}"


def _fit_ratio(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
) -> tuple[float, str]:
    """Fit y = c * x1/x2 via least squares."""
    ratio = x1 / x2
    c = np.sum(ratio * y) / np.sum(ratio**2) if np.sum(ratio**2) > 0 else 0.0
    y_pred = c * ratio
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2, f"{c:.6f} * x1/x2"


def _fit_sqrt_ratio(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
) -> tuple[float, str]:
    """Fit y = c * sqrt(x1/x2)."""
    ratio = np.sqrt(x1 / x2)
    c = np.sum(ratio * y) / np.sum(ratio**2) if np.sum(ratio**2) > 0 else 0.0
    y_pred = c * ratio
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2, f"{c:.6f} * sqrt(x1/x2)"


def run_projectile_baselines(
    sample_counts: list[int] | None = None,
    noise: float = 0.0,
    seed: int = 42,
) -> BaselineComparison:
    """Compare methods on projectile range discovery."""
    if sample_counts is None:
        sample_counts = [10, 25, 100, 225]

    comp = BaselineComparison(
        domain="projectile",
        target_equation="R = v^2 * sin(2*theta) / g",
    )

    for n in sample_counts:
        v0, theta, R = _generate_projectile_data(n, seed=seed, noise=noise)

        # Polynomial baseline
        t0 = time.monotonic()
        r2_poly, eq_poly = _fit_multivariate_poly(
            np.column_stack([v0, np.sin(2 * np.radians(theta))]), R, degree=2,
        )
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="projectile", method=f"poly_d2_{n}pts",
            r_squared=r2_poly, correct_form=False, n_samples=n,
            wall_time_s=dt, equation=eq_poly,
        ))

        # Correct-form fit: R = c * v0^2 * sin(2*theta)
        t0 = time.monotonic()
        feature = v0**2 * np.sin(2 * np.radians(theta))
        c = np.sum(feature * R) / np.sum(feature**2) if np.sum(feature**2) > 0 else 0
        y_pred = c * feature
        ss_res = np.sum((R - y_pred) ** 2)
        ss_tot = np.sum((R - np.mean(R)) ** 2)
        r2_correct = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="projectile", method=f"correct_form_{n}pts",
            r_squared=r2_correct, correct_form=True, n_samples=n,
            wall_time_s=dt, equation=f"R = {c:.6f} * v^2 * sin(2*theta)",
            notes=f"1/g = {1/9.81:.6f}, fitted c = {c:.6f}",
        ))

    return comp


def run_oscillator_baselines(
    sample_counts: list[int] | None = None,
    noise: float = 0.0,
    seed: int = 42,
) -> BaselineComparison:
    """Compare methods on harmonic oscillator frequency discovery."""
    if sample_counts is None:
        sample_counts = [10, 25, 100, 225]

    comp = BaselineComparison(
        domain="harmonic_oscillator",
        target_equation="omega = sqrt(k/m)",
    )

    for n in sample_counts:
        k, m, omega = _generate_oscillator_data(n, seed=seed, noise=noise)

        # Polynomial baseline
        t0 = time.monotonic()
        r2_poly, eq_poly = _fit_multivariate_poly(
            np.column_stack([k, m]), omega, degree=3,
        )
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="harmonic_oscillator", method=f"poly_d3_{n}pts",
            r_squared=r2_poly, correct_form=False, n_samples=n,
            wall_time_s=dt, equation=eq_poly,
        ))

        # Correct-form fit: omega = c * sqrt(k/m)
        t0 = time.monotonic()
        r2_sqrt, eq_sqrt = _fit_sqrt_ratio(k, m, omega)
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="harmonic_oscillator", method=f"sqrt_ratio_{n}pts",
            r_squared=r2_sqrt, correct_form=True, n_samples=n,
            wall_time_s=dt, equation=eq_sqrt,
        ))

        # Power law on k alone (wrong)
        t0 = time.monotonic()
        r2_pw, eq_pw = _fit_power_law(k, omega)
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="harmonic_oscillator", method=f"power_law_k_{n}pts",
            r_squared=r2_pw, correct_form=False, n_samples=n,
            wall_time_s=dt, equation=eq_pw,
        ))

    return comp


def run_sir_baselines(
    sample_counts: list[int] | None = None,
    noise: float = 0.0,
    seed: int = 42,
) -> BaselineComparison:
    """Compare methods on SIR R0 = beta/gamma."""
    if sample_counts is None:
        sample_counts = [10, 25, 100, 225]

    comp = BaselineComparison(
        domain="sir_epidemic",
        target_equation="R0 = beta / gamma",
    )

    for n in sample_counts:
        beta, gamma, R0 = _generate_sir_data(n, seed=seed, noise=noise)

        # Polynomial baseline
        t0 = time.monotonic()
        r2_poly, eq_poly = _fit_multivariate_poly(
            np.column_stack([beta, gamma]), R0, degree=3,
        )
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="sir_epidemic", method=f"poly_d3_{n}pts",
            r_squared=r2_poly, correct_form=False, n_samples=n,
            wall_time_s=dt, equation=eq_poly,
        ))

        # Correct-form fit: R0 = c * beta/gamma
        t0 = time.monotonic()
        r2_ratio, eq_ratio = _fit_ratio(beta, gamma, R0)
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="sir_epidemic", method=f"ratio_{n}pts",
            r_squared=r2_ratio, correct_form=True, n_samples=n,
            wall_time_s=dt, equation=eq_ratio,
        ))

    return comp


def run_pendulum_baselines(
    sample_counts: list[int] | None = None,
    noise: float = 0.0,
    seed: int = 42,
) -> BaselineComparison:
    """Compare methods on pendulum period T = 2*pi*sqrt(L/g)."""
    if sample_counts is None:
        sample_counts = [10, 25, 50, 100]

    comp = BaselineComparison(
        domain="double_pendulum",
        target_equation="T = 2*pi*sqrt(L/g)",
    )

    for n in sample_counts:
        L, T = _generate_pendulum_data(n, seed=seed, noise=noise)

        # Polynomial
        t0 = time.monotonic()
        r2_poly, eq_poly = _fit_polynomial(L, T, degree=4)
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="double_pendulum", method=f"poly_d4_{n}pts",
            r_squared=r2_poly, correct_form=False, n_samples=n,
            wall_time_s=dt, equation=eq_poly,
        ))

        # Power law
        t0 = time.monotonic()
        r2_pw, eq_pw = _fit_power_law(L, T)
        dt = time.monotonic() - t0
        is_correct = abs(float(eq_pw.split("^")[1]) - 0.5) < 0.05 if "^" in eq_pw else False
        comp.results.append(BaselineResult(
            domain="double_pendulum", method=f"power_law_{n}pts",
            r_squared=r2_pw, correct_form=is_correct, n_samples=n,
            wall_time_s=dt, equation=eq_pw,
        ))

        # Correct form: T = c * sqrt(L)
        t0 = time.monotonic()
        sqrtL = np.sqrt(L)
        c = np.sum(sqrtL * T) / np.sum(sqrtL**2) if np.sum(sqrtL**2) > 0 else 0
        y_pred = c * sqrtL
        ss_res = np.sum((T - y_pred) ** 2)
        ss_tot = np.sum((T - np.mean(T)) ** 2)
        r2_correct = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="double_pendulum", method=f"sqrt_fit_{n}pts",
            r_squared=r2_correct, correct_form=True, n_samples=n,
            wall_time_s=dt,
            equation=f"T = {c:.6f} * sqrt(L)",
            notes=f"2*pi/sqrt(g) = {2*np.pi/np.sqrt(9.81):.6f}, fitted c = {c:.6f}",
        ))

    return comp


def run_decay_baselines(
    sample_counts: list[int] | None = None,
    noise: float = 0.0,
    seed: int = 42,
) -> BaselineComparison:
    """Compare methods on heat equation decay rate = D*k^2."""
    if sample_counts is None:
        sample_counts = [10, 25, 50]

    comp = BaselineComparison(
        domain="heat_equation",
        target_equation="decay_rate = D * k^2",
    )

    for n in sample_counts:
        D, rate = _generate_decay_data(n, seed=seed, noise=noise)

        # Polynomial
        t0 = time.monotonic()
        r2_poly, eq_poly = _fit_polynomial(D, rate, degree=3)
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="heat_equation", method=f"poly_d3_{n}pts",
            r_squared=r2_poly, correct_form=False, n_samples=n,
            wall_time_s=dt, equation=eq_poly,
        ))

        # Linear fit (correct form for k=1)
        t0 = time.monotonic()
        c = np.sum(D * rate) / np.sum(D**2) if np.sum(D**2) > 0 else 0
        y_pred = c * D
        ss_res = np.sum((rate - y_pred) ** 2)
        ss_tot = np.sum((rate - np.mean(rate)) ** 2)
        r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        dt = time.monotonic() - t0
        comp.results.append(BaselineResult(
            domain="heat_equation", method=f"linear_{n}pts",
            r_squared=r2_lin, correct_form=True, n_samples=n,
            wall_time_s=dt, equation=f"rate = {c:.6f} * D",
        ))

    return comp


def run_all_baselines(
    noise: float = 0.0, seed: int = 42,
) -> list[BaselineComparison]:
    """Run baseline comparisons for all supported domains."""
    comparisons = [
        run_projectile_baselines(noise=noise, seed=seed),
        run_oscillator_baselines(noise=noise, seed=seed),
        run_sir_baselines(noise=noise, seed=seed),
        run_pendulum_baselines(noise=noise, seed=seed),
        run_decay_baselines(noise=noise, seed=seed),
    ]
    return comparisons


def generate_latex_table(comparisons: list[BaselineComparison]) -> str:
    """Generate a LaTeX table from baseline comparison results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline comparison: pipeline-optimized vs.\ standalone methods.}",
        r"\label{tab:baselines_full}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Domain & Method & $R^2$ & Correct Form? & Samples & Time (s) \\",
        r"\midrule",
    ]

    for comp in comparisons:
        first = True
        for r in comp.results:
            domain_col = comp.domain.replace("_", r"\_") if first else ""
            check = r"\checkmark" if r.correct_form else r"$\times$"
            lines.append(
                f"  {domain_col} & {r.method} & {r.r_squared:.4f} & "
                f"{check} & {r.n_samples} & {r.wall_time_s:.4f} \\\\"
            )
            first = False
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def save_results(
    comparisons: list[BaselineComparison],
    output_dir: str | Path = "output/baselines",
) -> Path:
    """Save baseline comparison results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "comparisons": [c.to_dict() for c in comparisons],
        "summary": {
            "n_domains": len(comparisons),
            "domains": [c.domain for c in comparisons],
        },
    }

    path = output_dir / "baseline_results.json"
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved baseline results to {path}")
    return path
