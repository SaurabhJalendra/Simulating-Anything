"""Meta-analysis across all 192 domain rediscovery results.

Provides four cross-cutting analyses that treat the full corpus of
rediscovery results as a dataset in its own right:

1. Universal Bifurcation Analysis -- classifies dynamics type
   (fixed point, limit cycle, chaos, hyperchaos) using Lyapunov
   exponent data extracted from every chaotic domain.

2. Hopf Scaling Law -- checks whether amplitude scales as
   sqrt(mu - mu_c) near the bifurcation for all domains that
   exhibit a Hopf transition.

3. R-squared Distribution Analysis -- loads every results.json,
   extracts the best R-squared per domain, and computes
   comprehensive statistics by mathematical class, method, and
   domain type.

4. Equation Complexity Analysis -- for domains with PySR
   equations, measures complexity (term count, operator types,
   nesting depth) and correlates with R-squared.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REDISCOVERY_DIR = PROJECT_ROOT / "output" / "rediscovery"

# Mathematical class heuristic: domain name pattern -> class label.
# Order matters -- first match wins.
_MATH_CLASS_PATTERNS: list[tuple[str, str]] = [
    # Discrete maps
    ("logistic_map", "Discrete Map"),
    ("henon_map", "Discrete Map"),
    ("standard_map", "Discrete Map"),
    ("ikeda_map", "Discrete Map"),
    ("tinkerbell_map", "Discrete Map"),
    ("lozi_map", "Discrete Map"),
    ("tent_map", "Discrete Map"),
    ("cubic_map", "Discrete Map"),
    ("ricker_map", "Discrete Map"),
    ("rulkov_map", "Discrete Map"),
    ("circle_map", "Discrete Map"),
    ("bouncing_ball", "Discrete Map"),
    ("coupled_map_lattice", "Discrete Map"),
    # Statistical mechanics / agent-based
    ("ising_model", "Statistical Mechanics"),
    ("bak_sneppen", "Statistical Mechanics"),
    ("vicsek", "Statistical Mechanics"),
    ("boltzmann_gas", "Statistical Mechanics"),
    ("lennard_jones", "Statistical Mechanics"),
    # PDEs / Reaction-diffusion
    ("gray_scott", "Reaction-Diffusion PDE"),
    ("brusselator_diffusion", "Reaction-Diffusion PDE"),
    ("brusselator_2d", "Reaction-Diffusion PDE"),
    ("brusselator_1d", "Reaction-Diffusion PDE"),
    ("schnakenberg", "Reaction-Diffusion PDE"),
    ("bz_spiral", "Reaction-Diffusion PDE"),
    ("fhn_spatial", "Reaction-Diffusion PDE"),
    ("fhn_lattice", "Reaction-Diffusion PDE"),
    ("oregonator_1d", "Reaction-Diffusion PDE"),
    ("gray_scott_1d", "Reaction-Diffusion PDE"),
    ("diffusive_lv", "Reaction-Diffusion PDE"),
    ("amari_neural_field", "Reaction-Diffusion PDE"),
    ("navier_stokes", "Fluid PDE"),
    ("turbulent_flow", "Fluid PDE"),
    ("shallow_water", "Hyperbolic PDE"),
    ("kuramoto_sivashinsky", "Chaotic PDE"),
    ("ginzburg_landau", "Complex PDE"),
    ("cahn_hilliard", "Phase Field PDE"),
    ("sine_gordon", "Soliton PDE"),
    ("heat_equation", "Linear PDE"),
    ("damped_wave", "Wave PDE"),
    ("cable_equation", "Linear PDE"),
    ("advection_1d", "Linear PDE"),
    ("burgers_1d", "Nonlinear PDE"),
    # Epidemiological
    ("sir", "Epidemiological ODE"),
    ("seir", "Epidemiological ODE"),
    ("network_sis", "Epidemiological ODE"),
    ("zombie", "Epidemiological ODE"),
    ("eco_epidemic", "Epidemiological ODE"),
    # Neuroscience
    ("hodgkin_huxley", "Neuroscience ODE"),
    ("hindmarsh_rose", "Neuroscience ODE"),
    ("morris_lecar", "Neuroscience ODE"),
    ("fitzhugh", "Neuroscience ODE"),
    ("izhikevich", "Neuroscience ODE"),
    ("wilson_cowan", "Neuroscience ODE"),
    ("rulkov", "Neuroscience ODE"),
    ("fhn_ring", "Neuroscience ODE"),
    # Ecology
    ("lotka_volterra", "Ecology ODE"),
    ("rosenzweig", "Ecology ODE"),
    ("competitive_lv", "Ecology ODE"),
    ("allee", "Ecology ODE"),
    ("bazykin", "Ecology ODE"),
    ("may_leonard", "Ecology ODE"),
    ("three_species", "Ecology ODE"),
    ("predator", "Ecology ODE"),
    ("four_species", "Ecology ODE"),
    ("harvested", "Ecology ODE"),
    ("chemostat", "Ecology ODE"),
    # Soliton / lattice
    ("toda_lattice", "Integrable Chain"),
    ("fput", "Nonlinear Chain"),
    ("spring_mass_chain", "Coupled ODE"),
    ("elastic_collision", "Coupled ODE"),
    # Celestial / GR
    ("kepler", "Celestial Mechanics"),
    ("schwarzschild", "GR Geodesic"),
    ("three_body", "Celestial Mechanics"),
    # Quantum
    ("quantum", "Quantum PDE"),
    # Algebraic
    ("projectile", "Algebraic"),
    # Chaotic ODE (catch-all for named attractors)
    ("lorenz", "Chaotic ODE"),
    ("rossler", "Chaotic ODE"),
    ("chua", "Chaotic ODE"),
    ("chen", "Chaotic ODE"),
    ("aizawa", "Chaotic ODE"),
    ("halvorsen", "Chaotic ODE"),
    ("burke_shaw", "Chaotic ODE"),
    ("sprott", "Chaotic ODE"),
    ("thomas", "Chaotic ODE"),
    ("duffing", "Nonlinear ODE"),
    ("driven_pendulum", "Nonlinear ODE"),
    ("double_pendulum", "Chaotic ODE"),
    ("colpitts", "Chaotic ODE"),
    ("rikitake", "Chaotic ODE"),
    ("rabinovich", "Chaotic ODE"),
    ("langford", "Chaotic ODE"),
    ("shimizu", "Chaotic ODE"),
    ("newton_leipnik", "Chaotic ODE"),
    ("wang", "Chaotic ODE"),
    ("arneodo", "Chaotic ODE"),
    ("rucklidge", "Chaotic ODE"),
    ("liu", "Chaotic ODE"),
    ("tigan", "Chaotic ODE"),
    ("sakarya", "Chaotic ODE"),
    ("dadras", "Chaotic ODE"),
    ("genesio", "Chaotic ODE"),
    ("lu_chen", "Chaotic ODE"),
    ("qi", "Chaotic ODE"),
    ("windmi", "Chaotic ODE"),
    ("finance", "Chaotic ODE"),
    ("hadley", "Chaotic ODE"),
    ("vallis", "Chaotic ODE"),
    ("nose_hoover", "Chaotic ODE"),
    ("lorenz_haken", "Chaotic ODE"),
    ("lorenz_stenflo", "Chaotic ODE"),
    ("magnetic_pendulum", "Chaotic ODE"),
    ("swinging_atwood", "Chaotic ODE"),
    ("mackey_glass", "Delay ODE"),
    ("delayed", "Delay ODE"),
    ("ueda", "Nonlinear ODE"),
    # Oscillators
    ("harmonic_oscillator", "Linear ODE"),
    ("van_der_pol", "Nonlinear ODE"),
    ("coupled_vdp", "Nonlinear ODE"),
    ("duffing_van_der_pol", "Nonlinear ODE"),
    ("brusselator", "Nonlinear ODE"),
    ("selkov", "Nonlinear ODE"),
    ("oregonator", "Nonlinear ODE"),
    ("stuart_landau", "Nonlinear ODE"),
    ("kuramoto", "Collective ODE"),
    ("coupled_oscillators", "Coupled ODE"),
    ("wilberforce", "Coupled ODE"),
    ("coupled_lorenz", "Chaotic ODE"),
    ("laser_rate", "Nonlinear ODE"),
    ("kapitza", "Nonlinear ODE"),
    ("rayleigh_benard", "Nonlinear ODE"),
    ("cart_pole", "Nonlinear ODE"),
    ("elastic_pendulum", "Nonlinear ODE"),
    ("circadian", "Nonlinear ODE"),
    ("autocatalator", "Nonlinear ODE"),
    ("double_well", "Nonlinear ODE"),
    ("hp_protein", "Monte Carlo"),
    ("bernoulli", "Linear ODE"),
    ("age_structured", "Structured Population"),
]

# Operators recognized in PySR expressions.
_PYSR_OPERATORS = {
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "square", "abs", "sign",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class DomainResult:
    """Parsed rediscovery result for one domain."""

    domain: str
    raw: dict
    best_r2: float | None = None
    method: str = "unknown"
    math_class: str = "Other"
    lyapunov: float | None = None
    is_chaotic: bool | None = None
    is_hyperchaotic: bool = False
    hopf_data: dict | None = None
    pysr_equations: list[dict] = field(default_factory=list)
    sindy_equations: list[dict] = field(default_factory=list)


def _classify_math_class(domain_name: str) -> str:
    """Assign a mathematical class label from the domain name."""
    for pattern, label in _MATH_CLASS_PATTERNS:
        if pattern in domain_name:
            return label
    return "Other"


def _extract_best_r2(data: dict) -> tuple[float | None, str]:
    """Extract the best R-squared and method from a results dict.

    Returns:
        (best_r2, method) where method is 'PySR', 'SINDy', or 'unknown'.
    """
    candidates: list[tuple[float, str]] = []

    # SINDy ODE R-squared
    sindy = data.get("sindy_ode", {})
    if "best_r2" in sindy:
        candidates.append((sindy["best_r2"], "SINDy"))
    elif "discoveries" in sindy:
        for d in sindy["discoveries"]:
            if "r_squared" in d:
                candidates.append((d["r_squared"], "SINDy"))
                break

    # PySR sections -- look for any key containing 'pysr'
    for key, val in data.items():
        if "pysr" not in key.lower():
            continue
        if isinstance(val, dict):
            if "best_r2" in val:
                candidates.append((val["best_r2"], "PySR"))
            elif "discoveries" in val:
                for d in val["discoveries"]:
                    if "r_squared" in d:
                        candidates.append((d["r_squared"], "PySR"))
                        break

    # Top-level discoveries (projectile style)
    if "discoveries" in data and isinstance(data["discoveries"], list):
        for d in data["discoveries"]:
            if "r_squared" in d:
                candidates.append((d["r_squared"], "PySR"))
                break
    if "best_r_squared" in data:
        candidates.append((data["best_r_squared"], "PySR"))

    # Correlation field (heat_equation style)
    for key in ("onsager_correlation", "correlation"):
        if key in data:
            candidates.append((data[key], "Correlation"))
    for key, val in data.items():
        if isinstance(val, dict) and "correlation" in val:
            candidates.append((val["correlation"], "Correlation"))

    if not candidates:
        return None, "unknown"

    # Pick the highest R-squared
    best = max(candidates, key=lambda x: x[0])
    return best


def _extract_lyapunov(data: dict) -> tuple[float | None, bool | None, bool]:
    """Extract Lyapunov exponent and classification from results dict.

    Returns:
        (lyapunov, is_chaotic, is_hyperchaotic)
    """
    lyap = None
    is_chaotic = None
    is_hyper = False

    # classic_parameters or default_parameters
    for key in ("classic_parameters", "default_parameters"):
        params = data.get(key, {})
        if "lyapunov_exponent" in params:
            lyap = params["lyapunov_exponent"]
            if "positive" in params:
                is_chaotic = bool(params["positive"])
            elif lyap is not None:
                is_chaotic = lyap > 0
            break

    # Sprott-style per-system Lyapunov
    if lyap is None:
        sprott_lyap = data.get("sprott_b_lyapunov", {})
        if "lyapunov_exponent" in sprott_lyap:
            lyap = sprott_lyap["lyapunov_exponent"]
            is_chaotic = sprott_lyap.get("is_chaotic")

    # classic_lyapunov (henon_map style)
    if lyap is None:
        cl = data.get("classic_lyapunov", {})
        if "lyapunov" in cl:
            lyap = cl["lyapunov"]
            is_chaotic = lyap > 0 if lyap is not None else None

    # Lyapunov analysis sub-dict
    if lyap is None:
        la = data.get("lyapunov_analysis", {})
        if "max_lyapunov" in la:
            lyap = la["max_lyapunov"]
            is_chaotic = lyap > 0 if lyap is not None else None

    # lyapunov sub-dict
    if lyap is None:
        ld = data.get("lyapunov", {})
        if isinstance(ld, dict) and "max_lyapunov" in ld:
            lyap = ld["max_lyapunov"]
            is_chaotic = lyap > 0 if lyap is not None else None

    # bifurcation_sweep max_lyapunov
    if lyap is None:
        bs = data.get("bifurcation_sweep", {})
        if "max_lyapunov" in bs:
            lyap = bs["max_lyapunov"]
            is_chaotic = lyap > 0 if lyap is not None else None

    # Hyperchaos detection
    spectrum = data.get("lyapunov_spectrum", {})
    if "n_positive" in spectrum and spectrum["n_positive"] >= 2:
        is_hyper = True
        is_chaotic = True
        if lyap is None and "exponents" in spectrum:
            lyap = max(spectrum["exponents"])

    return lyap, is_chaotic, is_hyper


def _extract_hopf_data(data: dict) -> dict | None:
    """Extract Hopf bifurcation data if present."""
    hopf = data.get("hopf_bifurcation", {})
    if hopf:
        return dict(hopf)
    # Some domains store it differently
    for key in ("hopf_pysr", "pysr_hopf"):
        if key in data and isinstance(data[key], dict):
            merged = dict(hopf) if hopf else {}
            merged["pysr"] = data[key]
            return merged
    return None if not hopf else dict(hopf)


def _extract_pysr_equations(data: dict) -> list[dict]:
    """Extract all PySR-discovered equations from the results."""
    eqs: list[dict] = []
    # Top-level discoveries
    if "discoveries" in data and isinstance(data["discoveries"], list):
        for d in data["discoveries"]:
            if "expression" in d and "r_squared" in d:
                eqs.append({
                    "expression": d["expression"],
                    "r_squared": d["r_squared"],
                    "source": "top_level",
                })
    # Keyed pysr sections
    for key, val in data.items():
        if "pysr" not in key.lower():
            continue
        if isinstance(val, dict) and "discoveries" in val:
            for d in val["discoveries"]:
                if "expression" in d and "r_squared" in d:
                    eqs.append({
                        "expression": d["expression"],
                        "r_squared": d["r_squared"],
                        "source": key,
                    })
    return eqs


def _extract_sindy_equations(data: dict) -> list[dict]:
    """Extract SINDy-discovered equations from the results."""
    sindy = data.get("sindy_ode", {})
    eqs: list[dict] = []
    for d in sindy.get("discoveries", []):
        if "expression" in d:
            eqs.append({
                "expression": d["expression"],
                "r_squared": d.get("r_squared"),
            })
    return eqs


def load_all_results(
    rediscovery_dir: Path | None = None,
) -> list[DomainResult]:
    """Load and parse all results.json files.

    Args:
        rediscovery_dir: Override for the rediscovery output directory.

    Returns:
        List of parsed DomainResult objects sorted by domain name.
    """
    base = rediscovery_dir or REDISCOVERY_DIR
    results: list[DomainResult] = []
    if not base.exists():
        logger.warning("Rediscovery dir not found: %s", base)
        return results

    for result_path in sorted(base.glob("*/results.json")):
        domain_name = result_path.parent.name
        try:
            with open(result_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", domain_name, exc)
            continue

        best_r2, method = _extract_best_r2(data)
        lyap, is_chaotic, is_hyper = _extract_lyapunov(data)
        hopf = _extract_hopf_data(data)
        pysr_eqs = _extract_pysr_equations(data)
        sindy_eqs = _extract_sindy_equations(data)

        results.append(DomainResult(
            domain=domain_name,
            raw=data,
            best_r2=best_r2,
            method=method,
            math_class=_classify_math_class(domain_name),
            lyapunov=lyap,
            is_chaotic=is_chaotic,
            is_hyperchaotic=is_hyper,
            hopf_data=hopf,
            pysr_equations=pysr_eqs,
            sindy_equations=sindy_eqs,
        ))

    logger.info("Loaded %d domain results from %s", len(results), base)
    return results


# ---------------------------------------------------------------------------
# 1. Universal Bifurcation Analysis
# ---------------------------------------------------------------------------

@dataclass
class DynamicsClassification:
    """Classification of a single domain's dynamics type."""

    domain: str
    lyapunov: float
    dynamics_type: str  # "fixed_point", "limit_cycle", "chaos", "hyperchaos"


@dataclass
class BifurcationSummary:
    """Aggregate statistics from the universal bifurcation analysis."""

    n_domains_with_lyapunov: int
    n_fixed_point: int
    n_limit_cycle: int
    n_chaos: int
    n_hyperchaos: int
    classifications: list[DynamicsClassification]
    lyapunov_stats: dict  # mean, median, std, min, max per type
    chaos_fraction: float


def _classify_dynamics(
    lyap: float, is_hyper: bool,
) -> str:
    """Classify dynamics type from Lyapunov exponent."""
    if is_hyper:
        return "hyperchaos"
    if lyap > 0.01:
        return "chaos"
    if lyap > -0.01:
        return "limit_cycle"
    return "fixed_point"


def run_bifurcation_analysis(
    results: list[DomainResult],
) -> BifurcationSummary:
    """Classify dynamics type for all domains with Lyapunov data.

    Args:
        results: Parsed domain results from load_all_results().

    Returns:
        BifurcationSummary with per-domain classifications and
        aggregate statistics.
    """
    classifications: list[DynamicsClassification] = []
    for r in results:
        if r.lyapunov is None:
            continue
        dtype = _classify_dynamics(r.lyapunov, r.is_hyperchaotic)
        classifications.append(DynamicsClassification(
            domain=r.domain,
            lyapunov=r.lyapunov,
            dynamics_type=dtype,
        ))

    counts = {"fixed_point": 0, "limit_cycle": 0, "chaos": 0, "hyperchaos": 0}
    lyap_by_type: dict[str, list[float]] = {k: [] for k in counts}
    for c in classifications:
        counts[c.dynamics_type] += 1
        lyap_by_type[c.dynamics_type].append(c.lyapunov)

    lyap_stats: dict[str, dict] = {}
    for dtype, vals in lyap_by_type.items():
        if vals:
            arr = np.array(vals)
            lyap_stats[dtype] = {
                "count": len(vals),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            lyap_stats[dtype] = {"count": 0}

    n_total = len(classifications)
    n_chaotic = counts["chaos"] + counts["hyperchaos"]
    chaos_frac = n_chaotic / n_total if n_total > 0 else 0.0

    logger.info(
        "Bifurcation analysis: %d domains, %d chaos, %d hyperchaos, "
        "%d limit cycle, %d fixed point",
        n_total, counts["chaos"], counts["hyperchaos"],
        counts["limit_cycle"], counts["fixed_point"],
    )

    return BifurcationSummary(
        n_domains_with_lyapunov=n_total,
        n_fixed_point=counts["fixed_point"],
        n_limit_cycle=counts["limit_cycle"],
        n_chaos=counts["chaos"],
        n_hyperchaos=counts["hyperchaos"],
        classifications=classifications,
        lyapunov_stats=lyap_stats,
        chaos_fraction=chaos_frac,
    )


# ---------------------------------------------------------------------------
# 2. Hopf Scaling Law
# ---------------------------------------------------------------------------

@dataclass
class HopfScalingResult:
    """Result of checking Hopf sqrt(mu-mu_c) scaling for one domain."""

    domain: str
    mu_c: float | None = None
    mu_c_theory: float | None = None
    has_amplitude_data: bool = False
    scaling_exponent: float | None = None
    scaling_r2: float | None = None
    consistent_with_sqrt: bool = False
    note: str = ""


@dataclass
class HopfSummary:
    """Aggregate Hopf bifurcation scaling analysis."""

    n_hopf_domains: int
    n_with_amplitude: int
    n_consistent_sqrt: int
    mean_exponent: float | None
    results: list[HopfScalingResult]


def _check_hopf_sqrt_scaling(dr: DomainResult) -> HopfScalingResult | None:
    """Check if a domain exhibits sqrt(mu - mu_c) amplitude scaling.

    We look for amplitude data in PySR equations that resemble
    sqrt(mu) or (mu - c)^0.5.  We also check Hopf bifurcation
    metadata for mu_c estimates.
    """
    if dr.hopf_data is None:
        return None

    result = HopfScalingResult(domain=dr.domain)

    # Extract critical parameter
    for key in ("b_c_estimate", "mu_c_estimate", "I_ext_c_estimate"):
        if key in dr.hopf_data:
            result.mu_c = dr.hopf_data[key]
            break

    for key in ("b_c_theory", "mu_c_theory"):
        if key in dr.hopf_data:
            result.mu_c_theory = dr.hopf_data[key]

    # Check amplitude PySR expressions for sqrt pattern
    amp_eqs = [
        eq for eq in dr.pysr_equations
        if "amplitude" in eq.get("source", "").lower()
    ]

    if not amp_eqs:
        # Also look for stuart_landau-style amplitude pysr
        amp_eqs = [
            eq for eq in dr.pysr_equations
            if "sqrt(mu)" in eq.get("expression", "").lower()
        ]

    if amp_eqs:
        result.has_amplitude_data = True
        best = max(amp_eqs, key=lambda e: e.get("r_squared", 0))
        expr = best.get("expression", "")
        r2 = best.get("r_squared", 0)

        # Check if expression contains sqrt
        if "sqrt" in expr.lower():
            result.scaling_exponent = 0.5
            result.scaling_r2 = r2
            result.consistent_with_sqrt = r2 > 0.9
            result.note = f"PySR: {expr} (R2={r2:.4f})"
        else:
            result.note = f"PySR (no sqrt): {expr} (R2={r2:.4f})"
    else:
        # Check if the hopf data itself has amplitude info
        raw = dr.raw
        for key in ("amplitude_data", "amplitude_pysr"):
            if key in raw and isinstance(raw[key], dict):
                result.has_amplitude_data = True
                if "mean_error_vs_theory" in raw[key]:
                    err = raw[key]["mean_error_vs_theory"]
                    if err < 0.01:
                        result.consistent_with_sqrt = True
                        result.scaling_exponent = 0.5
                        result.note = (
                            f"Amplitude matches sqrt theory "
                            f"(error={err:.2e})"
                        )

    return result


def run_hopf_analysis(results: list[DomainResult]) -> HopfSummary:
    """Analyze Hopf bifurcation scaling across all domains.

    Args:
        results: Parsed domain results from load_all_results().

    Returns:
        HopfSummary with per-domain scaling checks.
    """
    hopf_results: list[HopfScalingResult] = []
    for dr in results:
        hr = _check_hopf_sqrt_scaling(dr)
        if hr is not None:
            hopf_results.append(hr)

    n_amp = sum(1 for h in hopf_results if h.has_amplitude_data)
    n_sqrt = sum(1 for h in hopf_results if h.consistent_with_sqrt)
    exponents = [
        h.scaling_exponent for h in hopf_results
        if h.scaling_exponent is not None
    ]
    mean_exp = float(np.mean(exponents)) if exponents else None

    logger.info(
        "Hopf analysis: %d domains with Hopf, %d with amplitude data, "
        "%d consistent with sqrt scaling",
        len(hopf_results), n_amp, n_sqrt,
    )

    return HopfSummary(
        n_hopf_domains=len(hopf_results),
        n_with_amplitude=n_amp,
        n_consistent_sqrt=n_sqrt,
        mean_exponent=mean_exp,
        results=hopf_results,
    )


# ---------------------------------------------------------------------------
# 3. R-squared Distribution Analysis
# ---------------------------------------------------------------------------

@dataclass
class R2Stats:
    """Descriptive statistics for a group of R-squared values."""

    label: str
    count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    n_above_099: int
    n_above_0999: int
    n_negative: int = 0
    trimmed_mean: float = 0.0  # mean excluding R2 < 0
    domains: list[str] = field(default_factory=list)


@dataclass
class R2Summary:
    """Comprehensive R-squared distribution analysis."""

    n_total: int
    n_with_r2: int
    overall: R2Stats | None
    by_math_class: list[R2Stats]
    by_method: list[R2Stats]
    top_10: list[tuple[str, float]]
    bottom_10: list[tuple[str, float]]
    predictors: dict  # features that predict high R2


def _compute_r2_stats(
    label: str, pairs: list[tuple[str, float]],
) -> R2Stats:
    """Compute descriptive statistics for a list of (domain, R2) pairs."""
    vals = np.array([v for _, v in pairs])
    positive = vals[vals >= 0]
    trimmed = float(np.mean(positive)) if len(positive) > 0 else 0.0
    return R2Stats(
        label=label,
        count=len(vals),
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        std=float(np.std(vals)),
        min_val=float(np.min(vals)),
        max_val=float(np.max(vals)),
        q25=float(np.percentile(vals, 25)),
        q75=float(np.percentile(vals, 75)),
        n_above_099=int(np.sum(vals >= 0.99)),
        n_above_0999=int(np.sum(vals >= 0.999)),
        n_negative=int(np.sum(vals < 0)),
        trimmed_mean=trimmed,
        domains=[d for d, _ in pairs],
    )


def run_r2_analysis(results: list[DomainResult]) -> R2Summary:
    """Analyze R-squared distribution across all domains.

    Args:
        results: Parsed domain results from load_all_results().

    Returns:
        R2Summary with overall, per-class, and per-method breakdowns.
    """
    with_r2 = [(r.domain, r.best_r2) for r in results if r.best_r2 is not None]
    if not with_r2:
        return R2Summary(
            n_total=len(results), n_with_r2=0, overall=None,
            by_math_class=[], by_method=[], top_10=[], bottom_10=[],
            predictors={},
        )

    overall = _compute_r2_stats("all", with_r2)

    # Group by math class
    by_class: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        if r.best_r2 is not None:
            by_class.setdefault(r.math_class, []).append(
                (r.domain, r.best_r2)
            )
    class_stats = [
        _compute_r2_stats(cls, pairs)
        for cls, pairs in sorted(by_class.items())
    ]

    # Group by method
    by_method: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        if r.best_r2 is not None:
            by_method.setdefault(r.method, []).append(
                (r.domain, r.best_r2)
            )
    method_stats = [
        _compute_r2_stats(m, pairs)
        for m, pairs in sorted(by_method.items())
    ]

    # Top and bottom 10 (bottom sorted worst-first for readability)
    sorted_r2 = sorted(with_r2, key=lambda x: x[1], reverse=True)
    top_10 = sorted_r2[:10]
    bottom_10 = list(reversed(sorted_r2[-10:]))

    # Feature analysis: which domain properties predict high R2?
    predictors = _analyze_r2_predictors(results)

    logger.info(
        "R2 analysis: %d/%d with R2, median=%.4f, "
        "%d above 0.99, %d above 0.999",
        len(with_r2), len(results), overall.median,
        overall.n_above_099, overall.n_above_0999,
    )

    return R2Summary(
        n_total=len(results),
        n_with_r2=len(with_r2),
        overall=overall,
        by_math_class=class_stats,
        by_method=method_stats,
        top_10=top_10,
        bottom_10=bottom_10,
        predictors=predictors,
    )


def _analyze_r2_predictors(results: list[DomainResult]) -> dict:
    """Identify which domain features correlate with higher R-squared.

    Checks: math_class, presence of SINDy, chaotic vs non-chaotic,
    number of PySR equations found, state dimensionality hints.
    """
    # Chaotic vs non-chaotic
    chaotic_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and r.is_chaotic is True
    ]
    non_chaotic_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and r.is_chaotic is False
    ]
    no_lyap_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and r.is_chaotic is None
    ]

    # SINDy vs PySR only
    sindy_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and r.method == "SINDy"
    ]
    pysr_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and r.method == "PySR"
    ]

    # Has many PySR equations vs few
    many_eqs_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and len(r.pysr_equations) >= 5
    ]
    few_eqs_r2 = [
        r.best_r2 for r in results
        if r.best_r2 is not None and 0 < len(r.pysr_equations) < 5
    ]

    def _safe_stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        arr = np.array(vals)
        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "chaotic_vs_nonchaotic": {
            "chaotic": _safe_stats(chaotic_r2),
            "non_chaotic": _safe_stats(non_chaotic_r2),
            "no_lyapunov_data": _safe_stats(no_lyap_r2),
        },
        "method": {
            "sindy": _safe_stats(sindy_r2),
            "pysr": _safe_stats(pysr_r2),
        },
        "equation_count": {
            "many_pysr_eqs": _safe_stats(many_eqs_r2),
            "few_pysr_eqs": _safe_stats(few_eqs_r2),
        },
    }


# ---------------------------------------------------------------------------
# 4. Equation Complexity Analysis
# ---------------------------------------------------------------------------

@dataclass
class EquationComplexity:
    """Complexity metrics for a single PySR expression."""

    domain: str
    expression: str
    r_squared: float
    n_terms: int
    n_operators: int
    operator_types: list[str]
    nesting_depth: int
    char_length: int
    source: str = ""


@dataclass
class ComplexitySummary:
    """Aggregate equation complexity analysis."""

    n_equations: int
    n_domains: int
    complexity_r2_correlation: float
    mean_terms: float
    mean_depth: float
    mean_operators: float
    operator_frequency: dict[str, int]
    equations: list[EquationComplexity]
    # Best fit of R2 vs complexity
    complexity_trend: str  # "negative", "positive", "none"


def _count_terms(expr: str) -> int:
    """Count additive terms in an expression.

    Splits on top-level + and - signs (not inside parentheses).
    """
    depth = 0
    terms = 1
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in ("+", "-") and depth == 0:
            terms += 1
    return max(terms, 1)


def _nesting_depth(expr: str) -> int:
    """Compute maximum parenthesis nesting depth."""
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def _extract_operators(expr: str) -> list[str]:
    """Extract named mathematical operators from the expression."""
    found = []
    expr_lower = expr.lower()
    for op in _PYSR_OPERATORS:
        if op in expr_lower:
            found.append(op)
    # Also count basic arithmetic
    if "*" in expr:
        found.append("mul")
    if "/" in expr:
        found.append("div")
    if "^" in expr or "**" in expr:
        found.append("pow")
    return sorted(set(found))


def _measure_complexity(
    domain: str, eq: dict,
) -> EquationComplexity:
    """Compute complexity metrics for one PySR equation."""
    expr = eq.get("expression", "")
    r2 = eq.get("r_squared", 0.0)
    return EquationComplexity(
        domain=domain,
        expression=expr,
        r_squared=r2,
        n_terms=_count_terms(expr),
        n_operators=len(_extract_operators(expr)),
        operator_types=_extract_operators(expr),
        nesting_depth=_nesting_depth(expr),
        char_length=len(expr),
        source=eq.get("source", ""),
    )


def run_complexity_analysis(
    results: list[DomainResult],
) -> ComplexitySummary:
    """Analyze equation complexity vs R-squared for PySR equations.

    For each domain that has PySR equations, we take the best
    equation and measure its complexity, then correlate with R2.

    Args:
        results: Parsed domain results from load_all_results().

    Returns:
        ComplexitySummary with per-equation metrics and aggregate
        correlation.
    """
    measurements: list[EquationComplexity] = []
    seen_domains: set[str] = set()

    for dr in results:
        if not dr.pysr_equations:
            continue
        # Take the best PySR equation per domain
        best = max(
            dr.pysr_equations,
            key=lambda e: e.get("r_squared", 0),
        )
        if best.get("r_squared", 0) <= 0:
            continue
        ec = _measure_complexity(dr.domain, best)
        measurements.append(ec)
        seen_domains.add(dr.domain)

    if not measurements:
        return ComplexitySummary(
            n_equations=0, n_domains=0, complexity_r2_correlation=0.0,
            mean_terms=0.0, mean_depth=0.0, mean_operators=0.0,
            operator_frequency={}, equations=[],
            complexity_trend="none",
        )

    # Aggregate operator frequency
    op_freq: dict[str, int] = {}
    for ec in measurements:
        for op in ec.operator_types:
            op_freq[op] = op_freq.get(op, 0) + 1

    terms_arr = np.array([ec.n_terms for ec in measurements])
    depth_arr = np.array([ec.nesting_depth for ec in measurements])
    ops_arr = np.array([ec.n_operators for ec in measurements])
    r2_arr = np.array([ec.r_squared for ec in measurements])

    # Combined complexity score = n_terms + nesting_depth + n_operators
    complexity_score = terms_arr + depth_arr + ops_arr

    # Pearson correlation with R2
    if len(complexity_score) >= 3 and np.std(complexity_score) > 0:
        corr = float(np.corrcoef(complexity_score, r2_arr)[0, 1])
        if math.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    if corr > 0.1:
        trend = "positive"
    elif corr < -0.1:
        trend = "negative"
    else:
        trend = "none"

    logger.info(
        "Complexity analysis: %d equations from %d domains, "
        "complexity-R2 correlation=%.3f (%s)",
        len(measurements), len(seen_domains), corr, trend,
    )

    return ComplexitySummary(
        n_equations=len(measurements),
        n_domains=len(seen_domains),
        complexity_r2_correlation=corr,
        mean_terms=float(np.mean(terms_arr)),
        mean_depth=float(np.mean(depth_arr)),
        mean_operators=float(np.mean(ops_arr)),
        operator_frequency=dict(sorted(
            op_freq.items(), key=lambda x: -x[1],
        )),
        equations=measurements,
        complexity_trend=trend,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize(obj: object) -> object:
    """Recursively convert dataclasses and numpy types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: _serialize(v)
            for k, v in obj.__dict__.items()
            if k != "raw"  # skip bulky raw dicts
        }
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_meta_discovery(
    output_dir: str = "output/meta_discovery",
    rediscovery_dir: str | None = None,
) -> dict:
    """Run all four meta-analyses and save results.

    Args:
        output_dir: Directory (relative or absolute) for output files.
        rediscovery_dir: Override path to rediscovery results. If None,
            uses the default output/rediscovery/ under project root.

    Returns:
        Dictionary with keys 'bifurcation', 'hopf', 'r2', 'complexity',
        each containing the serialized analysis summary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rdir = Path(rediscovery_dir) if rediscovery_dir else None
    all_results = load_all_results(rdir)
    logger.info("Loaded %d domain results", len(all_results))

    # 1. Universal Bifurcation Analysis
    logger.info("--- Running universal bifurcation analysis ---")
    bifurcation = run_bifurcation_analysis(all_results)
    bif_dict = _serialize(bifurcation)
    _save_json(out_path / "bifurcation_analysis.json", bif_dict)

    # 2. Hopf Scaling Law
    logger.info("--- Running Hopf scaling analysis ---")
    hopf = run_hopf_analysis(all_results)
    hopf_dict = _serialize(hopf)
    _save_json(out_path / "hopf_scaling.json", hopf_dict)

    # 3. R-squared Distribution Analysis
    logger.info("--- Running R-squared distribution analysis ---")
    r2 = run_r2_analysis(all_results)
    r2_dict = _serialize(r2)
    _save_json(out_path / "r2_distribution.json", r2_dict)

    # 4. Equation Complexity Analysis
    logger.info("--- Running equation complexity analysis ---")
    complexity = run_complexity_analysis(all_results)
    comp_dict = _serialize(complexity)
    _save_json(out_path / "equation_complexity.json", comp_dict)

    # Unified summary
    summary = {
        "n_domains_loaded": len(all_results),
        "bifurcation": bif_dict,
        "hopf": hopf_dict,
        "r2": r2_dict,
        "complexity": comp_dict,
    }
    _save_json(out_path / "meta_discovery_summary.json", summary)

    _print_report(bifurcation, hopf, r2, complexity)
    logger.info("Meta-discovery results saved to %s", out_path)
    return summary


def _save_json(path: Path, data: object) -> None:
    """Write JSON with pretty formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _print_report(
    bif: BifurcationSummary,
    hopf: HopfSummary,
    r2: R2Summary,
    comp: ComplexitySummary,
) -> None:
    """Print a human-readable console summary."""
    print("\n" + "=" * 72)
    print("META-DISCOVERY ANALYSIS REPORT")
    print("=" * 72)

    print(f"\n1. UNIVERSAL BIFURCATION ANALYSIS"
          f" ({bif.n_domains_with_lyapunov} domains)")
    print(f"   Fixed point:  {bif.n_fixed_point}")
    print(f"   Limit cycle:  {bif.n_limit_cycle}")
    print(f"   Chaos:        {bif.n_chaos}")
    print(f"   Hyperchaos:   {bif.n_hyperchaos}")
    print(f"   Chaos fraction: {bif.chaos_fraction:.1%}")
    for dtype, stats in bif.lyapunov_stats.items():
        if stats.get("count", 0) > 0:
            print(f"   {dtype}: mean LE = {stats['mean']:.4f}, "
                  f"range [{stats['min']:.4f}, {stats['max']:.4f}]")

    print(f"\n2. HOPF SCALING ANALYSIS"
          f" ({hopf.n_hopf_domains} domains)")
    print(f"   With amplitude data: {hopf.n_with_amplitude}")
    print(f"   Consistent with sqrt(mu-mu_c): {hopf.n_consistent_sqrt}")
    if hopf.mean_exponent is not None:
        print(f"   Mean scaling exponent: {hopf.mean_exponent:.3f}")

    if r2.overall is not None:
        print(f"\n3. R-SQUARED DISTRIBUTION"
              f" ({r2.n_with_r2}/{r2.n_total} domains)")
        o = r2.overall
        print(f"   Mean:          {o.mean:.4f}")
        print(f"   Trimmed mean:  {o.trimmed_mean:.4f}"
              f"  (excluding {o.n_negative} negative R2)")
        print(f"   Median:        {o.median:.4f}")
        print(f"   Std:           {o.std:.4f}")
        print(f"   Range:         [{o.min_val:.4f}, {o.max_val:.4f}]")
        print(f"   R2 >= 0.99:    {o.n_above_099}")
        print(f"   R2 >= 0.999:   {o.n_above_0999}")
        print("   By math class (top 5 by median):")
        sorted_classes = sorted(
            r2.by_math_class,
            key=lambda s: s.median, reverse=True,
        )
        for s in sorted_classes[:5]:
            print(f"     {s.label:30s}  n={s.count:3d}  "
                  f"median={s.median:.4f}")
        print("   Top 5 domains:")
        for domain, val in r2.top_10[:5]:
            print(f"     {domain:35s} R2={val:.6f}")
        print("   Bottom 5 domains:")
        for domain, val in r2.bottom_10[:5]:
            print(f"     {domain:35s} R2={val:.6f}")

    print(f"\n4. EQUATION COMPLEXITY"
          f" ({comp.n_equations} equations, {comp.n_domains} domains)")
    print(f"   Mean terms:     {comp.mean_terms:.1f}")
    print(f"   Mean depth:     {comp.mean_depth:.1f}")
    print(f"   Mean operators: {comp.mean_operators:.1f}")
    print(f"   Complexity-R2 correlation: {comp.complexity_r2_correlation:.3f}"
          f" ({comp.complexity_trend})")
    if comp.operator_frequency:
        top_ops = list(comp.operator_frequency.items())[:5]
        print("   Most common operators: "
              + ", ".join(f"{op}({n})" for op, n in top_ops))

    print("\n" + "=" * 72)
