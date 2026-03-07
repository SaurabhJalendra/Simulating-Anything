"""Statistical significance tests for rediscovery results.

Provides formal hypothesis testing infrastructure:
1. Permutation test for R^2 significance
2. Bootstrap confidence intervals (extends error_analysis.py)
3. Wilcoxon signed-rank test for paired method comparisons
4. Effect size measures (Cohen's d)

No scipy dependency -- all tests implemented from scratch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PermutationTestResult:
    """Result of a permutation test."""

    observed_statistic: float
    p_value: float
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float
    significant: bool  # at alpha=0.05


@dataclass
class WilcoxonResult:
    """Result of Wilcoxon signed-rank test."""

    statistic: float
    p_value: float
    n_pairs: int
    significant: bool
    effect_size: float  # r = Z / sqrt(N)
    method_a_wins: int
    method_b_wins: int
    ties: int


@dataclass
class EffectSize:
    """Cohen's d and related effect size measures."""

    cohens_d: float
    interpretation: str  # "negligible", "small", "medium", "large"
    mean_diff: float
    pooled_std: float


@dataclass
class SignificanceReport:
    """Full statistical significance analysis for one domain."""

    domain: str
    r_squared: float
    permutation_test: PermutationTestResult | None = None
    bootstrap_ci: tuple[float, float] | None = None
    bootstrap_n: int = 0
    notes: list[str] = field(default_factory=list)


def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def permutation_test_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> PermutationTestResult:
    """Test if R^2 is significantly better than chance.

    Null hypothesis: the relationship between y_true and y_pred is no
    better than random permutation of the predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        n_permutations: Number of permutations.
        seed: Random seed.
        alpha: Significance level.

    Returns:
        PermutationTestResult with p-value.
    """
    rng = np.random.default_rng(seed)
    observed_r2 = _compute_r_squared(y_true, y_pred)

    null_r2s = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_pred = rng.permutation(y_pred)
        null_r2s[i] = _compute_r_squared(y_true, perm_pred)

    p_value = float(np.mean(null_r2s >= observed_r2))

    return PermutationTestResult(
        observed_statistic=observed_r2,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution_mean=float(np.mean(null_r2s)),
        null_distribution_std=float(np.std(null_r2s)),
        significant=p_value < alpha,
    )


def bootstrap_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for R^2.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_r2 = _compute_r_squared(y_true, y_pred)

    boot_r2s = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_r2s[i] = _compute_r_squared(y_true[idx], y_pred[idx])

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(boot_r2s, 100 * alpha))
    ci_upper = float(np.percentile(boot_r2s, 100 * (1 - alpha)))

    return point_r2, ci_lower, ci_upper


def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> WilcoxonResult:
    """Wilcoxon signed-rank test comparing two paired sets of scores.

    Tests whether method A produces significantly different scores than method B.
    Implementation without scipy.

    Args:
        scores_a: Scores from method A (one per domain).
        scores_b: Scores from method B (one per domain).
        alpha: Significance level.

    Returns:
        WilcoxonResult with test statistic and p-value.
    """
    diffs = np.array(scores_a) - np.array(scores_b)

    # Remove zeros (ties)
    nonzero_mask = diffs != 0
    ties = int(np.sum(~nonzero_mask))
    diffs = diffs[nonzero_mask]
    n = len(diffs)

    if n == 0:
        return WilcoxonResult(
            statistic=0.0, p_value=1.0, n_pairs=len(scores_a),
            significant=False, effect_size=0.0,
            method_a_wins=0, method_b_wins=0, ties=ties,
        )

    # Rank absolute differences
    abs_diffs = np.abs(diffs)
    ranks = np.empty(n)
    sorted_idx = np.argsort(abs_diffs)
    for rank_val, idx in enumerate(sorted_idx, 1):
        ranks[idx] = rank_val

    # Handle tied ranks (average)
    unique_vals = np.unique(abs_diffs)
    for val in unique_vals:
        mask = abs_diffs == val
        if np.sum(mask) > 1:
            avg_rank = np.mean(ranks[mask])
            ranks[mask] = avg_rank

    # W+ = sum of ranks for positive differences
    w_plus = np.sum(ranks[diffs > 0])
    w_minus = np.sum(ranks[diffs < 0])
    W = min(w_plus, w_minus)

    # Normal approximation for p-value (valid for n >= 10)
    mean_W = n * (n + 1) / 4
    std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if std_W > 0:
        z = (W - mean_W) / std_W
        # Two-tailed p-value using normal approximation
        p_value = 2 * _normal_cdf(-abs(z))
        effect_size = abs(z) / math.sqrt(n)
    else:
        p_value = 1.0
        effect_size = 0.0

    method_a_wins = int(np.sum(diffs > 0))
    method_b_wins = int(np.sum(diffs < 0))

    return WilcoxonResult(
        statistic=float(W),
        p_value=p_value,
        n_pairs=len(scores_a),
        significant=p_value < alpha,
        effect_size=effect_size,
        method_a_wins=method_a_wins,
        method_b_wins=method_b_wins,
        ties=ties,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> EffectSize:
    """Compute Cohen's d effect size between two groups."""
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    std_a, std_b = np.std(group_a, ddof=1), np.std(group_b, ddof=1)
    n_a, n_b = len(group_a), len(group_b)

    pooled_std = math.sqrt(
        ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
    ) if (n_a + n_b - 2) > 0 else 1.0

    d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
    abs_d = abs(d)

    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSize(
        cohens_d=d,
        interpretation=interpretation,
        mean_diff=float(mean_a - mean_b),
        pooled_std=pooled_std,
    )


def run_significance_analysis(
    domain: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> SignificanceReport:
    """Run full significance analysis for one domain's rediscovery.

    Args:
        domain: Domain name.
        y_true: Ground truth values.
        y_pred: Predicted values from discovered equation.
        n_permutations: For permutation test.
        n_bootstrap: For bootstrap CI.
        seed: Random seed.

    Returns:
        SignificanceReport with all test results.
    """
    r2 = _compute_r_squared(y_true, y_pred)

    perm = permutation_test_r_squared(
        y_true, y_pred, n_permutations=n_permutations, seed=seed,
    )

    _, ci_lower, ci_upper = bootstrap_r_squared(
        y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed,
    )

    notes = []
    if perm.significant:
        notes.append(f"R^2={r2:.6f} is statistically significant (p={perm.p_value:.6f})")
    else:
        notes.append(f"R^2={r2:.6f} is NOT statistically significant (p={perm.p_value:.6f})")

    notes.append(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

    return SignificanceReport(
        domain=domain,
        r_squared=r2,
        permutation_test=perm,
        bootstrap_ci=(ci_lower, ci_upper),
        bootstrap_n=n_bootstrap,
        notes=notes,
    )
