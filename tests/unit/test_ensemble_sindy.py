"""Test ensemble SINDy noise robustness (ADR-0001 Change #1).

The architectural commitment is that ensemble fitting is on by default.
The empirical claim is that ensemble fitting is at-least-as-accurate as
single-fit STLSQ on clean data and degrades more gracefully under noise.

Bounds in this file are evidence-based — they reflect actual observed
behavior on this Lotka-Volterra fixture across multiple seeds, not
aspirational numbers from the literature. Bounds will tighten as
WSINDy weak-form fitting is added (deferred work).

Coefficient extraction uses pysindy's model.coefficients() rather than
parsing equation strings, which is robust to spurious low-magnitude terms.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pysindy")
import pysindy as ps  # noqa: E402


def lotka_volterra_clean(n_steps: int = 1500, dt: float = 0.01) -> np.ndarray:
    """Generate a Lotka-Volterra trajectory.

    True dynamics:
        dx/dt = 1.5 x - 1.0 x y
        dy/dt = -3.0 y + 1.0 x y
    """
    state = np.array([10.0, 5.0])
    states = np.zeros((n_steps, 2))
    for i in range(n_steps):
        states[i] = state
        x, y = state
        state = state + dt * np.array(
            [1.5 * x - 1.0 * x * y, -3.0 * y + 1.0 * x * y]
        )
    return states


def add_gaussian_noise(states: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Add column-wise std-scaled Gaussian noise."""
    rng = np.random.default_rng(seed)
    return states + rng.normal(0.0, states.std(axis=0) * frac, size=states.shape)


def fit_pysindy(
    states: np.ndarray,
    dt: float,
    ensemble: bool,
    n_models: int = 30,
    np_seed: int | None = None,
):
    """Fit pysindy directly with or without EnsembleOptimizer wrapping.

    np_seed sets numpy global RNG state before fit to make EnsembleOptimizer's
    internal randomness reproducible (it does not expose its own RNG).
    Returns (coefficients_matrix, feature_names).
    """
    if np_seed is not None:
        np.random.seed(np_seed)
    base = ps.STLSQ(threshold=0.05, max_iter=20)
    opt = (
        ps.EnsembleOptimizer(
            opt=base, bagging=True, library_ensemble=True, n_models=n_models
        )
        if ensemble
        else base
    )
    model = ps.SINDy(
        optimizer=opt, feature_library=ps.PolynomialLibrary(degree=2)
    )
    model.fit(states, t=dt, feature_names=["x", "y"])
    return model.coefficients(), model.get_feature_names()


TRUE_COEFFS = {
    ("x", "x"): 1.5,
    ("x", "x y"): -1.0,
    ("y", "y"): -3.0,
    ("y", "x y"): 1.0,
}


def max_relative_error(coef: np.ndarray, names: list[str]) -> float:
    """Max relative error across the four dominant terms."""
    eq = {"x": 0, "y": 1}
    return max(
        abs(coef[eq[e], names.index(t)] - v) / abs(v)
        for (e, t), v in TRUE_COEFFS.items()
    )


def median_error_over_seeds(
    states_factory, ensemble: bool, n_models: int, fit_seeds: list[int]
) -> float:
    """Run fit across multiple numpy seeds and return median max-rel-error.

    Median is more robust than mean to outlier bootstraps.
    """
    errs = []
    for seed in fit_seeds:
        coef, names = fit_pysindy(
            states_factory(), dt=0.01, ensemble=ensemble, n_models=n_models, np_seed=seed
        )
        errs.append(max_relative_error(coef, names))
    return float(np.median(errs))


class TestEnsembleSINDyDefaults:
    """The architectural commitment: ensemble=True is the new default."""

    def test_run_sindy_default_is_ensemble_true(self):
        import inspect

        from simulating_anything.analysis.equation_discovery import run_sindy

        sig = inspect.signature(run_sindy)
        assert sig.parameters["ensemble"].default is True
        assert sig.parameters["bagging"].default is True
        assert sig.parameters["library_ensemble"].default is True

    def test_run_sindy_smoke_returns_discoveries(self):
        from simulating_anything.analysis.equation_discovery import run_sindy

        states = lotka_volterra_clean(n_steps=1000, dt=0.01)
        discoveries = run_sindy(
            states, dt=0.01, feature_names=["x", "y"], threshold=0.05, poly_degree=2
        )
        assert len(discoveries) == 2
        assert all(d.evidence.fit_r_squared > 0.9 for d in discoveries)


class TestEnsembleNoiseRobustness:
    """Empirical noise-robustness checks with seed averaging."""

    SEEDS = [0, 1, 2, 3, 4]

    def test_clean_data_ensemble_under_5pct_median(self):
        """Median across 5 seeds: ensemble error on clean data < 5%."""
        med = median_error_over_seeds(
            lotka_volterra_clean,
            ensemble=True, n_models=30, fit_seeds=self.SEEDS,
        )
        assert med < 0.05, f"Median ensemble error on clean data: {med:.4f}"

    def test_clean_data_single_under_5pct(self):
        """Single-fit baseline: deterministic, must be under 5% error."""
        coef, names = fit_pysindy(
            lotka_volterra_clean(), dt=0.01, ensemble=False
        )
        err = max_relative_error(coef, names)
        assert err < 0.05, f"Single-fit error on clean data: {err:.4f}"

    @pytest.mark.parametrize("noise_frac", [0.005, 0.01, 0.02])
    def test_ensemble_robust_at_low_noise(self, noise_frac):
        """At <=2% noise, median ensemble error stays under 15%.

        This bracket is where vanilla STLSQ already works; ensemble must
        not regress.
        """
        def factory():
            return add_gaussian_noise(lotka_volterra_clean(), noise_frac, seed=42)

        med = median_error_over_seeds(
            factory, ensemble=True, n_models=50, fit_seeds=self.SEEDS,
        )
        assert med < 0.15, (
            f"Median ensemble error at {noise_frac*100:.1f}% noise: {med:.3f}"
        )

    def test_ensemble_does_not_catastrophically_fail_at_5pct(self):
        """At 5% noise, median ensemble error stays under 50% (single-fit
        regularly produces 100%+ errors here per CLAUDE.md). This is the
        practical "graceful degradation" claim."""
        def factory():
            return add_gaussian_noise(lotka_volterra_clean(), 0.05, seed=42)

        med = median_error_over_seeds(
            factory, ensemble=True, n_models=50, fit_seeds=self.SEEDS,
        )
        assert med < 0.50, (
            f"Median ensemble error at 5% noise: {med:.3f} — "
            "ensemble is failing catastrophically"
        )
