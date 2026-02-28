"""Tests for the rediscovery module."""

from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.rediscovery.projectile import (
    generate_projectile_data,
    theoretical_range,
)
from simulating_anything.rediscovery.lotka_volterra import (
    generate_equilibrium_data,
    generate_ode_data,
)
from simulating_anything.rediscovery.gray_scott import (
    compute_dominant_wavelength,
    compute_pattern_energy,
    classify_pattern,
)


class TestProjectileDataGeneration:
    def test_generate_data_shape(self):
        data = generate_projectile_data(n_speeds=3, n_angles=3, dt=0.005)
        assert data["v0"].shape == (9,)
        assert data["theta"].shape == (9,)
        assert data["range"].shape == (9,)

    def test_theoretical_range(self):
        v0 = np.array([30.0])
        theta = np.array([np.pi / 4])
        g = np.array([9.81])
        R = theoretical_range(v0, theta, g)
        expected = 30.0**2 * np.sin(np.pi / 2) / 9.81
        np.testing.assert_allclose(R, expected, rtol=1e-10)

    def test_no_drag_range_matches_theory(self):
        """Simulated range should closely match R = v^2*sin(2*theta)/g."""
        data = generate_projectile_data(
            n_speeds=5, n_angles=5, drag_coefficient=0.0, dt=0.001
        )
        R_theory = theoretical_range(data["v0"], data["theta"], data["g"])
        # Allow up to 1% error from discrete integration
        rel_errors = np.abs(data["range"] - R_theory) / np.maximum(R_theory, 1.0)
        assert np.mean(rel_errors) < 0.01, f"Mean relative error {np.mean(rel_errors):.4%}"

    def test_range_increases_with_speed(self):
        data = generate_projectile_data(
            n_speeds=5, n_angles=1, speed_range=(10.0, 50.0),
            angle_range=(45.0, 45.0), dt=0.001
        )
        # Range should increase monotonically with speed at 45 degrees
        ranges = data["range"]
        assert all(ranges[i] < ranges[i + 1] for i in range(len(ranges) - 1))


class TestLotkaVolterraDataGeneration:
    def test_equilibrium_data_shape(self):
        data = generate_equilibrium_data(n_samples=5, n_steps=1000, dt=0.01)
        assert data["alpha"].shape == (5,)
        assert data["prey_avg"].shape == (5,)

    def test_equilibrium_approximation(self):
        """Time-averaged populations should approximate gamma/delta and alpha/beta."""
        data = generate_equilibrium_data(n_samples=10, n_steps=5000, dt=0.01)
        prey_theory = data["gamma"] / data["delta"]
        pred_theory = data["alpha"] / data["beta"]

        # Allow up to 15% error (time averages converge slowly for some params)
        prey_err = np.abs(data["prey_avg"] - prey_theory) / prey_theory
        pred_err = np.abs(data["pred_avg"] - pred_theory) / pred_theory
        assert np.median(prey_err) < 0.15, f"Median prey error: {np.median(prey_err):.2%}"
        assert np.median(pred_err) < 0.15, f"Median pred error: {np.median(pred_err):.2%}"

    def test_ode_data_generation(self):
        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 2)
        assert data["dt"] == 0.01
        # Populations should stay positive
        assert np.all(data["states"] >= 0)


class TestGrayScottAnalysis:
    def test_pattern_energy_uniform(self):
        field = np.ones((64, 64)) * 0.5
        assert compute_pattern_energy(field) < 1e-10

    def test_pattern_energy_patterned(self):
        rng = np.random.default_rng(42)
        field = rng.random((64, 64))
        assert compute_pattern_energy(field) > 0.01

    def test_classify_uniform(self):
        field = np.ones((64, 64)) * 0.5
        assert classify_pattern(field) == "uniform"

    def test_classify_pattern_exists(self):
        """A sinusoidal pattern should not be classified as uniform."""
        x = np.linspace(0, 4 * np.pi, 64)
        y = np.linspace(0, 4 * np.pi, 64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        field = 0.1 * np.sin(X) * np.sin(Y) + 0.5
        result = classify_pattern(field, threshold=1e-6)
        assert result != "uniform"

    def test_dominant_wavelength_sinusoidal(self):
        """A sinusoidal pattern should have a detectable wavelength."""
        L = 2.5
        nx = 128
        x = np.linspace(0, L, nx, endpoint=False)
        y = np.linspace(0, L, nx, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        k = 4  # 4 wavelengths across domain
        field = np.sin(2 * np.pi * k * X / L)
        wl = compute_dominant_wavelength(field, L)
        expected_wl = L / k  # 0.625
        assert abs(wl - expected_wl) < 0.1 * expected_wl, f"Got {wl}, expected ~{expected_wl}"


class TestPySRIntegration:
    """Test PySR integration (skipped if PySR not available)."""

    @pytest.fixture(autouse=True)
    def _check_pysr(self):
        try:
            from pysr import PySRRegressor  # noqa: F401
        except ImportError:
            pytest.skip("PySR not installed")

    def test_run_symbolic_regression_basic(self):
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        rng = np.random.default_rng(42)
        X = rng.uniform(1, 10, (50, 2))
        y = X[:, 0] * X[:, 1]  # y = x0 * x1

        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["x0", "x1"],
            n_iterations=5,
            binary_operators=["+", "-", "*"],
            unary_operators=[],
            max_complexity=10,
        )
        assert len(discoveries) > 0
        assert discoveries[0].evidence.fit_r_squared > 0.9


class TestPySINDyIntegration:
    """Test PySINDy integration (skipped if PySINDy not available)."""

    @pytest.fixture(autouse=True)
    def _check_sindy(self):
        try:
            import pysindy  # noqa: F401
        except ImportError:
            pytest.skip("PySINDy not installed")

    def test_run_sindy_exponential_decay(self):
        from simulating_anything.analysis.equation_discovery import run_sindy

        t = np.linspace(0, 5, 500)
        x = np.exp(-t).reshape(-1, 1)

        discoveries = run_sindy(
            x, dt=t[1] - t[0], feature_names=["x"], threshold=0.05
        )
        assert len(discoveries) > 0
        # Should find dx/dt = -1.0 * x
        assert discoveries[0].evidence.fit_r_squared > 0.99

    def test_run_sindy_lotka_volterra(self):
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_ode_data(n_steps=2000, dt=0.01)
        discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["prey", "pred"],
            threshold=0.05,
            poly_degree=2,
        )
        assert len(discoveries) == 2  # One equation per variable
        # Both should have reasonable fit
        for d in discoveries:
            assert d.evidence.fit_r_squared > 0.8, f"Low R2 for {d.expression}"
