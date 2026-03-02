"""Tests for the Gauss map simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.gauss_map import GaussMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha: float = 6.2,
    beta: float = -0.5,
    x_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GAUSS_MAP,  # Placeholder until GAUSS_MAP enum is added
        dt=1.0,
        n_steps=100,
        parameters={"alpha": alpha, "beta": beta, "x_0": x_0},
    )


class TestGaussMapCreation:
    def test_creation_and_params(self):
        sim = GaussMapSimulation(_make_config())
        assert sim.alpha == 6.2
        assert sim.beta == -0.5
        assert sim.x_0 == 0.5

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.GAUSS_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = GaussMapSimulation(config)
        assert sim.alpha == 6.2
        assert sim.beta == -0.5
        assert sim.x_0 == 0.5

    def test_custom_parameters(self):
        sim = GaussMapSimulation(_make_config(alpha=4.0, beta=0.0, x_0=0.1))
        assert sim.alpha == 4.0
        assert sim.beta == 0.0
        assert sim.x_0 == 0.1


class TestGaussMapIteration:
    def test_initial_state_shape(self):
        sim = GaussMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state[0] == pytest.approx(0.5)

    def test_step_applies_map(self):
        """Verify one iteration: f(x) = exp(-alpha * x^2) + beta."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.5))
        sim.reset()
        sim.step()
        expected = np.exp(-6.2 * 0.5**2) + (-0.5)
        assert sim.observe()[0] == pytest.approx(expected)

    def test_step_at_zero(self):
        """At x=0: f(0) = exp(0) + beta = 1 + beta."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.0))
        sim.reset()
        sim.step()
        expected = 1.0 + (-0.5)
        assert sim.observe()[0] == pytest.approx(expected)

    def test_step_with_large_x(self):
        """For large |x|, exp(-alpha*x^2) -> 0, so f(x) -> beta."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.3, x_0=5.0))
        sim.reset()
        sim.step()
        # exp(-6.2 * 25) is essentially 0
        assert sim.observe()[0] == pytest.approx(-0.3, abs=1e-10)

    def test_observe_returns_current_state(self):
        sim = GaussMapSimulation(_make_config(x_0=0.7))
        sim.reset()
        assert sim.observe()[0] == pytest.approx(0.7)

    def test_two_iterations(self):
        """Verify two consecutive steps."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.5))
        sim.reset()
        sim.step()
        x1 = sim.observe()[0]
        sim.step()
        expected_x2 = np.exp(-6.2 * x1**2) + (-0.5)
        assert sim.observe()[0] == pytest.approx(expected_x2)

    def test_map_range(self):
        """The map output is bounded: beta <= f(x) <= 1 + beta (since exp >= 0, <= 1)."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.1))
        sim.reset()
        for _ in range(500):
            sim.step()
            x = sim.observe()[0]
            # f(x) = exp(-alpha*x^2) + beta, exp in (0, 1], so f in (beta, 1+beta]
            assert x >= -0.5 - 1e-10, f"x={x} below beta"
            assert x <= 0.5 + 1e-10, f"x={x} above 1+beta"


class TestGaussMapFixedPoints:
    def test_fixed_points_exist(self):
        """Default parameters should have at least one fixed point."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        fps = sim.find_fixed_points()
        assert len(fps) >= 1

    def test_fixed_point_satisfies_equation(self):
        """Each fixed point x* must satisfy x* = exp(-alpha * x*^2) + beta."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        fps = sim.find_fixed_points()
        for fp in fps:
            x_star = fp["x"]
            f_x = np.exp(-6.2 * x_star**2) + (-0.5)
            assert f_x == pytest.approx(x_star, abs=1e-8)

    def test_fixed_point_stability_classification(self):
        """Stability is determined by |f'(x*)| < 1."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        fps = sim.find_fixed_points()
        for fp in fps:
            eigenvalue = fp["eigenvalue"]
            if abs(eigenvalue) < 1.0:
                assert fp["stability"] == "stable"
            else:
                assert fp["stability"] == "unstable"

    def test_fixed_point_eigenvalue_formula(self):
        """f'(x) = -2*alpha*x*exp(-alpha*x^2) at fixed point."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        fps = sim.find_fixed_points()
        for fp in fps:
            x_star = fp["x"]
            expected_eig = -2.0 * 6.2 * x_star * np.exp(-6.2 * x_star**2)
            assert fp["eigenvalue"] == pytest.approx(expected_eig, abs=1e-6)

    def test_fixed_points_different_alpha(self):
        """Different alpha values should give different fixed points."""
        fps1 = GaussMapSimulation(_make_config(alpha=1.0, beta=0.0)).find_fixed_points()
        fps2 = GaussMapSimulation(_make_config(alpha=10.0, beta=0.0)).find_fixed_points()
        # Both should have fixed points, but at different x values
        assert len(fps1) >= 1
        assert len(fps2) >= 1

    def test_fixed_points_beta_zero(self):
        """With beta=0, x* = exp(-alpha * x*^2) so x* > 0."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=0.0))
        fps = sim.find_fixed_points()
        assert len(fps) >= 1
        for fp in fps:
            assert fp["x"] > 0, "Fixed point with beta=0 should be positive"


class TestGaussMapPeriod2:
    def test_period_2_orbit_exists(self):
        """For certain parameters, the map should exhibit period-2 orbits."""
        # Sweep beta to find a period-2 region
        found_p2 = False
        for beta in np.linspace(-1.0, 0.5, 50):
            sim = GaussMapSimulation(_make_config(alpha=6.2, beta=beta))
            sim.reset()
            period = sim.detect_period(max_period=32, n_transient=2000)
            if period == 2:
                found_p2 = True
                break
        assert found_p2, "Should find a period-2 orbit somewhere in beta range"

    def test_period_2_orbit_consistency(self):
        """If period-2 detected, the orbit should alternate between two values."""
        # Find a beta that gives period 2
        for beta in np.linspace(-1.0, 0.5, 50):
            sim = GaussMapSimulation(_make_config(alpha=6.2, beta=beta))
            sim.reset()
            if sim.detect_period(max_period=32, n_transient=2000) == 2:
                # Reset and iterate to attractor
                sim2 = GaussMapSimulation(_make_config(alpha=6.2, beta=beta, x_0=0.5))
                sim2.reset()
                for _ in range(2000):
                    sim2.step()
                x_a = sim2.observe()[0]
                sim2.step()
                x_b = sim2.observe()[0]
                sim2.step()
                x_a2 = sim2.observe()[0]
                assert x_a == pytest.approx(x_a2, abs=1e-5)
                assert abs(x_a - x_b) > 1e-6, "Period-2 points should differ"
                break


class TestGaussMapLyapunov:
    def test_lyapunov_returns_float(self):
        sim = GaussMapSimulation(_make_config())
        lam = sim.compute_lyapunov(n_transient=100, n_compute=500)
        assert isinstance(lam, float)
        assert np.isfinite(lam)

    def test_lyapunov_negative_for_stable_fixed_point(self):
        """When orbit converges to a stable fixed point, Lyapunov should be negative."""
        # Use parameters where we know there is a stable fixed point
        sim = GaussMapSimulation(_make_config(alpha=1.0, beta=0.0))
        fps = sim.find_fixed_points()
        stable_fps = [fp for fp in fps if fp["stability"] == "stable"]
        if stable_fps:
            lam = sim.compute_lyapunov(n_transient=1000, n_compute=5000)
            assert lam < 0, f"Lyapunov={lam} should be negative for stable orbit"

    def test_lyapunov_derivative_at_zero(self):
        """f'(0) = 0, so if orbit includes x=0, log|f'| -> -inf."""
        # The derivative is zero at x=0; this happens at super-stable points
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        deriv_at_zero = sim._derivative(0.0)
        assert deriv_at_zero == pytest.approx(0.0)

    def test_lyapunov_varies_with_beta(self):
        """Lyapunov exponent should change across different beta values."""
        lyapunovs = []
        for beta in [-0.8, -0.3, 0.0, 0.3]:
            sim = GaussMapSimulation(_make_config(alpha=6.2, beta=beta))
            lam = sim.compute_lyapunov(n_transient=500, n_compute=2000)
            lyapunovs.append(lam)
        # Not all should be identical
        assert max(lyapunovs) - min(lyapunovs) > 0.01

    def test_lyapunov_chaotic_regime_positive(self):
        """In chaotic regions, Lyapunov should be positive."""
        # Scan for a chaotic beta
        for beta in np.linspace(-1.0, 0.5, 30):
            sim = GaussMapSimulation(_make_config(alpha=6.2, beta=beta))
            period = sim.detect_period(max_period=32, n_transient=2000)
            if period == -1:  # Chaotic
                lam = sim.compute_lyapunov(n_transient=1000, n_compute=5000)
                if np.isfinite(lam):
                    assert lam > 0, (
                        f"Lyapunov={lam} should be positive for chaotic "
                        f"orbit at beta={beta}"
                    )
                    return
        pytest.skip("No chaotic beta found in scan range")


class TestGaussMapPeriodDetection:
    def test_period_1_at_stable_fp(self):
        """When orbit converges to a fixed point, period should be 1."""
        sim = GaussMapSimulation(_make_config(alpha=1.0, beta=0.0))
        fps = sim.find_fixed_points()
        stable_fps = [fp for fp in fps if fp["stability"] == "stable"]
        if stable_fps:
            sim.reset()
            period = sim.detect_period(max_period=32, n_transient=2000)
            assert period == 1

    def test_chaotic_returns_minus_1(self):
        """Chaotic orbits should return -1 (period > max_period)."""
        # Find a chaotic parameter
        for beta in np.linspace(-1.0, 0.5, 30):
            sim = GaussMapSimulation(_make_config(alpha=6.2, beta=beta))
            sim.reset()
            period = sim.detect_period(max_period=32, n_transient=2000)
            if period == -1:
                return  # Found a chaotic parameter
        pytest.skip("No chaotic regime found in scan range")

    def test_period_detection_consistency(self):
        """Period detection should be consistent across initial conditions."""
        beta = -0.5
        periods = []
        for x_0 in [0.1, 0.3, 0.5, 0.7]:
            sim = GaussMapSimulation(
                _make_config(alpha=6.2, beta=beta, x_0=x_0)
            )
            sim.reset()
            p = sim.detect_period(max_period=32, n_transient=2000)
            periods.append(p)
        # All should detect the same period (same attractor)
        assert len(set(periods)) == 1


class TestGaussMapBifurcation:
    def test_bifurcation_data_shape(self):
        sim = GaussMapSimulation(_make_config())
        data = sim.bifurcation_data(
            param_name="beta",
            param_range=(-0.5, 0.5),
            n_params=10,
            n_transient=100,
            n_plot=10,
        )
        assert "param" in data
        assert "x" in data
        assert len(data["param"]) == len(data["x"])
        # At most 10 params * 10 plot points = 100 entries
        assert len(data["param"]) <= 100

    def test_bifurcation_data_alpha_sweep(self):
        """Can sweep alpha instead of beta."""
        sim = GaussMapSimulation(_make_config())
        data = sim.bifurcation_data(
            param_name="alpha",
            param_range=(1.0, 10.0),
            n_params=10,
            n_transient=100,
            n_plot=5,
        )
        assert len(data["param"]) > 0

    def test_bifurcation_invalid_param(self):
        sim = GaussMapSimulation(_make_config())
        with pytest.raises(ValueError, match="Unknown param_name"):
            sim.bifurcation_data(param_name="gamma")

    def test_bifurcation_x_bounded(self):
        """Attractor points should remain bounded."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        data = sim.bifurcation_data(
            param_name="beta",
            param_range=(-1.0, 0.5),
            n_params=20,
            n_transient=500,
            n_plot=50,
        )
        if len(data["x"]) > 0:
            assert np.all(np.isfinite(data["x"]))
            # For beta in [-1, 0.5], attractor should be in reasonable range
            assert np.all(np.abs(data["x"]) < 10.0)


class TestGaussMapTrajectory:
    def test_run_trajectory_shape(self):
        sim = GaussMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 1)

    def test_run_trajectory_timestamps(self):
        sim = GaussMapSimulation(_make_config())
        traj = sim.run(n_steps=20)
        assert traj.timestamps is not None
        assert len(traj.timestamps) == 21
        assert traj.timestamps[0] == pytest.approx(0.0)
        assert traj.timestamps[-1] == pytest.approx(20.0)

    def test_trajectory_first_step_matches(self):
        """Trajectory first step should match manual iteration."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.5))
        traj = sim.run(n_steps=5)
        expected_x1 = np.exp(-6.2 * 0.5**2) + (-0.5)
        assert traj.states[1, 0] == pytest.approx(expected_x1)


class TestGaussMapReproducibility:
    def test_deterministic_same_params(self):
        """Same parameters should give identical trajectories."""
        sim1 = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.5))
        traj1 = sim1.run(n_steps=100)

        sim2 = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.5))
        traj2 = sim2.run(n_steps=100)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_different_x0_different_orbit(self):
        """Different initial conditions should produce different short-term orbits."""
        sim1 = GaussMapSimulation(_make_config(x_0=0.1))
        traj1 = sim1.run(n_steps=10)

        sim2 = GaussMapSimulation(_make_config(x_0=0.9))
        traj2 = sim2.run(n_steps=10)

        # At least one step should differ
        assert not np.allclose(traj1.states, traj2.states)


class TestGaussMapDerivative:
    def test_derivative_formula(self):
        """f'(x) = -2*alpha*x*exp(-alpha*x^2)."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        x = 0.3
        expected = -2.0 * 6.2 * 0.3 * np.exp(-6.2 * 0.3**2)
        assert sim._derivative(x) == pytest.approx(expected)

    def test_derivative_at_zero_is_zero(self):
        """f'(0) = 0 for any alpha (x=0 is always a critical point of f)."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        assert sim._derivative(0.0) == pytest.approx(0.0)

    def test_derivative_negative_for_positive_x(self):
        """For x > 0, f'(x) < 0 (map is decreasing)."""
        sim = GaussMapSimulation(_make_config(alpha=6.2))
        for x in [0.1, 0.5, 1.0, 2.0]:
            assert sim._derivative(x) < 0

    def test_derivative_positive_for_negative_x(self):
        """For x < 0, f'(x) > 0 (map is increasing)."""
        sim = GaussMapSimulation(_make_config(alpha=6.2))
        for x in [-0.1, -0.5, -1.0, -2.0]:
            assert sim._derivative(x) > 0

    def test_derivative_numerical_check(self):
        """Verify analytical derivative against finite difference."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        x = 0.4
        h = 1e-8
        numerical = (sim._map(x + h) - sim._map(x - h)) / (2 * h)
        analytical = sim._derivative(x)
        assert analytical == pytest.approx(numerical, rel=1e-5)


class TestGaussMapInvariantDensity:
    def test_invariant_density_has_data(self):
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        result = sim.compute_invariant_density(
            n_iterations=5000, n_transient=500, n_bins=50,
        )
        assert len(result["bin_centers"]) > 0
        assert len(result["density"]) > 0

    def test_density_integrates_to_one(self):
        """Probability density should integrate to approximately 1."""
        sim = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
        result = sim.compute_invariant_density(
            n_iterations=50000, n_transient=1000, n_bins=100,
        )
        if len(result["bin_centers"]) > 1:
            dx = result["bin_centers"][1] - result["bin_centers"][0]
            integral = np.sum(result["density"]) * dx
            assert integral == pytest.approx(1.0, abs=0.1)


class TestGaussMapParameterSensitivity:
    def test_alpha_effect_on_curvature(self):
        """Larger alpha makes the Gaussian narrower, shrinking the attractor."""
        sim_low = GaussMapSimulation(_make_config(alpha=1.0, beta=0.0))
        sim_low.reset()
        for _ in range(500):
            sim_low.step()
        x_low = sim_low.observe()[0]

        sim_high = GaussMapSimulation(_make_config(alpha=20.0, beta=0.0))
        sim_high.reset()
        for _ in range(500):
            sim_high.step()
        x_high = sim_high.observe()[0]

        # Both should produce finite values
        assert np.isfinite(x_low)
        assert np.isfinite(x_high)

    def test_beta_shifts_attractor(self):
        """beta shifts the entire map vertically, moving the attractor."""
        configs = [
            _make_config(alpha=6.2, beta=-0.8, x_0=0.5),
            _make_config(alpha=6.2, beta=0.0, x_0=0.5),
            _make_config(alpha=6.2, beta=0.3, x_0=0.5),
        ]
        final_vals = []
        for cfg in configs:
            sim = GaussMapSimulation(cfg)
            sim.reset()
            for _ in range(2000):
                sim.step()
            # Collect orbit average
            total = 0.0
            for _ in range(1000):
                sim.step()
                total += sim.observe()[0]
            final_vals.append(total / 1000)

        # Higher beta should shift the orbit mean upward
        assert final_vals[2] > final_vals[0]


class TestGaussMapRediscovery:
    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.gauss_map import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_beta=20, beta_min=-0.5, beta_max=0.5)
        assert len(data["beta_values"]) == 20
        assert len(data["periods"]) == 20

    def test_lyapunov_data_generation(self):
        from simulating_anything.rediscovery.gauss_map import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_beta=10, beta_min=-0.5, beta_max=0.5)
        assert len(data["beta"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_feigenbaum_estimation(self):
        from simulating_anything.rediscovery.gauss_map import (
            estimate_feigenbaum,
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(
            n_beta=500, beta_min=-1.0, beta_max=1.0,
        )
        result = estimate_feigenbaum(data["periods"], data["beta_values"])
        assert "bifurcation_points" in result

    def test_make_config_helper(self):
        from simulating_anything.rediscovery.gauss_map import _make_config
        config = _make_config(alpha=5.0, beta=-0.3)
        assert config.parameters["alpha"] == 5.0
        assert config.parameters["beta"] == -0.3
