"""Tests for the cubic map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.cubic_map import CubicMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(r: float = 2.5, x_0: float = 0.5) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CUBIC_MAP,  
        dt=1.0,
        n_steps=100,
        parameters={"r": r, "x_0": x_0},
    )


class TestCubicMapConstruction:
    """Tests for CubicMapSimulation construction and defaults."""

    def test_creation_and_params(self):
        sim = CubicMapSimulation(_make_config())
        assert sim.r == 2.5
        assert sim.x_0 == 0.5

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.CUBIC_MAP,
            dt=1.0,
            n_steps=10,
            parameters={},
        )
        sim = CubicMapSimulation(config)
        assert sim.r == 2.5
        assert sim.x_0 == 0.5

    def test_custom_parameters(self):
        sim = CubicMapSimulation(_make_config(r=1.5, x_0=0.3))
        assert sim.r == 1.5
        assert sim.x_0 == 0.3


class TestCubicMapState:
    """Tests for state shape and basic iteration."""

    def test_initial_state_shape(self):
        sim = CubicMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state.dtype == np.float64

    def test_initial_state_value(self):
        sim = CubicMapSimulation(_make_config(x_0=0.7))
        state = sim.reset()
        assert state[0] == pytest.approx(0.7)

    def test_step_applies_map(self):
        sim = CubicMapSimulation(_make_config(r=2.0, x_0=0.5))
        sim.reset()
        sim.step()
        # x_1 = r * x_0 - x_0^3 = 2.0 * 0.5 - 0.125 = 0.875
        assert sim.observe()[0] == pytest.approx(0.875)

    def test_step_applies_map_negative(self):
        sim = CubicMapSimulation(_make_config(r=2.0, x_0=-0.5))
        sim.reset()
        sim.step()
        # x_1 = 2.0 * (-0.5) - (-0.5)^3 = -1.0 + 0.125 = -0.875
        assert sim.observe()[0] == pytest.approx(-0.875)

    def test_observe_returns_current_state(self):
        sim = CubicMapSimulation(_make_config(x_0=0.3))
        sim.reset()
        assert sim.observe()[0] == pytest.approx(0.3)

    def test_step_count_increments(self):
        sim = CubicMapSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


class TestCubicMapFixedPoints:
    """Tests for fixed point analysis."""

    def test_origin_is_always_fixed_point(self):
        """x*=0 is a fixed point: f(0) = r*0 - 0^3 = 0."""
        for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
            sim = CubicMapSimulation(_make_config(r=r, x_0=0.0))
            sim.reset()
            sim.step()
            assert sim.observe()[0] == pytest.approx(0.0, abs=1e-15)

    def test_nontrivial_fixed_points_exist_for_r_gt_1(self):
        """x* = +/-sqrt(r-1) are fixed points for r > 1."""
        r = 2.0
        x_star = np.sqrt(r - 1.0)  # = 1.0
        # f(x*) = r*x* - x*^3 = 2*1 - 1 = 1 = x*
        result = r * x_star - x_star**3
        assert result == pytest.approx(x_star, abs=1e-12)

    def test_nontrivial_fixed_points_negative(self):
        """Negative nontrivial fixed point: f(-sqrt(r-1)) = -sqrt(r-1)."""
        r = 2.5
        x_star = -np.sqrt(r - 1.0)
        result = r * x_star - x_star**3
        assert result == pytest.approx(x_star, abs=1e-12)

    def test_no_nontrivial_fixed_points_for_r_lt_1(self):
        sim = CubicMapSimulation(_make_config(r=0.5))
        fps = sim.find_fixed_points()
        assert len(fps) == 1  # Only origin

    def test_three_fixed_points_for_r_gt_1(self):
        sim = CubicMapSimulation(_make_config(r=2.0))
        fps = sim.find_fixed_points()
        assert len(fps) == 3  # Origin + two symmetric

    def test_fixed_point_values(self):
        r = 1.5
        sim = CubicMapSimulation(_make_config(r=r))
        fps = sim.find_fixed_points()
        x_star = np.sqrt(r - 1.0)
        x_values = sorted([fp["x"] for fp in fps])
        assert x_values[0] == pytest.approx(-x_star, abs=1e-12)
        assert x_values[1] == pytest.approx(0.0, abs=1e-12)
        assert x_values[2] == pytest.approx(x_star, abs=1e-12)

    def test_origin_stable_for_small_r(self):
        """Origin is stable when |r| < 1."""
        sim = CubicMapSimulation(_make_config(r=0.5))
        fps = sim.find_fixed_points()
        assert fps[0]["stability"] == "stable"
        assert fps[0]["eigenvalue"] == pytest.approx(0.5)

    def test_origin_unstable_for_r_gt_1(self):
        """Origin is unstable when |r| > 1."""
        sim = CubicMapSimulation(_make_config(r=2.0))
        fps = sim.find_fixed_points()
        assert fps[0]["stability"] == "unstable"

    def test_nontrivial_stable_for_1_lt_r_lt_2(self):
        """Nontrivial fixed points stable when |3-2r| < 1, i.e. 1 < r < 2."""
        sim = CubicMapSimulation(_make_config(r=1.5))
        fps = sim.find_fixed_points()
        # Nontrivial points are fps[1] and fps[2]
        assert fps[1]["stability"] == "stable"
        assert fps[2]["stability"] == "stable"
        # Eigenvalue: 3 - 2*1.5 = 0
        assert fps[1]["eigenvalue"] == pytest.approx(0.0)

    def test_nontrivial_unstable_for_r_gt_2(self):
        """Nontrivial fixed points unstable when r > 2 (eigenvalue < -1)."""
        sim = CubicMapSimulation(_make_config(r=2.5))
        fps = sim.find_fixed_points()
        assert fps[1]["stability"] == "unstable"
        # Eigenvalue: 3 - 2*2.5 = -2.0
        assert fps[1]["eigenvalue"] == pytest.approx(-2.0)


class TestCubicMapConvergence:
    """Tests for orbit convergence to fixed points."""

    def test_converges_to_origin_for_r_lt_1(self):
        """For |r| < 1, orbit converges to origin from any starting point."""
        sim = CubicMapSimulation(_make_config(r=0.5, x_0=0.3))
        sim.reset()
        for _ in range(100):
            sim.step()
        assert abs(sim.observe()[0]) < 1e-10

    def test_converges_to_nontrivial_for_1_lt_r_lt_2(self):
        """For 1 < r < 2, orbit converges to +/-sqrt(r-1)."""
        r = 1.5
        x_star = np.sqrt(r - 1.0)
        sim = CubicMapSimulation(_make_config(r=r, x_0=0.3))
        sim.reset()
        for _ in range(1000):
            sim.step()
        final_x = sim.observe()[0]
        error = min(abs(final_x - x_star), abs(final_x + x_star))
        assert error < 1e-6


class TestCubicMapSymmetry:
    """Tests for the odd symmetry property f(-x) = -f(x)."""

    def test_single_step_symmetry(self):
        """One step: f(-x_0) = -f(x_0)."""
        r = 2.5
        x_0 = 0.7
        f_pos = r * x_0 - x_0**3
        f_neg = r * (-x_0) - (-x_0)**3
        assert f_neg == pytest.approx(-f_pos, abs=1e-15)

    def test_orbit_symmetry(self):
        """Orbits starting at x_0 and -x_0 are negatives of each other."""
        r = 2.5
        sim_pos = CubicMapSimulation(_make_config(r=r, x_0=0.3))
        sim_neg = CubicMapSimulation(_make_config(r=r, x_0=-0.3))
        sim_pos.reset()
        sim_neg.reset()

        for _ in range(50):
            sim_pos.step()
            sim_neg.step()
            x_pos = sim_pos.observe()[0]
            x_neg = sim_neg.observe()[0]
            assert x_pos == pytest.approx(-x_neg, abs=1e-12)

    def test_f_of_zero_is_zero(self):
        """f(0) = r*0 - 0^3 = 0 for any r."""
        for r in [0.1, 1.0, 2.0, 3.0]:
            sim = CubicMapSimulation(_make_config(r=r, x_0=0.0))
            sim.reset()
            sim.step()
            assert sim.observe()[0] == pytest.approx(0.0, abs=1e-15)


class TestCubicMapPeriods:
    """Tests for period detection."""

    def test_period_1_for_stable_fixed_point(self):
        """For 1 < r < 2, orbit has period 1 (convergent to fixed point)."""
        sim = CubicMapSimulation(_make_config(r=1.5, x_0=0.3))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=2000)
        assert period == 1

    def test_period_2_for_r_above_2(self):
        """For r slightly above 2, orbit should have period 2."""
        sim = CubicMapSimulation(_make_config(r=2.2, x_0=0.5))
        sim.reset()
        period = sim.detect_period(max_period=64, n_transient=2000)
        assert period == 2

    def test_period_increases_with_r(self):
        """Period should generally increase (doubling) as r increases."""
        periods = []
        for r in [1.5, 2.2, 2.4]:
            sim = CubicMapSimulation(_make_config(r=r, x_0=0.5))
            sim.reset()
            p = sim.detect_period(max_period=64, n_transient=2000)
            periods.append(p)
        # Period should be non-decreasing
        for i in range(len(periods) - 1):
            assert periods[i] <= periods[i + 1] or periods[i + 1] == -1


class TestCubicMapLyapunov:
    """Tests for Lyapunov exponent computation."""

    def test_lyapunov_negative_for_stable(self):
        """Lyapunov exponent should be negative for r < 2 (stable fixed point)."""
        sim = CubicMapSimulation(_make_config(r=1.5))
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        assert lam < 0, f"Lyapunov={lam} should be negative for r=1.5"

    def test_lyapunov_at_nontrivial_fixed_point(self):
        """At x*=sqrt(r-1), theoretical Lyapunov is ln|3-2r|.

        For r=1.5: ln|3-3| = ln(0) = -inf. Use r=1.3 instead:
        ln|3-2.6| = ln(0.4) ~ -0.916.
        """
        r = 1.3
        sim = CubicMapSimulation(_make_config(r=r, x_0=0.3))
        lam = sim.compute_lyapunov(n_iterations=20000, n_transient=1000)
        theory = np.log(abs(3.0 - 2.0 * r))
        np.testing.assert_allclose(lam, theory, atol=0.1)

    def test_lyapunov_positive_for_chaos(self):
        """Lyapunov exponent should be positive in chaotic regime."""
        sim = CubicMapSimulation(_make_config(r=2.8, x_0=0.5))
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=1000)
        assert lam > 0, f"Lyapunov={lam} should be positive for r=2.8"

    def test_lyapunov_returns_nan_for_divergence(self):
        """Very large r causes orbit to diverge; should return NaN."""
        sim = CubicMapSimulation(_make_config(r=10.0, x_0=0.5))
        lam = sim.compute_lyapunov(n_iterations=100, n_transient=10)
        # Either NaN or a very large value due to divergence
        assert np.isnan(lam) or abs(lam) > 5


class TestCubicMapBifurcation:
    """Tests for bifurcation diagram generation."""

    def test_bifurcation_diagram_shape(self):
        sim = CubicMapSimulation(_make_config())
        r_vals = np.linspace(1.0, 2.5, 10)
        data = sim.bifurcation_diagram(r_vals, n_transient=100, n_plot=10)
        assert len(data["r"]) == len(data["x"])
        # Should have up to 10 r * 10 plot = 100 points (could be fewer if diverge)
        assert len(data["r"]) <= 100
        assert len(data["r"]) > 0

    def test_bifurcation_values_in_range(self):
        """For moderate r, attractor values should be bounded."""
        sim = CubicMapSimulation(_make_config())
        r_vals = np.linspace(1.0, 2.0, 5)
        data = sim.bifurcation_diagram(r_vals, n_transient=200, n_plot=20)
        # For 1 < r < 2, orbits converge to +/-sqrt(r-1) which is bounded
        assert np.all(np.abs(data["x"]) < 5.0)


class TestCubicMapInvariantDensity:
    """Tests for invariant density computation."""

    def test_invariant_density_shape(self):
        sim = CubicMapSimulation(_make_config(r=2.5, x_0=0.5))
        result = sim.compute_invariant_density(
            n_iterations=5000, n_transient=200, n_bins=50,
        )
        assert len(result["bin_centers"]) == 50
        assert len(result["density"]) == 50

    def test_invariant_density_integrates_to_1(self):
        """Invariant density should integrate approximately to 1."""
        sim = CubicMapSimulation(_make_config(r=2.5, x_0=0.5))
        result = sim.compute_invariant_density(
            n_iterations=50000, n_transient=500, n_bins=100,
        )
        if len(result["bin_centers"]) > 0:
            dx = result["bin_centers"][1] - result["bin_centers"][0]
            integral = np.sum(result["density"]) * dx
            np.testing.assert_allclose(integral, 1.0, atol=0.1)


class TestCubicMapTrajectory:
    """Tests for trajectory collection via run()."""

    def test_run_trajectory_shape(self):
        sim = CubicMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 1)

    def test_run_trajectory_initial_value(self):
        sim = CubicMapSimulation(_make_config(x_0=0.4))
        traj = sim.run(n_steps=10)
        assert traj.states[0, 0] == pytest.approx(0.4)

    def test_run_trajectory_timestamps(self):
        sim = CubicMapSimulation(_make_config())
        traj = sim.run(n_steps=20)
        assert len(traj.timestamps) == 21
        assert traj.timestamps[0] == 0.0


class TestCubicMapReproducibility:
    """Tests for deterministic behavior."""

    def test_same_params_same_trajectory(self):
        """Same parameters produce identical trajectories."""
        traj1 = CubicMapSimulation(_make_config(r=2.3, x_0=0.5)).run(n_steps=100)
        traj2 = CubicMapSimulation(_make_config(r=2.3, x_0=0.5)).run(n_steps=100)
        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_different_x0_different_trajectory(self):
        """Different initial conditions produce different trajectories."""
        traj1 = CubicMapSimulation(_make_config(r=2.5, x_0=0.3)).run(n_steps=10)
        traj2 = CubicMapSimulation(_make_config(r=2.5, x_0=0.4)).run(n_steps=10)
        assert not np.allclose(traj1.states, traj2.states)


class TestCubicMapRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.cubic_map import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_r=20, r_min=0.5, r_max=3.0)
        assert len(data["r_values"]) == 20
        assert len(data["periods"]) == 20

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.cubic_map import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_r=10, r_min=0.5, r_max=3.0)
        assert len(data["r"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_feigenbaum_estimation(self):
        from simulating_anything.rediscovery.cubic_map import (
            estimate_feigenbaum,
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_r=500, r_min=0.5, r_max=3.0)
        result = estimate_feigenbaum(data["periods"], data["r_values"])
        assert "bifurcation_points" in result
        assert len(result["bifurcation_points"]) >= 1

    def test_fixed_point_analysis(self):
        from simulating_anything.rediscovery.cubic_map import analyze_fixed_points
        result = analyze_fixed_points(r_values=np.array([0.5, 1.5, 2.5]))
        assert len(result["fixed_point_sweep"]) == 3
        # r=0.5: 1 fp, r=1.5: 3 fps, r=2.5: 3 fps
        assert result["fixed_point_sweep"][0]["n_fixed_points"] == 1
        assert result["fixed_point_sweep"][1]["n_fixed_points"] == 3

    def test_symmetry_verification(self):
        from simulating_anything.rediscovery.cubic_map import verify_symmetry
        result = verify_symmetry(r=2.5, n_steps=50)
        assert result["symmetric"] is True
        assert result["max_symmetry_error"] < 1e-10

    def test_make_config(self):
        from simulating_anything.rediscovery.cubic_map import _make_config
        config = _make_config(r=1.8, x_0=0.2)
        sim = CubicMapSimulation(config)
        assert sim.r == 1.8
        assert sim.x_0 == 0.2
