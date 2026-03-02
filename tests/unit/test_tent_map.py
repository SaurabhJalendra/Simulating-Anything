"""Tests for the tent map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.tent_map import TentMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(r: float = 2.0, x_0: float = 1.0 / np.pi) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.TENT_MAP,  
        dt=1.0,
        n_steps=100,
        parameters={"r": r, "x_0": x_0},
    )


class TestTentMapConstruction:
    def test_creation_and_params(self):
        sim = TentMapSimulation(_make_config(r=1.5, x_0=0.3))
        assert sim.r == 1.5
        assert sim.x_0 == 0.3

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.TENT_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = TentMapSimulation(config)
        assert sim.r == 2.0
        assert sim.x_0 == pytest.approx(1.0 / np.pi)

    def test_initial_state_shape(self):
        sim = TentMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state[0] == pytest.approx(1.0 / np.pi)

    def test_observe_returns_current_state(self):
        sim = TentMapSimulation(_make_config(x_0=0.7))
        sim.reset()
        assert sim.observe()[0] == pytest.approx(0.7)


class TestTentMapStep:
    def test_step_left_branch(self):
        """For x < 0.5: x_{n+1} = r * x."""
        sim = TentMapSimulation(_make_config(r=1.5, x_0=0.3))
        sim.reset()
        sim.step()
        # x_1 = 1.5 * 0.3 = 0.45
        assert sim.observe()[0] == pytest.approx(0.45)

    def test_step_right_branch(self):
        """For x >= 0.5: x_{n+1} = r * (1 - x)."""
        sim = TentMapSimulation(_make_config(r=1.5, x_0=0.7))
        sim.reset()
        sim.step()
        # x_1 = 1.5 * (1 - 0.7) = 1.5 * 0.3 = 0.45
        assert sim.observe()[0] == pytest.approx(0.45)

    def test_step_at_peak(self):
        """At x = 0.5, the map takes the right branch: r * (1 - 0.5) = r/2."""
        sim = TentMapSimulation(_make_config(r=2.0, x_0=0.5))
        sim.reset()
        sim.step()
        # x_1 = 2.0 * (1 - 0.5) = 1.0
        assert sim.observe()[0] == pytest.approx(1.0)

    def test_piecewise_linearity(self):
        """Verify piecewise linear structure: f is linear on [0,0.5] and [0.5,1]."""
        r = 1.8
        # Left branch: check linearity between two points
        sim1 = TentMapSimulation(_make_config(r=r, x_0=0.1))
        sim1.reset()
        sim1.step()
        y1 = sim1.observe()[0]

        sim2 = TentMapSimulation(_make_config(r=r, x_0=0.3))
        sim2.reset()
        sim2.step()
        y2 = sim2.observe()[0]

        # Slope on left branch should be r
        slope = (y2 - y1) / (0.3 - 0.1)
        assert slope == pytest.approx(r, abs=1e-10)

        # Right branch: check linearity
        sim3 = TentMapSimulation(_make_config(r=r, x_0=0.6))
        sim3.reset()
        sim3.step()
        y3 = sim3.observe()[0]

        sim4 = TentMapSimulation(_make_config(r=r, x_0=0.8))
        sim4.reset()
        sim4.step()
        y4 = sim4.observe()[0]

        # Slope on right branch should be -r
        slope_right = (y4 - y3) / (0.8 - 0.6)
        assert slope_right == pytest.approx(-r, abs=1e-10)

    def test_symmetry_about_half(self):
        """f(x) = f(1-x) for the tent map."""
        r = 1.7
        x_vals = [0.1, 0.2, 0.3, 0.4]
        for x in x_vals:
            sim_left = TentMapSimulation(_make_config(r=r, x_0=x))
            sim_left.reset()
            sim_left.step()

            sim_right = TentMapSimulation(_make_config(r=r, x_0=1.0 - x))
            sim_right.reset()
            sim_right.step()

            assert sim_left.observe()[0] == pytest.approx(
                sim_right.observe()[0], abs=1e-12
            )


class TestTentMapFixedPoints:
    def test_trivial_fixed_point(self):
        """x* = 0 is always a fixed point."""
        sim = TentMapSimulation(_make_config(r=1.5))
        fps = sim.find_fixed_points()
        assert fps[0]["x"] == pytest.approx(0.0)

    def test_nontrivial_fixed_point(self):
        """x* = r/(r+1) is a fixed point for r >= 1 (on right branch)."""
        for r in [1.0, 1.5, 2.0]:
            sim = TentMapSimulation(_make_config(r=r))
            fps = sim.find_fixed_points()
            assert len(fps) == 2
            expected = r / (r + 1.0)
            assert fps[1]["x"] == pytest.approx(expected, abs=1e-10)

    def test_no_nontrivial_fixed_point_for_r_lt_1(self):
        """For r < 1, only the trivial fixed point x*=0 exists."""
        sim = TentMapSimulation(_make_config(r=0.8))
        fps = sim.find_fixed_points()
        assert len(fps) == 1
        assert fps[0]["x"] == pytest.approx(0.0)

    def test_fixed_point_is_actually_fixed(self):
        """Verify f(x*) = x* for the nontrivial fixed point (r >= 1)."""
        for r in [1.0, 1.5, 1.8, 2.0]:
            x_star = r / (r + 1.0)
            # x_star >= 0.5 for r >= 1, so use right branch
            result = r * (1.0 - x_star)
            assert result == pytest.approx(x_star, abs=1e-10)

    def test_stability_for_r_less_than_1(self):
        """Only trivial fixed point exists for r < 1, and it is stable."""
        sim = TentMapSimulation(_make_config(r=0.8))
        fps = sim.find_fixed_points()
        assert len(fps) == 1
        assert fps[0]["stability"] == "stable"

    def test_instability_for_r_greater_than_1(self):
        """Both fixed points should be unstable for r > 1."""
        sim = TentMapSimulation(_make_config(r=1.5))
        fps = sim.find_fixed_points()
        assert fps[0]["stability"] == "unstable"
        assert fps[1]["stability"] == "unstable"

    def test_convergence_to_zero_r_less_than_1(self):
        """For r < 1, orbit should converge to x* = 0."""
        sim = TentMapSimulation(_make_config(r=0.8, x_0=0.4))
        sim.reset()
        for _ in range(500):
            sim.step()
        assert sim.observe()[0] == pytest.approx(0.0, abs=1e-10)


class TestTentMapLyapunov:
    def test_lyapunov_equals_ln_r_at_r2(self):
        """At r=2, Lyapunov exponent = ln(2) exactly."""
        sim = TentMapSimulation(_make_config(r=2.0, x_0=0.4))
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=500)
        np.testing.assert_allclose(lam, np.log(2), atol=1e-6)

    def test_lyapunov_equals_ln_r_general(self):
        """For r > 1, Lyapunov exponent should be ln(r)."""
        for r in [1.2, 1.5, 1.8, 2.0]:
            sim = TentMapSimulation(_make_config(r=r, x_0=0.4))
            lam = sim.compute_lyapunov(n_iterations=10000, n_transient=500)
            np.testing.assert_allclose(lam, np.log(r), atol=1e-6)

    def test_lyapunov_negative_for_r_less_than_1(self):
        """Lyapunov exponent should be negative (stable) for r < 1."""
        sim = TentMapSimulation(_make_config(r=0.8))
        lam = sim.compute_lyapunov(n_iterations=5000)
        assert lam < 0, f"Lyapunov={lam} should be negative for r=0.8"

    def test_lyapunov_positive_for_r_greater_than_1(self):
        """Lyapunov exponent should be positive (chaotic) for r > 1."""
        sim = TentMapSimulation(_make_config(r=1.5))
        lam = sim.compute_lyapunov(n_iterations=5000)
        assert lam > 0, f"Lyapunov={lam} should be positive for r=1.5"

    def test_lyapunov_zero_at_r1(self):
        """At r=1, Lyapunov exponent = ln(1) = 0."""
        sim = TentMapSimulation(_make_config(r=1.0, x_0=0.4))
        lam = sim.compute_lyapunov(n_iterations=10000, n_transient=500)
        np.testing.assert_allclose(lam, 0.0, atol=1e-6)

    def test_lyapunov_monotonically_increases_with_r(self):
        """Lyapunov exponent = ln(r) is monotonically increasing."""
        r_vals = [1.1, 1.3, 1.5, 1.7, 2.0]
        lyaps = []
        for r in r_vals:
            sim = TentMapSimulation(_make_config(r=r))
            lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
            lyaps.append(lam)

        for i in range(len(lyaps) - 1):
            assert lyaps[i] < lyaps[i + 1], (
                f"Lyapunov should increase: {lyaps[i]} < {lyaps[i+1]} at "
                f"r={r_vals[i]}, {r_vals[i+1]}"
            )


class TestTentMapInvariantDensity:
    def test_uniform_density_at_r2(self):
        """At r=2, the invariant density should be approximately uniform.

        The tent map at r=2 is conjugate to a binary shift, which causes
        finite-precision orbits to hit 0 after ~52 iterations. Random
        perturbation maintains ergodicity but introduces slight non-uniformity
        near the map's peak (x=0.5). We use a 35% tolerance to account for
        this numerical artifact while still verifying approximate uniformity.
        """
        sim = TentMapSimulation(_make_config(r=2.0))
        density_data = sim.compute_invariant_density(
            n_bins=20,
            n_iterations=200000,
            n_transient=1000,
        )
        density = density_data["density"]
        # Uniform density on [0,1] has value 1.0
        # Allow 35% tolerance due to finite-precision perturbation artifacts
        np.testing.assert_allclose(density, 1.0, atol=0.35)

    def test_density_integrates_to_one(self):
        """The density histogram should integrate to approximately 1."""
        sim = TentMapSimulation(_make_config(r=2.0))
        density_data = sim.compute_invariant_density(
            n_bins=50,
            n_iterations=100000,
        )
        # density=True in np.histogram means integral = 1
        bin_width = 1.0 / 50
        total = np.sum(density_data["density"]) * bin_width
        np.testing.assert_allclose(total, 1.0, atol=0.02)

    def test_density_bin_centers(self):
        """Check that bin centers are correctly computed."""
        sim = TentMapSimulation(_make_config(r=2.0))
        density_data = sim.compute_invariant_density(n_bins=10)
        centers = density_data["bin_centers"]
        assert len(centers) == 10
        assert centers[0] == pytest.approx(0.05)
        assert centers[-1] == pytest.approx(0.95)


class TestTentMapBifurcation:
    def test_bifurcation_diagram_shape(self):
        sim = TentMapSimulation(_make_config())
        r_vals = np.linspace(0.5, 2.0, 10)
        data = sim.bifurcation_diagram(r_vals, n_transient=100, n_plot=10)
        assert len(data["r"]) == 100  # 10 r * 10 plot points
        assert len(data["x"]) == 100

    def test_bifurcation_values_bounded(self):
        """All bifurcation x values should be in [0, 1] for r <= 2."""
        sim = TentMapSimulation(_make_config())
        r_vals = np.linspace(0.1, 2.0, 20)
        data = sim.bifurcation_diagram(r_vals, n_transient=200, n_plot=20)
        assert np.all(data["x"] >= -0.001)
        assert np.all(data["x"] <= 1.001)

    def test_no_period_doubling(self):
        """Tent map goes directly to chaos -- no intermediate period-2 regime.

        Unlike the logistic map, for r slightly above 1 the tent map is
        immediately chaotic (Lyapunov > 0).
        """
        # At r=1.1, logistic map has period 1 but tent map has Lyapunov > 0
        sim = TentMapSimulation(_make_config(r=1.1))
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        assert lam > 0, (
            f"Tent map at r=1.1 should be chaotic (Lyapunov={lam}), "
            "no period-doubling cascade"
        )


class TestTentMapTrajectory:
    def test_run_trajectory_shape(self):
        sim = TentMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 1)  # initial + 50 steps

    def test_trajectory_bounded_for_r_le_2(self):
        """Iterates should stay in [0, 1] for r <= 2."""
        sim = TentMapSimulation(_make_config(r=2.0, x_0=0.4))
        sim.reset()
        for _ in range(1000):
            sim.step()
            x = sim.observe()[0]
            assert 0 <= x <= 1.0, f"x={x} outside [0,1]"

    def test_reproducibility(self):
        """Same initial condition should give identical trajectory."""
        sim1 = TentMapSimulation(_make_config(r=1.8, x_0=0.3))
        traj1 = sim1.run(n_steps=100)

        sim2 = TentMapSimulation(_make_config(r=1.8, x_0=0.3))
        traj2 = sim2.run(n_steps=100)

        np.testing.assert_array_equal(traj1.states, traj2.states)


class TestTentMapPeriodDetection:
    def test_period_1_for_r_less_than_1(self):
        """For r < 1, orbit converges to fixed point (period 1)."""
        sim = TentMapSimulation(_make_config(r=0.8))
        sim.reset()
        period = sim.detect_period()
        assert period == 1

    def test_chaotic_for_r_2(self):
        """For r=2, orbit should be chaotic (no detectable period).

        Must use irrational x_0 to avoid periodic orbits caused by
        the binary shift structure of the tent map at r=2.
        """
        sim = TentMapSimulation(_make_config(r=2.0))  # Default x_0=1/pi
        sim.reset()
        period = sim.detect_period(max_period=64)
        assert period == -1


class TestTentMapRediscovery:
    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.tent_map import generate_lyapunov_data
        data = generate_lyapunov_data(n_r=10, r_min=0.5, r_max=2.0)
        assert len(data["r"]) == 10
        assert len(data["lyapunov"]) == 10
        assert len(data["lyapunov_theory"]) == 10

    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.tent_map import generate_bifurcation_data
        data = generate_bifurcation_data(n_r=20, r_min=0.5, r_max=2.0)
        assert len(data["r_values"]) == 20
        assert len(data["periods"]) == 20

    def test_fixed_point_verification(self):
        from simulating_anything.rediscovery.tent_map import verify_fixed_points
        results = verify_fixed_points(r_values=np.array([1.0, 1.5, 2.0]))
        assert len(results) == 3
        for res in results:
            assert res["error"] is not None
            assert res["error"] < 1e-10

    def test_invariant_density_verification(self):
        from simulating_anything.rediscovery.tent_map import verify_invariant_density
        result = verify_invariant_density(n_bins=10, n_iterations=50000)
        assert "mean_density" in result
        assert "max_deviation_from_uniform" in result
        assert result["mean_density"] == pytest.approx(1.0, abs=0.2)
