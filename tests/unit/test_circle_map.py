"""Tests for the circle map (Arnold tongue) simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.circle_map import CircleMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    omega: float = 0.5,
    K: float = 1.0,
    theta_0: float = 0.1,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CIRCLE_MAP,  # Placeholder until CIRCLE_MAP enum added
        dt=1.0,
        n_steps=100,
        parameters={"omega": omega, "K": K, "theta_0": theta_0},
    )


class TestCircleMapSimulation:
    """Tests for the CircleMapSimulation class."""

    def test_creation_and_params(self):
        """Verify parameters are read from config correctly."""
        sim = CircleMapSimulation(_make_config(omega=0.3, K=2.0, theta_0=0.5))
        assert sim.omega == 0.3
        assert sim.K == 2.0
        assert sim.theta_0 == 0.5

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.CIRCLE_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = CircleMapSimulation(config)
        assert sim.omega == 0.5
        assert sim.K == 1.0
        assert sim.theta_0 == 0.1

    def test_initial_state_shape(self):
        """State should be shape (1,)."""
        sim = CircleMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)

    def test_initial_state_value(self):
        """Initial state should equal theta_0."""
        sim = CircleMapSimulation(_make_config(theta_0=0.7))
        state = sim.reset()
        assert state[0] == pytest.approx(0.7)

    def test_map_formula_one_step(self):
        """Verify one step matches the circle map formula."""
        omega, K, theta_0 = 0.3, 0.5, 0.2
        sim = CircleMapSimulation(_make_config(omega=omega, K=K, theta_0=theta_0))
        sim.reset()
        sim.step()

        expected = (theta_0 + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta_0)) % 1.0
        assert sim.observe()[0] == pytest.approx(expected, abs=1e-14)

    def test_map_formula_two_steps(self):
        """Verify two consecutive steps match iterated formula."""
        omega, K, theta_0 = 0.4, 0.8, 0.15
        sim = CircleMapSimulation(_make_config(omega=omega, K=K, theta_0=theta_0))
        sim.reset()

        # Step 1
        theta_1 = (theta_0 + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta_0)) % 1.0
        sim.step()
        assert sim.observe()[0] == pytest.approx(theta_1, abs=1e-14)

        # Step 2
        theta_2 = (theta_1 + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta_1)) % 1.0
        sim.step()
        assert sim.observe()[0] == pytest.approx(theta_2, abs=1e-14)

    def test_theta_stays_in_unit_interval(self):
        """All iterates should lie in [0, 1)."""
        sim = CircleMapSimulation(_make_config(omega=0.7, K=1.5, theta_0=0.9))
        sim.reset()
        for _ in range(1000):
            sim.step()
            theta = sim.observe()[0]
            assert 0.0 <= theta < 1.0, f"theta={theta} not in [0, 1)"

    def test_observe_returns_current_state(self):
        """observe() should return the same state as the last step()."""
        sim = CircleMapSimulation(_make_config())
        sim.reset()
        state = sim.step()
        np.testing.assert_array_equal(sim.observe(), state)

    def test_run_trajectory(self):
        """run() should return a TrajectoryData with correct shape."""
        sim = CircleMapSimulation(_make_config())
        traj = sim.run(50)
        assert traj.states.shape == (51, 1)  # 50 steps + initial
        assert traj.timestamps.shape == (51,)

    def test_step_count_increments(self):
        """Step count should increment with each step."""
        sim = CircleMapSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


class TestCircleMapK0PureRotation:
    """K=0: map reduces to pure rotation theta_{n+1} = (theta_n + Omega) mod 1."""

    def test_k0_pure_rotation_one_step(self):
        """At K=0, one step should just add Omega."""
        omega = 0.3
        sim = CircleMapSimulation(_make_config(omega=omega, K=0.0, theta_0=0.2))
        sim.reset()
        sim.step()
        assert sim.observe()[0] == pytest.approx(0.5, abs=1e-14)

    def test_k0_winding_number_equals_omega(self):
        """At K=0, winding number should exactly equal Omega."""
        for omega in [0.1, 0.25, 0.333, 0.5, 0.7, 0.9]:
            sim = CircleMapSimulation(_make_config(omega=omega, K=0.0))
            w = sim.compute_winding_number(n_transient=100, n_compute=500, K=0.0)
            assert w == pytest.approx(omega, abs=1e-6), (
                f"w={w} != Omega={omega} at K=0"
            )

    def test_k0_irrational_omega(self):
        """At K=0, irrational Omega should give irrational winding number."""
        omega = 1.0 / np.sqrt(2)  # Irrational
        sim = CircleMapSimulation(_make_config(omega=omega, K=0.0))
        w = sim.compute_winding_number(n_transient=100, n_compute=1000, K=0.0)
        assert w == pytest.approx(omega, abs=1e-5)


class TestCircleMapWindingNumber:
    """Tests for winding number computation."""

    def test_winding_number_range(self):
        """Winding number should be in [0, 1) for Omega in [0, 1)."""
        sim = CircleMapSimulation(_make_config())
        for omega in np.linspace(0.05, 0.95, 10):
            w = sim.compute_winding_number(omega=omega, K=1.0)
            assert 0.0 <= w <= 1.0, f"w={w} outside [0, 1] at Omega={omega}"

    def test_winding_number_at_omega_0(self):
        """At Omega=0, winding number should be 0 (fixed point mode-locking)."""
        sim = CircleMapSimulation(_make_config())
        w = sim.compute_winding_number(omega=0.0, K=1.0)
        assert abs(w) < 0.01, f"w={w} should be ~0 at Omega=0"

    def test_winding_number_at_omega_half(self):
        """At Omega=0.5, K=1, should mode-lock to w=1/2."""
        sim = CircleMapSimulation(_make_config())
        w = sim.compute_winding_number(omega=0.5, K=1.0, n_compute=5000)
        assert abs(w - 0.5) < 0.01, f"w={w} should be ~0.5 at Omega=0.5"

    def test_winding_number_monotone_at_k0(self):
        """At K=0, winding number should be strictly monotone in Omega."""
        omegas = np.linspace(0.01, 0.99, 30)
        sim = CircleMapSimulation(_make_config(K=0.0))
        ws = [sim.compute_winding_number(omega=om, K=0.0, n_compute=500) for om in omegas]
        diffs = np.diff(ws)
        assert np.all(diffs > 0), "w(Omega) should be strictly increasing at K=0"

    def test_winding_number_nondecreasing_at_k1(self):
        """At K=1, winding number should be non-decreasing in Omega."""
        omegas = np.linspace(0.0, 1.0, 50)
        sim = CircleMapSimulation(_make_config(K=1.0))
        ws = [sim.compute_winding_number(omega=om, K=1.0, n_compute=2000) for om in omegas]
        diffs = np.diff(ws)
        # Allow small numerical noise
        assert np.all(diffs >= -1e-3), "w(Omega) should be non-decreasing at K=1"


class TestCircleMapLyapunov:
    """Tests for Lyapunov exponent computation."""

    def test_lyapunov_at_k0_is_zero(self):
        """At K=0, derivative is 1 everywhere, so lambda = log(1) = 0."""
        sim = CircleMapSimulation(_make_config(K=0.0))
        lam = sim.compute_lyapunov(K=0.0, n_compute=5000)
        assert abs(lam) < 1e-10, f"Lyapunov={lam} should be 0 at K=0"

    def test_lyapunov_at_k_small_nonnegative(self):
        """For 0 < K < 1, Lyapunov should be <= 0 (no chaos)."""
        sim = CircleMapSimulation(_make_config())
        for K in [0.1, 0.3, 0.5, 0.8]:
            lam = sim.compute_lyapunov(K=K, omega=0.5, n_compute=5000)
            assert lam <= 0.1, f"Lyapunov={lam} should be <= 0 for K={K}<1"

    def test_lyapunov_positive_for_chaos(self):
        """For K > 1, some Omega values should give positive Lyapunov."""
        sim = CircleMapSimulation(_make_config())
        # At K=2.0, many Omega values produce chaos
        found_positive = False
        for omega in np.linspace(0.1, 0.9, 20):
            lam = sim.compute_lyapunov(K=2.0, omega=omega, n_compute=5000)
            if lam > 0.01:
                found_positive = True
                break
        assert found_positive, "Should find positive Lyapunov for K=2.0"

    def test_lyapunov_at_k0_versus_above_critical(self):
        """At K=0, Lyapunov is exactly 0. Above critical K, max Lyapunov > 0.

        For K > 1, the map is not a diffeomorphism and chaos is possible.
        We check that the maximum Lyapunov over several Omega values is
        positive for K=1.5 but zero for K=0.
        """
        sim = CircleMapSimulation(_make_config())
        omegas = np.linspace(0.05, 0.95, 20)

        max_k0 = max(
            sim.compute_lyapunov(K=0.0, omega=om, n_compute=2000) for om in omegas
        )
        max_k15 = max(
            sim.compute_lyapunov(K=1.5, omega=om, n_compute=3000) for om in omegas
        )
        assert abs(max_k0) < 1e-8, f"Max Lyapunov at K=0 should be ~0, got {max_k0}"
        assert max_k15 > 0.01, f"Max Lyapunov at K=1.5 should be positive, got {max_k15}"


class TestCircleMapModeLocking:
    """Tests for mode-locking (Arnold tongue) detection."""

    def test_devils_staircase_returns_data(self):
        """devils_staircase should return omega and winding_number arrays."""
        sim = CircleMapSimulation(_make_config())
        data = sim.devils_staircase(n_omega=50, K=1.0)
        assert "omega" in data
        assert "winding_number" in data
        assert len(data["omega"]) == 50
        assert len(data["winding_number"]) == 50

    def test_devils_staircase_has_plateaus(self):
        """At K=1, devil's staircase should show flat regions (mode-locking)."""
        sim = CircleMapSimulation(_make_config())
        data = sim.devils_staircase(n_omega=200, K=1.0)
        ws = data["winding_number"]

        # Count approximately constant regions (|dw| < epsilon)
        diffs = np.abs(np.diff(ws))
        n_flat = np.sum(diffs < 1e-5)
        # At K=1, most of the staircase should be flat (mode-locked)
        assert n_flat > 50, f"Only {n_flat} flat points -- expected plateaus"

    def test_detect_mode_locking_finds_tongues(self):
        """Mode-locking detection should find major tongues at K=1."""
        sim = CircleMapSimulation(_make_config())
        data = sim.devils_staircase(n_omega=500, K=1.0)
        tongues = sim.detect_mode_locking(
            data["omega"], data["winding_number"],
            tolerance=1e-3, max_q=6,
        )
        assert len(tongues) > 0, "Should detect at least one tongue"

        # The 0/1 and 1/2 tongues should always be present
        winding_values = [t["winding_number"] for t in tongues]
        assert 0.0 in winding_values, "Should detect w=0 tongue"
        assert 0.5 in winding_values, "Should detect w=1/2 tongue"

    def test_mode_locking_tongue_structure(self):
        """Each detected tongue should have valid structure."""
        sim = CircleMapSimulation(_make_config())
        data = sim.devils_staircase(n_omega=300, K=1.0)
        tongues = sim.detect_mode_locking(
            data["omega"], data["winding_number"],
            tolerance=1e-3, max_q=4,
        )
        for t in tongues:
            assert "p" in t and "q" in t
            assert "omega_center" in t
            assert "omega_width" in t
            assert t["q"] >= 1
            assert 0 <= t["p"] <= t["q"]
            assert t["omega_width"] >= 0.0


class TestCircleMapArnoldTongues:
    """Tests for Arnold tongue grid generation."""

    def test_arnold_tongue_data_shape(self):
        """Arnold tongue data should have correct grid shape."""
        sim = CircleMapSimulation(_make_config())
        data = sim.arnold_tongue_data(n_omega=20, n_K=10)
        assert data["omega_values"].shape == (20,)
        assert data["K_values"].shape == (10,)
        assert data["winding_numbers"].shape == (10, 20)

    def test_arnold_tongue_k0_row_equals_omega(self):
        """At K=0 (first row if K starts at 0), winding numbers should match Omega."""
        sim = CircleMapSimulation(_make_config())
        data = sim.arnold_tongue_data(
            omega_range=(0.05, 0.95), K_range=(0.0, 1.0), n_omega=20, n_K=5,
        )
        # K=0 is the first row
        k0_winding = data["winding_numbers"][0, :]
        omegas = data["omega_values"]
        np.testing.assert_allclose(k0_winding, omegas, atol=0.02)

    def test_arnold_tongue_winding_bounded(self):
        """All winding numbers should be in [0, 1]."""
        sim = CircleMapSimulation(_make_config())
        data = sim.arnold_tongue_data(n_omega=15, n_K=8)
        assert np.all(data["winding_numbers"] >= -0.01)
        assert np.all(data["winding_numbers"] <= 1.01)


class TestCircleMapDeterminism:
    """Tests for reproducibility."""

    def test_deterministic_iteration(self):
        """Same parameters should produce identical trajectories."""
        sim1 = CircleMapSimulation(_make_config(omega=0.3, K=0.8, theta_0=0.2))
        sim2 = CircleMapSimulation(_make_config(omega=0.3, K=0.8, theta_0=0.2))

        traj1 = sim1.run(100)
        traj2 = sim2.run(100)
        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_winding_number_reproducible(self):
        """Winding number computation should be deterministic."""
        sim = CircleMapSimulation(_make_config())
        w1 = sim.compute_winding_number(omega=0.3, K=1.0)
        w2 = sim.compute_winding_number(omega=0.3, K=1.0)
        assert w1 == w2

    def test_lyapunov_reproducible(self):
        """Lyapunov exponent should be deterministic."""
        sim = CircleMapSimulation(_make_config())
        l1 = sim.compute_lyapunov(omega=0.5, K=1.5)
        l2 = sim.compute_lyapunov(omega=0.5, K=1.5)
        assert l1 == l2


class TestCircleMapRediscovery:
    """Tests for the rediscovery module."""

    def test_devils_staircase_data_generation(self):
        from simulating_anything.rediscovery.circle_map import (
            generate_devils_staircase_data,
        )
        data = generate_devils_staircase_data(n_omega=30, K=1.0)
        assert len(data["omega"]) == 30
        assert len(data["winding_number"]) == 30

    def test_lyapunov_sweep_generation(self):
        from simulating_anything.rediscovery.circle_map import generate_lyapunov_sweep
        data = generate_lyapunov_sweep(n_omega=10, K=1.5)
        assert len(data["omega"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_lyapunov_K_sweep_generation(self):
        from simulating_anything.rediscovery.circle_map import generate_lyapunov_K_sweep
        data = generate_lyapunov_K_sweep(n_K=10, omega=0.5)
        assert len(data["K"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_arnold_tongue_data_generation(self):
        from simulating_anything.rediscovery.circle_map import (
            generate_arnold_tongue_data,
        )
        data = generate_arnold_tongue_data(n_omega=10, n_K=5)
        assert data["winding_numbers"].shape == (5, 10)
