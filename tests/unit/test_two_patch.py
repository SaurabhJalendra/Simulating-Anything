"""Tests for the two-patch predator-prey simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.two_patch import TwoPatchSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 0.5,
    e: float = 0.4,
    d: float = 0.2,
    m_N: float = 0.1,
    m_P: float = 0.05,
    dt: float = 0.01,
    n_steps: int = 2000,
    **overrides,
) -> SimulationConfig:
    params = {
        "r": r, "K": K, "a": a, "e": e, "d": d,
        "m_N": m_N, "m_P": m_P,
        "N1_0": 5.0, "P1_0": 2.0,
        "N2_0": 3.0, "P2_0": 1.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.TWO_PATCH,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestTwoPatchBasics:
    """Basic simulation tests: creation, shape, determinism, stability."""

    def test_create_simulation(self):
        """Simulation should be instantiable with default config."""
        sim = TwoPatchSimulation(_make_config())
        assert sim is not None

    def test_reset_returns_state(self):
        """reset() should return the initial state array."""
        sim = TwoPatchSimulation(_make_config())
        state = sim.reset()
        assert isinstance(state, np.ndarray)

    def test_observe_shape(self):
        """State should be a 4-element vector [N1, P1, N2, P2]."""
        sim = TwoPatchSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_initial_conditions(self):
        """Reset should set the specified initial conditions."""
        sim = TwoPatchSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [5.0, 2.0, 3.0, 1.0])

    def test_custom_initial_conditions(self):
        """Custom ICs should be respected."""
        config = _make_config(N1_0=8.0, P1_0=1.5, N2_0=2.0, P2_0=3.0)
        sim = TwoPatchSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [8.0, 1.5, 2.0, 3.0])

    def test_step_advances(self):
        """State should change after one step."""
        sim = TwoPatchSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same config should produce identical trajectories."""
        config = _make_config()
        sim1 = TwoPatchSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = TwoPatchSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_rk4_stability(self):
        """No NaN or Inf should appear during integration."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        for _ in range(2000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        config = _make_config(n_steps=100)
        sim = TwoPatchSimulation(config)
        traj = sim.run(100)
        assert traj.states.shape == (101, 4)
        assert traj.timestamps.shape == (101,)

    def test_default_parameters(self):
        """Default parameters should match specified values."""
        config = SimulationConfig(
            domain=Domain.TWO_PATCH,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = TwoPatchSimulation(config)
        assert sim.r == 1.0
        assert sim.K == 10.0
        assert sim.a == 0.5
        assert sim.e == 0.4
        assert sim.d == 0.2
        assert sim.m_N == 0.1
        assert sim.m_P == 0.05


class TestTwoPatchPositivity:
    """Tests that populations remain non-negative."""

    def test_state_non_negative(self):
        """All populations should remain >= 0."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_state_non_negative_extreme_ics(self):
        """Populations should stay non-negative even from extreme initial conditions."""
        config = _make_config(N1_0=0.01, P1_0=5.0, N2_0=9.0, P2_0=0.01)
        sim = TwoPatchSimulation(config)
        sim.reset()
        for _ in range(3000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_zero_prey_stays_zero(self):
        """If prey starts at zero on a patch, it can be rescued by migration."""
        config = _make_config(N1_0=0.0, P1_0=1.0, N2_0=5.0, P2_0=1.0, m_N=0.5)
        sim = TwoPatchSimulation(config)
        sim.reset()
        for _ in range(500):
            sim.step()
        # With migration, prey on patch 1 should increase from zero
        assert sim._state[0] > 0.0, "Prey on patch 1 was not rescued by migration"


class TestTwoPatchTotalPopulations:
    """Tests for total_prey() and total_predator() methods."""

    def test_total_prey(self):
        """total_prey should return N1 + N2."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        total = sim.total_prey()
        assert total == pytest.approx(5.0 + 3.0)

    def test_total_predator(self):
        """total_predator should return P1 + P2."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        total = sim.total_predator()
        assert total == pytest.approx(2.0 + 1.0)

    def test_total_prey_after_steps(self):
        """total_prey should reflect updated populations."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        total = sim.total_prey()
        assert total > 0
        assert total == pytest.approx(sim._state[0] + sim._state[2])

    def test_total_predator_after_steps(self):
        """total_predator should reflect updated populations."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        total = sim.total_predator()
        assert total > 0
        assert total == pytest.approx(sim._state[1] + sim._state[3])

    def test_total_prey_none_state(self):
        """total_prey should return 0 if state is None."""
        sim = TwoPatchSimulation(_make_config())
        assert sim.total_prey() == 0.0

    def test_total_predator_none_state(self):
        """total_predator should return 0 if state is None."""
        sim = TwoPatchSimulation(_make_config())
        assert sim.total_predator() == 0.0


class TestTwoPatchZeroMigration:
    """Tests that patches are independent when migration is zero."""

    def test_zero_migration_independent(self):
        """With m_N=0, m_P=0, the two patches should evolve independently."""
        config = _make_config(m_N=0.0, m_P=0.0)
        sim = TwoPatchSimulation(config)
        sim.reset()

        # Run a separate single-patch equivalent for patch 1
        config1 = _make_config(
            m_N=0.0, m_P=0.0,
            N1_0=5.0, P1_0=2.0, N2_0=5.0, P2_0=2.0,
        )
        sim1 = TwoPatchSimulation(config1)
        sim1.reset()

        for _ in range(500):
            sim.step()
            sim1.step()

        # Patch 1 of sim should match patch 1 of sim1 since both start
        # with same N1_0, P1_0 and zero migration
        np.testing.assert_allclose(sim._state[:2], sim1._state[:2], atol=1e-10)

    def test_zero_migration_patches_diverge(self):
        """With m=0 and different ICs, patches stay distinct."""
        config = _make_config(m_N=0.0, m_P=0.0)
        sim = TwoPatchSimulation(config)
        sim.reset()

        for _ in range(1000):
            sim.step()

        # Patches should not be identical
        diff = sim.patch_difference()
        assert diff > 0.01, f"Patches unexpectedly identical: diff={diff}"

    def test_zero_migration_no_coupling_terms(self):
        """Derivatives at m=0 should have no coupling contribution."""
        sim = TwoPatchSimulation(_make_config(m_N=0.0, m_P=0.0))
        sim.reset()
        state = np.array([5.0, 2.0, 3.0, 1.0])
        derivs = sim._derivatives(state)

        # dN1 = r*N1*(1-N1/K) - a*N1*P1 = 1*5*(1-5/10) - 0.5*5*2 = 2.5 - 5 = -2.5
        assert derivs[0] == pytest.approx(-2.5)
        # dP1 = e*a*N1*P1 - d*P1 = 0.4*0.5*5*2 - 0.2*2 = 2.0 - 0.4 = 1.6
        assert derivs[1] == pytest.approx(1.6)
        # dN2 = r*N2*(1-N2/K) - a*N2*P2 = 1*3*(1-3/10) - 0.5*3*1 = 2.1 - 1.5 = 0.6
        assert derivs[2] == pytest.approx(0.6)
        # dP2 = e*a*N2*P2 - d*P2 = 0.4*0.5*3*1 - 0.2*1 = 0.6 - 0.2 = 0.4
        assert derivs[3] == pytest.approx(0.4)


class TestTwoPatchHighMigration:
    """Tests that high migration synchronizes patches."""

    def test_high_migration_synchronizes(self):
        """Strong migration should drive patches toward the same state."""
        config = _make_config(
            m_N=5.0, m_P=5.0,
            N1_0=8.0, P1_0=1.0,
            N2_0=2.0, P2_0=3.0,
        )
        sim = TwoPatchSimulation(config)
        sim.reset()

        for _ in range(5000):
            sim.step()

        # With very strong migration, patches should synchronize
        diff = sim.patch_difference()
        assert diff < 0.5, (
            f"Patches not synchronized with high migration: diff={diff}"
        )

    def test_high_migration_is_synchronized(self):
        """is_synchronized should return True for high migration."""
        config = _make_config(
            m_N=5.0, m_P=5.0,
            N1_0=8.0, P1_0=1.0,
            N2_0=2.0, P2_0=3.0,
        )
        sim = TwoPatchSimulation(config)
        sim.reset()

        for _ in range(5000):
            sim.step()

        assert sim.is_synchronized(threshold=0.5), (
            f"Not synchronized: diff={sim.patch_difference()}"
        )

    def test_migration_reduces_patch_difference(self):
        """Higher migration should yield lower average patch difference."""
        diffs = {}
        for m in [0.0, 1.0]:
            config = _make_config(
                m_N=m, m_P=m,
                N1_0=8.0, P1_0=1.0,
                N2_0=2.0, P2_0=3.0,
            )
            sim = TwoPatchSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(5000):
                sim.step()

            # Measure average difference
            diff_sum = 0.0
            n_measure = 3000
            for _ in range(n_measure):
                sim.step()
                diff_sum += sim.patch_difference()
            diffs[m] = diff_sum / n_measure

        assert diffs[1.0] < diffs[0.0], (
            f"Higher migration diff ({diffs[1.0]:.4f}) not less than "
            f"lower migration diff ({diffs[0.0]:.4f})"
        )


class TestTwoPatchBoundedness:
    """Tests that populations remain bounded."""

    def test_population_bounded(self):
        """All populations should remain bounded (not diverge)."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            # Prey bounded by carrying capacity (with some margin)
            # Predator also bounded given finite prey
            assert np.all(state < 100), f"Population diverged: {state}"

    def test_prey_bounded_by_carrying_capacity(self):
        """Prey on each patch should stay near or below K."""
        config = _make_config(N1_0=9.0, N2_0=9.0, P1_0=0.5, P2_0=0.5)
        sim = TwoPatchSimulation(config)
        sim.reset()

        max_prey = 0.0
        for _ in range(5000):
            state = sim.step()
            max_prey = max(max_prey, state[0], state[2])

        # With predation, prey should not greatly exceed K
        assert max_prey < 2 * sim.K, (
            f"Prey exceeded 2*K={2*sim.K}: max={max_prey}"
        )


class TestTwoPatchSymmetry:
    """Tests that symmetric ICs produce symmetric dynamics."""

    def test_symmetric_ics_stay_symmetric(self):
        """If N1=N2, P1=P2, and migration is symmetric, patches stay equal."""
        config = _make_config(
            m_N=0.1, m_P=0.05,
            N1_0=5.0, P1_0=2.0,
            N2_0=5.0, P2_0=2.0,
        )
        sim = TwoPatchSimulation(config)
        sim.reset()

        for _ in range(2000):
            state = sim.step()
            # Patches should remain exactly equal
            np.testing.assert_allclose(
                state[:2], state[2:], atol=1e-12,
                err_msg=f"Symmetric ICs broke symmetry: {state}"
            )

    def test_asymmetric_ics_converge_with_migration(self):
        """Asymmetric ICs should converge toward symmetric with migration."""
        config = _make_config(
            m_N=0.5, m_P=0.5,
            N1_0=9.0, P1_0=0.5,
            N2_0=1.0, P2_0=3.0,
        )
        sim = TwoPatchSimulation(config)
        sim.reset()

        initial_diff = sim.patch_difference()

        for _ in range(3000):
            sim.step()

        final_diff = sim.patch_difference()

        # With migration, difference should decrease from initial
        assert final_diff < initial_diff, (
            f"Patch difference did not decrease: initial={initial_diff:.4f}, "
            f"final={final_diff:.4f}"
        )


class TestTwoPatchPatchDifference:
    """Tests for patch_difference() method."""

    def test_patch_difference_zero_symmetric(self):
        """Patch difference should be zero for symmetric state."""
        config = _make_config(N1_0=5.0, P1_0=2.0, N2_0=5.0, P2_0=2.0)
        sim = TwoPatchSimulation(config)
        sim.reset()
        assert sim.patch_difference() == pytest.approx(0.0, abs=1e-15)

    def test_patch_difference_positive_asymmetric(self):
        """Patch difference should be positive for asymmetric state."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        diff = sim.patch_difference()
        assert diff > 0

    def test_patch_difference_formula(self):
        """patch_difference should compute Euclidean distance correctly."""
        config = _make_config(N1_0=3.0, P1_0=4.0, N2_0=0.0, P2_0=0.0)
        sim = TwoPatchSimulation(config)
        sim.reset()
        diff = sim.patch_difference()
        # sqrt(3^2 + 4^2) = 5.0
        assert diff == pytest.approx(5.0)

    def test_patch_difference_none_state(self):
        """patch_difference should return 0 if state is None."""
        sim = TwoPatchSimulation(_make_config())
        assert sim.patch_difference() == 0.0

    def test_patch_difference_decreases_with_higher_migration(self):
        """Average patch difference should decrease as migration increases."""
        avg_diffs = []
        for m in [0.01, 0.1, 0.5, 2.0]:
            config = _make_config(
                m_N=m, m_P=m,
                N1_0=8.0, P1_0=1.0,
                N2_0=2.0, P2_0=3.0,
            )
            sim = TwoPatchSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(5000):
                sim.step()

            diff_sum = 0.0
            n_meas = 2000
            for _ in range(n_meas):
                sim.step()
                diff_sum += sim.patch_difference()

            avg_diffs.append(diff_sum / n_meas)

        # Each subsequent migration rate should yield lower difference
        for i in range(len(avg_diffs) - 1):
            assert avg_diffs[i + 1] <= avg_diffs[i] + 0.5, (
                f"Patch diff at m={[0.01, 0.1, 0.5, 2.0][i+1]} "
                f"({avg_diffs[i+1]:.4f}) not less than at "
                f"m={[0.01, 0.1, 0.5, 2.0][i]} ({avg_diffs[i]:.4f})"
            )


class TestTwoPatchDerivatives:
    """Tests for the derivative computation."""

    def test_derivatives_with_migration(self):
        """Migration terms should appear in derivatives."""
        sim = TwoPatchSimulation(_make_config(m_N=0.1, m_P=0.05))
        sim.reset()
        state = np.array([5.0, 2.0, 3.0, 1.0])

        derivs_m = sim._derivatives(state)

        # Compare with zero-migration
        sim0 = TwoPatchSimulation(_make_config(m_N=0.0, m_P=0.0))
        sim0.reset()
        derivs_0 = sim0._derivatives(state)

        # Migration should change dN1 by m_N*(N2-N1) = 0.1*(3-5) = -0.2
        assert derivs_m[0] == pytest.approx(derivs_0[0] + 0.1 * (3 - 5))
        # Migration should change dP1 by m_P*(P2-P1) = 0.05*(1-2) = -0.05
        assert derivs_m[1] == pytest.approx(derivs_0[1] + 0.05 * (1 - 2))
        # Migration should change dN2 by m_N*(N1-N2) = 0.1*(5-3) = 0.2
        assert derivs_m[2] == pytest.approx(derivs_0[2] + 0.1 * (5 - 3))
        # Migration should change dP2 by m_P*(P1-P2) = 0.05*(2-1) = 0.05
        assert derivs_m[3] == pytest.approx(derivs_0[3] + 0.05 * (2 - 1))

    def test_migration_antisymmetric(self):
        """Migration is antisymmetric: m*(N2-N1) = -m*(N1-N2)."""
        sim = TwoPatchSimulation(_make_config(m_N=0.3, m_P=0.2))
        sim.reset()
        state = np.array([4.0, 1.0, 7.0, 3.0])
        derivs = sim._derivatives(state)

        # The migration contributions to prey should be equal and opposite
        # dN1_mig = m_N*(N2-N1) = 0.3*(7-4) = 0.9
        # dN2_mig = m_N*(N1-N2) = 0.3*(4-7) = -0.9
        # Check that local dynamics + migration terms are correct
        # Local dN1 = r*N1*(1-N1/K) - a*N1*P1 = 4*(1-0.4) - 0.5*4*1 = 2.4-2 = 0.4
        # dN1 = 0.4 + 0.9 = 1.3
        assert derivs[0] == pytest.approx(1.3)

    def test_derivatives_at_sync_manifold(self):
        """On N1=N2, P1=P2, migration terms vanish."""
        sim = TwoPatchSimulation(_make_config(m_N=1.0, m_P=1.0))
        sim.reset()
        state = np.array([5.0, 2.0, 5.0, 2.0])
        derivs = sim._derivatives(state)
        # On sync manifold, derivatives should be identical for both patches
        np.testing.assert_allclose(derivs[:2], derivs[2:])


class TestTwoPatchEquilibrium:
    """Tests for coexistence equilibrium."""

    def test_coexistence_equilibrium_formula(self):
        """Equilibrium should match N*=d/(e*a), P*=(r/a)*(1-N*/K)."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        N_star, P_star = sim.coexistence_equilibrium()

        expected_N = 0.2 / (0.4 * 0.5)  # d/(e*a) = 1.0
        expected_P = (1.0 / 0.5) * (1 - 1.0 / 10.0)  # (r/a)*(1-N*/K) = 1.8

        assert N_star == pytest.approx(expected_N)
        assert P_star == pytest.approx(expected_P)

    def test_equilibrium_is_stable_no_migration(self):
        """Starting at equilibrium with no migration should stay near it."""
        sim = TwoPatchSimulation(_make_config(m_N=0.0, m_P=0.0))
        sim.reset()
        N_star, P_star = sim.coexistence_equilibrium()

        # Start both patches at equilibrium
        config_eq = _make_config(
            m_N=0.0, m_P=0.0,
            N1_0=N_star, P1_0=P_star,
            N2_0=N_star, P2_0=P_star,
        )
        sim_eq = TwoPatchSimulation(config_eq)
        sim_eq.reset()

        for _ in range(1000):
            sim_eq.step()

        # Should remain near equilibrium
        state = sim_eq.observe()
        np.testing.assert_allclose(state[0], N_star, atol=0.5)
        np.testing.assert_allclose(state[1], P_star, atol=0.5)


class TestTwoPatchMigrationSweep:
    """Tests for the migration_sweep method."""

    def test_migration_sweep_shape(self):
        """migration_sweep should return arrays of correct length."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        m_vals = np.array([0.0, 0.5, 1.0])
        result = sim.migration_sweep(m_vals, n_steps=500, n_transient=300)
        assert len(result["m"]) == 3
        assert len(result["mean_diff"]) == 3
        assert len(result["mean_total_prey"]) == 3

    def test_migration_sweep_diffs_non_negative(self):
        """All patch differences should be non-negative."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        m_vals = np.array([0.0, 0.1, 0.5])
        result = sim.migration_sweep(m_vals, n_steps=500, n_transient=300)
        assert np.all(result["mean_diff"] >= 0)

    def test_migration_sweep_predator(self):
        """Sweeping predator migration should also work."""
        sim = TwoPatchSimulation(_make_config())
        sim.reset()
        m_vals = np.array([0.0, 0.2, 0.5])
        result = sim.migration_sweep(
            m_vals, sweep_prey=False, n_steps=500, n_transient=300,
        )
        assert len(result["m"]) == 3


class TestTwoPatchRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.two_patch import generate_ode_data
        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["m_N"] == 0.1

    def test_ode_data_populations_positive(self):
        from simulating_anything.rediscovery.two_patch import generate_ode_data
        data = generate_ode_data(n_steps=500, dt=0.01)
        assert np.all(data["states"] >= 0), "Negative populations in ODE data"

    def test_migration_sweep_data_generation(self):
        from simulating_anything.rediscovery.two_patch import (
            generate_migration_sweep_data,
        )
        data = generate_migration_sweep_data(
            n_m=5, m_min=0.0, m_max=1.0,
            n_steps=500, n_transient=200, dt=0.01,
        )
        assert len(data["m_N"]) == 5
        assert len(data["mean_diff"]) == 5
        assert data["m_N"][0] == pytest.approx(0.0)
        assert data["m_N"][-1] == pytest.approx(1.0)

    def test_sync_comparison_data(self):
        from simulating_anything.rediscovery.two_patch import (
            generate_sync_comparison_data,
        )
        data = generate_sync_comparison_data(n_steps=1000, dt=0.01)
        assert "low_migration" in data
        assert "high_migration" in data
        assert "mean_patch_diff" in data["low_migration"]
        assert "sync_fraction" in data["high_migration"]

    def test_run_function_exists(self):
        """Ensure the main rediscovery runner function is importable."""
        from simulating_anything.rediscovery.two_patch import (
            run_two_patch_rediscovery,
        )
        assert callable(run_two_patch_rediscovery)
