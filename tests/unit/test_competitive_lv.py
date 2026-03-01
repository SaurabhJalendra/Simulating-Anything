"""Tests for the Competitive Lotka-Volterra (4-species) simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.competitive_lv import CompetitiveLVSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    n_species: int = 4,
    r: list[float] | None = None,
    K: list[float] | None = None,
    alpha: list[list[float]] | None = None,
    N_init: list[float] | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Helper to build a SimulationConfig for the competitive LV model."""
    if r is None:
        r = [1.0, 0.72, 1.53, 1.27]
    if K is None:
        K = [100.0, 100.0, 100.0, 100.0]
    if alpha is None:
        alpha = [
            [1.0, 0.5, 0.4, 0.3],
            [0.4, 1.0, 0.6, 0.3],
            [0.3, 0.4, 1.0, 0.5],
            [0.5, 0.3, 0.4, 1.0],
        ]

    params: dict[str, float] = {"n_species": float(n_species)}
    for i in range(n_species):
        params[f"r_{i}"] = r[i]
        params[f"K_{i}"] = K[i]
        for j in range(n_species):
            params[f"alpha_{i}_{j}"] = alpha[i][j]
        if N_init is not None:
            params[f"N_0_{i}"] = N_init[i]

    return SimulationConfig(
        domain=Domain.COMPETITIVE_LV,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSingleSpeciesLogistic:
    """One species alone should follow logistic growth."""

    def test_single_species_logistic(self):
        """A single species with no competitors follows dN/dt = rN(1-N/K)."""
        config = _make_config(
            n_species=1,
            r=[1.0],
            K=[100.0],
            alpha=[[1.0]],
            N_init=[10.0],
            dt=0.01,
            n_steps=5000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        N_final = sim.observe()[0]
        # Should converge to K=100
        assert N_final == pytest.approx(100.0, rel=0.01)


class TestCarryingCapacity:
    def test_carrying_capacity(self):
        """A single species starting above K should decrease toward K."""
        config = _make_config(
            n_species=1,
            r=[1.0],
            K=[50.0],
            alpha=[[1.0]],
            N_init=[80.0],
            dt=0.01,
            n_steps=5000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        N_final = sim.observe()[0]
        assert N_final == pytest.approx(50.0, rel=0.01)


class TestTwoSpeciesExclusion:
    def test_two_species_exclusion(self):
        """Strong inter-specific competition leads to exclusion of one species."""
        # alpha_12 = alpha_21 = 1.5 (stronger than intra-specific = 1)
        config = _make_config(
            n_species=2,
            r=[1.0, 1.0],
            K=[100.0, 100.0],
            alpha=[[1.0, 1.5], [1.5, 1.0]],
            N_init=[51.0, 49.0],  # Slight asymmetry to break tie
            dt=0.01,
            n_steps=50000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        # One species should dominate, the other near zero
        assert sim.n_surviving(threshold=1.0) <= 1


class TestTwoSpeciesCoexistence:
    def test_two_species_coexistence(self):
        """Weak inter-specific competition allows both species to coexist."""
        # alpha_12 = alpha_21 = 0.3 (weaker than intra-specific = 1)
        config = _make_config(
            n_species=2,
            r=[1.0, 1.0],
            K=[100.0, 100.0],
            alpha=[[1.0, 0.3], [0.3, 1.0]],
            N_init=[50.0, 50.0],
            dt=0.01,
            n_steps=20000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
        pops = sim.observe()
        # Both species should persist
        assert pops[0] > 10.0
        assert pops[1] > 10.0


class TestFourSpeciesCoexistence:
    def test_four_species_coexistence(self):
        """Default parameters should allow all 4 species to persist."""
        config = _make_config(dt=0.01, n_steps=80000)
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(80000):
            sim.step()
        pops = sim.observe()
        # All 4 species should be present (above threshold)
        assert sim.n_surviving(threshold=1.0) == 4
        # Total population bounded
        assert np.sum(pops) < 500.0


class TestNonNegative:
    def test_non_negative(self):
        """Populations must stay >= 0 at all times."""
        config = _make_config(
            N_init=[1.0, 1.0, 1.0, 1.0],  # Start small
            dt=0.01,
            n_steps=5000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative population: {state}"


class TestTotalBounded:
    def test_total_bounded(self):
        """Total population should stay bounded (no divergence)."""
        config = _make_config(dt=0.01, n_steps=10000)
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
            total = np.sum(sim.observe())
            assert total < 1e6, f"Population diverged: total={total}"


class TestEquilibriumComputation:
    def test_equilibrium_computation(self):
        """N* = alpha^{-1} @ K should satisfy the equilibrium equations."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        N_star = sim.equilibrium_point()

        # At equilibrium: alpha @ N* = K
        product = sim.alpha @ N_star
        np.testing.assert_allclose(product, sim.K, rtol=1e-10)

    def test_equilibrium_all_positive(self):
        """Default parameters should give a feasible (all positive) equilibrium."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        N_star = sim.equilibrium_point()
        assert np.all(N_star > 0), f"Non-feasible equilibrium: {N_star}"


class TestCommunityMatrixShape:
    def test_community_matrix_shape(self):
        """Community matrix should be n_species x n_species."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        J = sim.community_matrix()
        assert J.shape == (4, 4)


class TestStabilityEigenvalues:
    def test_stability_eigenvalues_at_stable_equilibrium(self):
        """All eigenvalue real parts should be negative at stable coexistence."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        eigs = sim.stability_eigenvalues()
        assert len(eigs) == 4
        # For default (coexistence) parameters, all eigenvalues should have
        # negative real parts
        assert np.all(np.real(eigs) < 0), f"Unstable eigenvalues: {eigs}"


class TestCompetitiveExclusion:
    def test_competitive_exclusion(self):
        """Increasing competition should reduce diversity."""
        diversities = []
        for alpha_off in [0.3, 0.8, 1.3]:
            alpha = [
                [1.0, alpha_off, alpha_off, alpha_off],
                [alpha_off, 1.0, alpha_off, alpha_off],
                [alpha_off, alpha_off, 1.0, alpha_off],
                [alpha_off, alpha_off, alpha_off, 1.0],
            ]
            config = _make_config(
                alpha=alpha,
                N_init=[51.0, 49.0, 48.0, 52.0],  # Break symmetry
                dt=0.01,
                n_steps=80000,
            )
            sim = CompetitiveLVSimulation(config)
            sim.reset()
            for _ in range(80000):
                sim.step()
            diversities.append(sim.diversity_index())

        # Diversity should generally decrease or stay same as competition increases
        # At minimum, the strongest competition should have lower diversity
        assert diversities[-1] <= diversities[0] + 0.1


class TestDiversityIndex:
    def test_diversity_index_range(self):
        """Shannon diversity should be in [0, ln(n_species)]."""
        config = _make_config(dt=0.01, n_steps=50000)
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        H = sim.diversity_index()
        max_H = np.log(4)  # Maximum for 4 species
        assert 0.0 <= H <= max_H + 0.01, f"Diversity out of range: {H}"

    def test_diversity_zero_when_one_species(self):
        """Diversity should be ~0 when only one species survives."""
        config = _make_config(
            n_species=1,
            r=[1.0],
            K=[100.0],
            alpha=[[1.0]],
            N_init=[50.0],
            dt=0.01,
            n_steps=1000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(1000):
            sim.step()
        H = sim.diversity_index()
        # With only 1 species, p=1, H = -1*ln(1) = 0
        assert H == pytest.approx(0.0, abs=1e-10)


class TestRK4Stability:
    def test_rk4_stability(self):
        """No NaN or Inf after many steps."""
        config = _make_config(dt=0.01, n_steps=50000)
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN/Inf in state: {state}"


class TestDeterministic:
    def test_deterministic(self):
        """Same parameters should produce identical trajectories."""
        config = _make_config(dt=0.01, n_steps=500)
        sim1 = CompetitiveLVSimulation(config)
        sim2 = CompetitiveLVSimulation(config)

        sim1.reset()
        sim2.reset()

        states1 = []
        states2 = []
        for _ in range(500):
            states1.append(sim1.step().copy())
            states2.append(sim2.step().copy())

        np.testing.assert_array_equal(np.array(states1), np.array(states2))


class TestObserveShape:
    def test_observe_shape(self):
        """observe() should return an array of length n_species."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_observe_shape_custom_n(self):
        """observe() shape should match n_species for non-default n."""
        config = _make_config(
            n_species=3,
            r=[1.0, 1.0, 1.0],
            K=[100.0, 100.0, 100.0],
            alpha=[[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]],
            N_init=[50.0, 50.0, 50.0],
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        assert sim.observe().shape == (3,)


class TestResetState:
    def test_reset_state(self):
        """After reset, state should be the initial conditions."""
        config = _make_config(N_init=[10.0, 20.0, 30.0, 40.0])
        sim = CompetitiveLVSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [10.0, 20.0, 30.0, 40.0])

    def test_reset_default_k_over_2(self):
        """Default initial conditions should be K/2."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [50.0, 50.0, 50.0, 50.0])


class TestStepAdvances:
    def test_step_advances(self):
        """State should change after a step."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)


class TestSymmetry:
    def test_symmetry(self):
        """Equal r, K, and symmetric alpha should give equal equilibria."""
        alpha_sym = [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.5],
            [0.5, 0.5, 0.5, 1.0],
        ]
        config = _make_config(
            r=[1.0, 1.0, 1.0, 1.0],
            K=[100.0, 100.0, 100.0, 100.0],
            alpha=alpha_sym,
            N_init=[50.0, 50.0, 50.0, 50.0],
            dt=0.01,
            n_steps=50000,
        )
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        pops = sim.observe()
        # All populations should be equal (by symmetry)
        np.testing.assert_allclose(pops, pops[0], rtol=0.01)


class TestRediscoveryData:
    def test_rediscovery_data(self):
        """generate_coexistence_trajectory should return valid data."""
        from simulating_anything.rediscovery.competitive_lv import (
            generate_coexistence_trajectory,
        )
        data = generate_coexistence_trajectory(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["N_star_analytical"].shape == (4,)
        assert data["n_species"] == 4


class TestExclusionSweep:
    def test_exclusion_sweep(self):
        """generate_exclusion_sweep_data should produce valid sweep results."""
        from simulating_anything.rediscovery.competitive_lv import (
            generate_exclusion_sweep_data,
        )
        data = generate_exclusion_sweep_data(n_alpha=5, n_steps=5000, dt=0.01)
        assert len(data["alpha_12"]) == 5
        assert len(data["n_surviving"]) == 5
        assert len(data["diversity"]) == 5
        assert data["final_populations"].shape == (5, 4)


class TestIsStableCoexistence:
    def test_is_stable_coexistence_default(self):
        """Default parameters should give stable coexistence."""
        config = _make_config()
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        assert sim.is_stable_coexistence()

    def test_is_unstable_strong_competition(self):
        """Very strong competition should not give stable coexistence."""
        strong_alpha = [
            [1.0, 2.0, 2.0, 2.0],
            [2.0, 1.0, 2.0, 2.0],
            [2.0, 2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0, 1.0],
        ]
        config = _make_config(alpha=strong_alpha)
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        # With alpha_ij > 1 for all i!=j, the interior equilibrium
        # is not feasible (negative N*) so this should not be stable
        assert not sim.is_stable_coexistence()


class TestNSurviving:
    def test_n_surviving_all(self):
        """All species present initially should register as surviving."""
        config = _make_config(N_init=[50.0, 50.0, 50.0, 50.0])
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        assert sim.n_surviving() == 4

    def test_n_surviving_partial(self):
        """Species below threshold should not be counted."""
        config = _make_config(N_init=[50.0, 0.0001, 50.0, 0.0001])
        sim = CompetitiveLVSimulation(config)
        sim.reset()
        assert sim.n_surviving(threshold=1e-3) == 2
