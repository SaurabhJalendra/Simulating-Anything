"""Tests for the Ratio-Dependent (Arditi-Ginzburg) predator-prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.ratio_dependent import RatioDependentSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 100.0,
    a: float = 0.5,
    b: float = 1.0,
    e: float = 0.5,
    d: float = 0.2,
    N_0: float = 50.0,
    P_0: float = 10.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.RATIO_DEPENDENT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "e": e, "d": d,
            "N_0": N_0, "P_0": P_0,
        },
    )


# ---------------------------------------------------------------------------
# Creation and initialization
# ---------------------------------------------------------------------------

class TestRatioDependentCreation:
    def test_creation_default_params(self):
        sim = RatioDependentSimulation(_make_config())
        assert sim.r == 1.0
        assert sim.K == 100.0
        assert sim.a == 0.5
        assert sim.b == 1.0
        assert sim.e == 0.5
        assert sim.d == 0.2

    def test_creation_custom_params(self):
        sim = RatioDependentSimulation(_make_config(r=2.0, K=200.0, e=0.8))
        assert sim.r == 2.0
        assert sim.K == 200.0
        assert sim.e == 0.8

    def test_initial_state_shape(self):
        sim = RatioDependentSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = RatioDependentSimulation(_make_config(N_0=60.0, P_0=15.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [60.0, 15.0])

    def test_observe_shape(self):
        sim = RatioDependentSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_observe_equals_reset(self):
        sim = RatioDependentSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_allclose(obs, state)

    def test_state_dtype_float64(self):
        sim = RatioDependentSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestRatioDependentProperties:
    def test_prey_population(self):
        sim = RatioDependentSimulation(_make_config(N_0=60.0, P_0=15.0))
        sim.reset()
        assert sim.prey_population == pytest.approx(60.0)

    def test_predator_population(self):
        sim = RatioDependentSimulation(_make_config(N_0=60.0, P_0=15.0))
        sim.reset()
        assert sim.predator_population == pytest.approx(15.0)

    def test_total_population(self):
        sim = RatioDependentSimulation(_make_config(N_0=60.0, P_0=15.0))
        sim.reset()
        assert sim.total_population == pytest.approx(75.0)

    def test_properties_before_reset(self):
        sim = RatioDependentSimulation(_make_config())
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0
        assert sim.total_population == 0.0


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

class TestRatioDependentEquilibrium:
    def test_coexistence_equilibrium_exists(self):
        """With default params, coexistence equilibrium should exist."""
        sim = RatioDependentSimulation(_make_config())
        N_star, P_star = sim.coexistence_equilibrium()
        assert N_star > 0
        assert P_star > 0

    def test_coexistence_equilibrium_N_star(self):
        """Verify N* = K*(1 - a*(1-q)/(b*r)) where q = d/(e*a)."""
        sim = RatioDependentSimulation(_make_config(
            r=1.0, K=100.0, a=0.5, b=1.0, e=0.5, d=0.2,
        ))
        N_star, _ = sim.coexistence_equilibrium()
        # q = 0.2/(0.5*0.5) = 0.8, 1-q = 0.2
        # N* = 100*(1 - 0.5*0.2/(1.0*1.0)) = 100*0.9 = 90
        assert N_star == pytest.approx(90.0, rel=1e-10)

    def test_coexistence_equilibrium_P_star(self):
        """Verify P* = N*(1-q)/(q*b) where q = d/(e*a)."""
        sim = RatioDependentSimulation(_make_config(
            r=1.0, K=100.0, a=0.5, b=1.0, e=0.5, d=0.2,
        ))
        N_star, P_star = sim.coexistence_equilibrium()
        # P* = 90*0.2/(0.8*1.0) = 90*0.25 = 22.5
        assert P_star == pytest.approx(22.5, rel=1e-10)

    def test_derivatives_zero_at_equilibrium(self):
        """At coexistence equilibrium, derivatives should be zero."""
        sim = RatioDependentSimulation(_make_config())
        N_star, P_star = sim.coexistence_equilibrium()
        state = np.array([N_star, P_star])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_no_coexistence_when_predator_unsustainable(self):
        """When e*a <= d, no coexistence equilibrium exists."""
        # e*a = 0.01*0.5 = 0.005 < d = 0.2
        sim = RatioDependentSimulation(_make_config(e=0.01))
        with pytest.raises(ValueError, match="cannot sustain"):
            sim.coexistence_equilibrium()

    def test_no_coexistence_when_predation_too_high(self):
        """When a*(1-q)/(b*r) >= 1, N* <= 0 so no coexistence."""
        # a=2.0, d=0.1, e=0.5: q=0.1, a*(1-q)/(b*r) = 2.0*0.9/1.0 = 1.8 > 1
        sim = RatioDependentSimulation(_make_config(a=2.0, d=0.1, e=0.5))
        with pytest.raises(ValueError, match="N\\*"):
            sim.coexistence_equilibrium()

    def test_equilibrium_scales_with_K(self):
        """N* should scale linearly with K."""
        sim1 = RatioDependentSimulation(_make_config(K=100.0))
        sim2 = RatioDependentSimulation(_make_config(K=200.0))
        N1, _ = sim1.coexistence_equilibrium()
        N2, _ = sim2.coexistence_equilibrium()
        assert N2 == pytest.approx(2 * N1, rel=1e-10)


# ---------------------------------------------------------------------------
# Functional response
# ---------------------------------------------------------------------------

class TestFunctionalResponse:
    def test_functional_response_basic(self):
        """f(N, P) = a*N/(N + b*P)."""
        sim = RatioDependentSimulation(_make_config(a=0.5, b=1.0))
        # f(10, 5) = 0.5*10/(10 + 1.0*5) = 5/15 = 1/3
        assert sim.functional_response(10.0, 5.0) == pytest.approx(1.0 / 3.0)

    def test_functional_response_zero_predator(self):
        """With P=0, f(N, 0) = a*N/N = a."""
        sim = RatioDependentSimulation(_make_config(a=0.5))
        assert sim.functional_response(10.0, 0.0) == pytest.approx(0.5)

    def test_functional_response_zero_both(self):
        """f(0, 0) should return 0 (degenerate case)."""
        sim = RatioDependentSimulation(_make_config())
        assert sim.functional_response(0.0, 0.0) == 0.0

    def test_functional_response_saturates(self):
        """As N/P -> inf, f -> a (maximum predation rate)."""
        sim = RatioDependentSimulation(_make_config(a=0.5, b=1.0))
        assert sim.functional_response(1e6, 1.0) == pytest.approx(0.5, rel=1e-4)

    def test_functional_response_depends_on_ratio(self):
        """Doubling both N and P should give the same functional response."""
        sim = RatioDependentSimulation(_make_config(a=0.5, b=1.0))
        f1 = sim.functional_response(10.0, 5.0)
        f2 = sim.functional_response(20.0, 10.0)
        assert f1 == pytest.approx(f2)


# ---------------------------------------------------------------------------
# Stability (no paradox of enrichment)
# ---------------------------------------------------------------------------

class TestNoParadoxOfEnrichment:
    def test_is_stable_default_params(self):
        """Equilibrium should be stable with default parameters."""
        sim = RatioDependentSimulation(_make_config())
        assert sim.is_stable()

    def test_is_stable_large_K(self):
        """Equilibrium should remain stable even for very large K."""
        sim = RatioDependentSimulation(_make_config(K=1000.0))
        assert sim.is_stable()

    def test_is_stable_very_large_K(self):
        """Equilibrium should remain stable for K=10000."""
        sim = RatioDependentSimulation(_make_config(K=10000.0))
        assert sim.is_stable()

    def test_not_stable_when_no_equilibrium(self):
        """is_stable returns False when no equilibrium exists."""
        sim = RatioDependentSimulation(_make_config(e=0.01))
        assert not sim.is_stable()

    def test_no_limit_cycle_at_large_K(self):
        """With K=500, trajectory should converge to equilibrium, not oscillate."""
        sim = RatioDependentSimulation(_make_config(
            K=500.0, dt=0.01, n_steps=30000, N_0=50.0, P_0=10.0,
        ))
        sim.reset()

        states = []
        for _ in range(30000):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)
        # Check second half for convergence
        half = len(trajectory) // 2
        N_half = trajectory[half:, 0]
        amplitude = np.max(N_half) - np.min(N_half)
        # N* = 500*0.9 = 450. Amplitude should be very small (converged).
        N_eq = 500.0 * 0.9
        assert amplitude / N_eq < 0.05, (
            f"Oscillation amplitude {amplitude:.2f} too large for K=500 "
            f"(relative: {amplitude / N_eq:.4f})"
        )

    def test_stable_across_K_sweep(self):
        """Amplitude should remain small across a range of K values."""
        # q = d/(e*a) = 0.2/(0.5*0.5) = 0.8; N* = K*(1 - 0.5*0.2/1.0) = K*0.9
        for K_val in [50.0, 100.0, 200.0, 500.0]:
            sim = RatioDependentSimulation(_make_config(
                K=K_val, dt=0.01, n_steps=20000, N_0=K_val * 0.5, P_0=10.0,
            ))
            sim.reset()
            for _ in range(20000):
                sim.step()
                state = sim.observe()
            # Should converge near equilibrium N* = K*0.9
            N_eq = K_val * 0.9
            assert abs(state[0] - N_eq) / N_eq < 0.1, (
                f"K={K_val}: N={state[0]:.2f} not near N*={N_eq:.2f}"
            )


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class TestRatioDependentDynamics:
    def test_step_advances_state(self):
        sim = RatioDependentSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = RatioDependentSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up."""
        sim = RatioDependentSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            # Prey bounded by K, predator bounded
            assert state[0] < 200, f"Prey blow-up: {state[0]}"
            assert state[1] < 200, f"Predator blow-up: {state[1]}"

    def test_convergence_to_equilibrium(self):
        """From default ICs, trajectory should converge to equilibrium."""
        sim = RatioDependentSimulation(_make_config(dt=0.01, n_steps=30000))
        sim.reset()
        for _ in range(30000):
            sim.step()
        final = sim.observe()
        N_star, P_star = sim.coexistence_equilibrium()
        np.testing.assert_allclose(final, [N_star, P_star], rtol=0.02)

    def test_convergence_from_different_ic(self):
        """Convergence should occur from various initial conditions."""
        N_star_expected = 90.0

        for N_0, P_0 in [(10.0, 2.0), (80.0, 30.0), (30.0, 50.0)]:
            sim = RatioDependentSimulation(_make_config(
                N_0=N_0, P_0=P_0, dt=0.01, n_steps=30000,
            ))
            sim.reset()
            for _ in range(30000):
                sim.step()
            final = sim.observe()
            assert abs(final[0] - N_star_expected) / N_star_expected < 0.05, (
                f"IC ({N_0}, {P_0}): N={final[0]:.2f} not near N*={N_star_expected}"
            )

    def test_prey_bounded_by_K(self):
        """Prey should never exceed carrying capacity K."""
        sim = RatioDependentSimulation(_make_config(
            K=100.0, N_0=80.0, P_0=1.0, dt=0.01,
        ))
        sim.reset()
        for _ in range(5000):
            sim.step()
            assert sim.observe()[0] <= 100.0 + 1e-6, (
                f"Prey exceeded K: {sim.observe()[0]}"
            )


# ---------------------------------------------------------------------------
# Trajectory and run
# ---------------------------------------------------------------------------

class TestRatioDependentTrajectory:
    def test_run_trajectory_shape(self):
        sim = RatioDependentSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201

    def test_trajectory_timestamps(self):
        sim = RatioDependentSimulation(_make_config(dt=0.01, n_steps=100))
        traj = sim.run(n_steps=100)
        expected = np.arange(101) * 0.01
        np.testing.assert_allclose(traj.timestamps, expected, atol=1e-12)

    def test_trajectory_parameters(self):
        sim = RatioDependentSimulation(_make_config())
        traj = sim.run(n_steps=10)
        assert "r" in traj.parameters
        assert "K" in traj.parameters
        assert traj.parameters["r"] == 1.0

    def test_reproducibility_with_same_config(self):
        cfg = _make_config(n_steps=100)
        sim1 = RatioDependentSimulation(cfg)
        traj1 = sim1.run(100)
        sim2 = RatioDependentSimulation(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_long_run_no_nan(self):
        """No NaN after many steps."""
        sim = RatioDependentSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"


# ---------------------------------------------------------------------------
# Reset determinism
# ---------------------------------------------------------------------------

class TestResetDeterminism:
    def test_reset_restores_initial_state(self):
        sim = RatioDependentSimulation(_make_config(N_0=50.0, P_0=10.0))
        sim.reset()
        for _ in range(100):
            sim.step()
        state_after_steps = sim.observe().copy()
        sim.reset()
        state_after_reset = sim.observe()
        np.testing.assert_allclose(state_after_reset, [50.0, 10.0])
        assert not np.allclose(state_after_steps, state_after_reset)

    def test_reset_produces_identical_trajectory(self):
        sim = RatioDependentSimulation(_make_config())
        sim.reset()
        states1 = []
        for _ in range(50):
            states1.append(sim.step().copy())

        sim.reset()
        states2 = []
        for _ in range(50):
            states2.append(sim.step().copy())

        np.testing.assert_allclose(np.array(states1), np.array(states2))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_predator(self):
        """With P=0, prey should grow logistically."""
        sim = RatioDependentSimulation(_make_config(
            N_0=10.0, P_0=0.0, K=100.0, r=1.0, dt=0.01,
        ))
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        # Prey should approach K, predator stays 0
        assert state[0] == pytest.approx(100.0, rel=0.01)
        assert state[1] == pytest.approx(0.0, abs=1e-10)

    def test_very_small_populations(self):
        """Small populations should not produce NaN."""
        sim = RatioDependentSimulation(_make_config(
            N_0=0.01, P_0=0.01, dt=0.01,
        ))
        sim.reset()
        for _ in range(1000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))
        assert np.all(state >= 0)

    def test_predator_dies_without_prey(self):
        """With very low N_0 and high P_0, predator should decline."""
        sim = RatioDependentSimulation(_make_config(
            N_0=0.001, P_0=100.0, d=0.5, dt=0.01,
        ))
        sim.reset()
        initial_P = sim.observe()[1]
        for _ in range(100):
            sim.step()
        assert sim.observe()[1] < initial_P


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

class TestParameterSweep:
    def test_different_r_values(self):
        """Higher r should lead to higher equilibrium N*."""
        N_stars = []
        for r_val in [0.5, 1.0, 2.0]:
            sim = RatioDependentSimulation(_make_config(
                r=r_val, dt=0.01, n_steps=30000,
            ))
            if not sim.is_stable():
                continue
            sim.reset()
            for _ in range(30000):
                sim.step()
            N_stars.append(sim.observe()[0])
        # Higher r -> higher N* = K*(1 - d/(e*r))
        for i in range(len(N_stars) - 1):
            assert N_stars[i + 1] > N_stars[i]

    def test_different_d_values(self):
        """Higher d should lead to lower equilibrium P* (fewer predators)."""
        P_stars = []
        for d_val in [0.1, 0.2]:
            sim = RatioDependentSimulation(_make_config(
                d=d_val, r=1.0, e=0.5, a=0.5, dt=0.01, n_steps=30000,
            ))
            if not sim.is_stable():
                continue
            sim.reset()
            for _ in range(30000):
                sim.step()
            P_stars.append(sim.observe()[1])
        # Higher d -> lower P*
        for i in range(len(P_stars) - 1):
            assert P_stars[i + 1] < P_stars[i]

    def test_b_parameter_effect(self):
        """Varying b should affect predator equilibrium."""
        P_stars = []
        for b_val in [0.5, 1.0, 2.0]:
            sim = RatioDependentSimulation(_make_config(
                b=b_val, dt=0.01, n_steps=30000,
            ))
            N_star, P_star = sim.coexistence_equilibrium()
            P_stars.append(P_star)
        # Higher b -> lower P* (P* = N*(e*a - d)/(b*d))
        for i in range(len(P_stars) - 1):
            assert P_stars[i + 1] < P_stars[i]


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------

class TestRediscoveryDataGeneration:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.ratio_dependent import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 100.0
        assert data["a"] == 0.5
        assert data["b"] == 1.0

    def test_trajectory_data_custom_params(self):
        from simulating_anything.rediscovery.ratio_dependent import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(r=2.0, K=200.0, n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 2)
        assert data["r"] == 2.0
        assert data["K"] == 200.0

    def test_enrichment_sweep_data(self):
        from simulating_anything.rediscovery.ratio_dependent import (
            generate_enrichment_sweep,
        )
        data = generate_enrichment_sweep(n_K_values=5, n_steps=500, dt=0.01)
        assert len(data["K"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["P_amplitude"]) == 5
        assert len(data["N_equilibrium"]) == 5

    def test_enrichment_sweep_amplitudes_small(self):
        """Enrichment sweep amplitudes should be small (no paradox)."""
        from simulating_anything.rediscovery.ratio_dependent import (
            generate_enrichment_sweep,
        )
        data = generate_enrichment_sweep(
            n_K_values=5, K_min=50.0, K_max=200.0,
            n_steps=10000, dt=0.01,
        )
        # Amplitudes should be small relative to equilibria
        for i in range(len(data["K"])):
            N_eq = data["N_equilibrium"][i]
            if N_eq > 0:
                relative_amp = data["N_amplitude"][i] / N_eq
                assert relative_amp < 0.2, (
                    f"K={data['K'][i]}: relative amplitude {relative_amp:.4f} too large"
                )

    def test_equilibrium_sweep_data(self):
        from simulating_anything.rediscovery.ratio_dependent import (
            generate_equilibrium_sweep,
        )
        data = generate_equilibrium_sweep(n_samples=10, n_steps=5000, dt=0.01)
        assert "N_measured" in data
        assert "P_measured" in data
        assert "N_theory" in data
        assert "P_theory" in data
        # Should have at least some valid samples
        assert len(data["N_measured"]) > 0
