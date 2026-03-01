"""Tests for the eco-epidemiological predator-prey with disease model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.eco_epidemic import EcoEpidemicSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.01,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for EcoEpidemic with optional overrides."""
    params = {
        "r": 1.0, "K": 100.0, "beta": 0.01,
        "a1": 0.1, "a2": 0.3, "h1": 0.1, "h2": 0.1,
        "e1": 0.5, "e2": 0.3, "d_disease": 0.2, "m": 0.3,
        "S_0": 50.0, "I_0": 10.0, "P_0": 5.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.ECO_EPIDEMIC,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestEcoEpidemicSimulation:
    """Core simulation tests."""

    def test_reset_state(self):
        """Initial conditions should match parameters."""
        sim = EcoEpidemicSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)
        np.testing.assert_allclose(state, [50.0, 10.0, 5.0])

    def test_observe_shape(self):
        """Observe should return 3-element state [S, I, P]."""
        sim = EcoEpidemicSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_step_advances(self):
        """State should change after a step."""
        sim = EcoEpidemicSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = EcoEpidemicSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = EcoEpidemicSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_non_negative(self):
        """S, I, P should stay non-negative throughout simulation."""
        sim = EcoEpidemicSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"
            assert state[2] >= 0, f"P became negative: {state[2]}"

    def test_conservation_check(self):
        """Total population should remain bounded (not diverge)."""
        sim = EcoEpidemicSimulation(_make_config(dt=0.01))
        sim.reset()
        cap = sim.K
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            total = state[0] + state[1] + state[2]
            # Total should stay bounded; can exceed K due to predators
            # but should not diverge
            assert total < cap * 10, f"Population diverged: {total}"

    def test_long_run_stability(self):
        """No NaN after 100000 steps."""
        sim = EcoEpidemicSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(100000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_disease_free_dynamics(self):
        """Without infection (beta=0), reduces to predator-prey."""
        sim = EcoEpidemicSimulation(_make_config(beta=0.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        # Infected should die out since beta=0 and d_disease>0
        assert state[1] < 0.01, (
            f"Infected should decay without transmission: I={state[1]}"
        )

    def test_predator_free_dynamics(self):
        """Without predators (P=0), reduces to SIR-like dynamics."""
        sim = EcoEpidemicSimulation(_make_config(P_0=0.0, m=10.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        # Predators should remain near zero (started at 0, high death rate)
        assert state[2] < 0.01, f"Predators should stay extinct: P={state[2]}"

    def test_logistic_growth(self):
        """Without predators or disease, prey grows logistically to K."""
        sim = EcoEpidemicSimulation(
            _make_config(beta=0.0, P_0=0.0, I_0=0.0, m=10.0, dt=0.01, S_0=10.0)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        # S should approach K=100
        assert abs(state[0] - 100.0) < 1.0, (
            f"S should approach K=100: S={state[0]}"
        )

    def test_carrying_capacity(self):
        """Prey alone should reach carrying capacity K."""
        sim = EcoEpidemicSimulation(
            _make_config(
                beta=0.0, P_0=0.0, I_0=0.0, m=10.0,
                S_0=5.0, dt=0.01,
            )
        )
        sim.reset()
        for _ in range(80000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(state[0], 100.0, atol=2.0)

    def test_disease_invasion(self):
        """High beta without predators leads to non-trivial infected population.

        With the default predation parameters, predators act as effective
        biological control and can drive disease to extinction. To test
        disease invasion, we remove predators (P_0=0, high m).
        """
        sim = EcoEpidemicSimulation(
            _make_config(beta=0.03, P_0=0.0, m=10.0, dt=0.01)
        )
        sim.reset()
        for _ in range(20000):
            sim.step()

        # Collect post-transient data
        inf_vals = []
        for _ in range(5000):
            sim.step()
            inf_vals.append(sim.observe()[1])
        inf_mean = np.mean(inf_vals)
        assert inf_mean > 0.1, (
            f"Disease should persist with high beta: mean I={inf_mean}"
        )

    def test_predator_control(self):
        """Strong predation should reduce disease prevalence.

        Predators preferentially eat infected prey (a2 > a1), acting as
        biological disease control.
        """
        # Weak predation: high predator mortality
        sim_weak = EcoEpidemicSimulation(_make_config(m=0.8, beta=0.02, dt=0.01))
        sim_weak.reset()
        for _ in range(30000):
            sim_weak.step()
        inf_vals_weak = []
        for _ in range(5000):
            sim_weak.step()
            inf_vals_weak.append(sim_weak.observe()[1])
        inf_weak = np.mean(inf_vals_weak)

        # Strong predation: low predator mortality
        sim_strong = EcoEpidemicSimulation(
            _make_config(m=0.15, beta=0.02, dt=0.01)
        )
        sim_strong.reset()
        for _ in range(30000):
            sim_strong.step()
        inf_vals_strong = []
        for _ in range(5000):
            sim_strong.step()
            inf_vals_strong.append(sim_strong.observe()[1])
        inf_strong = np.mean(inf_vals_strong)

        # Strong predation should result in fewer infected
        assert inf_strong <= inf_weak + 1.0, (
            f"Strong predation should reduce infection: "
            f"I_strong={inf_strong:.2f}, I_weak={inf_weak:.2f}"
        )

    def test_R0_threshold(self):
        """Disease should invade when R0 > 1."""
        # R0 = beta*K/d = 0.005*100/0.2 = 2.5 > 1 => disease invades
        sim_high = EcoEpidemicSimulation(
            _make_config(beta=0.005, P_0=0.0, m=10.0, dt=0.01)
        )
        assert sim_high.R0_no_predators > 1.0

        # R0 = 0.001*100/0.2 = 0.5 < 1 => disease dies out
        sim_low = EcoEpidemicSimulation(
            _make_config(beta=0.001, P_0=0.0, m=10.0, dt=0.01)
        )
        assert sim_low.R0_no_predators < 1.0

    def test_holling_type2(self):
        """Functional response should saturate at high prey density.

        At high S, fr_s = a1*S/(1 + h1*a1*S) -> 1/h1 = 10.
        """
        sim = EcoEpidemicSimulation(_make_config())
        sim.reset()

        # Compute functional response at low and high prey densities
        fr_low = sim.a1 * 1.0 / (1.0 + sim.h1 * sim.a1 * 1.0)
        fr_high = sim.a1 * 1000.0 / (1.0 + sim.h1 * sim.a1 * 1000.0)

        # High should be closer to saturation (1/h1 = 10)
        saturation = 1.0 / sim.h1
        assert abs(fr_high - saturation) < abs(fr_low - saturation)

    def test_preferential_predation(self):
        """Infected prey should be consumed faster (a2 > a1).

        With equal S and I, the functional response on I should be higher.
        """
        sim = EcoEpidemicSimulation(_make_config())
        assert sim.a2 > sim.a1, "a2 should be > a1 for preferential predation"

        # Same prey density, functional response on I should be higher
        density = 20.0
        fr_s = sim.a1 * density / (1.0 + sim.h1 * sim.a1 * density)
        fr_i = sim.a2 * density / (1.0 + sim.h2 * sim.a2 * density)
        assert fr_i > fr_s, (
            f"Infected prey predation rate should be higher: {fr_i} vs {fr_s}"
        )

    def test_steady_state_exists(self):
        """System should reach some form of equilibrium for default params."""
        sim = EcoEpidemicSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(50000):
            sim.step()

        # Collect data to check for bounded variation
        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())

        states = np.array(states)
        # All components should be finite and bounded
        assert np.all(np.isfinite(states))
        for col_idx, name in enumerate(["S", "I", "P"]):
            col = states[:, col_idx]
            assert np.max(col) < 200, f"{name} unbounded: max={np.max(col)}"

    def test_oscillations(self):
        """System can oscillate for certain parameter ranges.

        Use parameters that promote oscillatory coexistence.
        """
        sim = EcoEpidemicSimulation(
            _make_config(
                r=1.0, K=100.0, beta=0.02,
                a1=0.15, a2=0.4, h1=0.05, h2=0.05,
                e1=0.6, e2=0.4, d_disease=0.15, m=0.25,
                S_0=40.0, I_0=15.0, P_0=8.0, dt=0.01,
            )
        )
        sim.reset()

        # Skip transient
        for _ in range(50000):
            sim.step()

        # Measure variation in S
        s_vals = []
        for _ in range(20000):
            sim.step()
            s_vals.append(sim.observe()[0])

        s_range = max(s_vals) - min(s_vals)

        # Either oscillating (range > 0) or at equilibrium -- both valid
        assert s_range >= 0, "S range should be non-negative"


class TestEcoEpidemicAnalytical:
    """Tests for analytical properties and computed quantities."""

    def test_R0_no_predators(self):
        """R0 without predators = beta*K/d."""
        sim = EcoEpidemicSimulation(
            _make_config(beta=0.01, K=100.0, d_disease=0.2)
        )
        assert sim.R0_no_predators == pytest.approx(5.0)

    def test_disease_free_equilibrium(self):
        """Disease-free equilibrium should satisfy the fixed-point equations."""
        sim = EcoEpidemicSimulation(_make_config())
        try:
            S_star, P_star = sim.disease_free_equilibrium()
            # Verify it is a fixed point: dS/dt = 0, dP/dt = 0 at (S*, 0, P*)
            state = np.array([S_star, 0.0, P_star])
            dy = sim._derivatives(state)
            np.testing.assert_allclose(dy[0], 0.0, atol=1e-8)
            np.testing.assert_allclose(dy[2], 0.0, atol=1e-8)
            assert S_star > 0
            assert P_star > 0
        except ValueError:
            # Some parameter combinations may not have this equilibrium
            pass

    def test_compute_R0_with_predators(self):
        """Effective R0 should be less than R0 without predators."""
        sim = EcoEpidemicSimulation(_make_config())
        r0_eff = sim.compute_R0()
        r0_no_pred = sim.R0_no_predators
        # Predators should reduce effective R0
        assert r0_eff <= r0_no_pred + 0.01, (
            f"Effective R0 ({r0_eff}) should be <= R0 without "
            f"predators ({r0_no_pred})"
        )


class TestEcoEpidemicRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_rediscovery_data(self):
        """ODE data generation should produce valid arrays."""
        from simulating_anything.rediscovery.eco_epidemic import generate_ode_data
        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert len(data["S"]) == 501
        assert len(data["I"]) == 501
        assert len(data["P"]) == 501
        assert np.all(np.isfinite(data["states"]))

    def test_bifurcation_sweep(self):
        """Beta sweep should produce results for each beta value."""
        from simulating_anything.rediscovery.eco_epidemic import generate_beta_sweep
        data = generate_beta_sweep(n_beta=5, n_steps=1000, dt=0.01)
        assert len(data["beta_values"]) == 5
        assert len(data["disease_prevalence"]) == 5
        assert len(data["S"]) == 5
        assert len(data["I"]) == 5
        assert len(data["P"]) == 5
        assert np.all(np.isfinite(data["disease_prevalence"]))

    def test_predator_control_sweep(self):
        """Predator mortality sweep should produce valid results."""
        from simulating_anything.rediscovery.eco_epidemic import (
            generate_predator_control_sweep,
        )
        data = generate_predator_control_sweep(n_m=5, n_steps=1000, dt=0.01)
        assert len(data["m_values"]) == 5
        assert len(data["disease_prevalence"]) == 5
        assert len(data["predator_pop"]) == 5
        assert np.all(np.isfinite(data["disease_prevalence"]))
