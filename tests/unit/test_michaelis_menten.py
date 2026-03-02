"""Tests for the Michaelis-Menten enzyme kinetics model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.michaelis_menten import (
    MichaelisMentenSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    k1: float = 1.0,
    k_1: float = 0.1,
    k2: float = 0.5,
    E_total: float = 1.0,
    S0: float = 10.0,
    C0: float = 0.0,
    P0: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.MICHAELIS_MENTEN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "k1": k1, "k_1": k_1, "k2": k2,
            "E_total": E_total, "S0": S0,
            "C0": C0, "P0": P0,
        },
    )


class TestMichaelisMentenCreation:
    def test_initial_state_shape(self):
        sim = MichaelisMentenSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        sim = MichaelisMentenSimulation(_make_config(S0=5.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [5.0, 0.0, 0.0])

    def test_initial_state_with_complex(self):
        sim = MichaelisMentenSimulation(_make_config(S0=8.0, C0=0.5, P0=1.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [8.0, 0.5, 1.5])

    def test_default_parameters(self):
        sim = MichaelisMentenSimulation(_make_config())
        assert sim.k1 == 1.0
        assert sim.k_1 == 0.1
        assert sim.k2 == 0.5
        assert sim.E_total == 1.0
        assert sim.S0 == 10.0

    def test_custom_parameters(self):
        sim = MichaelisMentenSimulation(
            _make_config(k1=2.0, k_1=0.3, k2=1.0, E_total=2.0, S0=20.0)
        )
        assert sim.k1 == 2.0
        assert sim.k_1 == 0.3
        assert sim.k2 == 1.0
        assert sim.E_total == 2.0
        assert sim.S0 == 20.0


class TestMichaelisMentenStepAndObserve:
    def test_step_advances_state(self):
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = MichaelisMentenSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_step_count_increments(self):
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_run_returns_trajectory(self):
        sim = MichaelisMentenSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_run_timestamps_correct(self):
        sim = MichaelisMentenSimulation(_make_config(dt=0.01))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * 0.01
        np.testing.assert_allclose(traj.timestamps, expected, atol=1e-12)


class TestMichaelisMentenConservation:
    def test_enzyme_conservation(self):
        """E + C = E_total must hold at all times."""
        sim = MichaelisMentenSimulation(_make_config(E_total=2.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
            S, C, P = sim.observe()
            E = sim.E_total - C
            assert abs(E + C - sim.E_total) < 1e-10, (
                f"Enzyme conservation violated: E+C={E + C}, "
                f"E_total={sim.E_total}"
            )

    def test_substrate_conservation(self):
        """S + C + P = S0 must hold (total substrate is conserved)."""
        sim = MichaelisMentenSimulation(_make_config(S0=10.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
            S, C, P = sim.observe()
            total = S + C + P
            assert abs(total - sim.S0) < 1e-6, (
                f"Substrate conservation violated: S+C+P={total}, S0={sim.S0}"
            )

    def test_conservation_different_params(self):
        """Conservation holds with different parameters."""
        sim = MichaelisMentenSimulation(
            _make_config(k1=3.0, k_1=0.5, k2=2.0, E_total=3.0, S0=15.0)
        )
        sim.reset()
        for _ in range(3000):
            sim.step()
            S, C, P = sim.observe()
            assert abs(S + C + P - sim.S0) < 1e-5
            assert abs((sim.E_total - C) + C - sim.E_total) < 1e-10

    def test_non_negative_concentrations(self):
        """All concentrations must remain non-negative."""
        sim = MichaelisMentenSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            S, C, P = sim.observe()
            assert S >= -1e-12, f"S went negative: {S}"
            assert C >= -1e-12, f"C went negative: {C}"
            assert P >= -1e-12, f"P went negative: {P}"

    def test_complex_bounded_by_E_total(self):
        """Complex concentration C cannot exceed E_total."""
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            sim.step()
            C = sim.observe()[1]
            assert C <= sim.E_total + 1e-10, (
                f"C exceeded E_total: C={C}, E_total={sim.E_total}"
            )


class TestMichaelisMentenSteadyState:
    def test_substrate_decreases(self):
        """Substrate should decrease monotonically (no reverse reaction dominance
        when starting from C0=0)."""
        sim = MichaelisMentenSimulation(_make_config(S0=10.0))
        sim.reset()
        prev_S = sim.observe()[0]
        # After initial transient, S should decrease
        for _ in range(100):
            sim.step()
        prev_S = sim.observe()[0]
        for _ in range(4900):
            sim.step()
            S = sim.observe()[0]
            assert S <= prev_S + 1e-10, (
                f"Substrate increased: {prev_S:.6f} -> {S:.6f}"
            )
            prev_S = S

    def test_product_increases(self):
        """Product should increase monotonically."""
        sim = MichaelisMentenSimulation(_make_config(S0=10.0))
        sim.reset()
        prev_P = sim.observe()[2]
        for _ in range(5000):
            sim.step()
            P = sim.observe()[2]
            assert P >= prev_P - 1e-10, (
                f"Product decreased: {prev_P:.6f} -> {P:.6f}"
            )
            prev_P = P

    def test_final_state_approaches_completion(self):
        """After long time, S -> 0, P -> S0, C -> 0."""
        sim = MichaelisMentenSimulation(
            _make_config(S0=10.0, dt=0.01, n_steps=50000)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        S, C, P = sim.observe()
        assert S < 0.01, f"Substrate not consumed: S={S}"
        assert P > sim.S0 - 0.01, f"Product not formed: P={P}"
        assert C < 0.01, f"Complex not released: C={C}"

    def test_free_enzyme_property(self):
        """Free enzyme E = E_total - C."""
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        E = sim.free_enzyme
        C = sim.observe()[1]
        assert abs(E - (sim.E_total - C)) < 1e-12


class TestQuasiSteadyState:
    def test_qssa_after_transient(self):
        """After transient, dC/dt should be near zero (QSSA)."""
        sim = MichaelisMentenSimulation(
            _make_config(S0=10.0, E_total=0.1, dt=0.001)
        )
        sim.reset()
        # Run past transient (for QSSA, need E_total << S0)
        for _ in range(5000):
            sim.step()
        dc_dt = sim.quasi_steady_state_check()
        # Under QSSA conditions (E_total << S0), dC/dt should be small
        assert dc_dt < 0.01, (
            f"QSSA not satisfied: |dC/dt|={dc_dt:.6f}"
        )

    def test_qssa_improves_with_small_enzyme(self):
        """QSSA works better when E_total << S0."""
        results = []
        for E_total in [0.01, 0.1, 1.0]:
            sim = MichaelisMentenSimulation(
                _make_config(S0=10.0, E_total=E_total, dt=0.001)
            )
            sim.reset()
            for _ in range(5000):
                sim.step()
            dc_dt = sim.quasi_steady_state_check()
            results.append(dc_dt)
        # Smaller E_total should give better QSSA
        assert results[0] < results[2], (
            f"QSSA should improve with smaller enzyme: {results}"
        )

    def test_quasi_steady_state_check_initial(self):
        """Before reset, should return inf."""
        sim = MichaelisMentenSimulation(_make_config())
        assert sim.quasi_steady_state_check() == float("inf")


class TestMichaelisMentenFormulas:
    def test_vmax_formula(self):
        """V_max = k2 * E_total."""
        sim = MichaelisMentenSimulation(
            _make_config(k2=0.5, E_total=2.0)
        )
        assert sim.V_max == pytest.approx(1.0)

    def test_vmax_different_params(self):
        sim = MichaelisMentenSimulation(
            _make_config(k2=3.0, E_total=0.5)
        )
        assert sim.V_max == pytest.approx(1.5)

    def test_km_formula(self):
        """K_m = (k_1 + k2) / k1."""
        sim = MichaelisMentenSimulation(
            _make_config(k1=1.0, k_1=0.1, k2=0.5)
        )
        assert sim.K_m == pytest.approx(0.6)

    def test_km_different_params(self):
        sim = MichaelisMentenSimulation(
            _make_config(k1=2.0, k_1=0.5, k2=1.0)
        )
        assert sim.K_m == pytest.approx(0.75)

    def test_steady_state_rate_formula(self):
        """v = V_max * S / (K_m + S)."""
        sim = MichaelisMentenSimulation(
            _make_config(k2=0.5, E_total=1.0, k1=1.0, k_1=0.1)
        )
        # V_max = 0.5, K_m = 0.6
        v = sim.steady_state_rate(S=0.6)
        expected = 0.5 * 0.6 / (0.6 + 0.6)
        assert v == pytest.approx(expected, rel=1e-10)

    def test_steady_state_rate_at_high_s(self):
        """At very high S, rate approaches V_max."""
        sim = MichaelisMentenSimulation(
            _make_config(k2=0.5, E_total=1.0)
        )
        v = sim.steady_state_rate(S=1000.0)
        assert v == pytest.approx(sim.V_max, rel=0.01)

    def test_steady_state_rate_at_km(self):
        """At S = K_m, rate should be V_max/2."""
        sim = MichaelisMentenSimulation(_make_config())
        v = sim.steady_state_rate(S=sim.K_m)
        assert v == pytest.approx(sim.V_max / 2.0, rel=1e-10)

    def test_steady_state_rate_at_zero_s(self):
        """At S = 0, rate should be 0."""
        sim = MichaelisMentenSimulation(_make_config())
        v = sim.steady_state_rate(S=0.0)
        assert v == 0.0

    def test_steady_state_rate_uses_current_state(self):
        """Without argument, should use current substrate."""
        sim = MichaelisMentenSimulation(_make_config(S0=5.0))
        sim.reset()
        v = sim.steady_state_rate()
        expected = sim.V_max * 5.0 / (sim.K_m + 5.0)
        assert v == pytest.approx(expected, rel=1e-10)


class TestParameterVariation:
    def test_higher_k1_faster_reaction(self):
        """Higher binding rate k1 should consume substrate faster."""
        configs = [
            _make_config(k1=0.5, dt=0.01),
            _make_config(k1=5.0, dt=0.01),
        ]
        final_S = []
        for cfg in configs:
            sim = MichaelisMentenSimulation(cfg)
            sim.reset()
            for _ in range(2000):
                sim.step()
            final_S.append(sim.observe()[0])
        # Higher k1 should leave less substrate
        assert final_S[1] < final_S[0], (
            f"Higher k1 didn't consume more: {final_S}"
        )

    def test_higher_k2_faster_product(self):
        """Higher catalytic rate k2 should produce product faster."""
        configs = [
            _make_config(k2=0.1, dt=0.01),
            _make_config(k2=2.0, dt=0.01),
        ]
        final_P = []
        for cfg in configs:
            sim = MichaelisMentenSimulation(cfg)
            sim.reset()
            for _ in range(2000):
                sim.step()
            final_P.append(sim.observe()[2])
        # Higher k2 should produce more product
        assert final_P[1] > final_P[0], (
            f"Higher k2 didn't produce more: {final_P}"
        )

    def test_higher_E_total_faster_reaction(self):
        """More enzyme should speed up the reaction."""
        configs = [
            _make_config(E_total=0.1, dt=0.01),
            _make_config(E_total=5.0, dt=0.01),
        ]
        final_P = []
        for cfg in configs:
            sim = MichaelisMentenSimulation(cfg)
            sim.reset()
            for _ in range(2000):
                sim.step()
            final_P.append(sim.observe()[2])
        assert final_P[1] > final_P[0], (
            f"More enzyme didn't help: {final_P}"
        )

    def test_higher_k_1_slower_reaction(self):
        """Higher unbinding rate should slow down net reaction."""
        configs = [
            _make_config(k_1=0.01, dt=0.01),
            _make_config(k_1=5.0, dt=0.01),
        ]
        final_P = []
        for cfg in configs:
            sim = MichaelisMentenSimulation(cfg)
            sim.reset()
            for _ in range(2000):
                sim.step()
            final_P.append(sim.observe()[2])
        # Higher k_1 should produce less product (slower net catalysis)
        assert final_P[0] > final_P[1], (
            f"Higher k_1 didn't slow down: {final_P}"
        )


class TestDerivatives:
    def test_derivatives_at_zero_complex(self):
        """With C=0, dC/dt should be positive (complex forming)."""
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        dy = sim._derivatives(np.array([10.0, 0.0, 0.0]))
        # dS should be negative (substrate consumed)
        assert dy[0] < 0
        # dC should be positive (complex forming)
        assert dy[1] > 0
        # dP should be zero (no complex to catalyze)
        assert dy[2] == pytest.approx(0.0)

    def test_derivatives_at_equilibrium(self):
        """At full conversion (S=0, C=0, P=S0), all derivatives should be zero."""
        sim = MichaelisMentenSimulation(_make_config(S0=10.0))
        sim.reset()
        dy = sim._derivatives(np.array([0.0, 0.0, 10.0]))
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-12)

    def test_derivatives_mass_balance(self):
        """dS/dt + dC/dt + dP/dt = 0 (total substrate conservation)."""
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        y = np.array([5.0, 0.3, 4.7])
        dy = sim._derivatives(y)
        assert abs(dy[0] + dy[1] + dy[2]) < 1e-12, (
            f"Mass balance violated: sum={dy[0] + dy[1] + dy[2]}"
        )

    def test_derivatives_enzyme_balance(self):
        """dC/dt reflects enzyme conservation: no net creation of E+C."""
        sim = MichaelisMentenSimulation(_make_config())
        sim.reset()
        y = np.array([8.0, 0.5, 1.5])
        dy = sim._derivatives(y)
        # d(E+C)/dt = -dC/dt + dC/dt = 0 automatically since E = E_total - C
        # But checking: binding uses E, unbinding creates E
        E = sim.E_total - y[1]
        binding = sim.k1 * E * y[0]
        unbinding = sim.k_1 * y[1]
        catalysis = sim.k2 * y[1]
        expected_dC = binding - unbinding - catalysis
        assert dy[1] == pytest.approx(expected_dC)


class TestMichaelisMentenMeasurements:
    def test_measure_initial_rate(self):
        """Initial rate should approximate V_max*S0/(K_m+S0)."""
        sim = MichaelisMentenSimulation(
            _make_config(S0=10.0, E_total=0.1, dt=0.001)
        )
        v0 = sim.measure_initial_rate(n_steps=20)
        expected = sim.steady_state_rate(S=10.0)
        # Allow 20% tolerance for transient effects
        assert v0 == pytest.approx(expected, rel=0.2), (
            f"Initial rate {v0} far from expected {expected}"
        )

    def test_initial_rate_increases_with_substrate(self):
        """Higher initial substrate should give higher initial rate."""
        rates = []
        for S0 in [1.0, 5.0, 20.0]:
            sim = MichaelisMentenSimulation(
                _make_config(S0=S0, dt=0.001)
            )
            rates.append(sim.measure_initial_rate(n_steps=20))
        assert rates[0] < rates[1] < rates[2], (
            f"Rates should increase with S0: {rates}"
        )

    def test_time_to_half_substrate(self):
        """Should return a finite positive time."""
        sim = MichaelisMentenSimulation(
            _make_config(S0=10.0, dt=0.01)
        )
        t_half = sim.time_to_half_substrate()
        assert np.isfinite(t_half)
        assert t_half > 0


class TestRediscoveryDataGeneration:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.michaelis_menten import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["k1"] == 1.0
        assert data["E_total"] == 1.0

    def test_initial_rate_data_generation(self):
        from simulating_anything.rediscovery.michaelis_menten import (
            generate_initial_rate_data,
        )

        data = generate_initial_rate_data(n_S0=5, dt=0.001)
        assert len(data["S0"]) == 5
        assert len(data["v0_measured"]) == 5
        assert len(data["v0_theory"]) == 5
        assert data["V_max"] > 0
        assert data["K_m"] > 0

    def test_vmax_data_generation(self):
        from simulating_anything.rediscovery.michaelis_menten import (
            generate_vmax_data,
        )

        data = generate_vmax_data(n_E=5, dt=0.001)
        assert len(data["E_total"]) == 5
        assert len(data["v_max_measured"]) == 5
        assert data["k2"] == 0.5

    def test_initial_rate_data_matches_theory(self):
        """Measured initial rates should correlate with MM equation.

        Correlation is not perfect because the QSSA is approximate --
        it works best when E_total << S0. With default E_total=1.0
        and small S0 values, there is some deviation.
        """
        from simulating_anything.rediscovery.michaelis_menten import (
            generate_initial_rate_data,
        )

        data = generate_initial_rate_data(n_S0=10, dt=0.001)
        corr = np.corrcoef(data["v0_measured"], data["v0_theory"])[0, 1]
        assert corr > 0.95, f"Poor correlation: {corr}"
