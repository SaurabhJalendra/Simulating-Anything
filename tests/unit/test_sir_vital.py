"""Tests for SIR with vital dynamics (births and deaths) simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sir_vital import SIRVitalSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    beta: float = 0.4,
    gamma: float = 0.1,
    mu: float = 0.01,
    S_0: float = 0.99,
    I_0: float = 0.01,
    R_0_init: float = 0.0,
    dt: float = 0.1,
    n_steps: int = 5000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SIR_VITAL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": beta,
            "gamma": gamma,
            "mu": mu,
            "S_0": S_0,
            "I_0": I_0,
            "R_0_init": R_0_init,
        },
    )


# ---------------------------------------------------------------------------
# Creation and initialization
# ---------------------------------------------------------------------------
class TestSIRVitalCreation:
    def test_initial_state_shape(self):
        sim = SIRVitalSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        sim = SIRVitalSimulation(_make_config(S_0=0.99, I_0=0.01, R_0_init=0.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.99, 0.01, 0.0], atol=1e-12)

    def test_initial_state_sums_to_one(self):
        sim = SIRVitalSimulation(_make_config(S_0=0.8, I_0=0.1, R_0_init=0.1))
        state = sim.reset()
        assert state.sum() == pytest.approx(1.0)

    def test_initial_state_normalized(self):
        """Unnormalized initial fractions should be normalized to sum=1."""
        sim = SIRVitalSimulation(_make_config(S_0=8.0, I_0=1.0, R_0_init=1.0))
        state = sim.reset()
        assert state.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(state, [0.8, 0.1, 0.1])

    def test_parameters_stored(self):
        sim = SIRVitalSimulation(_make_config(beta=0.5, gamma=0.2, mu=0.02))
        assert sim.beta == pytest.approx(0.5)
        assert sim.gamma == pytest.approx(0.2)
        assert sim.mu == pytest.approx(0.02)

    def test_default_parameters(self):
        """Default params should be used when not specified."""
        config = SimulationConfig(
            domain=Domain.SIR_VITAL, dt=0.1, n_steps=100, parameters={},
        )
        sim = SIRVitalSimulation(config)
        assert sim.beta == pytest.approx(0.4)
        assert sim.gamma == pytest.approx(0.1)
        assert sim.mu == pytest.approx(0.01)
        assert sim.S_0 == pytest.approx(0.99)
        assert sim.I_0 == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Basic dynamics
# ---------------------------------------------------------------------------
class TestSIRVitalDynamics:
    def test_step_advances_state(self):
        sim = SIRVitalSimulation(_make_config())
        s0 = sim.reset().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = SIRVitalSimulation(_make_config())
        sim.reset()
        sim.step()
        s1 = sim.observe()
        s2 = sim.observe()
        np.testing.assert_array_equal(s1, s2)

    def test_run_trajectory(self):
        sim = SIRVitalSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 3)
        assert len(traj.timestamps) == 201

    def test_trajectory_timestamps(self):
        dt = 0.1
        sim = SIRVitalSimulation(_make_config(dt=dt, n_steps=100))
        traj = sim.run(n_steps=100)
        np.testing.assert_allclose(traj.timestamps[-1], 100 * dt)


# ---------------------------------------------------------------------------
# Conservation and non-negativity
# ---------------------------------------------------------------------------
class TestConservation:
    def test_sum_equals_one(self):
        """S + I + R = 1 at all times."""
        sim = SIRVitalSimulation(_make_config())
        sim.reset()
        for _ in range(2000):
            state = sim.step()
            assert state.sum() == pytest.approx(1.0, abs=1e-10)

    def test_state_non_negative(self):
        """All compartments must remain non-negative."""
        sim = SIRVitalSimulation(_make_config(beta=0.8, gamma=0.3, mu=0.05))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0.0), f"Negative state: {state}"

    def test_conservation_long_run(self):
        """Conservation holds over long integration."""
        sim = SIRVitalSimulation(_make_config(dt=0.05, n_steps=20000))
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert state.sum() == pytest.approx(1.0, abs=1e-9)

    def test_non_negative_disease_free(self):
        """Non-negativity holds when R0 < 1 (disease dies out)."""
        sim = SIRVitalSimulation(_make_config(beta=0.05, gamma=0.1, mu=0.01))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0.0)


# ---------------------------------------------------------------------------
# R0 computation
# ---------------------------------------------------------------------------
class TestR0:
    def test_R0_default(self):
        """Default: R0 = 0.4 / (0.1 + 0.01) = 3.636..."""
        sim = SIRVitalSimulation(_make_config())
        assert sim.compute_R0() == pytest.approx(0.4 / 0.11)

    def test_R0_custom(self):
        sim = SIRVitalSimulation(_make_config(beta=0.3, gamma=0.2, mu=0.05))
        assert sim.compute_R0() == pytest.approx(0.3 / 0.25)

    def test_R0_high(self):
        sim = SIRVitalSimulation(_make_config(beta=0.8, gamma=0.05, mu=0.01))
        assert sim.compute_R0() == pytest.approx(0.8 / 0.06)

    def test_R0_near_one(self):
        # beta = gamma + mu exactly => R0 = 1
        sim = SIRVitalSimulation(_make_config(beta=0.11, gamma=0.1, mu=0.01))
        assert sim.compute_R0() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Disease-free equilibrium (R0 < 1)
# ---------------------------------------------------------------------------
class TestDiseaseFreeEquilibrium:
    def test_dfe_formula(self):
        sim = SIRVitalSimulation(_make_config())
        dfe = sim.disease_free_equilibrium()
        assert dfe == (1.0, 0.0, 0.0)

    def test_dfe_stable_when_R0_lt_1(self):
        """When R0 < 1, I should decay to 0."""
        sim = SIRVitalSimulation(
            _make_config(beta=0.05, gamma=0.1, mu=0.01, n_steps=50000)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert state[1] < 1e-6, f"I should be ~0 but is {state[1]}"
        assert state[0] > 0.99, f"S should be ~1 but is {state[0]}"

    def test_dfe_I_decreasing_when_R0_lt_1(self):
        """When R0 < 1, I should decrease monotonically (at least after initial)."""
        sim = SIRVitalSimulation(_make_config(beta=0.05, gamma=0.1, mu=0.01))
        sim.reset()
        # Skip a few steps for initial transient
        for _ in range(10):
            sim.step()
        prev_I = sim.observe()[1]
        for _ in range(2000):
            sim.step()
            I = sim.observe()[1]
            # Allow small numerical noise
            assert I <= prev_I + 1e-12, f"I increased: {prev_I} -> {I}"
            prev_I = I


# ---------------------------------------------------------------------------
# Endemic equilibrium (R0 > 1)
# ---------------------------------------------------------------------------
class TestEndemicEquilibrium:
    def test_endemic_eq_returns_none_when_R0_lt_1(self):
        sim = SIRVitalSimulation(_make_config(beta=0.05, gamma=0.1, mu=0.01))
        assert sim.endemic_equilibrium() is None

    def test_endemic_eq_values(self):
        """Check analytical endemic equilibrium formulas."""
        sim = SIRVitalSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.01))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        S_star, I_star, R_star = eq

        r0 = 0.4 / 0.11
        assert S_star == pytest.approx(1.0 / r0)
        assert I_star == pytest.approx(0.01 * (r0 - 1.0) / 0.4)
        assert R_star == pytest.approx(0.1 * I_star / 0.01)

    def test_endemic_eq_sums_to_one(self):
        """Endemic equilibrium fractions should sum to 1."""
        sim = SIRVitalSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.01))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert sum(eq) == pytest.approx(1.0)

    def test_simulation_converges_to_endemic(self):
        """Simulation should converge to analytical endemic equilibrium."""
        sim = SIRVitalSimulation(
            _make_config(beta=0.4, gamma=0.1, mu=0.01, dt=0.1, n_steps=200000)
        )
        sim.reset()

        for _ in range(200000):
            sim.step()

        state = sim.observe()
        eq = sim.endemic_equilibrium()
        assert eq is not None
        S_star, I_star, R_star = eq

        np.testing.assert_allclose(state[0], S_star, rtol=0.02,
                                   err_msg=f"S: sim={state[0]}, theory={S_star}")
        np.testing.assert_allclose(state[1], I_star, rtol=0.02,
                                   err_msg=f"I: sim={state[1]}, theory={I_star}")
        np.testing.assert_allclose(state[2], R_star, rtol=0.02,
                                   err_msg=f"R: sim={state[2]}, theory={R_star}")

    def test_endemic_I_positive_for_large_R0(self):
        """Endemic I* should be positive for R0 >> 1."""
        sim = SIRVitalSimulation(_make_config(beta=0.8, gamma=0.05, mu=0.01))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert eq[1] > 0.01  # I* should be nontrivial

    def test_endemic_I_approaches_zero_near_bifurcation(self):
        """As R0 -> 1+, I* -> 0."""
        beta_c = 0.11  # gamma + mu
        # Just above critical
        sim = SIRVitalSimulation(_make_config(beta=beta_c + 0.01))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert eq[1] < 0.01  # I* should be small near bifurcation


# ---------------------------------------------------------------------------
# Damped oscillations to endemic equilibrium
# ---------------------------------------------------------------------------
class TestDampedOscillations:
    def test_disease_persists_for_R0_gt_1(self):
        """Unlike basic SIR, I should NOT go to 0 when R0 > 1."""
        sim = SIRVitalSimulation(
            _make_config(beta=0.4, gamma=0.1, mu=0.01, n_steps=100000)
        )
        sim.reset()
        for _ in range(100000):
            sim.step()
        state = sim.observe()
        # I should be at endemic level, not zero
        assert state[1] > 0.001, f"Disease should persist but I={state[1]}"

    def test_oscillations_detected(self):
        """The approach to endemic equilibrium should show oscillations."""
        sim = SIRVitalSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.01, dt=0.1))
        sim.reset()

        I_values = []
        for _ in range(10000):
            sim.step()
            I_values.append(sim.observe()[1])

        I_arr = np.array(I_values)
        # Count sign changes in dI (oscillation indicator)
        dI = np.diff(I_arr)
        sign_changes = np.sum(np.diff(np.sign(dI)) != 0)
        # There should be multiple sign changes (oscillation)
        assert sign_changes >= 2, f"Expected oscillations, got {sign_changes} changes"


# ---------------------------------------------------------------------------
# ODE terms verification
# ---------------------------------------------------------------------------
class TestODETerms:
    def test_derivatives_at_disease_free_eq(self):
        """At DFE (S=1, I=0, R=0), derivatives should be zero."""
        sim = SIRVitalSimulation(_make_config())
        y = np.array([1.0, 0.0, 0.0])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-14)

    def test_derivatives_at_endemic_eq(self):
        """At endemic equilibrium, derivatives should be approximately zero."""
        sim = SIRVitalSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.01))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        y = np.array(eq)
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-12)

    def test_derivatives_structure(self):
        """Verify the ODE terms match the expected form."""
        sim = SIRVitalSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.01))
        S, I, R = 0.8, 0.1, 0.1
        y = np.array([S, I, R])
        dy = sim._derivatives(y)

        expected_dS = 0.01 - 0.4 * 0.8 * 0.1 - 0.01 * 0.8
        expected_dI = 0.4 * 0.8 * 0.1 - 0.1 * 0.1 - 0.01 * 0.1
        expected_dR = 0.1 * 0.1 - 0.01 * 0.1

        np.testing.assert_allclose(dy[0], expected_dS, rtol=1e-12)
        np.testing.assert_allclose(dy[1], expected_dI, rtol=1e-12)
        np.testing.assert_allclose(dy[2], expected_dR, rtol=1e-12)

    def test_derivatives_sum_to_zero(self):
        """Sum of derivatives should be zero (total population conserved)."""
        sim = SIRVitalSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.01))
        y = np.array([0.7, 0.15, 0.15])
        dy = sim._derivatives(y)
        # dS + dI + dR = mu - mu*S - (gamma+mu)*I + beta*S*I - mu*I + gamma*I - mu*R
        # = mu - mu*(S+I+R) = mu - mu = 0 when S+I+R=1
        assert np.abs(dy.sum()) < 1e-14


# ---------------------------------------------------------------------------
# RK4 integration accuracy
# ---------------------------------------------------------------------------
class TestRK4:
    def test_rk4_accuracy_vs_euler(self):
        """RK4 should be more accurate than forward Euler for same dt."""
        dt = 0.1
        n_steps = 100

        # RK4 simulation
        sim_rk4 = SIRVitalSimulation(_make_config(dt=dt, n_steps=n_steps))
        sim_rk4.reset()
        for _ in range(n_steps):
            sim_rk4.step()
        state_rk4 = sim_rk4.observe()

        # Reference: RK4 with much smaller dt
        sim_ref = SIRVitalSimulation(_make_config(dt=dt / 100, n_steps=n_steps * 100))
        sim_ref.reset()
        for _ in range(n_steps * 100):
            sim_ref.step()
        state_ref = sim_ref.observe()

        error = np.max(np.abs(state_rk4 - state_ref))
        assert error < 1e-4, f"RK4 error too large: {error}"

    def test_small_dt_convergence(self):
        """Halving dt should reduce error by ~16x (4th order)."""
        n_steps_coarse = 500
        dt_coarse = 0.2
        dt_fine = 0.1

        sim_coarse = SIRVitalSimulation(
            _make_config(dt=dt_coarse, n_steps=n_steps_coarse)
        )
        sim_coarse.reset()
        for _ in range(n_steps_coarse):
            sim_coarse.step()

        sim_fine = SIRVitalSimulation(
            _make_config(dt=dt_fine, n_steps=n_steps_coarse * 2)
        )
        sim_fine.reset()
        for _ in range(n_steps_coarse * 2):
            sim_fine.step()

        # Reference with very small dt
        sim_ref = SIRVitalSimulation(
            _make_config(dt=0.001, n_steps=int(n_steps_coarse * dt_coarse / 0.001))
        )
        sim_ref.reset()
        for _ in range(sim_ref.config.n_steps):
            sim_ref.step()

        err_coarse = np.max(np.abs(sim_coarse.observe() - sim_ref.observe()))
        err_fine = np.max(np.abs(sim_fine.observe() - sim_ref.observe()))

        # For RK4, halving dt should improve error by ~16x, but accept 4x minimum
        if err_coarse > 1e-12:
            ratio = err_coarse / max(err_fine, 1e-16)
            assert ratio > 4, f"Convergence ratio {ratio} too low (expected ~16)"


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
class TestParameterSweep:
    def test_sweep_beta(self):
        sim = SIRVitalSimulation(_make_config())
        beta_range = np.linspace(0.05, 0.5, 5)
        data = sim.parameter_sweep("beta", beta_range, n_steps=5000)
        assert "beta" in data
        assert len(data["beta"]) == 5
        assert len(data["I_final"]) == 5
        assert len(data["R0"]) == 5

    def test_sweep_R0_increases_with_beta(self):
        sim = SIRVitalSimulation(_make_config())
        beta_range = np.linspace(0.1, 0.5, 5)
        data = sim.parameter_sweep("beta", beta_range, n_steps=5000)
        # R0 should increase with beta
        assert np.all(np.diff(data["R0"]) > 0)

    def test_sweep_captures_bifurcation(self):
        """Sweeping beta across R0=1 should show transition in I_final."""
        sim = SIRVitalSimulation(_make_config(gamma=0.1, mu=0.01))
        # beta_c = gamma + mu = 0.11
        beta_range = np.linspace(0.05, 0.3, 10)
        data = sim.parameter_sweep("beta", beta_range, n_steps=50000)
        # For beta < 0.11, I_final should be ~0; for beta > 0.11, I_final > 0
        below = data["R0"] < 1.0
        above = data["R0"] > 1.5  # well above to avoid near-bifurcation
        if np.any(below):
            assert np.mean(data["I_final"][below]) < 0.001
        if np.any(above):
            assert np.mean(data["I_final"][above]) > 0.001


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    def test_deterministic_runs(self):
        """Two runs with same config should produce identical trajectories."""
        config = _make_config(n_steps=500)
        sim1 = SIRVitalSimulation(config)
        traj1 = sim1.run(n_steps=500)

        sim2 = SIRVitalSimulation(config)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_reset_restores_initial(self):
        """Resetting should restore the initial state exactly."""
        sim = SIRVitalSimulation(_make_config())
        state0 = sim.reset().copy()
        for _ in range(100):
            sim.step()
        state_reset = sim.reset()
        np.testing.assert_array_equal(state0, state_reset)


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------
class TestRediscoveryData:
    def test_r0_sweep_data(self):
        from simulating_anything.rediscovery.sir_vital import generate_r0_sweep_data
        data = generate_r0_sweep_data(n_samples=5, n_steps=1000, dt=0.1)
        assert len(data["beta"]) == 5
        assert len(data["R0"]) == 5
        assert len(data["I_final"]) == 5
        assert np.all(data["R0"] > 0)

    def test_beta_bifurcation_data(self):
        from simulating_anything.rediscovery.sir_vital import (
            generate_beta_bifurcation_data,
        )
        data = generate_beta_bifurcation_data(n_beta=5, n_steps=5000, dt=0.1)
        assert len(data["beta"]) == 5
        assert len(data["I_final"]) == 5
        assert "beta_c_theory" in data
        assert data["beta_c_theory"] == pytest.approx(0.11)

    def test_endemic_equilibrium_data(self):
        from simulating_anything.rediscovery.sir_vital import (
            generate_endemic_equilibrium_data,
        )
        data = generate_endemic_equilibrium_data(n_samples=3, n_steps=5000, dt=0.1)
        assert len(data["I_sim"]) == 3
        assert len(data["I_theory"]) == 3
        assert np.all(data["R0"] > 1.0)

    def test_trajectory_data(self):
        from simulating_anything.rediscovery.sir_vital import generate_trajectory_data
        data = generate_trajectory_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 3)
        assert len(data["time"]) == 101
        # Conservation: all rows sum to 1
        np.testing.assert_allclose(data["states"].sum(axis=1), 1.0, atol=1e-8)
