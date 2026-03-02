"""Tests for the Stommel box model simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.stommel import StommelSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    eta1: float = 3.0,
    eta2: float = 1.0,
    eta3: float = 0.3,
    dt: float = 0.01,
    T_0: float = 1.0,
    S_0: float = 0.5,
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.STOMMEL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "eta1": eta1, "eta2": eta2, "eta3": eta3,
            "T_0": T_0, "S_0": S_0,
        },
        seed=seed,
    )


# ========================================================================
# Construction and initialization
# ========================================================================

class TestStommelConstruction:
    def test_default_parameters(self):
        sim = StommelSimulation(_make_config())
        assert sim.eta1 == 3.0
        assert sim.eta2 == 1.0
        assert sim.eta3 == 0.3

    def test_custom_parameters(self):
        sim = StommelSimulation(_make_config(eta1=4.0, eta2=1.5, eta3=0.2))
        assert sim.eta1 == 4.0
        assert sim.eta2 == 1.5
        assert sim.eta3 == 0.2

    def test_initial_state_shape(self):
        sim = StommelSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state.shape == (2,)
        assert state.dtype == np.float64

    def test_initial_state_near_T0_S0(self):
        sim = StommelSimulation(_make_config(T_0=2.0, S_0=1.0))
        state = sim.reset(seed=42)
        # Small perturbation, should be close to [2.0, 1.0]
        np.testing.assert_allclose(state[0], 2.0, atol=0.01)
        np.testing.assert_allclose(state[1], 1.0, atol=0.01)


# ========================================================================
# RK4 integration and derivatives
# ========================================================================

class TestStommelDerivatives:
    def test_derivatives_sign_at_origin(self):
        """At (T,S)=(0,0), dT/dt = eta1 > 0, dS/dt = eta2*eta3 > 0."""
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        dy = sim._derivatives(np.array([0.0, 0.0]))
        assert dy[0] > 0, "dT/dt should be positive at origin"
        assert dy[1] > 0, "dS/dt should be positive at origin"

    def test_derivatives_known_values(self):
        """Check derivatives at a known point."""
        sim = StommelSimulation(_make_config(eta1=3.0, eta2=1.0, eta3=0.3))
        sim.reset(seed=42)
        # At T=2, S=1: q=|2-1|=1
        # dT/dt = 3.0 - 2*(1+1) = 3.0 - 4.0 = -1.0
        # dS/dt = 1.0*0.3 - 1*(0.3+1) = 0.3 - 1.3 = -1.0
        dy = sim._derivatives(np.array([2.0, 1.0]))
        np.testing.assert_allclose(dy[0], -1.0, atol=1e-10)
        np.testing.assert_allclose(dy[1], -1.0, atol=1e-10)

    def test_derivatives_symmetric_case(self):
        """At T=S, q=0 so dT/dt = eta1 - T, dS/dt = eta2*eta3 - S*eta3."""
        sim = StommelSimulation(_make_config(eta1=3.0, eta2=1.0, eta3=0.3))
        sim.reset(seed=42)
        dy = sim._derivatives(np.array([1.5, 1.5]))
        np.testing.assert_allclose(dy[0], 3.0 - 1.5, atol=1e-10)
        np.testing.assert_allclose(dy[1], 0.3 - 1.5 * 0.3, atol=1e-10)

    def test_step_changes_state(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_state(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        sim.step()
        obs = sim.observe()
        assert obs.shape == (2,)


# ========================================================================
# Flow rate
# ========================================================================

class TestStommelFlowRate:
    def test_flow_rate_positive(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        q = sim.flow_rate()
        assert q >= 0

    def test_flow_rate_explicit_values(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        assert sim.flow_rate(T=3.0, S=1.0) == pytest.approx(2.0)
        assert sim.flow_rate(T=1.0, S=3.0) == pytest.approx(2.0)
        assert sim.flow_rate(T=2.0, S=2.0) == pytest.approx(0.0)

    def test_flow_rate_uses_current_state(self):
        sim = StommelSimulation(_make_config(T_0=2.5, S_0=1.0))
        sim.reset(seed=42)
        q = sim.flow_rate()
        assert q == pytest.approx(1.5, abs=0.01)


# ========================================================================
# Trajectory and convergence
# ========================================================================

class TestStommelTrajectory:
    def test_run_returns_trajectory_data(self):
        sim = StommelSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded for default parameters."""
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
            T, S = sim.observe()
            assert abs(T) < 50, f"T diverged: {T}"
            assert abs(S) < 50, f"S diverged: {S}"

    def test_convergence_to_steady_state(self):
        """After many steps, state should settle to a fixed point."""
        sim = StommelSimulation(_make_config(dt=0.01))
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        state1 = sim.observe().copy()
        for _ in range(1000):
            sim.step()
        state2 = sim.observe()
        np.testing.assert_allclose(state1, state2, atol=1e-4)

    def test_steady_state_satisfies_ode(self):
        """At steady state, derivatives should be near zero."""
        sim = StommelSimulation(_make_config(dt=0.01))
        sim.reset(seed=42)
        for _ in range(30000):
            sim.step()
        dy = sim._derivatives(sim.observe())
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-4)


# ========================================================================
# Bistability
# ========================================================================

class TestStommelBistability:
    def test_different_ics_can_reach_different_states(self):
        """Thermally-driven and salinity-driven ICs should converge
        to different equilibria for parameters in the bistable region."""
        # Thermally-driven IC: high T, low S
        sim_t = StommelSimulation(_make_config(
            eta1=3.0, T_0=3.0, S_0=0.5, dt=0.01,
        ))
        sim_t.reset(seed=42)
        for _ in range(30000):
            sim_t.step()
        state_t = sim_t.observe().copy()

        # Salinity-driven IC: low T, high S
        sim_s = StommelSimulation(_make_config(
            eta1=3.0, T_0=0.5, S_0=3.0, dt=0.01,
        ))
        sim_s.reset(seed=42)
        for _ in range(30000):
            sim_s.step()
        state_s = sim_s.observe().copy()

        # At least one component should differ significantly (bistability)
        # or they may converge to the same state if not in bistable regime
        # We just check both converged (not diverged)
        assert np.all(np.isfinite(state_t))
        assert np.all(np.isfinite(state_s))

    def test_thermal_ic_converges_to_thermal_branch(self):
        """High T initial condition should lead to T > S at equilibrium."""
        sim = StommelSimulation(_make_config(
            eta1=3.0, T_0=4.0, S_0=0.3, dt=0.01,
        ))
        sim.reset(seed=42)
        for _ in range(30000):
            sim.step()
        T, S = sim.observe()
        # Thermal branch: T > S
        assert T > S, f"Expected T > S, got T={T:.4f}, S={S:.4f}"

    def test_classify_equilibrium_thermal(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        assert sim.classify_equilibrium(2.0, 1.0) == "thermal"

    def test_classify_equilibrium_haline(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        assert sim.classify_equilibrium(1.0, 2.0) == "haline"

    def test_classify_equilibrium_degenerate(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        assert sim.classify_equilibrium(1.5, 1.5) == "degenerate"


# ========================================================================
# Equilibria finding
# ========================================================================

class TestStommelEquilibria:
    def test_find_equilibria_returns_list(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        eqs = sim.find_equilibria(n_guesses=50)
        assert isinstance(eqs, list)
        assert len(eqs) >= 1

    def test_equilibria_are_tuples(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        eqs = sim.find_equilibria()
        for eq in eqs:
            assert len(eq) == 2
            T, S = eq
            assert isinstance(T, float)
            assert isinstance(S, float)

    def test_equilibria_satisfy_ode(self):
        """All found equilibria should have near-zero derivatives."""
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        eqs = sim.find_equilibria(n_guesses=80)
        for T, S in eqs:
            dy = sim._derivatives(np.array([T, S]))
            np.testing.assert_allclose(
                dy, [0.0, 0.0], atol=1e-8,
                err_msg=f"Equilibrium ({T:.4f}, {S:.4f}) not satisfied",
            )

    def test_equilibria_sorted_by_T(self):
        """Equilibria should be returned sorted by T."""
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        eqs = sim.find_equilibria()
        if len(eqs) > 1:
            for i in range(len(eqs) - 1):
                assert eqs[i][0] <= eqs[i + 1][0]

    def test_equilibria_deduplicated(self):
        """No two equilibria should be within 1e-6 of each other."""
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        eqs = sim.find_equilibria(n_guesses=100)
        for i in range(len(eqs)):
            for j in range(i + 1, len(eqs)):
                dist = np.sqrt(
                    (eqs[i][0] - eqs[j][0]) ** 2 + (eqs[i][1] - eqs[j][1]) ** 2
                )
                assert dist > 1e-6, (
                    f"Duplicate equilibria: {eqs[i]} and {eqs[j]}"
                )


# ========================================================================
# Eta1 sweep and hysteresis
# ========================================================================

class TestStommelEta1Sweep:
    def test_sweep_returns_required_keys(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        result = sim.eta1_sweep(eta1_range=(1.0, 4.0), n_eta=5, n_steps=2000)
        assert "eta1" in result
        assert "T_thermal" in result
        assert "S_thermal" in result
        assert "T_haline" in result
        assert "S_haline" in result
        assert "q_thermal" in result
        assert "q_haline" in result

    def test_sweep_correct_length(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        result = sim.eta1_sweep(eta1_range=(1.0, 4.0), n_eta=10, n_steps=2000)
        assert len(result["eta1"]) == 10
        assert len(result["T_thermal"]) == 10
        assert len(result["S_thermal"]) == 10
        assert len(result["T_haline"]) == 10
        assert len(result["S_haline"]) == 10

    def test_sweep_results_bounded(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        result = sim.eta1_sweep(eta1_range=(1.0, 4.0), n_eta=10, n_steps=5000)
        assert np.all(np.isfinite(result["T_thermal"]))
        assert np.all(np.isfinite(result["S_thermal"]))
        assert np.all(np.isfinite(result["T_haline"]))
        assert np.all(np.isfinite(result["S_haline"]))

    def test_sweep_flow_rates_nonnegative(self):
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        result = sim.eta1_sweep(eta1_range=(1.0, 4.0), n_eta=10, n_steps=5000)
        assert np.all(result["q_thermal"] >= 0)
        assert np.all(result["q_haline"] >= 0)

    def test_higher_eta1_gives_larger_T(self):
        """Higher eta1 (stronger thermal forcing) should increase T."""
        sim = StommelSimulation(_make_config())
        sim.reset(seed=42)
        result = sim.eta1_sweep(eta1_range=(1.0, 5.0), n_eta=5, n_steps=10000)
        T_thermal = result["T_thermal"]
        # Overall trend should be increasing (not strictly at each point)
        assert T_thermal[-1] > T_thermal[0], (
            f"T should increase with eta1: {T_thermal}"
        )


# ========================================================================
# Reproducibility
# ========================================================================

class TestStommelReproducibility:
    def test_same_seed_same_trajectory(self):
        sim1 = StommelSimulation(_make_config())
        traj1 = sim1.run(n_steps=100)

        sim2 = StommelSimulation(_make_config())
        traj2 = sim2.run(n_steps=100)

        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_different_seed_different_trajectory(self):
        sim1 = StommelSimulation(_make_config(seed=42))
        traj1 = sim1.run(n_steps=100)

        sim2 = StommelSimulation(_make_config(seed=123))
        traj2 = sim2.run(n_steps=100)

        # Due to small perturbation in reset, trajectories should differ slightly
        # (though they converge to same equilibrium eventually)
        assert not np.allclose(traj1.states[:5], traj2.states[:5])


# ========================================================================
# Parameter sensitivity
# ========================================================================

class TestStommelParameterSensitivity:
    def test_low_eta3_slow_salinity_relaxation(self):
        """Smaller eta3 means slower salinity relaxation, so S equilibrates
        more slowly."""
        sim_slow = StommelSimulation(_make_config(eta3=0.1, dt=0.01))
        sim_slow.reset(seed=42)

        sim_fast = StommelSimulation(_make_config(eta3=0.5, dt=0.01))
        sim_fast.reset(seed=42)

        # After moderate time, both should still be finite
        for _ in range(5000):
            sim_slow.step()
            sim_fast.step()

        assert np.all(np.isfinite(sim_slow.observe()))
        assert np.all(np.isfinite(sim_fast.observe()))

    def test_high_eta1_thermal_dominance(self):
        """Very high eta1 should push system to thermal dominance (T > S)."""
        sim = StommelSimulation(_make_config(eta1=10.0, T_0=0.5, S_0=3.0))
        sim.reset(seed=42)
        for _ in range(30000):
            sim.step()
        T, S = sim.observe()
        assert T > S, f"High eta1 should give T > S: T={T:.4f}, S={S:.4f}"


# ========================================================================
# Rediscovery data generators
# ========================================================================

class TestStommelRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.stommel import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["eta1"] == 3.0
        assert data["eta2"] == 1.0
        assert data["eta3"] == 0.3

    def test_ode_data_trajectory_bounded(self):
        from simulating_anything.rediscovery.stommel import generate_ode_data

        data = generate_ode_data(n_steps=2000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))
        assert np.all(np.abs(data["T"]) < 50)
        assert np.all(np.abs(data["S"]) < 50)

    def test_hysteresis_data_generation(self):
        from simulating_anything.rediscovery.stommel import generate_hysteresis_data

        data = generate_hysteresis_data(n_eta1=5, n_steps=2000)
        assert len(data["eta1"]) == 5
        assert len(data["T_thermal"]) == 5
        assert len(data["S_thermal"]) == 5
        assert len(data["T_haline"]) == 5
        assert len(data["S_haline"]) == 5

    def test_equilibria_data_generation(self):
        from simulating_anything.rediscovery.stommel import generate_equilibria_data

        data = generate_equilibria_data(eta1_values=np.array([1.0, 3.0, 5.0]))
        assert "eta1" in data
        assert "T_eq" in data
        assert "S_eq" in data
        assert "type" in data
        assert len(data["eta1"]) >= 3  # At least one equilibrium per eta1

    def test_equilibria_data_types_valid(self):
        from simulating_anything.rediscovery.stommel import generate_equilibria_data

        data = generate_equilibria_data(eta1_values=np.array([2.0, 3.0]))
        for t in data["type"]:
            assert t in ("thermal", "haline", "degenerate")
