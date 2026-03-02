"""Tests for the Zombie SIR epidemic model (Munz et al. 2009)."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.zombie_sir import ZombieSIRSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for ZombieSIR with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.0095, "alpha": 0.005, "zeta": 0.0001,
        "delta": 0.0001, "rho": 0.5, "Pi": 0.0, "N": 500.0,
        "S_0": 499.0, "I_0": 0.0, "Z_0": 1.0, "R_0_init": 0.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.ZOMBIE_SIR,  
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestZombieSIRConstruction:
    """Tests for construction and parameter defaults."""

    def test_default_construction(self):
        """Simulation should construct with default parameters."""
        sim = ZombieSIRSimulation(_make_config())
        assert sim.beta == 0.0095
        assert sim.alpha == 0.005
        assert sim.zeta == 0.0001
        assert sim.delta == 0.0001
        assert sim.rho == 0.5
        assert sim.Pi == 0.0
        assert sim.N == 500.0

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = ZombieSIRSimulation(_make_config(beta=0.02, alpha=0.01))
        assert sim.beta == 0.02
        assert sim.alpha == 0.01

    def test_initial_population(self):
        """Default initial conditions: S=499, I=0, Z=1, R=0."""
        sim = ZombieSIRSimulation(_make_config())
        assert sim.S_0 == 499.0
        assert sim.I_0 == 0.0
        assert sim.Z_0 == 1.0
        assert sim.R_0_init == 0.0


class TestZombieSIRState:
    """Tests for state shape and initialization."""

    def test_reset_state_shape(self):
        """Reset should return 4-element state [S, I, Z, R]."""
        sim = ZombieSIRSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)

    def test_reset_state_values(self):
        """Initial state should match parameters."""
        sim = ZombieSIRSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [499.0, 0.0, 1.0, 0.0])

    def test_observe_shape(self):
        """Observe should return 4-element state."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)


class TestZombieSIRConservation:
    """Tests for population conservation and boundedness."""

    def test_population_conservation_no_birth_no_resurrection(self):
        """S+I+Z+R = N when Pi=0 and zeta=0 (no births, no resurrection)."""
        sim = ZombieSIRSimulation(
            _make_config(Pi=0.0, zeta=0.0, dt=0.05)
        )
        sim.reset()
        N = sim.N
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            total = state.sum()
            np.testing.assert_allclose(
                total, N, rtol=1e-4,
                err_msg=f"Population not conserved: {total} != {N}",
            )

    def test_population_conservation_default(self):
        """With small zeta and Pi=0, total should be approximately conserved.

        Zeta redistributes from R to Z (no net change), so total is exact.
        """
        sim = ZombieSIRSimulation(_make_config(Pi=0.0, dt=0.05))
        sim.reset()
        N = sim.N
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            total = state.sum()
            np.testing.assert_allclose(
                total, N, rtol=1e-3,
                err_msg=f"Population changed: {total} vs {N}",
            )

    def test_population_grows_with_births(self):
        """With Pi > 0, total population should increase over time."""
        sim = ZombieSIRSimulation(_make_config(Pi=1.0, dt=0.1))
        sim.reset()
        N_initial = sim.observe().sum()
        for _ in range(1000):
            sim.step()
        N_final = sim.observe().sum()
        assert N_final > N_initial, (
            f"Population should grow with Pi>0: {N_final} <= {N_initial}"
        )

    def test_non_negative_populations(self):
        """All compartments should remain non-negative."""
        sim = ZombieSIRSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), (
                f"Negative population detected: {state}"
            )

    def test_bounded_populations(self):
        """Populations should not diverge (stay bounded by N when Pi=0)."""
        sim = ZombieSIRSimulation(_make_config(Pi=0.0, dt=0.1))
        sim.reset()
        N = sim.N
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            total = state.sum()
            assert total < N * 2, f"Population diverged: {total}"


class TestZombieSIRDynamics:
    """Tests for outbreak dynamics and equilibria."""

    def test_zombie_outbreak_occurs(self):
        """Starting with Z=1, zombie count should initially increase."""
        sim = ZombieSIRSimulation(_make_config(dt=0.1))
        sim.reset()
        initial_Z = sim.observe()[2]
        for _ in range(100):
            sim.step()
        Z_after = sim.observe()[2]
        assert Z_after > initial_Z, (
            f"Zombie outbreak should occur: Z went from {initial_Z} to {Z_after}"
        )

    def test_susceptibles_decrease(self):
        """Susceptibles should decrease during outbreak."""
        sim = ZombieSIRSimulation(_make_config(dt=0.1))
        sim.reset()
        initial_S = sim.observe()[0]
        for _ in range(500):
            sim.step()
        S_after = sim.observe()[0]
        assert S_after < initial_S, (
            f"S should decrease: {S_after} >= {initial_S}"
        )

    def test_infected_transient(self):
        """Infected should appear as transient (bitten but not yet zombie)."""
        sim = ZombieSIRSimulation(_make_config(dt=0.1))
        sim.reset()
        max_I = 0.0
        for _ in range(2000):
            sim.step()
            I_now = sim.observe()[1]
            if I_now > max_I:
                max_I = I_now
        # With high rho=0.5, infected turn to zombies quickly
        # but there should be some transient infected
        assert max_I > 0.0, "Infected should appear transiently"

    def test_doomsday_equilibrium_default(self):
        """With default params (beta > alpha), zombies eventually dominate."""
        sim = ZombieSIRSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        # With beta=0.0095 > alpha=0.005, zombies should win
        assert state[2] > state[0], (
            f"Zombies should dominate: Z={state[2]}, S={state[0]}"
        )

    def test_no_zombies_no_outbreak(self):
        """With Z=0 initially, no outbreak should occur."""
        sim = ZombieSIRSimulation(
            _make_config(Z_0=0.0, dt=0.1)
        )
        sim.reset()
        for _ in range(1000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(state[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(state[1], 0.0, atol=1e-10)

    def test_high_alpha_humans_survive(self):
        """With very high kill rate, humans should survive."""
        sim = ZombieSIRSimulation(
            _make_config(alpha=0.1, beta=0.0095, dt=0.1)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        # High alpha means humans kill zombies faster than zombies bite
        assert state[0] > 100, (
            f"Humans should survive with high alpha: S={state[0]}"
        )

    def test_low_alpha_zombies_win(self):
        """With very low kill rate, zombies should win."""
        sim = ZombieSIRSimulation(
            _make_config(alpha=0.001, beta=0.0095, dt=0.1)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        # Low alpha: zombies overwhelm humans
        assert state[0] < 10, (
            f"Zombies should dominate with low alpha: S={state[0]}"
        )


class TestZombieSIRParameterSensitivity:
    """Tests for parameter sensitivity."""

    def test_beta_sensitivity(self):
        """Higher beta should lead to faster zombie outbreak."""
        # Low beta
        sim_low = ZombieSIRSimulation(_make_config(beta=0.005, dt=0.1))
        sim_low.reset()
        for _ in range(2000):
            sim_low.step()
        Z_low = sim_low.observe()[2]

        # High beta
        sim_high = ZombieSIRSimulation(_make_config(beta=0.02, dt=0.1))
        sim_high.reset()
        for _ in range(2000):
            sim_high.step()
        Z_high = sim_high.observe()[2]

        assert Z_high > Z_low, (
            f"Higher beta should mean more zombies: Z_high={Z_high}, Z_low={Z_low}"
        )

    def test_rho_sensitivity(self):
        """Higher rho means faster zombification of infected."""
        # Low rho: infected linger longer
        sim_low = ZombieSIRSimulation(_make_config(rho=0.1, dt=0.1))
        sim_low.reset()
        max_I_low = 0.0
        for _ in range(5000):
            sim_low.step()
            I_now = sim_low.observe()[1]
            if I_now > max_I_low:
                max_I_low = I_now

        # High rho: infected turn quickly
        sim_high = ZombieSIRSimulation(_make_config(rho=1.0, dt=0.1))
        sim_high.reset()
        max_I_high = 0.0
        for _ in range(5000):
            sim_high.step()
            I_now = sim_high.observe()[1]
            if I_now > max_I_high:
                max_I_high = I_now

        # Lower rho should allow more infected to accumulate
        assert max_I_low > max_I_high, (
            f"Lower rho should give higher peak I: "
            f"max_I_low={max_I_low}, max_I_high={max_I_high}"
        )

    def test_resurrection_increases_zombies(self):
        """Higher zeta (resurrection) should increase final zombie count."""
        # No resurrection
        sim_no = ZombieSIRSimulation(_make_config(zeta=0.0, dt=0.1))
        sim_no.reset()
        for _ in range(30000):
            sim_no.step()
        Z_no = sim_no.observe()[2]

        # High resurrection
        sim_high = ZombieSIRSimulation(_make_config(zeta=0.01, dt=0.1))
        sim_high.reset()
        for _ in range(30000):
            sim_high.step()
        Z_high = sim_high.observe()[2]

        assert Z_high >= Z_no - 1.0, (
            f"Resurrection should not reduce zombies: "
            f"Z_high={Z_high}, Z_no={Z_no}"
        )

    def test_delta_sensitivity(self):
        """Higher delta (natural death of infected) diverts from zombification."""
        # Low delta: most infected become zombies
        sim_low = ZombieSIRSimulation(
            _make_config(delta=0.0, rho=0.5, dt=0.1)
        )
        sim_low.reset()
        for _ in range(20000):
            sim_low.step()
        Z_low_delta = sim_low.observe()[2]

        # High delta: infected die naturally instead of becoming zombies
        sim_high = ZombieSIRSimulation(
            _make_config(delta=5.0, rho=0.5, dt=0.1)
        )
        sim_high.reset()
        for _ in range(20000):
            sim_high.step()
        Z_high_delta = sim_high.observe()[2]

        # Higher delta should mean fewer zombies (infected die before turning)
        assert Z_high_delta <= Z_low_delta + 1.0, (
            f"Higher delta should reduce zombification: "
            f"Z_high_delta={Z_high_delta}, Z_low_delta={Z_low_delta}"
        )


class TestZombieSIRReproduction:
    """Tests for reproduction number and equilibria."""

    def test_R0_zombie_formula(self):
        """R0_zombie = beta / alpha."""
        sim = ZombieSIRSimulation(
            _make_config(beta=0.0095, alpha=0.005)
        )
        r0 = sim.compute_basic_reproduction()
        assert r0 == pytest.approx(0.0095 / 0.005)

    def test_R0_zombie_greater_than_one(self):
        """Default params give R0_z > 1 (outbreak grows)."""
        sim = ZombieSIRSimulation(_make_config())
        r0 = sim.compute_basic_reproduction()
        assert r0 > 1.0, f"R0_z should be > 1 for default params: {r0}"

    def test_R0_zombie_less_than_one(self):
        """High alpha gives R0_z < 1 (outbreak controlled)."""
        sim = ZombieSIRSimulation(_make_config(alpha=0.02, beta=0.0095))
        r0 = sim.compute_basic_reproduction()
        assert r0 < 1.0, f"R0_z should be < 1 with high alpha: {r0}"

    def test_R0_infinite_when_no_kill(self):
        """R0_z = inf when alpha = 0 (no zombie killing)."""
        sim = ZombieSIRSimulation(_make_config(alpha=0.0))
        r0 = sim.compute_basic_reproduction()
        assert r0 == float("inf")

    def test_equilibria_disease_free(self):
        """Disease-free equilibrium should be [N, 0, 0, 0]."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        eq = sim.compute_equilibria()
        np.testing.assert_allclose(eq["disease_free"], [500.0, 0.0, 0.0, 0.0])

    def test_equilibria_doomsday(self):
        """Doomsday equilibrium should be [0, 0, N, 0]."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        eq = sim.compute_equilibria()
        np.testing.assert_allclose(eq["doomsday"], [0.0, 0.0, 500.0, 0.0])


class TestZombieSIRTrajectory:
    """Tests for trajectory collection and reproducibility."""

    def test_trajectory_collection(self):
        """run() should collect trajectory with correct shapes."""
        sim = ZombieSIRSimulation(_make_config(n_steps=100, dt=0.1))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps(self):
        """Timestamps should be evenly spaced at dt intervals."""
        sim = ZombieSIRSimulation(_make_config(n_steps=50, dt=0.1))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * 0.1
        np.testing.assert_allclose(traj.timestamps, expected, rtol=1e-10)

    def test_deterministic(self):
        """Same config produces identical trajectories."""
        config = _make_config(n_steps=200, dt=0.1)
        sim1 = ZombieSIRSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = ZombieSIRSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_no_nan(self):
        """No NaN or Inf after long simulation."""
        sim = ZombieSIRSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(100000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf: {state}"


class TestZombieSIRAnalysisMethods:
    """Tests for sweep and analysis methods."""

    def test_alpha_sweep(self):
        """Alpha sweep should return arrays of correct length."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        alpha_vals = np.linspace(0.001, 0.05, 5)
        result = sim.alpha_sweep(alpha_vals, n_steps=1000, dt=0.1)
        assert len(result["alpha_values"]) == 5
        assert len(result["final_S"]) == 5
        assert len(result["final_Z"]) == 5
        assert len(result["survived"]) == 5
        assert np.all(np.isfinite(result["final_S"]))
        assert np.all(np.isfinite(result["final_Z"]))

    def test_alpha_sweep_monotonic(self):
        """Higher alpha should generally lead to more survivors."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        alpha_vals = np.array([0.002, 0.01, 0.05])
        result = sim.alpha_sweep(alpha_vals, n_steps=20000, dt=0.1)
        # Higher alpha should give more surviving humans
        assert result["final_S"][2] >= result["final_S"][0], (
            f"Higher alpha should mean more survivors: "
            f"S(0.05)={result['final_S'][2]}, S(0.002)={result['final_S'][0]}"
        )

    def test_outbreak_dynamics(self):
        """Outbreak dynamics should return all expected arrays."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        result = sim.outbreak_dynamics(n_steps=500, dt=0.1)
        assert len(result["time"]) == 501
        assert len(result["S"]) == 501
        assert len(result["I"]) == 501
        assert len(result["Z"]) == 501
        assert len(result["R"]) == 501
        assert "peak_Z" in result
        assert "peak_Z_time" in result
        assert "final_S" in result
        assert "final_Z" in result

    def test_outbreak_dynamics_peak(self):
        """Peak zombie count should be positive and occur after t=0."""
        sim = ZombieSIRSimulation(_make_config())
        sim.reset()
        result = sim.outbreak_dynamics(n_steps=5000, dt=0.1)
        assert result["peak_Z"] > 1.0, (
            f"Peak Z should be > 1: {result['peak_Z']}"
        )
        assert result["peak_Z_time"] > 0.0, (
            f"Peak should occur after t=0: {result['peak_Z_time']}"
        )


class TestZombieSIRRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        """ODE data should produce valid arrays."""
        from simulating_anything.rediscovery.zombie_sir import generate_ode_data
        data = generate_ode_data(n_steps=500, dt=0.1)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert len(data["S"]) == 501
        assert len(data["I"]) == 501
        assert len(data["Z"]) == 501
        assert len(data["R"]) == 501
        assert np.all(np.isfinite(data["states"]))

    def test_alpha_sweep_data(self):
        """Alpha sweep should produce results for each value."""
        from simulating_anything.rediscovery.zombie_sir import generate_alpha_sweep
        data = generate_alpha_sweep(n_alpha=5, n_steps=1000, dt=0.1)
        assert len(data["alpha_values"]) == 5
        assert len(data["final_S"]) == 5
        assert len(data["final_Z"]) == 5
        assert len(data["survived"]) == 5
        assert len(data["R0_z"]) == 5
        assert np.all(np.isfinite(data["final_S"]))

    def test_resurrection_sweep_data(self):
        """Resurrection sweep should produce valid results."""
        from simulating_anything.rediscovery.zombie_sir import (
            generate_resurrection_sweep,
        )
        data = generate_resurrection_sweep(n_zeta=5, n_steps=1000, dt=0.1)
        assert len(data["zeta_values"]) == 5
        assert len(data["final_S"]) == 5
        assert len(data["final_Z"]) == 5
        assert len(data["peak_Z"]) == 5
        assert np.all(np.isfinite(data["final_Z"]))
