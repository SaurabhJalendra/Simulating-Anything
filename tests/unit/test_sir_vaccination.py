"""Tests for the SIR with vaccination and vital dynamics model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sir_vaccination import (
    SIRVaccinationSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(
    dt: float = 0.1,
    n_steps: int = 5000,
    **param_overrides: float,
) -> SIRVaccinationSimulation:
    """Create an SIRVaccinationSimulation with default SIR-vaccination parameters.

    Default parameters: beta=0.3, gamma=0.1, mu=0.01, nu=0.0,
    N=1000, S_0=990, I_0=10, R_0_init=0.
    """
    params: dict[str, float] = {
        "beta": 0.3,
        "gamma": 0.1,
        "mu": 0.01,
        "nu": 0.0,
        "N": 1000.0,
        "S_0": 990.0,
        "I_0": 10.0,
        "R_0_init": 0.0,
    }
    params.update(param_overrides)
    config = SimulationConfig(
        domain=Domain.SIR_VACCINATION,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    return SIRVaccinationSimulation(config)


class TestSIRVaccinationSimulation:
    """Core simulation behaviour and epidemiological property tests."""

    def test_initial_state(self):
        """State after reset has shape (3,) and values match S_0, I_0, R_0_init."""
        sim = _make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        np.testing.assert_allclose(state, [990.0, 10.0, 0.0])

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = _make_sim()
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_trajectory_shape(self):
        """run(n) returns TrajectoryData with states of shape (n+1, 3)."""
        n = 200
        sim = _make_sim(n_steps=n)
        traj = sim.run(n_steps=n)
        assert traj.states.shape == (n + 1, 3)

    def test_state_dim(self):
        """State vector is 3-dimensional: [S, I, R]."""
        sim = _make_sim()
        state = sim.reset()
        assert state.ndim == 1
        assert state.shape[0] == 3

    def test_observe(self):
        """observe() returns the current state vector."""
        sim = _make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs, sim._state)

    def test_population_conservation(self):
        """S + I + R should remain approximately N over many steps.

        With balanced vital dynamics (birth rate = death rate = mu),
        total population is conserved by the ODEs.
        """
        sim = _make_sim()
        sim.reset()
        N0 = sim.total_population
        for _ in range(2000):
            sim.step()
        N_final = sim.total_population
        rel_drift = abs(N_final - N0) / N0
        assert rel_drift < 0.01, f"Population drift {rel_drift:.6f} too large"

    def test_non_negative_populations(self):
        """All compartments should remain >= 0 throughout the simulation."""
        sim = _make_sim(dt=0.1)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative state: {state}"

    def test_disease_free_equilibrium_property(self):
        """DFE should be [S*, 0, R*] with correct analytical values."""
        nu, mu, N = 0.05, 0.01, 1000.0
        sim = _make_sim(nu=nu, mu=mu, N=N)
        dfe = sim.disease_free_equilibrium()
        assert dfe.shape == (3,)
        # I* = 0
        assert dfe[1] == pytest.approx(0.0)
        # S* = mu*N / (nu + mu)
        assert dfe[0] == pytest.approx(mu * N / (nu + mu))
        # R* = nu*N / (nu + mu)
        assert dfe[2] == pytest.approx(nu * N / (nu + mu))

    def test_disease_free_convergence(self):
        """With no initial infected, system should converge to the DFE."""
        sim = _make_sim(
            nu=0.5, mu=0.01, N=1000.0,
            S_0=900.0, I_0=0.0, R_0_init=100.0,
            dt=0.1, n_steps=100000,
        )
        sim.reset()
        for _ in range(100000):
            sim.step()
        state = sim.observe()
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(state, dfe, atol=1.0)

    def test_endemic_behavior(self):
        """With no vaccination and R0 > 1, infection grows significantly."""
        sim = _make_sim(
            beta=0.3, gamma=0.1, mu=0.01, nu=0.0,
            N=1000.0, S_0=990.0, I_0=10.0,
        )
        assert sim.compute_r0() > 1.0
        sim.reset()
        peak_I = 0.0
        for _ in range(10000):
            state = sim.step()
            peak_I = max(peak_I, state[1])
        assert peak_I > 50.0

    def test_vaccination_reduces_peak(self):
        """Higher vaccination rate should reduce peak infected count."""
        sim_no = _make_sim(nu=0.0)
        sim_no.reset()
        peak_no = 0.0
        for _ in range(10000):
            peak_no = max(peak_no, sim_no.step()[1])

        sim_vacc = _make_sim(nu=0.05)
        sim_vacc.reset()
        peak_vacc = 0.0
        for _ in range(10000):
            peak_vacc = max(peak_vacc, sim_vacc.step()[1])

        assert peak_vacc < peak_no

    def test_waning_immunity_effect(self):
        """Without vaccination, recovered individuals can be reinfected via mu.

        In this model mu represents vital dynamics (death/birth), not
        classical waning immunity. Individuals leave R by dying (rate mu)
        and newborns enter S (rate mu*N). With nu>0, vaccination moves
        S to R, so more nu means less S available -- effectively the R
        compartment grows.
        """
        sim_with_nu = _make_sim(nu=0.05, mu=0.01)
        sim_with_nu.reset()
        for _ in range(5000):
            sim_with_nu.step()
        state_vacc = sim_with_nu.observe()
        # With vaccination, R should be larger than without
        sim_no_nu = _make_sim(nu=0.0, mu=0.01)
        sim_no_nu.reset()
        for _ in range(5000):
            sim_no_nu.step()
        state_no = sim_no_nu.observe()
        # Vaccination transfers S to R, so R should be higher with nu > 0
        assert state_vacc[2] > state_no[2]

    def test_different_parameters(self):
        """Simulation should accept different parameter combinations."""
        sim = _make_sim(beta=0.6, gamma=0.2, mu=0.005, nu=0.1, N=500.0)
        state = sim.reset()
        assert state.shape == (3,)
        sim.step()
        assert np.all(sim.observe() >= 0)

    def test_reset_deterministic(self):
        """Same configuration produces identical trajectories."""
        sim1 = _make_sim()
        sim1.reset()
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = _make_sim()
        sim2.reset()
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2)

    def test_trajectory_data_metadata(self):
        """TrajectoryData should contain parameters from the config."""
        sim = _make_sim(n_steps=50)
        traj = sim.run(n_steps=50)
        assert "beta" in traj.parameters
        assert "gamma" in traj.parameters
        assert traj.parameters["beta"] == pytest.approx(0.3)
        assert traj.parameters["gamma"] == pytest.approx(0.1)

    def test_r0_property(self):
        """R0 = beta / (gamma + mu)."""
        sim = _make_sim(beta=0.3, gamma=0.1, mu=0.01)
        expected = 0.3 / (0.1 + 0.01)
        assert sim.compute_r0() == pytest.approx(expected, rel=1e-10)

    def test_r_eff_property(self):
        """R_eff = R0 * mu / (nu + mu)."""
        sim = _make_sim(beta=0.3, gamma=0.1, mu=0.01, nu=0.05)
        r0 = sim.compute_r0()
        expected = r0 * 0.01 / (0.05 + 0.01)
        assert sim.compute_r_eff() == pytest.approx(expected, rel=1e-10)

    def test_r_eff_decreases_with_vaccination(self):
        """Higher vaccination rate should lower R_eff."""
        sim_low = _make_sim(nu=0.01)
        sim_high = _make_sim(nu=0.1)
        assert sim_high.compute_r_eff() < sim_low.compute_r_eff()

    def test_critical_vaccination_rate(self):
        """nu_c = mu * (R0 - 1)."""
        sim = _make_sim(beta=0.3, gamma=0.1, mu=0.01)
        r0 = sim.compute_r0()
        expected = 0.01 * (r0 - 1.0)
        assert sim.critical_vaccination_rate() == pytest.approx(expected)

    def test_critical_vaccination_rate_low_r0(self):
        """When R0 <= 1, critical vaccination rate should be 0."""
        sim = _make_sim(beta=0.05, gamma=0.1, mu=0.01)
        assert sim.compute_r0() < 1.0
        assert sim.critical_vaccination_rate() == 0.0

    def test_herd_immunity_fraction(self):
        """Herd immunity threshold: p_c = 1 - 1/R0."""
        sim = _make_sim(beta=0.3, gamma=0.1, mu=0.01)
        r0 = sim.compute_r0()
        expected = 1.0 - 1.0 / r0
        assert sim.herd_immunity_threshold() == pytest.approx(expected)

    def test_measure_peak_infected(self):
        """Peak infected should be a positive float in an epidemic."""
        sim = _make_sim(nu=0.0)
        sim.reset()
        peak_I = 0.0
        for _ in range(10000):
            state = sim.step()
            peak_I = max(peak_I, state[1])
        assert isinstance(peak_I, float)
        assert peak_I > 0.0

    def test_measure_final_size(self):
        """After an epidemic, the recovered compartment should be positive."""
        sim = _make_sim(nu=0.0, mu=0.0)
        sim.reset()
        for _ in range(5000):
            sim.step()
        final_R = float(sim.observe()[2])
        assert isinstance(final_R, float)
        assert final_R > 0.0

    def test_nu_c_makes_r_eff_unity(self):
        """At nu = nu_c, R_eff should be approximately 1."""
        sim_base = _make_sim()
        nu_c = sim_base.critical_vaccination_rate()
        sim_crit = _make_sim(nu=nu_c)
        assert sim_crit.compute_r_eff() == pytest.approx(1.0, abs=1e-8)

    def test_endemic_equilibrium_population_sum(self):
        """Endemic equilibrium should have S* + I* + R* = N."""
        sim = _make_sim(nu=0.0)
        ee = sim.endemic_equilibrium()
        assert ee is not None
        assert ee.sum() == pytest.approx(1000.0, rel=1e-6)

    def test_endemic_equilibrium_none_when_eliminated(self):
        """No endemic equilibrium when R_eff < 1 (disease eliminated)."""
        sim = _make_sim(nu=1.0)
        assert sim.compute_r_eff() < 1.0
        assert sim.endemic_equilibrium() is None


class TestSIRVaccinationRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_generate_ode_data(self):
        """ODE data should have correct shape, keys, and parameter values."""
        from simulating_anything.rediscovery.sir_vaccination import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 3)
        assert data["beta"] == 0.3
        assert data["gamma"] == 0.1
        assert data["mu"] == 0.01
        assert data["nu"] == 0.02
        assert np.all(data["states"] >= 0)

    def test_vaccination_sweep(self):
        """Vaccination sweep should return valid arrays with expected keys."""
        from simulating_anything.rediscovery.sir_vaccination import (
            generate_vaccination_sweep_data,
        )

        data = generate_vaccination_sweep_data(
            n_nu=5, n_steps=1000, dt=0.1,
        )
        assert "nu" in data
        assert "final_I_frac" in data
        assert "final_S_frac" in data
        assert "r_eff" in data
        assert "nu_c_theory" in data
        assert len(data["nu"]) == 5
        assert np.all(data["final_I_frac"] >= 0)
        assert data["nu_c_theory"] > 0

    def test_run_rediscovery_structure(self):
        """run_sir_vaccination_rediscovery returns a dict with required keys."""
        from simulating_anything.rediscovery.sir_vaccination import (
            run_sir_vaccination_rediscovery,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_sir_vaccination_rediscovery(
                output_dir=tmpdir,
                n_iterations=5,
            )

        assert isinstance(results, dict)
        assert results["domain"] == "sir_vaccination"
        assert "targets" in results
        assert "r0_data" in results
        assert "vaccination_sweep" in results
