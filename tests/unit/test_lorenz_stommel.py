"""Tests for the coupled Lorenz-Stommel atmosphere-ocean simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.lorenz_stommel import (
    LorenzStommelSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestLorenzStommelSimulation:
    """Tests for the LorenzStommelSimulation class."""

    def _make_sim(self, **kwargs) -> LorenzStommelSimulation:
        """Create a simulation with default or overridden parameters."""
        defaults = {
            "sigma": 10.0,
            "rho": 28.0,
            "beta": 8.0 / 3.0,
            "eta1": 3.0,
            "eta2": 1.0,
            "delta": 0.3,
            "coupling_strength": 0.1,
            "ocean_feedback": 0.01,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 1.0,
            "T_0": 2.0,
            "S_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzStommelSimulation(config)

    def test_initial_state_shape(self):
        """State vector should have 5 components: [x, y, z, T, S]."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (5,)

    def test_initial_state_values(self):
        """Initial state should match the configured initial conditions."""
        sim = self._make_sim(
            x_0=2.0, y_0=-1.0, z_0=5.0, T_0=3.0, S_0=0.5
        )
        state = sim.reset()
        np.testing.assert_allclose(state, [2.0, -1.0, 5.0, 3.0, 0.5])

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_current_state(self):
        """observe() should return the same state as the last step."""
        sim = self._make_sim()
        sim.reset()
        state = sim.step()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_state_dimensions(self):
        """State should remain 5D after many steps."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(500):
            state = sim.step()
        assert state.shape == (5,)

    def test_trajectory_bounded(self):
        """Coupled system should remain bounded for default parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), (
                f"State became NaN/Inf: {state}"
            )
            # Lorenz attractor is bounded within ~50 for default params
            assert np.linalg.norm(state[:3]) < 200, (
                f"Atmosphere diverged: {state[:3]}"
            )
            # Ocean variables should stay positive and bounded
            assert state[3] > -10.0, f"T went too negative: {state[3]}"
            assert state[4] > -10.0, f"S went too negative: {state[4]}"

    def test_ocean_variables_bounded(self):
        """Ocean T and S should remain bounded and physical."""
        sim = self._make_sim()
        sim.reset()
        T_vals = []
        S_vals = []
        for _ in range(5000):
            state = sim.step()
            T_vals.append(state[3])
            S_vals.append(state[4])
        T_arr = np.array(T_vals)
        S_arr = np.array(S_vals)
        assert np.all(np.isfinite(T_arr))
        assert np.all(np.isfinite(S_arr))
        # T and S should be bounded for physical ocean dynamics
        assert np.max(np.abs(T_arr)) < 100.0
        assert np.max(np.abs(S_arr)) < 100.0

    def test_run_returns_trajectory_data(self):
        """run() should return a proper TrajectoryData object."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 5)
        assert len(traj.timestamps) == 101

    def test_atmosphere_state_accessor(self):
        """atmosphere_state() should return only Lorenz variables."""
        sim = self._make_sim()
        sim.reset()
        sim.step()
        atm = sim.atmosphere_state()
        assert atm.shape == (3,)
        np.testing.assert_array_equal(atm, sim._state[:3])

    def test_ocean_state_accessor(self):
        """ocean_state() should return only Stommel variables."""
        sim = self._make_sim()
        sim.reset()
        sim.step()
        ocean = sim.ocean_state()
        assert ocean.shape == (2,)
        np.testing.assert_array_equal(ocean, sim._state[3:])

    def test_flow_rate_nonnegative(self):
        """Flow rate q = |T - S| should always be non-negative."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(1000):
            sim.step()
            q = sim.flow_rate()
            assert q >= 0.0


class TestDecoupledLimit:
    """Tests verifying that coupling_strength=0 gives independent subsystems."""

    def _make_decoupled_sim(self) -> LorenzStommelSimulation:
        """Create a simulation with zero coupling."""
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.005,
            n_steps=10000,
            parameters={
                "sigma": 10.0,
                "rho": 28.0,
                "beta": 8.0 / 3.0,
                "eta1": 3.0,
                "eta2": 1.0,
                "delta": 0.3,
                "coupling_strength": 0.0,
                "ocean_feedback": 0.0,
                "x_0": 1.0,
                "y_0": 1.0,
                "z_0": 1.0,
                "T_0": 2.0,
                "S_0": 1.0,
            },
        )
        return LorenzStommelSimulation(config)

    def test_decoupled_lorenz_matches_standalone(self):
        """With zero coupling, Lorenz subsystem should match standalone Lorenz."""
        from simulating_anything.simulation.lorenz import LorenzSimulation

        # Run coupled system with zero coupling
        sim_coupled = self._make_decoupled_sim()
        sim_coupled.reset()

        # Run standalone Lorenz with same initial conditions
        config_lorenz = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=0.005,
            n_steps=10000,
            parameters={
                "sigma": 10.0,
                "rho": 28.0,
                "beta": 8.0 / 3.0,
                "x_0": 1.0,
                "y_0": 1.0,
                "z_0": 1.0,
            },
        )
        sim_lorenz = LorenzSimulation(config_lorenz)
        sim_lorenz.reset()

        # Step both and compare atmosphere variables
        for _ in range(500):
            state_coupled = sim_coupled.step()
            state_lorenz = sim_lorenz.step()
            np.testing.assert_allclose(
                state_coupled[:3], state_lorenz,
                atol=1e-12,
                err_msg="Decoupled Lorenz diverged from standalone",
            )

    def test_decoupled_ocean_converges_to_equilibrium(self):
        """With zero coupling, ocean subsystem should converge to a steady state.

        The decoupled Stommel equations (dT/dt = eta1 - T - |T-S|*T,
        dS/dt = eta2 - delta*S - |T-S|*S) should converge to an equilibrium
        where dT/dt = dS/dt = 0.

        Note: The coupled model uses a simplified Stommel parametrization
        that differs from the standalone StommelSimulation (which uses
        eta3 as both multiplier and relaxation rate). This test verifies
        the decoupled ocean reaches its own internal equilibrium.
        """
        sim = self._make_decoupled_sim()
        sim.reset()

        # Run to steady state
        for _ in range(10000):
            sim.step()

        T_eq, S_eq = sim.ocean_state()
        # Verify equilibrium: dT/dt and dS/dt should be ~0
        q = abs(T_eq - S_eq)
        dT = 3.0 - T_eq - q * T_eq
        dS = 1.0 - 0.3 * S_eq - q * S_eq
        assert abs(dT) < 1e-6, f"dT/dt = {dT} at equilibrium, expected ~0"
        assert abs(dS) < 1e-6, f"dS/dt = {dS} at equilibrium, expected ~0"

    def test_decoupled_no_cross_influence(self):
        """With zero coupling, changing ocean IC should not affect atmosphere."""
        config_a = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.005,
            n_steps=1000,
            parameters={
                "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
                "eta1": 3.0, "eta2": 1.0, "delta": 0.3,
                "coupling_strength": 0.0, "ocean_feedback": 0.0,
                "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
                "T_0": 2.0, "S_0": 1.0,
            },
        )
        config_b = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.005,
            n_steps=1000,
            parameters={
                "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
                "eta1": 3.0, "eta2": 1.0, "delta": 0.3,
                "coupling_strength": 0.0, "ocean_feedback": 0.0,
                "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
                "T_0": 5.0, "S_0": 3.0,  # Different ocean IC
            },
        )
        sim_a = LorenzStommelSimulation(config_a)
        sim_b = LorenzStommelSimulation(config_b)
        sim_a.reset()
        sim_b.reset()

        for _ in range(500):
            sa = sim_a.step()
            sb = sim_b.step()
            # Atmosphere should be identical despite different ocean ICs
            np.testing.assert_allclose(
                sa[:3], sb[:3], atol=1e-12,
                err_msg="Ocean IC affected atmosphere in decoupled mode",
            )
            # Ocean should differ due to different ICs
            assert not np.allclose(sa[3:], sb[3:], atol=1e-6)


class TestCoupledDynamics:
    """Tests verifying that coupling produces measurable effects."""

    def _make_sim(self, coupling_strength, ocean_feedback=0.01):
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.005,
            n_steps=10000,
            parameters={
                "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
                "eta1": 3.0, "eta2": 1.0, "delta": 0.3,
                "coupling_strength": coupling_strength,
                "ocean_feedback": ocean_feedback,
                "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
                "T_0": 2.0, "S_0": 1.0,
            },
        )
        return LorenzStommelSimulation(config)

    def test_coupling_changes_ocean_trajectory(self):
        """Non-zero coupling should alter the ocean trajectory."""
        sim_uncoupled = self._make_sim(
            coupling_strength=0.0, ocean_feedback=0.0
        )
        sim_coupled = self._make_sim(
            coupling_strength=0.5, ocean_feedback=0.0
        )

        sim_uncoupled.reset()
        sim_coupled.reset()

        for _ in range(500):
            sim_uncoupled.step()
            sim_coupled.step()

        # Ocean states should differ
        ocean_u = sim_uncoupled.ocean_state()
        ocean_c = sim_coupled.ocean_state()
        assert not np.allclose(ocean_u, ocean_c, atol=0.01), (
            "Coupling did not change ocean trajectory"
        )

    def test_ocean_feedback_changes_atmosphere(self):
        """Non-zero ocean_feedback should alter the atmosphere trajectory."""
        sim_no_fb = self._make_sim(
            coupling_strength=0.0, ocean_feedback=0.0
        )
        sim_fb = self._make_sim(
            coupling_strength=0.0, ocean_feedback=0.5
        )

        sim_no_fb.reset()
        sim_fb.reset()

        # Run enough steps for feedback to accumulate
        for _ in range(2000):
            sim_no_fb.step()
            sim_fb.step()

        atm_no_fb = sim_no_fb.atmosphere_state()
        atm_fb = sim_fb.atmosphere_state()
        # Due to chaotic sensitivity, even small feedback should diverge
        assert not np.allclose(atm_no_fb, atm_fb, atol=0.1), (
            "Ocean feedback did not change atmosphere trajectory"
        )

    def test_lyapunov_positive_for_chaotic_params(self):
        """The coupled system should have positive Lyapunov for default params."""
        sim = self._make_sim(coupling_strength=0.1)
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.005)
        assert lam > 0.3, (
            f"Lyapunov {lam:.4f} too small -- expected chaotic dynamics"
        )
        assert lam < 5.0, (
            f"Lyapunov {lam:.4f} unreasonably large"
        )

    def test_atmosphere_ocean_correlation(self):
        """Correlation measurement should return valid coefficients."""
        sim = self._make_sim(coupling_strength=0.1)
        corr = sim.atmosphere_ocean_correlation(
            n_steps=3000, n_transient=1000
        )
        assert "corr_x_T" in corr
        assert "corr_x_S" in corr
        assert "mean_q" in corr
        # Correlations should be in [-1, 1]
        assert -1.0 <= corr["corr_x_T"] <= 1.0
        assert -1.0 <= corr["corr_x_S"] <= 1.0
        # Mean flow rate should be positive
        assert corr["mean_q"] > 0.0


class TestLorenzStommelRediscoveryData:
    """Tests for the rediscovery data generation functions."""

    def test_ode_data_shape(self):
        """ODE data should have correct shape."""
        from simulating_anything.rediscovery.lorenz_stommel import (
            generate_ode_data,
        )

        data = generate_ode_data(
            coupling_strength=0.1, n_steps=200, dt=0.005
        )
        assert data["states"].shape == (201, 5)
        assert data["dt"] == 0.005
        assert data["coupling_strength"] == 0.1

    def test_ode_data_finite(self):
        """Generated ODE data should remain finite."""
        from simulating_anything.rediscovery.lorenz_stommel import (
            generate_ode_data,
        )

        data = generate_ode_data(
            coupling_strength=0.1, n_steps=1000, dt=0.005
        )
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_sweep_shape(self):
        """Lyapunov sweep should return arrays of matching length."""
        from simulating_anything.rediscovery.lorenz_stommel import (
            generate_lyapunov_vs_coupling,
        )

        couplings = np.array([0.0, 0.1, 0.5])
        data = generate_lyapunov_vs_coupling(
            coupling_values=couplings, n_steps=2000,
            n_transient=500, dt=0.005,
        )
        assert len(data["coupling"]) == 3
        assert len(data["lyapunov_exponent"]) == 3
        assert np.all(np.isfinite(data["lyapunov_exponent"]))
