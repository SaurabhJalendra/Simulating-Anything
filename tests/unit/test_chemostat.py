"""Tests for the chemostat (continuous bioreactor) simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.chemostat import Chemostat
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    D: float = 0.1,
    S_in: float = 10.0,
    mu_max: float = 0.5,
    K_s: float = 2.0,
    Y_xs: float = 0.5,
    S_0: float = 5.0,
    X_0: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CHEMOSTAT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "D": D, "S_in": S_in, "mu_max": mu_max,
            "K_s": K_s, "Y_xs": Y_xs, "S_0": S_0, "X_0": X_0,
        },
    )


class TestChemostatCreation:
    def test_initial_state_shape(self):
        sim = Chemostat(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = Chemostat(_make_config(S_0=5.0, X_0=1.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [5.0, 1.0])

    def test_parameters_stored(self):
        sim = Chemostat(_make_config(D=0.2, S_in=15.0, mu_max=0.6))
        assert sim.D == pytest.approx(0.2)
        assert sim.S_in == pytest.approx(15.0)
        assert sim.mu_max == pytest.approx(0.6)
        assert sim.K_s == pytest.approx(2.0)
        assert sim.Y_xs == pytest.approx(0.5)

    def test_default_parameters(self):
        """Default params should be used when not specified."""
        config = SimulationConfig(
            domain=Domain.CHEMOSTAT, dt=0.01, n_steps=100, parameters={},
        )
        sim = Chemostat(config)
        assert sim.D == pytest.approx(0.1)
        assert sim.S_in == pytest.approx(10.0)
        assert sim.mu_max == pytest.approx(0.5)
        assert sim.K_s == pytest.approx(2.0)
        assert sim.Y_xs == pytest.approx(0.5)


class TestChemostatDynamics:
    def test_step_advances_state(self):
        sim = Chemostat(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_state_non_negative(self):
        """Concentrations must remain non-negative throughout simulation."""
        sim = Chemostat(_make_config(D=0.3, S_0=0.1, X_0=5.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
            S, X = sim.observe()
            assert S >= 0.0, f"Substrate went negative: S={S}"
            assert X >= 0.0, f"Biomass went negative: X={X}"

    def test_trajectory_bounded(self):
        """State should remain bounded (no divergence)."""
        sim = Chemostat(_make_config())
        sim.reset()
        for _ in range(5000):
            sim.step()
            S, X = sim.observe()
            assert S <= sim.S_in + 1.0, f"Substrate exceeded S_in: S={S}"
            assert X < 100.0, f"Biomass diverged: X={X}"

    def test_observe_returns_current_state(self):
        sim = Chemostat(_make_config())
        sim.reset()
        sim.step()
        s1 = sim.observe()
        # observe should return same reference without advancing
        s2 = sim.observe()
        np.testing.assert_array_equal(s1, s2)

    def test_run_trajectory(self):
        sim = Chemostat(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201


class TestMonodKinetics:
    def test_growth_rate_at_zero_substrate(self):
        """Monod rate should be zero when S=0."""
        sim = Chemostat(_make_config(S_0=0.0, X_0=1.0))
        sim.reset()
        assert sim.growth_rate == pytest.approx(0.0)

    def test_growth_rate_at_half_saturation(self):
        """At S = K_s, mu should be mu_max / 2."""
        sim = Chemostat(_make_config(S_0=2.0, X_0=1.0))  # K_s=2.0
        sim.reset()
        assert sim.growth_rate == pytest.approx(0.25)  # 0.5 * 2 / (2+2)

    def test_growth_rate_high_substrate(self):
        """At S >> K_s, mu should approach mu_max."""
        sim = Chemostat(_make_config(S_0=1000.0, X_0=1.0))
        sim.reset()
        assert sim.growth_rate == pytest.approx(0.5, rel=0.01)

    def test_monod_rate_method(self):
        sim = Chemostat(_make_config())
        assert sim.monod_rate(0.0) == pytest.approx(0.0)
        assert sim.monod_rate(2.0) == pytest.approx(0.25)
        assert sim.monod_rate(1000.0) == pytest.approx(0.5, rel=0.01)


class TestWashoutBifurcation:
    def test_washout_D_value(self):
        """D_c = mu_max * S_in / (K_s + S_in) = 0.5 * 10 / 12 = 5/12."""
        sim = Chemostat(_make_config())
        expected_Dc = 0.5 * 10.0 / (2.0 + 10.0)
        assert sim.washout_D == pytest.approx(expected_Dc)

    def test_washout_at_high_dilution(self):
        """Above D_c, biomass should wash out to zero."""
        D_c = 0.5 * 10.0 / (2.0 + 10.0)  # ~0.4167
        sim = Chemostat(_make_config(D=D_c + 0.05, S_0=5.0, X_0=2.0))
        sim.reset()
        for _ in range(50000):
            sim.step()
        _, X = sim.observe()
        assert X < 0.01, f"Biomass should wash out but X={X}"

    def test_survival_below_washout(self):
        """Below D_c, biomass should survive at steady state."""
        sim = Chemostat(_make_config(D=0.1, S_0=5.0, X_0=1.0))
        sim.reset()
        for _ in range(50000):
            sim.step()
        _, X = sim.observe()
        assert X > 0.1, f"Biomass should survive but X={X}"


class TestSteadyState:
    def test_analytical_steady_state(self):
        """Verify analytical steady-state formulas."""
        sim = Chemostat(_make_config(D=0.1))
        S_star, X_star = sim.steady_state

        # S* = K_s * D / (mu_max - D) = 2 * 0.1 / 0.4 = 0.5
        assert S_star == pytest.approx(0.5)
        # X* = Y_xs * (S_in - S*) = 0.5 * (10 - 0.5) = 4.75
        assert X_star == pytest.approx(4.75)

    def test_steady_state_washout(self):
        """Above washout, steady state should be (S_in, 0)."""
        sim = Chemostat(_make_config(D=0.45))
        S_star, X_star = sim.steady_state
        assert S_star == pytest.approx(10.0)
        assert X_star == pytest.approx(0.0)

    def test_simulation_converges_to_steady_state(self):
        """Simulation should converge to analytical steady state."""
        sim = Chemostat(_make_config(D=0.1, dt=0.01))
        sim.reset()

        for _ in range(50000):
            sim.step()

        S_sim, X_sim = sim.observe()
        S_star, X_star = sim.steady_state

        np.testing.assert_allclose(S_sim, S_star, rtol=0.01,
                                   err_msg=f"S: sim={S_sim}, theory={S_star}")
        np.testing.assert_allclose(X_sim, X_star, rtol=0.01,
                                   err_msg=f"X: sim={X_sim}, theory={X_star}")

    def test_derivatives_at_steady_state(self):
        """Derivatives should be approximately zero at analytical steady state."""
        sim = Chemostat(_make_config(D=0.1))
        S_star, X_star = sim.steady_state
        y = np.array([S_star, X_star])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)


class TestReproducibility:
    def test_deterministic_runs(self):
        """Two runs with same config should produce identical trajectories."""
        config = _make_config(n_steps=500)
        sim1 = Chemostat(config)
        traj1 = sim1.run(n_steps=500)

        sim2 = Chemostat(config)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_array_equal(traj1.states, traj2.states)


class TestRediscoveryData:
    def test_steady_state_data_generation(self):
        from simulating_anything.rediscovery.chemostat import generate_steady_state_data
        data = generate_steady_state_data(n_D=5, dt=0.01, n_settle=5000)
        assert len(data["D"]) == 5
        assert len(data["S_steady"]) == 5
        assert len(data["X_steady"]) == 5
        assert "D_c_theory" in data

    def test_washout_data_generation(self):
        from simulating_anything.rediscovery.chemostat import generate_washout_data
        data = generate_washout_data(n_D=5, dt=0.01, n_settle=5000)
        assert len(data["D"]) == 5
        assert len(data["X_final"]) == 5
        assert "D_c_theory" in data

    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.chemostat import generate_trajectory_data
        data = generate_trajectory_data(D=0.1, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["D"] == 0.1
