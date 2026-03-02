"""Tests for the logistic growth with harvesting simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.harvested_population import (
    HarvestedPopulationSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 1.0,
    H: float = 0.0,
    x_0: float | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    params: dict[str, float] = {"r": r, "K": K, "H": H}
    if x_0 is not None:
        params["x_0"] = x_0
    return SimulationConfig(
        domain=Domain.HARVESTED_POPULATION,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestHarvestedPopulationSimulation:
    def test_creation_and_params(self):
        sim = HarvestedPopulationSimulation(_make_config(r=2.0, K=3.0, H=0.5))
        assert sim.r == 2.0
        assert sim.K == 3.0
        assert sim.H == 0.5

    def test_default_parameters(self):
        config = SimulationConfig(
            domain=Domain.HARVESTED_POPULATION, dt=0.01, n_steps=10,
            parameters={},
        )
        sim = HarvestedPopulationSimulation(config)
        assert sim.r == 1.0
        assert sim.K == 1.0
        assert sim.H == 0.0

    def test_default_x0_is_half_K(self):
        """Default initial condition should be K/2."""
        sim = HarvestedPopulationSimulation(_make_config(K=4.0))
        state = sim.reset()
        assert state[0] == pytest.approx(2.0)

    def test_initial_state_shape(self):
        sim = HarvestedPopulationSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)

    def test_initial_state_value(self):
        sim = HarvestedPopulationSimulation(_make_config(x_0=0.7))
        state = sim.reset()
        assert state[0] == pytest.approx(0.7)

    def test_step_advances_state(self):
        sim = HarvestedPopulationSimulation(_make_config(r=1.0, K=1.0, H=0.0, x_0=0.3))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = HarvestedPopulationSimulation(_make_config(x_0=0.8))
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = HarvestedPopulationSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 1)
        assert len(traj.timestamps) == 101


class TestHarvestedPopulationEquilibria:
    def test_no_harvesting_converges_to_K(self):
        """Without harvesting, logistic growth converges to K."""
        sim = HarvestedPopulationSimulation(
            _make_config(r=1.0, K=2.0, H=0.0, x_0=0.3, dt=0.01)
        )
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(2.0, abs=0.01)

    def test_find_equilibria_no_harvesting(self):
        """H=0: stable eq at K, unstable at 0 (but 0 is trivial)."""
        sim = HarvestedPopulationSimulation(_make_config(r=1.0, K=2.0, H=0.0))
        sim.reset()
        eqs = sim.find_equilibria()
        # Upper equilibrium is K
        assert len(eqs) >= 1
        x_values = [eq["x"] for eq in eqs]
        assert any(abs(x - 2.0) < 0.01 for x in x_values)

    def test_find_equilibria_moderate_harvesting(self):
        """H < H_c: two equilibria (upper stable, lower unstable)."""
        r, K, H = 1.0, 2.0, 0.3
        sim = HarvestedPopulationSimulation(_make_config(r=r, K=K, H=H))
        sim.reset()
        eqs = sim.find_equilibria()
        assert len(eqs) == 2
        # Upper should be stable, lower unstable
        stabilities = {eq["stability"] for eq in eqs}
        assert "stable" in stabilities
        assert "unstable" in stabilities

    def test_find_equilibria_at_msy(self):
        """H = H_c: saddle-node at K/2."""
        r, K = 1.0, 2.0
        H_c = r * K / 4.0
        sim = HarvestedPopulationSimulation(_make_config(r=r, K=K, H=H_c))
        sim.reset()
        eqs = sim.find_equilibria()
        assert len(eqs) == 1
        assert eqs[0]["stability"] == "saddle-node"
        assert eqs[0]["x"] == pytest.approx(K / 2.0, abs=1e-8)

    def test_find_equilibria_above_msy(self):
        """H > H_c: no positive equilibria."""
        r, K = 1.0, 2.0
        H = r * K / 4.0 + 0.1  # Above MSY
        sim = HarvestedPopulationSimulation(_make_config(r=r, K=K, H=H))
        sim.reset()
        eqs = sim.find_equilibria()
        assert len(eqs) == 0

    def test_equilibrium_formula(self):
        """Verify equilibria match x* = K/2 +/- sqrt(K^2/4 - H*K/r)."""
        r, K, H = 2.0, 4.0, 1.0
        disc = K**2 / 4.0 - H * K / r
        x_upper = K / 2.0 + np.sqrt(disc)
        x_lower = K / 2.0 - np.sqrt(disc)

        sim = HarvestedPopulationSimulation(_make_config(r=r, K=K, H=H))
        sim.reset()
        eqs = sim.find_equilibria()
        eq_x = sorted([eq["x"] for eq in eqs])
        np.testing.assert_allclose(eq_x, [x_lower, x_upper], atol=1e-10)

    def test_stable_equilibrium_convergence(self):
        """Starting near upper equilibrium should converge there."""
        r, K, H = 1.0, 2.0, 0.2
        sim = HarvestedPopulationSimulation(
            _make_config(r=r, K=K, H=H, x_0=K * 0.9, dt=0.01)
        )
        sim.reset()
        eqs = sim.find_equilibria()
        x_stable = max(eq["x"] for eq in eqs)

        for _ in range(20000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(x_stable, abs=0.01)

    def test_derivatives_zero_at_equilibrium(self):
        """At any equilibrium, dx/dt should be zero."""
        r, K, H = 1.5, 3.0, 0.5
        sim = HarvestedPopulationSimulation(_make_config(r=r, K=K, H=H))
        sim.reset()
        eqs = sim.find_equilibria()
        for eq in eqs:
            dy = sim._derivatives(np.array([eq["x"]]))
            np.testing.assert_allclose(dy, [0.0], atol=1e-10)


class TestMSY:
    def test_msy_formula(self):
        """MSY = r*K/4."""
        sim = HarvestedPopulationSimulation(_make_config(r=2.0, K=4.0))
        assert sim.compute_msy() == pytest.approx(2.0)

    def test_msy_scales_with_r(self):
        """MSY should scale linearly with r."""
        msy_1 = HarvestedPopulationSimulation(_make_config(r=1.0, K=1.0)).compute_msy()
        msy_2 = HarvestedPopulationSimulation(_make_config(r=2.0, K=1.0)).compute_msy()
        assert msy_2 == pytest.approx(2 * msy_1)

    def test_msy_scales_with_K(self):
        """MSY should scale linearly with K."""
        msy_1 = HarvestedPopulationSimulation(_make_config(r=1.0, K=1.0)).compute_msy()
        msy_3 = HarvestedPopulationSimulation(_make_config(r=1.0, K=3.0)).compute_msy()
        assert msy_3 == pytest.approx(3 * msy_1)


class TestExtinction:
    def test_clamp_prevents_negative(self):
        """Population should be clamped to x >= 0."""
        sim = HarvestedPopulationSimulation(
            _make_config(r=1.0, K=1.0, H=10.0, x_0=0.1, dt=0.01)
        )
        sim.reset()
        for _ in range(500):
            sim.step()
        assert sim.observe()[0] >= 0.0

    def test_extinction_above_msy(self):
        """Population should go extinct when H > H_c."""
        r, K = 1.0, 2.0
        H = r * K / 4.0 + 0.2  # Well above MSY
        sim = HarvestedPopulationSimulation(
            _make_config(r=r, K=K, H=H, x_0=K * 0.9, dt=0.01)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(0.0, abs=1e-10)

    def test_survival_below_msy(self):
        """Population should survive when H < H_c (started near upper eq)."""
        r, K = 1.0, 2.0
        H = r * K / 4.0 * 0.5  # Below MSY
        sim = HarvestedPopulationSimulation(
            _make_config(r=r, K=K, H=H, x_0=K * 0.9, dt=0.01)
        )
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim.observe()[0] > 0.1

    def test_is_extinct_flag(self):
        sim = HarvestedPopulationSimulation(
            _make_config(r=1.0, K=1.0, H=5.0, x_0=0.1, dt=0.01)
        )
        sim.reset()
        assert not sim.is_extinct()
        for _ in range(1000):
            sim.step()
        assert sim.is_extinct()

    def test_time_to_extinction_finite_above_msy(self):
        r, K = 1.0, 2.0
        H = r * K / 4.0 + 0.5
        sim = HarvestedPopulationSimulation(
            _make_config(r=r, K=K, H=H, x_0=K * 0.9, dt=0.01)
        )
        t_ext = sim.time_to_extinction(max_steps=50000)
        assert t_ext < float("inf")

    def test_time_to_extinction_inf_below_msy(self):
        r, K = 1.0, 2.0
        H = r * K / 4.0 * 0.5
        sim = HarvestedPopulationSimulation(
            _make_config(r=r, K=K, H=H, x_0=K * 0.9, dt=0.01)
        )
        t_ext = sim.time_to_extinction(max_steps=20000)
        assert t_ext == float("inf")


class TestHarvestedPopulationRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.harvested_population import (
            generate_ode_data,
        )
        data = generate_ode_data(r=1.0, K=2.0, H=0.2, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 1)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 2.0
        assert data["H"] == 0.2

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.harvested_population import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(r=1.0, K=2.0, n_H=5, dt=0.01)
        assert len(data["H"]) == 5
        assert len(data["final_pop"]) == 5
        assert data["H_c_theory"] == pytest.approx(0.5)

    def test_msy_data_generation(self):
        from simulating_anything.rediscovery.harvested_population import (
            generate_msy_data,
        )
        data = generate_msy_data(n_r=3, n_K=3, dt=0.01)
        assert len(data["r"]) == 9  # 3 * 3
        assert len(data["H_c_measured"]) == 9
        # Measured H_c should roughly match r*K/4
        correlation = np.corrcoef(data["H_c_measured"], data["H_c_theory"])[0, 1]
        assert correlation > 0.95
