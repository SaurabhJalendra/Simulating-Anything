"""Tests for the Allee effect predator-prey model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.allee_predator_prey import (
    AlleePredatorPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N_0: float = 50.0,
    P_0: float = 5.0,
    r: float = 1.0,
    A: float = 10.0,
    K: float = 100.0,
    a: float = 0.01,
    h: float = 0.1,
    e: float = 0.5,
    m: float = 0.3,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.ALLEE_PREDATOR_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "A": A, "K": K,
            "a": a, "h": h, "e": e, "m": m,
            "N_0": N_0, "P_0": P_0,
        },
    )


class TestAlleePredatorPreySimulation:
    """Core simulation behavior tests."""

    def test_allee_effect(self):
        """N < A and no predators -> prey declines toward extinction."""
        config = _make_config(N_0=5.0, P_0=0.0, A=10.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()
        initial_N = sim.observe()[0]

        for _ in range(10000):
            sim.step()

        final_N = sim.observe()[0]
        assert final_N < initial_N, (
            f"Prey should decline below Allee threshold: "
            f"initial={initial_N}, final={final_N}"
        )
        assert final_N < 1.0, f"Prey should approach extinction: N={final_N}"

    def test_above_allee(self):
        """N > A and no predators -> prey grows toward K."""
        config = _make_config(N_0=20.0, P_0=0.0, A=10.0, K=100.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()
        initial_N = sim.observe()[0]

        for _ in range(50000):
            sim.step()

        final_N = sim.observe()[0]
        assert final_N > initial_N, (
            f"Prey should grow above Allee threshold: "
            f"initial={initial_N}, final={final_N}"
        )

    def test_carrying_capacity(self):
        """No predators, N > A -> prey reaches K eventually."""
        config = _make_config(N_0=20.0, P_0=0.0, A=10.0, K=100.0, dt=0.01)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(100000):
            sim.step()

        final_N = sim.observe()[0]
        assert final_N == pytest.approx(100.0, abs=2.0), (
            f"Prey should reach carrying capacity: N={final_N}"
        )

    def test_prey_extinction(self):
        """Predators push prey below A -> extinction."""
        # Start prey just above Allee threshold with high predation
        # Using higher attack rate a=0.05 so predation overwhelms growth
        config = _make_config(N_0=12.0, P_0=20.0, A=10.0, a=0.05)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(100000):
            sim.step()
            N, _ = sim.observe()
            if N < 0.01:
                break

        final_N = sim.observe()[0]
        assert final_N < 1.0, (
            f"Prey should go extinct with heavy predation: N={final_N}"
        )

    def test_coexistence(self):
        """Moderate predation allows coexistence above A."""
        config = _make_config(N_0=50.0, P_0=2.0, A=10.0, K=100.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(100000):
            sim.step()

        N, P = sim.observe()
        assert N > 1.0, f"Prey should survive: N={N}"
        assert P > 0.1 or N > 10.0, (
            f"System should coexist or prey wins: N={N}, P={P}"
        )

    def test_bistability(self):
        """Different ICs reach different outcomes."""
        # Above Allee threshold
        config_above = _make_config(N_0=50.0, P_0=5.0, A=10.0)
        sim_above = AlleePredatorPreySimulation(config_above)
        sim_above.reset()
        for _ in range(50000):
            sim_above.step()
        N_above = sim_above.observe()[0]

        # Below Allee threshold
        config_below = _make_config(N_0=5.0, P_0=5.0, A=10.0)
        sim_below = AlleePredatorPreySimulation(config_below)
        sim_below.reset()
        for _ in range(50000):
            sim_below.step()
        N_below = sim_below.observe()[0]

        # They should end up in different states
        assert abs(N_above - N_below) > 5.0, (
            f"Bistability not observed: N_above={N_above:.2f}, "
            f"N_below={N_below:.2f}"
        )

    def test_non_negative(self):
        """N, P >= 0 at all times."""
        config = _make_config(N_0=15.0, P_0=30.0, A=10.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(10000):
            sim.step()
            N, P = sim.observe()
            assert N >= 0, f"Prey went negative: N={N}"
            assert P >= 0, f"Predator went negative: P={P}"

    def test_bounded(self):
        """Populations remain bounded."""
        config = _make_config(N_0=50.0, P_0=5.0, A=10.0, K=100.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(50000):
            sim.step()
            N, P = sim.observe()
            assert N < 500, f"Prey diverged: N={N}"
            assert P < 500, f"Predator diverged: P={P}"

    def test_allee_growth_zero_at_critical_points(self):
        """Growth is zero at N=0, N=A, N=K."""
        config = _make_config()
        sim = AlleePredatorPreySimulation(config)

        assert sim.allee_growth(0.0) == pytest.approx(0.0, abs=1e-10)
        assert sim.allee_growth(sim.A) == pytest.approx(0.0, abs=1e-10)
        assert sim.allee_growth(sim.K) == pytest.approx(0.0, abs=1e-10)

    def test_allee_growth_negative_below_A(self):
        """Growth rate negative for 0 < N < A."""
        config = _make_config(A=10.0, K=100.0)
        sim = AlleePredatorPreySimulation(config)

        # Test several points between 0 and A
        for N in [1.0, 3.0, 5.0, 8.0, 9.9]:
            growth = sim.allee_growth(N)
            assert growth < 0, (
                f"Growth should be negative below A: N={N}, growth={growth}"
            )

    def test_allee_growth_positive_above_A(self):
        """Growth rate positive for A < N < K."""
        config = _make_config(A=10.0, K=100.0)
        sim = AlleePredatorPreySimulation(config)

        # Test several points between A and K
        for N in [15.0, 30.0, 50.0, 70.0, 90.0]:
            growth = sim.allee_growth(N)
            assert growth > 0, (
                f"Growth should be positive above A: N={N}, growth={growth}"
            )

    def test_rk4_stability(self):
        """No NaN or Inf in simulation output."""
        config = _make_config(N_0=50.0, P_0=10.0, dt=0.01)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_deterministic(self):
        """Same parameters produce same result."""
        config = _make_config(N_0=50.0, P_0=5.0)

        sim1 = AlleePredatorPreySimulation(config)
        sim1.reset()
        for _ in range(1000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = AlleePredatorPreySimulation(config)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_observe_shape(self):
        """Observe returns 2-element state."""
        config = _make_config()
        sim = AlleePredatorPreySimulation(config)
        state = sim.reset()
        assert state.shape == (2,)
        assert sim.observe().shape == (2,)

    def test_reset_state(self):
        """Initial conditions are set correctly."""
        config = _make_config(N_0=42.0, P_0=7.5)
        sim = AlleePredatorPreySimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [42.0, 7.5])

    def test_step_advances(self):
        """State changes after a step."""
        config = _make_config(N_0=50.0, P_0=5.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_holling_type2(self):
        """Functional response saturates at 1/h for large N."""
        config = _make_config(h=0.1, a=0.01)
        sim = AlleePredatorPreySimulation(config)

        # Small N: approximately linear (a*N)
        fr_small = sim.holling_type2(1.0)
        assert fr_small == pytest.approx(0.01 / 1.001, rel=0.01)

        # Large N: saturates at 1/h = 10
        fr_large = sim.holling_type2(100000.0)
        assert fr_large == pytest.approx(1.0 / 0.1, rel=0.01)

    def test_extinction_sweep(self):
        """Extinction sweep produces coherent results."""
        config = _make_config(N_0=50.0, P_0=5.0)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        P0_values = np.array([0.0, 5.0, 20.0, 40.0])
        result = sim.extinction_sweep(P0_values, N_0=50.0, n_steps=10000)

        assert len(result["P0_values"]) == 4
        assert len(result["final_N"]) == 4
        assert len(result["extinct"]) == 4
        # No predators -> prey should survive
        assert result["final_N"][0] > 1.0

    def test_bifurcation(self):
        """Parameter sweep changes stability."""
        config = _make_config()
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        A_values = np.array([5.0, 20.0, 40.0, 60.0])
        result = sim.bifurcation_analysis(A_values, n_steps=10000)

        assert len(result["A_values"]) == 4
        assert len(result["final_N"]) == 4
        assert len(result["extinct"]) == 4


class TestAlleePredatorPreyEquilibria:
    """Tests for equilibrium computation."""

    def test_find_equilibria_basic(self):
        """Should always find extinction and boundary equilibria."""
        config = _make_config()
        sim = AlleePredatorPreySimulation(config)
        eq = sim.find_equilibria()

        types = [e["type"] for e in eq]
        assert "extinction" in types
        assert "allee_threshold" in types
        assert "carrying_capacity" in types

    def test_extinction_equilibrium(self):
        """Origin (0,0) is always an equilibrium."""
        config = _make_config()
        sim = AlleePredatorPreySimulation(config)
        eq = sim.find_equilibria()
        ext = [e for e in eq if e["type"] == "extinction"][0]
        assert ext["N"] == 0.0
        assert ext["P"] == 0.0

    def test_separatrix_distance(self):
        """Separatrix distance is positive above A, negative below."""
        config = _make_config(A=10.0)
        sim = AlleePredatorPreySimulation(config)

        assert sim.separatrix_distance(15.0, 5.0) > 0
        assert sim.separatrix_distance(5.0, 5.0) < 0
        assert sim.separatrix_distance(10.0, 5.0) == pytest.approx(0.0)


class TestAlleePredatorPreyRediscovery:
    """Tests for rediscovery data generation."""

    def test_rediscovery_data(self):
        """ODE data generation works."""
        from simulating_anything.rediscovery.allee_predator_prey import (
            generate_ode_data,
        )

        data = generate_ode_data(N_0=50.0, P_0=5.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["dt"] == 0.01

    def test_bistability_data(self):
        """Bistability data generation works."""
        from simulating_anything.rediscovery.allee_predator_prey import (
            generate_bistability_data,
        )

        data = generate_bistability_data(
            N_above=50.0, N_below=5.0, P_0=5.0, n_steps=1000, dt=0.01,
        )
        assert "above_trajectory" in data
        assert "below_trajectory" in data
        assert data["above_trajectory"].shape[1] == 2
        assert data["below_trajectory"].shape[1] == 2

    def test_predator_impact_data(self):
        """Predator impact sweep works."""
        from simulating_anything.rediscovery.allee_predator_prey import (
            generate_predator_impact_data,
        )

        data = generate_predator_impact_data(
            P0_values=np.array([0.0, 10.0, 30.0]),
            N_0=50.0, n_steps=1000, dt=0.01,
        )
        assert len(data["P0_values"]) == 3
        assert len(data["final_N"]) == 3
        assert "critical_P" in data
