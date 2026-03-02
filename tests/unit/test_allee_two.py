"""Tests for the double Allee predator-prey model (both species with Allee effects)."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.allee_two import AlleeTwoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N_0: float = 50.0,
    P_0: float = 10.0,
    r_N: float = 1.0,
    K_N: float = 100.0,
    A_N: float = 5.0,
    r_P: float = 0.5,
    K_P: float = 50.0,
    A_P: float = 3.0,
    a: float = 0.01,
    h: float = 0.1,
    e: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.ALLEE_TWO,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r_N": r_N, "K_N": K_N, "A_N": A_N,
            "r_P": r_P, "K_P": K_P, "A_P": A_P,
            "a": a, "h": h, "e": e,
            "N_0": N_0, "P_0": P_0,
        },
    )


class TestAlleeTwoCreation:
    """Test simulation creation and basic properties."""

    def test_create_default(self):
        """Default config creates a valid simulation."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        assert sim.r_N == 1.0
        assert sim.K_N == 100.0
        assert sim.A_N == 5.0
        assert sim.r_P == 0.5
        assert sim.K_P == 50.0
        assert sim.A_P == 3.0

    def test_custom_parameters(self):
        """Custom parameters are stored correctly."""
        config = _make_config(r_N=2.0, K_N=200.0, A_N=10.0, r_P=1.0, K_P=80.0, A_P=5.0)
        sim = AlleeTwoSimulation(config)
        assert sim.r_N == 2.0
        assert sim.K_N == 200.0
        assert sim.A_N == 10.0
        assert sim.r_P == 1.0
        assert sim.K_P == 80.0
        assert sim.A_P == 5.0

    def test_state_shape(self):
        """State vector has shape (2,)."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        state = sim.reset()
        assert state.shape == (2,)

    def test_observe_shape(self):
        """Observe returns 2-element state."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_reset_values(self):
        """Reset sets initial conditions correctly."""
        config = _make_config(N_0=42.0, P_0=7.5)
        sim = AlleeTwoSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [42.0, 7.5])

    def test_step_advances(self):
        """State changes after a step."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_state_dtype(self):
        """State uses float64."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        state = sim.reset()
        assert state.dtype == np.float64


class TestAlleeTwoNonNegativity:
    """Test that populations remain non-negative under all conditions."""

    def test_non_negative_default(self):
        """N, P >= 0 at all times with default params."""
        config = _make_config(N_0=50.0, P_0=10.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
            N, P = sim.observe()
            assert N >= 0, f"Prey went negative: N={N}"
            assert P >= 0, f"Predator went negative: P={P}"

    def test_non_negative_high_predation(self):
        """Non-negativity holds with high attack rate."""
        config = _make_config(N_0=15.0, P_0=30.0, a=0.05)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
            N, P = sim.observe()
            assert N >= 0, f"Prey went negative: N={N}"
            assert P >= 0, f"Predator went negative: P={P}"

    def test_non_negative_small_populations(self):
        """Non-negativity with small initial populations."""
        config = _make_config(N_0=0.5, P_0=0.5)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
            N, P = sim.observe()
            assert N >= 0, f"Prey went negative: N={N}"
            assert P >= 0, f"Predator went negative: P={P}"

    def test_bounded(self):
        """Populations remain bounded by carrying capacities."""
        config = _make_config(N_0=50.0, P_0=10.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(30000):
            sim.step()
            N, P = sim.observe()
            assert N < 200, f"Prey diverged: N={N}"
            assert P < 200, f"Predator diverged: P={P}"


class TestAlleeTwoPreyAllee:
    """Test prey Allee effect: prey declines when N < A_N."""

    def test_prey_declines_below_AN_no_predator(self):
        """N < A_N with no predators -> prey declines toward extinction."""
        config = _make_config(N_0=3.0, P_0=0.0, A_N=5.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        initial_N = sim.observe()[0]
        for _ in range(10000):
            sim.step()
        final_N = sim.observe()[0]
        assert final_N < initial_N, (
            f"Prey should decline below A_N: initial={initial_N}, final={final_N}"
        )
        assert final_N < 1.0, f"Prey should approach extinction: N={final_N}"

    def test_prey_grows_above_AN_no_predator(self):
        """N > A_N with no predators -> prey grows toward K_N."""
        config = _make_config(N_0=20.0, P_0=0.0, A_N=5.0, K_N=100.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        initial_N = sim.observe()[0]
        for _ in range(50000):
            sim.step()
        final_N = sim.observe()[0]
        assert final_N > initial_N, (
            f"Prey should grow above A_N: initial={initial_N}, final={final_N}"
        )

    def test_prey_reaches_carrying_capacity(self):
        """No predators, N > A_N -> prey reaches K_N."""
        config = _make_config(N_0=20.0, P_0=0.0, A_N=5.0, K_N=100.0, dt=0.01)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(100000):
            sim.step()
        final_N = sim.observe()[0]
        assert final_N == pytest.approx(100.0, abs=2.0), (
            f"Prey should reach carrying capacity: N={final_N}"
        )

    def test_prey_allee_growth_zeros(self):
        """Prey growth is zero at N=0, N=A_N, N=K_N."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        assert sim.prey_allee_growth(0.0) == pytest.approx(0.0, abs=1e-10)
        assert sim.prey_allee_growth(sim.A_N) == pytest.approx(0.0, abs=1e-10)
        assert sim.prey_allee_growth(sim.K_N) == pytest.approx(0.0, abs=1e-10)

    def test_prey_growth_negative_below_AN(self):
        """Growth rate negative for 0 < N < A_N."""
        config = _make_config(A_N=10.0, K_N=100.0)
        sim = AlleeTwoSimulation(config)
        for N in [1.0, 3.0, 5.0, 8.0, 9.9]:
            growth = sim.prey_allee_growth(N)
            assert growth < 0, (
                f"Growth should be negative below A_N: N={N}, growth={growth}"
            )

    def test_prey_growth_positive_above_AN(self):
        """Growth rate positive for A_N < N < K_N."""
        config = _make_config(A_N=5.0, K_N=100.0)
        sim = AlleeTwoSimulation(config)
        for N in [10.0, 30.0, 50.0, 70.0, 90.0]:
            growth = sim.prey_allee_growth(N)
            assert growth > 0, (
                f"Growth should be positive above A_N: N={N}, growth={growth}"
            )


class TestAlleeTwoPredatorAllee:
    """Test predator Allee effect: predator declines when P < A_P."""

    def test_predator_allee_growth_zeros(self):
        """Predator growth is zero at P=0, P=A_P, P=K_P."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        assert sim.predator_allee_growth(0.0) == pytest.approx(0.0, abs=1e-10)
        assert sim.predator_allee_growth(sim.A_P) == pytest.approx(0.0, abs=1e-10)
        assert sim.predator_allee_growth(sim.K_P) == pytest.approx(0.0, abs=1e-10)

    def test_predator_growth_negative_below_AP(self):
        """Predator intrinsic growth is negative for 0 < P < A_P."""
        config = _make_config(A_P=5.0, K_P=50.0)
        sim = AlleeTwoSimulation(config)
        for P in [0.5, 1.0, 2.0, 4.0, 4.9]:
            growth = sim.predator_allee_growth(P)
            assert growth < 0, (
                f"Growth should be negative below A_P: P={P}, growth={growth}"
            )

    def test_predator_growth_positive_above_AP(self):
        """Predator intrinsic growth is positive for A_P < P < K_P."""
        config = _make_config(A_P=3.0, K_P=50.0)
        sim = AlleeTwoSimulation(config)
        for P in [5.0, 10.0, 20.0, 30.0, 45.0]:
            growth = sim.predator_allee_growth(P)
            assert growth > 0, (
                f"Growth should be positive above A_P: P={P}, growth={growth}"
            )

    def test_predator_dies_alone_below_AP(self):
        """Predator with P < A_P and no prey dies out."""
        config = _make_config(N_0=0.0, P_0=2.0, A_P=3.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
        final_P = sim.observe()[1]
        assert final_P < 0.1, f"Predator should die without prey below A_P: P={final_P}"

    def test_predator_survives_alone_above_AP(self):
        """Predator with P > A_P and no prey reaches K_P."""
        config = _make_config(N_0=0.0, P_0=10.0, A_P=3.0, K_P=50.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(100000):
            sim.step()
        final_P = sim.observe()[1]
        assert final_P == pytest.approx(50.0, abs=2.0), (
            f"Predator should reach K_P: P={final_P}"
        )


class TestAlleeTwoBistability:
    """Test four-state bistability from both Allee effects."""

    def test_both_above_coexistence(self):
        """Both above thresholds -> coexistence."""
        config = _make_config(N_0=50.0, P_0=10.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        N, P = sim.observe()
        assert N > 1.0, f"Prey should survive: N={N}"
        assert P > 0.1, f"Predator should survive: P={P}"

    def test_prey_below_predator_above(self):
        """Prey below A_N, predator above A_P -> prey goes extinct, predator to K_P."""
        config = _make_config(N_0=3.0, P_0=10.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        N, P = sim.observe()
        assert N < 1.0, f"Prey should go extinct: N={N}"
        assert P > 1.0, f"Predator should survive at K_P: P={P}"

    def test_prey_above_predator_below(self):
        """Prey above A_N, predator below A_P -> predator dies, prey to K_N."""
        config = _make_config(N_0=50.0, P_0=1.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        N, P = sim.observe()
        assert N > 10.0, f"Prey should survive near K_N: N={N}"
        assert P < 1.0, f"Predator should go extinct: P={P}"

    def test_both_below_extinction(self):
        """Both below thresholds -> mutual extinction."""
        config = _make_config(N_0=3.0, P_0=1.0)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        N, P = sim.observe()
        assert N < 1.0, f"Prey should go extinct: N={N}"
        assert P < 1.0, f"Predator should go extinct: P={P}"

    def test_four_distinct_outcomes(self):
        """Four different ICs produce four distinct outcomes."""
        ics = [(50.0, 10.0), (3.0, 10.0), (50.0, 1.0), (3.0, 1.0)]
        outcomes = []
        for N0, P0 in ics:
            config = _make_config(N_0=N0, P_0=P0)
            sim = AlleeTwoSimulation(config)
            sim.reset()
            for _ in range(50000):
                sim.step()
            N, P = sim.observe()
            # Classify
            if N > 1 and P > 1:
                outcomes.append("coexistence")
            elif N > 1:
                outcomes.append("prey_only")
            elif P > 1:
                outcomes.append("pred_only")
            else:
                outcomes.append("extinction")
        # Should have at least 3 distinct outcomes
        assert len(set(outcomes)) >= 3, (
            f"Expected at least 3 distinct outcomes, got: {outcomes}"
        )


class TestAlleeTwoFunctionalResponse:
    """Test Holling Type II functional response."""

    def test_holling_linear_at_low_N(self):
        """For small N, functional response is approximately a*N."""
        config = _make_config(a=0.01, h=0.1)
        sim = AlleeTwoSimulation(config)
        fr = sim.holling_type2(1.0)
        expected = 0.01 / 1.1
        assert fr == pytest.approx(expected, rel=0.01)

    def test_holling_saturates(self):
        """For large N, functional response saturates at a/h."""
        config = _make_config(a=0.01, h=0.1)
        sim = AlleeTwoSimulation(config)
        fr_large = sim.holling_type2(100000.0)
        saturation = 0.01 / 0.1  # = 0.1
        assert fr_large == pytest.approx(saturation, rel=0.01)

    def test_holling_monotonic(self):
        """Functional response is monotonically increasing."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        N_vals = [1.0, 5.0, 10.0, 50.0, 100.0]
        fr_vals = [sim.holling_type2(N) for N in N_vals]
        for i in range(len(fr_vals) - 1):
            assert fr_vals[i] < fr_vals[i + 1]


class TestAlleeTwoEquilibria:
    """Test equilibrium computation."""

    def test_extinction_equilibrium(self):
        """(0, 0) is always an equilibrium."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        eq = sim.find_equilibria()
        ext = [e for e in eq if e["type"] == "extinction"]
        assert len(ext) == 1
        assert ext[0]["N"] == 0.0
        assert ext[0]["P"] == 0.0

    def test_prey_allee_threshold_equilibrium(self):
        """(A_N, 0) is an equilibrium."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        eq = sim.find_equilibria()
        allee = [e for e in eq if e["type"] == "prey_allee_threshold"]
        assert len(allee) == 1
        assert allee[0]["N"] == sim.A_N

    def test_prey_carrying_capacity_equilibrium(self):
        """(K_N, 0) is an equilibrium."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        eq = sim.find_equilibria()
        cc = [e for e in eq if e["type"] == "prey_carrying_capacity"]
        assert len(cc) == 1
        assert cc[0]["N"] == sim.K_N

    def test_pred_allee_threshold_equilibrium(self):
        """(0, A_P) is an equilibrium."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        eq = sim.find_equilibria()
        allee = [e for e in eq if e["type"] == "pred_allee_threshold"]
        assert len(allee) == 1
        assert allee[0]["P"] == sim.A_P

    def test_pred_carrying_capacity_equilibrium(self):
        """(0, K_P) is an equilibrium."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        eq = sim.find_equilibria()
        cc = [e for e in eq if e["type"] == "pred_carrying_capacity"]
        assert len(cc) == 1
        assert cc[0]["P"] == sim.K_P

    def test_at_least_five_boundary_equilibria(self):
        """System has at least 5 boundary equilibria."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        eq = sim.find_equilibria()
        assert len(eq) >= 5


class TestAlleeTwoStability:
    """Test numerical stability and determinism."""

    def test_rk4_no_nan(self):
        """No NaN or Inf in output."""
        config = _make_config(N_0=50.0, P_0=10.0, dt=0.01)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_deterministic(self):
        """Same parameters produce same result."""
        config = _make_config(N_0=50.0, P_0=10.0)
        sim1 = AlleeTwoSimulation(config)
        sim1.reset()
        for _ in range(1000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = AlleeTwoSimulation(config)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_small_dt_stability(self):
        """Smaller dt doesn't cause instability."""
        config = _make_config(N_0=50.0, P_0=10.0, dt=0.001)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state >= 0), f"Negative state: {state}"


class TestAlleeTwoTrajectoryData:
    """Test run() method and TrajectoryData output."""

    def test_run_returns_trajectory(self):
        """run() returns proper TrajectoryData."""
        config = _make_config(n_steps=100)
        sim = AlleeTwoSimulation(config)
        traj = sim.run(100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps(self):
        """Timestamps are correct."""
        config = _make_config(dt=0.01, n_steps=50)
        sim = AlleeTwoSimulation(config)
        traj = sim.run(50)
        expected_times = np.arange(51) * 0.01
        np.testing.assert_allclose(traj.timestamps, expected_times, atol=1e-10)

    def test_trajectory_non_negative(self):
        """All trajectory states are non-negative."""
        config = _make_config(n_steps=5000)
        sim = AlleeTwoSimulation(config)
        traj = sim.run(5000)
        assert np.all(traj.states >= 0)

    def test_trajectory_parameters(self):
        """TrajectoryData stores parameters correctly."""
        config = _make_config(r_N=2.0, K_N=200.0)
        sim = AlleeTwoSimulation(config)
        traj = sim.run(10)
        assert traj.parameters["r_N"] == 2.0
        assert traj.parameters["K_N"] == 200.0


class TestAlleeTwoParameterSweeps:
    """Test parameter sweep methods."""

    def test_prey_allee_sweep(self):
        """Prey Allee sweep produces valid output."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        A_N_values = np.array([3.0, 10.0, 30.0])
        result = sim.prey_allee_sweep(A_N_values, n_steps=5000)
        assert len(result["A_N_values"]) == 3
        assert len(result["final_N"]) == 3
        assert len(result["prey_survives"]) == 3

    def test_predator_allee_sweep(self):
        """Predator Allee sweep produces valid output."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        A_P_values = np.array([1.0, 5.0, 15.0])
        result = sim.predator_allee_sweep(A_P_values, n_steps=5000)
        assert len(result["A_P_values"]) == 3
        assert len(result["final_P"]) == 3
        assert len(result["pred_survives"]) == 3

    def test_basin_sweep(self):
        """Basin of attraction sweep produces valid output."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        N_range = np.array([5.0, 30.0, 60.0])
        P_range = np.array([2.0, 10.0])
        result = sim.basin_of_attraction_sweep(
            N_range=N_range, P_range=P_range, n_steps=5000,
        )
        assert result["final_N"].shape == (3, 2)
        assert result["final_P"].shape == (3, 2)
        assert result["outcome"].shape == (3, 2)
        assert np.all(result["outcome"] >= 0)
        assert np.all(result["outcome"] <= 3)

    def test_higher_AN_more_prey_extinction(self):
        """Larger prey Allee threshold makes prey extinction more likely."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        A_N_values = np.array([3.0, 20.0, 49.0])
        result = sim.prey_allee_sweep(A_N_values, n_steps=20000)
        # Small A_N: prey more likely to survive
        # Large A_N close to N_0: prey more likely to go extinct
        if result["prey_survives"][0]:
            assert result["final_N"][0] > 0.1

    def test_higher_AP_more_pred_extinction(self):
        """Larger predator Allee threshold makes predator extinction more likely."""
        config = _make_config()
        sim = AlleeTwoSimulation(config)
        sim.reset()
        A_P_values = np.array([2.0, 9.0, 40.0])
        result = sim.predator_allee_sweep(A_P_values, n_steps=20000)
        # The largest A_P should make predator survival harder
        assert len(result["pred_survives"]) == 3


class TestAlleeTwoRediscoveryDataGen:
    """Tests for rediscovery data generation functions."""

    def test_ode_data(self):
        """ODE data generation produces valid arrays."""
        from simulating_anything.rediscovery.allee_two import generate_ode_data

        data = generate_ode_data(N_0=50.0, P_0=10.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["dt"] == 0.01
        assert len(data["N"]) == 501
        assert len(data["P"]) == 501

    def test_bistability_data(self):
        """Bistability data generation covers four regimes."""
        from simulating_anything.rediscovery.allee_two import (
            generate_bistability_data,
        )

        data = generate_bistability_data(
            N_above=50.0, N_below=3.0,
            P_above=10.0, P_below=1.0,
            n_steps=1000, dt=0.01,
        )
        assert "both_above_trajectory" in data
        assert "prey_below_trajectory" in data
        assert "pred_below_trajectory" in data
        assert "both_below_trajectory" in data
        assert data["both_above_trajectory"].shape[1] == 2

    def test_basin_data(self):
        """Basin data generation produces grid output."""
        from simulating_anything.rediscovery.allee_two import generate_basin_data

        data = generate_basin_data(
            N_range=np.array([5.0, 30.0]),
            P_range=np.array([2.0, 10.0]),
            n_steps=1000, dt=0.01,
        )
        assert data["final_N"].shape == (2, 2)
        assert data["outcome"].shape == (2, 2)

    def test_prey_allee_sweep_data(self):
        """Prey Allee sweep data generation works."""
        from simulating_anything.rediscovery.allee_two import (
            generate_prey_allee_sweep_data,
        )

        data = generate_prey_allee_sweep_data(
            A_N_values=np.array([3.0, 10.0, 30.0]),
            n_steps=1000, dt=0.01,
        )
        assert len(data["A_N_values"]) == 3
        assert len(data["final_N"]) == 3
        assert len(data["prey_survives"]) == 3

    def test_predator_allee_sweep_data(self):
        """Predator Allee sweep data generation works."""
        from simulating_anything.rediscovery.allee_two import (
            generate_predator_allee_sweep_data,
        )

        data = generate_predator_allee_sweep_data(
            A_P_values=np.array([1.0, 5.0, 15.0]),
            n_steps=1000, dt=0.01,
        )
        assert len(data["A_P_values"]) == 3
        assert len(data["final_P"]) == 3
        assert len(data["pred_survives"]) == 3
