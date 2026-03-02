"""Tests for the predator-prey model with toxic defense."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.predator_prey_toxin import (
    PredatorPreyToxinSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N_0: float = 50.0,
    P_0: float = 10.0,
    r: float = 1.0,
    K: float = 100.0,
    a: float = 0.5,
    h: float = 0.02,
    e: float = 0.4,
    d: float = 0.2,
    c: float = 0.001,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.PREDATOR_PREY_TOXIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "h": h, "e": e, "d": d, "c": c,
            "N_0": N_0, "P_0": P_0,
        },
    )


class TestSimulation:
    """Core simulation creation and interface tests."""

    def test_init_default_params(self):
        """Default parameters are set correctly."""
        sim = PredatorPreyToxinSimulation(_make_config())
        assert sim.r == 1.0
        assert sim.K == 100.0
        assert sim.a == 0.5
        assert sim.h == 0.02
        assert sim.e == 0.4
        assert sim.d == 0.2
        assert sim.c == 0.001

    def test_init_custom_params(self):
        """Custom parameters override defaults."""
        sim = PredatorPreyToxinSimulation(_make_config(r=2.0, K=200.0, c=0.005))
        assert sim.r == 2.0
        assert sim.K == 200.0
        assert sim.c == 0.005

    def test_reset_returns_state(self):
        """Reset returns initial state array."""
        sim = PredatorPreyToxinSimulation(_make_config(N_0=42.0, P_0=7.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [42.0, 7.5])

    def test_reset_shape(self):
        """Reset returns 2-element state."""
        sim = PredatorPreyToxinSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_step_returns_state(self):
        """Step returns a numpy array."""
        sim = PredatorPreyToxinSimulation(_make_config())
        sim.reset()
        state = sim.step()
        assert isinstance(state, np.ndarray)
        assert state.shape == (2,)

    def test_step_advances_state(self):
        """State changes after a step."""
        sim = PredatorPreyToxinSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """Observe returns the current state."""
        sim = PredatorPreyToxinSimulation(_make_config(N_0=50.0, P_0=10.0))
        sim.reset()
        obs = sim.observe()
        np.testing.assert_allclose(obs, [50.0, 10.0])
        assert obs.shape == (2,)

    def test_run_trajectory_shape(self):
        """Run returns trajectory with correct shape."""
        sim = PredatorPreyToxinSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201

    def test_run_trajectory_initial(self):
        """Run trajectory starts at initial conditions."""
        sim = PredatorPreyToxinSimulation(
            _make_config(N_0=50.0, P_0=10.0, n_steps=100)
        )
        traj = sim.run(n_steps=100)
        np.testing.assert_allclose(traj.states[0], [50.0, 10.0])

    def test_deterministic(self):
        """Same parameters produce identical results."""
        config = _make_config(N_0=50.0, P_0=10.0)

        sim1 = PredatorPreyToxinSimulation(config)
        sim1.reset()
        for _ in range(1000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = PredatorPreyToxinSimulation(config)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)


class TestRK4:
    """RK4 integration tests: non-negativity, boundedness, stability."""

    def test_populations_non_negative(self):
        """N, P >= 0 at all times."""
        sim = PredatorPreyToxinSimulation(_make_config(N_0=10.0, P_0=30.0))
        sim.reset()

        for _ in range(10000):
            sim.step()
            N, P = sim.observe()
            assert N >= 0, f"Prey went negative: N={N}"
            assert P >= 0, f"Predator went negative: P={P}"

    def test_populations_bounded(self):
        """Populations do not diverge."""
        sim = PredatorPreyToxinSimulation(_make_config(N_0=80.0, P_0=5.0))
        sim.reset()

        for _ in range(50000):
            sim.step()
            N, P = sim.observe()
            assert N < 500, f"Prey diverged: N={N}"
            assert P < 500, f"Predator diverged: P={P}"

    def test_carrying_capacity_respected(self):
        """Without predators, prey reaches K."""
        sim = PredatorPreyToxinSimulation(
            _make_config(N_0=10.0, P_0=0.0, K=100.0)
        )
        sim.reset()

        for _ in range(100000):
            sim.step()

        final_N = sim.observe()[0]
        assert final_N == pytest.approx(100.0, abs=2.0), (
            f"Prey should reach K: N={final_N}"
        )

    def test_no_nan_or_inf(self):
        """No NaN or Inf in simulation output."""
        sim = PredatorPreyToxinSimulation(_make_config(N_0=50.0, P_0=10.0, dt=0.01))
        sim.reset()

        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_predator_extinction_without_prey(self):
        """Predators die off without prey."""
        sim = PredatorPreyToxinSimulation(_make_config(N_0=0.0, P_0=10.0))
        sim.reset()

        for _ in range(50000):
            sim.step()

        final_P = sim.observe()[1]
        assert final_P < 0.01, f"Predator should die without prey: P={final_P}"

    def test_prey_grows_without_predators(self):
        """Prey grows logistically without predators."""
        sim = PredatorPreyToxinSimulation(
            _make_config(N_0=10.0, P_0=0.0, K=100.0)
        )
        sim.reset()
        initial_N = sim.observe()[0]

        for _ in range(5000):
            sim.step()

        final_N = sim.observe()[0]
        assert final_N > initial_N, (
            f"Prey should grow: initial={initial_N}, final={final_N}"
        )


class TestFunctionalResponse:
    """Tests for the modified functional response."""

    def test_zero_at_zero(self):
        """f(0) = 0."""
        sim = PredatorPreyToxinSimulation(_make_config())
        assert sim.functional_response(0.0) == pytest.approx(0.0)

    def test_positive_for_positive_N(self):
        """f(N) > 0 for N > 0."""
        sim = PredatorPreyToxinSimulation(_make_config())
        for N in [1.0, 10.0, 50.0, 100.0]:
            assert sim.functional_response(N) > 0, f"f({N}) should be positive"

    def test_matches_holling2_when_c_zero(self):
        """When c=0, matches standard Holling Type II."""
        sim = PredatorPreyToxinSimulation(_make_config(c=0.0))
        N_values = np.array([1.0, 10.0, 50.0, 100.0, 200.0])

        fr_toxin = sim.functional_response(N_values)
        fr_holling = sim.functional_response_holling2(N_values)

        np.testing.assert_allclose(fr_toxin, fr_holling, rtol=1e-10)

    def test_decreases_at_high_N_with_toxin(self):
        """With c > 0, f(N) eventually decreases for large N."""
        sim = PredatorPreyToxinSimulation(_make_config(c=0.001))

        # Find peak and verify decline after peak
        N_values = np.linspace(1, 500, 1000)
        fr_values = sim.functional_response(N_values)
        peak_idx = np.argmax(fr_values)

        # There should be a peak before N=500
        assert peak_idx > 0, "Should have a peak in functional response"
        assert peak_idx < len(N_values) - 1, (
            "Peak should be before maximum N tested"
        )

        # Value after peak should be less than peak
        assert fr_values[-1] < fr_values[peak_idx], (
            "f(N) should decrease after peak with toxin"
        )

    def test_holling2_saturates(self):
        """Holling Type II saturates at 1/h for large N."""
        sim = PredatorPreyToxinSimulation(_make_config(h=0.02, a=0.5))
        fr_large = sim.functional_response_holling2(1000000.0)
        expected = 1.0 / 0.02  # 1/h = 50
        assert fr_large == pytest.approx(expected, rel=0.01)

    def test_toxin_reduces_response(self):
        """At high N, toxin version gives lower response than Holling II."""
        sim = PredatorPreyToxinSimulation(_make_config(c=0.001))
        N_high = 200.0

        fr_toxin = sim.functional_response(N_high)
        fr_holling = sim.functional_response_holling2(N_high)

        assert fr_toxin < fr_holling, (
            f"Toxin should reduce predation: "
            f"toxin={fr_toxin:.4f}, holling={fr_holling:.4f}"
        )

    def test_array_input(self):
        """Functional response works with numpy array input."""
        sim = PredatorPreyToxinSimulation(_make_config())
        N_values = np.array([0.0, 10.0, 50.0, 100.0])
        result = sim.functional_response(N_values)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(0.0)

    def test_stronger_toxin_more_reduction(self):
        """Higher c gives more reduction in functional response at high N."""
        sim_low = PredatorPreyToxinSimulation(_make_config(c=0.0005))
        sim_high = PredatorPreyToxinSimulation(_make_config(c=0.005))
        N_test = 150.0

        fr_low = sim_low.functional_response(N_test)
        fr_high = sim_high.functional_response(N_test)

        assert fr_high < fr_low, (
            f"Stronger toxin should reduce more: "
            f"c=0.0005: {fr_low:.4f}, c=0.005: {fr_high:.4f}"
        )


class TestToxinEffect:
    """Tests for the toxin defense mechanism."""

    def test_c_zero_gives_standard_dynamics(self):
        """With c=0, system behaves like Rosenzweig-MacArthur."""
        # Use lower attack rate so coexistence is possible without toxin
        sim_toxin = PredatorPreyToxinSimulation(
            _make_config(c=0.0, N_0=50.0, P_0=5.0, a=0.1, h=0.1, e=0.5, d=0.1)
        )
        sim_toxin.reset()

        # Run and collect trajectory
        for _ in range(5000):
            sim_toxin.step()

        state = sim_toxin.observe()
        # Both populations should survive with moderate predation parameters
        assert state[0] > 0.1, f"Prey should survive with c=0: N={state[0]}"
        assert state[1] > 0.1, f"Predator should survive with c=0: P={state[1]}"

    def test_high_c_prey_persists(self):
        """With high toxin, prey can persist at higher density."""
        # High toxin: prey should have higher average density
        sim_high = PredatorPreyToxinSimulation(
            _make_config(c=0.008, N_0=50.0, P_0=10.0)
        )
        sim_high.reset()

        sim_low = PredatorPreyToxinSimulation(
            _make_config(c=0.0001, N_0=50.0, P_0=10.0)
        )
        sim_low.reset()

        n_steps = 50000
        states_high = [sim_high.observe().copy()]
        states_low = [sim_low.observe().copy()]
        for _ in range(n_steps):
            sim_high.step()
            sim_low.step()
            states_high.append(sim_high.observe().copy())
            states_low.append(sim_low.observe().copy())

        traj_high = np.array(states_high)
        traj_low = np.array(states_low)

        # Compare average prey density in second half
        half = n_steps // 2
        N_avg_high = np.mean(traj_high[half:, 0])
        N_avg_low = np.mean(traj_low[half:, 0])

        assert N_avg_high > N_avg_low, (
            f"Higher toxin should support higher prey density: "
            f"high c avg={N_avg_high:.2f}, low c avg={N_avg_low:.2f}"
        )

    def test_toxin_reduces_predator_density(self):
        """Strong toxin reduces average predator density."""
        sim_c0 = PredatorPreyToxinSimulation(
            _make_config(c=0.0, N_0=50.0, P_0=10.0)
        )
        sim_c0.reset()

        sim_ctox = PredatorPreyToxinSimulation(
            _make_config(c=0.008, N_0=50.0, P_0=10.0)
        )
        sim_ctox.reset()

        n_steps = 50000
        states_c0 = [sim_c0.observe().copy()]
        states_ctox = [sim_ctox.observe().copy()]
        for _ in range(n_steps):
            sim_c0.step()
            sim_ctox.step()
            states_c0.append(sim_c0.observe().copy())
            states_ctox.append(sim_ctox.observe().copy())

        traj_c0 = np.array(states_c0)
        traj_ctox = np.array(states_ctox)

        half = n_steps // 2
        P_avg_c0 = np.mean(traj_c0[half:, 1])
        P_avg_ctox = np.mean(traj_ctox[half:, 1])

        assert P_avg_ctox < P_avg_c0, (
            f"Toxin should reduce predator density: "
            f"c=0 avg={P_avg_c0:.2f}, c=0.008 avg={P_avg_ctox:.2f}"
        )

    def test_derivatives_match_equations(self):
        """Verify derivatives manually at a specific point."""
        sim = PredatorPreyToxinSimulation(
            _make_config(r=1.0, K=100.0, a=0.5, h=0.02, e=0.4, d=0.2, c=0.001)
        )

        N, P = 50.0, 10.0
        state = np.array([N, P])
        dy = sim._derivatives(state)

        # Manually compute expected derivatives
        growth = 1.0 * 50.0 * (1.0 - 50.0 / 100.0)  # = 25.0
        fr = 0.5 * 50.0 / (1.0 + 0.5 * 0.02 * 50.0 + 0.001 * 50.0**2)
        # denominator = 1 + 0.5 + 2.5 = 4.0
        # fr = 25 / 4.0 = 6.25
        expected_dN = growth - fr * P  # 25.0 - 6.25 * 10 = -37.5
        expected_dP = 0.4 * fr * P - 0.2 * P  # 0.4*6.25*10 - 2 = 23.0

        assert dy[0] == pytest.approx(expected_dN, rel=1e-10)
        assert dy[1] == pytest.approx(expected_dP, rel=1e-10)


class TestEquilibrium:
    """Tests for equilibrium computation."""

    def test_find_equilibria_includes_extinction(self):
        """Extinction (0,0) is always an equilibrium."""
        sim = PredatorPreyToxinSimulation(_make_config())
        eq = sim.find_equilibria()
        types = [e["type"] for e in eq]
        assert "extinction" in types

    def test_find_equilibria_includes_prey_only(self):
        """Prey-only (K, 0) is always an equilibrium."""
        sim = PredatorPreyToxinSimulation(_make_config(K=100.0))
        eq = sim.find_equilibria()
        prey_only = [e for e in eq if e["type"] == "prey_only"]
        assert len(prey_only) == 1
        assert prey_only[0]["N"] == pytest.approx(100.0)
        assert prey_only[0]["P"] == pytest.approx(0.0)

    def test_coexistence_equilibrium_exists(self):
        """With default parameters, a coexistence equilibrium should exist."""
        sim = PredatorPreyToxinSimulation(_make_config())
        eq = sim.find_equilibria()
        coexist = [e for e in eq if e["type"] == "coexistence"]
        assert len(coexist) >= 1, "Should find coexistence equilibrium"
        assert coexist[0]["N"] > 0
        assert coexist[0]["P"] > 0

    def test_equilibrium_prey_with_c_zero(self):
        """With c=0, equilibrium_prey gives standard Holling II result."""
        sim = PredatorPreyToxinSimulation(
            _make_config(c=0.0, a=0.5, h=0.02, e=0.4, d=0.2)
        )
        N_star = sim.equilibrium_prey()
        # N* = d / (e*a - a*h*d) = 0.2 / (0.2 - 0.002) = 0.2/0.198 = 1.0101...
        expected = 0.2 / (0.4 * 0.5 - 0.5 * 0.02 * 0.2)
        assert N_star is not None
        assert N_star == pytest.approx(expected, rel=1e-6)

    def test_equilibrium_is_fixed_point(self):
        """Derivatives are near zero at computed equilibrium."""
        sim = PredatorPreyToxinSimulation(_make_config())
        N_star = sim.equilibrium_prey()
        if N_star is not None:
            P_star = sim.equilibrium_predator(N_star)
            if P_star > 0:
                state = np.array([N_star, P_star])
                dy = sim._derivatives(state)
                np.testing.assert_allclose(
                    dy, [0.0, 0.0], atol=1e-8,
                    err_msg=f"Derivatives not zero at equilibrium: {dy}"
                )

    def test_equilibrium_predator_positive(self):
        """Predator equilibrium is positive for valid parameters."""
        sim = PredatorPreyToxinSimulation(_make_config())
        N_star = sim.equilibrium_prey()
        assert N_star is not None
        P_star = sim.equilibrium_predator(N_star)
        assert P_star > 0, f"P* should be positive: P*={P_star}"

    def test_equilibrium_prey_below_K(self):
        """Prey equilibrium N* < K."""
        sim = PredatorPreyToxinSimulation(_make_config(K=100.0))
        N_star = sim.equilibrium_prey()
        assert N_star is not None
        assert N_star < 100.0, f"N* should be below K: N*={N_star}"


class TestIsOscillatory:
    """Tests for oscillation detection."""

    def test_constant_trajectory_not_oscillatory(self):
        """A constant trajectory is not oscillatory."""
        sim = PredatorPreyToxinSimulation(_make_config(K=100.0))
        # Constant state
        states = np.ones((1000, 2)) * 50.0
        assert not sim.is_oscillatory(states)

    def test_oscillating_trajectory_detected(self):
        """A sinusoidal trajectory is detected as oscillatory."""
        sim = PredatorPreyToxinSimulation(_make_config(K=100.0))
        t = np.linspace(0, 20 * np.pi, 2000)
        states = np.column_stack([
            50.0 + 20.0 * np.sin(t),  # N oscillating with amplitude 20
            10.0 + 5.0 * np.cos(t),   # P oscillating
        ])
        assert sim.is_oscillatory(states, threshold=0.05)

    def test_short_trajectory_not_oscillatory(self):
        """Very short trajectories return False."""
        sim = PredatorPreyToxinSimulation(_make_config(K=100.0))
        states = np.array([[50.0, 10.0], [51.0, 9.0]])
        assert not sim.is_oscillatory(states)

    def test_damped_to_steady_not_oscillatory(self):
        """Trajectory that damps to steady state should not be oscillatory."""
        sim = PredatorPreyToxinSimulation(_make_config(K=100.0))
        n = 2000
        t = np.linspace(0, 10, n)
        # Exponentially damped oscillation (amplitude < threshold in second half)
        amp = 0.1 * np.exp(-2.0 * t)
        states = np.column_stack([
            50.0 + amp * np.sin(10 * t),
            10.0 + 0.01 * amp * np.cos(10 * t),
        ])
        assert not sim.is_oscillatory(states, threshold=0.05)


class TestToxinSweep:
    """Tests for the toxin_sweep method on the simulation class."""

    def test_sweep_returns_correct_keys(self):
        """Toxin sweep returns all expected keys."""
        sim = PredatorPreyToxinSimulation(_make_config())
        sim.reset()
        c_values = np.array([0.0, 0.001, 0.005])
        result = sim.toxin_sweep(c_values, n_steps=1000)

        assert "c_values" in result
        assert "final_N" in result
        assert "final_P" in result
        assert "oscillatory" in result

    def test_sweep_shapes(self):
        """Sweep output arrays have correct shapes."""
        sim = PredatorPreyToxinSimulation(_make_config())
        sim.reset()
        c_values = np.array([0.0, 0.002, 0.004, 0.006, 0.008])
        result = sim.toxin_sweep(c_values, n_steps=1000)

        assert len(result["c_values"]) == 5
        assert len(result["final_N"]) == 5
        assert len(result["final_P"]) == 5
        assert len(result["oscillatory"]) == 5


class TestRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_toxin_sweep_data_generation(self):
        """Toxin sweep data generation produces valid output."""
        from simulating_anything.rediscovery.predator_prey_toxin import (
            generate_toxin_sweep_data,
        )

        data = generate_toxin_sweep_data(
            n_c_values=5, c_min=0.0, c_max=0.01, n_steps=1000, dt=0.01,
        )
        assert len(data["c_values"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["P_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["oscillatory"]) == 5
        assert data["c_values"][0] == pytest.approx(0.0)
        assert data["c_values"][-1] == pytest.approx(0.01)

    def test_enrichment_data_generation(self):
        """Enrichment sweep data generation produces valid output."""
        from simulating_anything.rediscovery.predator_prey_toxin import (
            generate_enrichment_data,
        )

        data = generate_enrichment_data(
            n_r_values=5, r_min=0.5, r_max=3.0, n_steps=1000, dt=0.01,
        )
        assert len(data["r_values"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["P_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["oscillatory"]) == 5

    def test_trajectory_data_generation(self):
        """Trajectory data generation produces valid output."""
        from simulating_anything.rediscovery.predator_prey_toxin import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["dt"] == 0.01
        assert data["c"] == 0.001

    def test_functional_response_comparison(self):
        """Functional response comparison data is valid."""
        from simulating_anything.rediscovery.predator_prey_toxin import (
            generate_functional_response_comparison,
        )

        data = generate_functional_response_comparison(
            N_values=np.linspace(0, 100, 50),
            c_values=np.array([0.0, 0.001]),
        )
        assert "N_values" in data
        assert "holling2" in data
        assert len(data["N_values"]) == 50
        assert len(data["holling2"]) == 50

    def test_trajectory_populations_valid(self):
        """Generated trajectory has non-negative populations."""
        from simulating_anything.rediscovery.predator_prey_toxin import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=2000, dt=0.01)
        assert np.all(data["N"] >= 0), "Prey should be non-negative"
        assert np.all(data["P"] >= 0), "Predator should be non-negative"
        assert np.all(np.isfinite(data["states"])), "All states should be finite"
