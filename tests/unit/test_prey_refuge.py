"""Tests for the predator-prey with refuge simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.prey_refuge import PreyRefugeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 10.0,
    alpha: float = 0.5,
    m: float = 0.3,
    beta: float = 0.4,
    delta: float = 0.2,
    N_0: float = 2.0,
    P_0: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 2000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.PREY_REFUGE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "alpha": alpha, "m": m,
            "beta": beta, "delta": delta,
            "N_0": N_0, "P_0": P_0,
        },
    )


# ---------------------------------------------------------------------------
# Creation and basic interface tests
# ---------------------------------------------------------------------------

class TestPreyRefugeCreation:
    def test_creation_default_params(self):
        sim = PreyRefugeSimulation(_make_config())
        assert sim.r == 1.0
        assert sim.K == 10.0
        assert sim.alpha == 0.5
        assert sim.m == 0.3
        assert sim.beta == 0.4
        assert sim.delta == 0.2

    def test_creation_custom_params(self):
        sim = PreyRefugeSimulation(_make_config(r=2.0, K=20.0, m=0.5))
        assert sim.r == 2.0
        assert sim.K == 20.0
        assert sim.m == 0.5

    def test_reset_returns_initial_state(self):
        sim = PreyRefugeSimulation(_make_config(N_0=3.0, P_0=2.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [3.0, 2.0])

    def test_initial_state_shape(self):
        sim = PreyRefugeSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_observe_shape(self):
        sim = PreyRefugeSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_observe_matches_reset(self):
        sim = PreyRefugeSimulation(_make_config(N_0=5.0, P_0=3.0))
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_allclose(state, obs)

    def test_step_advances_state(self):
        sim = PreyRefugeSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestPreyRefugeProperties:
    def test_prey_population(self):
        sim = PreyRefugeSimulation(_make_config(N_0=5.0, P_0=3.0))
        sim.reset()
        assert sim.prey_population == pytest.approx(5.0)

    def test_predator_population(self):
        sim = PreyRefugeSimulation(_make_config(N_0=5.0, P_0=3.0))
        sim.reset()
        assert sim.predator_population == pytest.approx(3.0)

    def test_total_population(self):
        sim = PreyRefugeSimulation(_make_config(N_0=5.0, P_0=3.0))
        sim.reset()
        assert sim.total_population == pytest.approx(8.0)

    def test_properties_before_reset(self):
        sim = PreyRefugeSimulation(_make_config())
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0
        assert sim.total_population == 0.0


# ---------------------------------------------------------------------------
# Effective predation rate
# ---------------------------------------------------------------------------

class TestEffectivePredationRate:
    def test_effective_predation_default(self):
        sim = PreyRefugeSimulation(_make_config(alpha=0.5, m=0.3))
        assert sim.effective_predation_rate() == pytest.approx(0.35)

    def test_effective_predation_no_refuge(self):
        sim = PreyRefugeSimulation(_make_config(alpha=0.5, m=0.0))
        assert sim.effective_predation_rate() == pytest.approx(0.5)

    def test_effective_predation_high_refuge(self):
        sim = PreyRefugeSimulation(_make_config(alpha=0.5, m=0.9))
        assert sim.effective_predation_rate() == pytest.approx(0.05)

    def test_effective_predation_decreases_with_m(self):
        """Higher refuge fraction means lower effective predation."""
        rates = []
        for m_val in [0.0, 0.2, 0.4, 0.6, 0.8]:
            sim = PreyRefugeSimulation(_make_config(m=m_val))
            rates.append(sim.effective_predation_rate())
        # Strictly decreasing
        for i in range(len(rates) - 1):
            assert rates[i] > rates[i + 1]


# ---------------------------------------------------------------------------
# Equilibrium tests
# ---------------------------------------------------------------------------

class TestPreyRefugeEquilibrium:
    def test_equilibrium_exists(self):
        """With default params, coexistence equilibrium should exist."""
        sim = PreyRefugeSimulation(_make_config())
        N_star, P_star = sim.equilibrium()
        assert N_star > 0
        assert P_star > 0

    def test_equilibrium_formula(self):
        """Verify equilibrium matches the analytical formula."""
        sim = PreyRefugeSimulation(_make_config(
            r=1.0, K=10.0, alpha=0.5, m=0.3, beta=0.4, delta=0.2,
        ))
        N_star, P_star = sim.equilibrium()
        # N* = delta / (beta * alpha * (1-m)) = 0.2 / (0.4 * 0.5 * 0.7)
        expected_N = 0.2 / (0.4 * 0.5 * 0.7)
        # P* = r * (1 - N*/K) / (alpha*(1-m))
        expected_P = 1.0 * (1.0 - expected_N / 10.0) / (0.5 * 0.7)
        assert N_star == pytest.approx(expected_N, rel=1e-10)
        assert P_star == pytest.approx(expected_P, rel=1e-10)

    def test_derivatives_zero_at_equilibrium(self):
        """At coexistence equilibrium, derivatives should be zero."""
        sim = PreyRefugeSimulation(_make_config())
        N_star, P_star = sim.equilibrium()
        state = np.array([N_star, P_star])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_equilibrium_increases_prey_with_m(self):
        """Increasing refuge fraction m raises prey equilibrium N*."""
        N_stars = []
        for m_val in [0.0, 0.2, 0.4, 0.6]:
            sim = PreyRefugeSimulation(_make_config(m=m_val))
            N_star, _ = sim.equilibrium()
            N_stars.append(N_star)
        for i in range(len(N_stars) - 1):
            assert N_stars[i] < N_stars[i + 1], (
                f"N* should increase with m: m_vals=0.0..0.6, N*={N_stars}"
            )

    def test_predator_equilibrium_varies_with_m(self):
        """P* = r*(1-N*/K) / (alpha*(1-m)) changes with m.

        As m increases, N* increases (more prey survive), but the denominator
        alpha*(1-m) decreases faster than the numerator. So P* can increase
        or decrease depending on parameters. Verify the formula is consistent.
        """
        for m_val in [0.0, 0.2, 0.4, 0.6]:
            sim = PreyRefugeSimulation(_make_config(m=m_val))
            N_star, P_star = sim.equilibrium()
            eff = sim.effective_predation_rate()
            # Verify P* formula: P* = r*(1 - N*/K) / eff
            expected_P = sim.r * (1.0 - N_star / sim.K) / eff
            assert P_star == pytest.approx(expected_P, rel=1e-10), (
                f"P* formula mismatch at m={m_val}"
            )

    def test_predator_equilibrium_high_refuge_eventually_fails(self):
        """At very high m, N* approaches K and P* drops toward zero
        eventually the equilibrium ceases to exist."""
        # With m close to 1, N* = delta/(beta*alpha*(1-m)) grows unbounded
        # and eventually exceeds K, so equilibrium disappears
        sim = PreyRefugeSimulation(_make_config(m=0.95))
        with pytest.raises(ValueError):
            sim.equilibrium()

    def test_no_equilibrium_when_predator_unsustainable(self):
        """When delta is too high relative to predation, no coexistence."""
        # N* = delta/(beta*alpha*(1-m)) = 5.0/(0.4*0.5*0.7) = 35.71 > K=10
        sim = PreyRefugeSimulation(_make_config(delta=5.0))
        with pytest.raises(ValueError, match="N\\*.*>=.*K"):
            sim.equilibrium()


# ---------------------------------------------------------------------------
# Stability tests
# ---------------------------------------------------------------------------

class TestPreyRefugeStability:
    def test_is_stable_default(self):
        """Default parameters should produce a well-defined stability answer."""
        sim = PreyRefugeSimulation(_make_config())
        # Just check it doesn't crash
        result = sim.is_stable()
        assert isinstance(result, bool)

    def test_high_refuge_stabilizes(self):
        """High refuge fraction tends to stabilize the equilibrium."""
        sim = PreyRefugeSimulation(_make_config(m=0.8))
        assert sim.is_stable(), "High refuge should stabilize equilibrium"

    def test_jacobian_eigenvalues(self):
        """Eigenvalues should exist and have correct properties."""
        sim = PreyRefugeSimulation(_make_config())
        e1, e2 = sim.jacobian_eigenvalues()
        # For a 2D system, eigenvalues come in conjugate pairs or are real
        # Their product is the determinant (should be positive)
        det = e1 * e2
        assert det.real > 0, "Determinant should be positive"

    def test_stable_eigenvalues_negative_real_part(self):
        """When stable, eigenvalues should have negative real part."""
        sim = PreyRefugeSimulation(_make_config(m=0.8))
        if sim.is_stable():
            e1, e2 = sim.jacobian_eigenvalues()
            assert e1.real < 0
            assert e2.real < 0


# ---------------------------------------------------------------------------
# Dynamics and trajectory tests
# ---------------------------------------------------------------------------

class TestPreyRefugeDynamics:
    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = PreyRefugeSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up."""
        sim = PreyRefugeSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 200), f"Population blow-up: {state}"

    def test_prey_bounded_by_K(self):
        """Prey should not significantly exceed carrying capacity K."""
        sim = PreyRefugeSimulation(_make_config(K=10.0, N_0=1.0, P_0=0.5))
        sim.reset()
        max_N = 0.0
        for _ in range(20000):
            sim.step()
            N = sim.observe()[0]
            max_N = max(max_N, N)
        # Prey may overshoot K slightly in oscillations but not by much
        assert max_N < 2.0 * sim.K, f"Prey exceeded 2*K: max_N={max_N}"

    def test_no_refuge_matches_standard_model(self):
        """With m=0, should match standard LV with logistic growth.

        dN/dt = r*N*(1-N/K) - alpha*N*P
        dP/dt = beta*alpha*N*P - delta*P
        """
        sim = PreyRefugeSimulation(_make_config(m=0.0, N_0=2.0, P_0=1.0))
        sim.reset()

        # Manually compute one RK4 step of the standard model
        N0, P0 = 2.0, 1.0
        r, K, alpha, beta, delta = 1.0, 10.0, 0.5, 0.4, 0.2
        dt = 0.01

        def derivs(N, P):
            dN = r * N * (1 - N / K) - alpha * N * P
            dP = beta * alpha * N * P - delta * P
            return dN, dP

        # RK4
        k1_N, k1_P = derivs(N0, P0)
        k2_N, k2_P = derivs(N0 + 0.5 * dt * k1_N, P0 + 0.5 * dt * k1_P)
        k3_N, k3_P = derivs(N0 + 0.5 * dt * k2_N, P0 + 0.5 * dt * k2_P)
        k4_N, k4_P = derivs(N0 + dt * k3_N, P0 + dt * k3_P)

        N1 = N0 + (dt / 6) * (k1_N + 2 * k2_N + 2 * k3_N + k4_N)
        P1 = P0 + (dt / 6) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)

        sim.step()
        state = sim.observe()
        np.testing.assert_allclose(state, [N1, P1], rtol=1e-12)

    def test_high_refuge_reduces_oscillation(self):
        """High refuge fraction should reduce oscillation amplitude."""
        amps = []
        for m_val in [0.0, 0.5, 0.8]:
            config = _make_config(m=m_val, n_steps=30000, dt=0.01, N_0=2.0, P_0=1.0)
            sim = PreyRefugeSimulation(config)
            sim.reset()

            states = []
            for _ in range(30000):
                sim.step()
                states.append(sim.observe().copy())

            traj = np.array(states)
            half = len(traj) // 2
            amp = np.max(traj[half:, 0]) - np.min(traj[half:, 0])
            amps.append(amp)

        # Amplitude should generally decrease with m
        assert amps[2] < amps[0], (
            f"High refuge should reduce oscillation: amps={amps}"
        )

    def test_predator_extinction_high_delta(self):
        """Very high predator death rate should lead to predator extinction."""
        config = _make_config(delta=5.0, N_0=5.0, P_0=2.0, n_steps=10000)
        sim = PreyRefugeSimulation(config)
        sim.reset()

        for _ in range(10000):
            sim.step()

        _, P = sim.observe()
        assert P < 0.01, f"Predator should go extinct with high delta: P={P}"

    def test_prey_grows_without_predator(self):
        """Without predators, prey should reach carrying capacity."""
        config = _make_config(N_0=1.0, P_0=0.0, K=10.0, n_steps=50000, dt=0.01)
        sim = PreyRefugeSimulation(config)
        sim.reset()

        for _ in range(50000):
            sim.step()

        N, P = sim.observe()
        assert N == pytest.approx(10.0, abs=0.1), f"Prey should reach K: N={N}"
        assert P == pytest.approx(0.0), f"Predator should stay at 0: P={P}"


# ---------------------------------------------------------------------------
# Trajectory collection tests
# ---------------------------------------------------------------------------

class TestPreyRefugeTrajectory:
    def test_run_trajectory_shape(self):
        sim = PreyRefugeSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201

    def test_run_timestamps(self):
        sim = PreyRefugeSimulation(_make_config(dt=0.01, n_steps=100))
        traj = sim.run(n_steps=100)
        expected_t = np.arange(101) * 0.01
        np.testing.assert_allclose(traj.timestamps, expected_t, atol=1e-10)

    def test_reproducibility(self):
        cfg = _make_config(n_steps=100)
        sim1 = PreyRefugeSimulation(cfg)
        traj1 = sim1.run(100)
        sim2 = PreyRefugeSimulation(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_deterministic(self):
        """Same parameters produce identical result after many steps."""
        config = _make_config(N_0=3.0, P_0=2.0)

        sim1 = PreyRefugeSimulation(config)
        sim1.reset()
        for _ in range(2000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = PreyRefugeSimulation(config)
        sim2.reset()
        for _ in range(2000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_rk4_stability_no_nan(self):
        """No NaN or Inf in simulation output."""
        config = _make_config(N_0=5.0, P_0=3.0, dt=0.01)
        sim = PreyRefugeSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"


# ---------------------------------------------------------------------------
# Rediscovery data generation tests
# ---------------------------------------------------------------------------

class TestPreyRefugeRediscovery:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.prey_refuge import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 10.0
        assert data["alpha"] == 0.5
        assert data["m"] == 0.3

    def test_trajectory_data_positivity(self):
        from simulating_anything.rediscovery.prey_refuge import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=2000, dt=0.01)
        assert np.all(data["states"] >= 0), "All states should be non-negative"

    def test_refuge_sweep_data_generation(self):
        from simulating_anything.rediscovery.prey_refuge import (
            generate_refuge_sweep_data,
        )
        data = generate_refuge_sweep_data(
            n_m_values=5, n_steps=1000, dt=0.01,
        )
        assert len(data["m"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["P_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["N_star_theory"]) == 5

    def test_refuge_sweep_prey_trend(self):
        """Prey average should generally increase with m in sweep data."""
        from simulating_anything.rediscovery.prey_refuge import (
            generate_refuge_sweep_data,
        )
        data = generate_refuge_sweep_data(
            n_m_values=5, m_min=0.0, m_max=0.8,
            n_steps=10000, dt=0.01,
        )
        # Average prey at highest m should exceed that at lowest m
        assert data["N_avg"][-1] > data["N_avg"][0], (
            f"Prey should increase with refuge: "
            f"N_avg[0]={data['N_avg'][0]:.3f}, N_avg[-1]={data['N_avg'][-1]:.3f}"
        )
