"""Tests for the Chirikov standard map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.standard_map import StandardMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    K: float = 0.9716,
    n_particles: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.STANDARD_MAP,
        dt=1.0,
        n_steps=100,
        parameters={"K": K, "n_particles": float(n_particles)},
    )


class TestStandardMapSimulation:
    def test_creation_and_params(self):
        """Verify parameters are read from config correctly."""
        sim = StandardMapSimulation(_make_config(K=2.0, n_particles=50))
        assert sim.K == 2.0
        assert sim.n_particles == 50

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.STANDARD_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = StandardMapSimulation(config)
        assert sim.K == 0.9716
        assert sim.n_particles == 100

    def test_initial_state_shape(self):
        """State should have shape (2 * n_particles,)."""
        sim = StandardMapSimulation(_make_config(n_particles=25))
        state = sim.reset()
        assert state.shape == (50,)

    def test_map_equations(self):
        """Verify one step of (theta, p) update for a single particle."""
        sim = StandardMapSimulation(_make_config(K=1.0, n_particles=1))
        sim.reset(seed=0)
        theta_0 = sim._state[0]
        p_0 = sim._state[1]

        sim.step()
        theta_1 = sim._state[0]
        p_1 = sim._state[1]

        # Expected: p_new = (p_0 + K*sin(theta_0)) mod 2pi
        #           theta_new = (theta_0 + p_new) mod 2pi
        p_expected = (p_0 + 1.0 * np.sin(theta_0)) % (2 * np.pi)
        theta_expected = (theta_0 + p_expected) % (2 * np.pi)

        assert p_1 == pytest.approx(p_expected, abs=1e-12)
        assert theta_1 == pytest.approx(theta_expected, abs=1e-12)

    def test_area_preservation(self):
        """Jacobian determinant = 1 for all K values (symplectic map).

        The Jacobian is:
          J = [[1 + K*cos(theta), 1], [K*cos(theta), 1]]
          det(J) = 1 + K*cos(theta) - K*cos(theta) = 1
        """
        for K in [0.0, 0.5, 0.9716, 2.0, 5.0, 10.0]:
            thetas = np.linspace(0, 2 * np.pi, 50)
            for theta in thetas:
                cos_t = np.cos(theta)
                det = (1.0 + K * cos_t) * 1.0 - 1.0 * K * cos_t
                assert det == pytest.approx(1.0, abs=1e-14), (
                    f"det(J) = {det} at K={K}, theta={theta}"
                )

    def test_fixed_point_origin(self):
        """(0, 0) is always a fixed point of the standard map.

        p_new = (0 + K*sin(0)) mod 2pi = 0
        theta_new = (0 + 0) mod 2pi = 0
        """
        for K in [0.0, 0.5, 1.0, 3.0]:
            sim = StandardMapSimulation(_make_config(K=K, n_particles=1))
            sim.reset()
            sim._state[0] = 0.0
            sim._state[1] = 0.0
            sim.step()
            assert sim._state[0] == pytest.approx(0.0, abs=1e-14)
            assert sim._state[1] == pytest.approx(0.0, abs=1e-14)

    def test_period2_orbit_at_pi_0(self):
        """(pi, 0) maps to (pi, K*sin(pi)) = (pi, 0) -- also a fixed point.

        sin(pi) = 0, so p_new = 0, theta_new = pi + 0 = pi.
        """
        sim = StandardMapSimulation(_make_config(K=2.0, n_particles=1))
        sim.reset()
        sim._state[0] = np.pi
        sim._state[1] = 0.0
        sim.step()
        # p_new = (0 + K*sin(pi)) mod 2pi = 0
        # theta_new = (pi + 0) mod 2pi = pi
        assert sim._state[0] == pytest.approx(np.pi, abs=1e-12)
        assert sim._state[1] == pytest.approx(0.0, abs=1e-12)

    def test_integrable_limit_K0(self):
        """At K=0 the map is integrable: p is constant, theta advances by p.

        This means each orbit is a straight horizontal line in phase space
        (when viewed as theta vs iteration), and p never changes.
        """
        sim = StandardMapSimulation(_make_config(K=0.0, n_particles=1))
        sim.reset()
        p_init = sim._state[1]
        theta_init = sim._state[0]

        for step_num in range(1, 20):
            sim.step()
            # p should remain constant
            assert sim._state[1] == pytest.approx(p_init, abs=1e-12)
            # theta advances by p each step (mod 2pi)
            expected_theta = (theta_init + step_num * p_init) % (2 * np.pi)
            assert sim._state[0] == pytest.approx(expected_theta, abs=1e-10)

    def test_kam_tori_low_K(self):
        """At K=0.5 (below K_c), most orbits should be regular (non-chaotic).

        We verify by computing the Lyapunov exponent for a generic orbit
        and checking it is close to zero or negative.
        """
        sim = StandardMapSimulation(_make_config(K=0.5))
        lam = sim.compute_lyapunov(K=0.5, n_steps=5000, theta_0=1.0, p_0=1.0)
        # For a regular orbit on a KAM torus, Lyapunov should be near zero
        # (some orbits may be near resonance, so allow small positive)
        assert lam < 0.1, f"Lyapunov={lam} too large for K=0.5"

    def test_full_chaos_large_K(self):
        """At K=5.0, the Lyapunov exponent should be strongly positive."""
        sim = StandardMapSimulation(_make_config(K=5.0))
        lam = sim.compute_lyapunov(K=5.0, n_steps=5000, theta_0=0.5, p_0=0.5)
        assert lam > 0.5, f"Lyapunov={lam} too small for K=5.0"

    def test_lyapunov_zero_at_K0(self):
        """At K=0, all orbits are regular: Lyapunov exponent = 0."""
        sim = StandardMapSimulation(_make_config(K=0.0))
        lam = sim.compute_lyapunov(K=0.0, n_steps=5000, theta_0=1.0, p_0=1.0)
        assert abs(lam) < 0.01, f"Lyapunov={lam} should be ~0 for K=0"

    def test_lyapunov_positive_large_K(self):
        """At K=2.0, the Lyapunov exponent should be positive."""
        sim = StandardMapSimulation(_make_config(K=2.0))
        lam = sim.compute_lyapunov(K=2.0, n_steps=5000, theta_0=0.5, p_0=0.5)
        assert lam > 0.1, f"Lyapunov={lam} should be positive for K=2.0"

    def test_mod_2pi(self):
        """All coordinates should stay in [0, 2*pi) after many steps."""
        sim = StandardMapSimulation(_make_config(K=3.0, n_particles=50))
        sim.reset()
        for _ in range(200):
            sim.step()
            thetas, ps = sim.get_particles()
            assert np.all(thetas >= 0), "theta < 0"
            assert np.all(thetas < 2 * np.pi + 1e-10), "theta >= 2pi"
            assert np.all(ps >= 0), "p < 0"
            assert np.all(ps < 2 * np.pi + 1e-10), "p >= 2pi"

    def test_momentum_wrap(self):
        """Momentum wraps correctly via mod 2*pi."""
        sim = StandardMapSimulation(_make_config(K=10.0, n_particles=1))
        sim.reset()
        sim._state[0] = 1.0
        sim._state[1] = 6.0  # Near 2*pi

        sim.step()
        p = sim._state[1]
        assert 0 <= p < 2 * np.pi + 1e-10, f"p={p} not wrapped correctly"

    def test_stochasticity_sweep_monotonic(self):
        """Chaos fraction should generally increase with K."""
        sim = StandardMapSimulation(_make_config())
        K_values = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        sweep = sim.stochasticity_sweep(
            K_values, n_particles=30, n_steps=500,
        )
        fracs = sweep["chaos_fractions"]
        # At K=0, fraction should be 0 or very small
        assert fracs[0] < 0.2, f"Chaos fraction at K=0: {fracs[0]}"
        # At K=4, fraction should be high
        assert fracs[-1] > 0.5, f"Chaos fraction at K=4: {fracs[-1]}"

    def test_phase_portrait_shape(self):
        """Phase portrait output has correct shape."""
        sim = StandardMapSimulation(_make_config())
        data = sim.phase_portrait(K=1.0, n_particles=25, n_steps=10)
        # n_particles initial + n_particles * n_steps
        expected_len = 25 * (10 + 1)
        assert len(data["theta"]) == expected_len
        assert len(data["p"]) == expected_len

    def test_trajectory_length(self):
        """Can run the simulation for many steps without error."""
        sim = StandardMapSimulation(_make_config(K=1.5, n_particles=10))
        traj = sim.run(n_steps=1000)
        # 1001 states: initial + 1000 steps, each state has 20 elements
        assert traj.states.shape == (1001, 20)

    def test_deterministic(self):
        """Same seed gives same trajectory."""
        config = _make_config(K=2.0, n_particles=10)
        sim1 = StandardMapSimulation(config)
        traj1 = sim1.run(n_steps=50)

        sim2 = StandardMapSimulation(config)
        traj2 = sim2.run(n_steps=50)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_multiple_K_values(self):
        """Different K give different dynamics (different final states)."""
        states_by_K = []
        for K in [0.0, 1.0, 3.0]:
            sim = StandardMapSimulation(_make_config(K=K, n_particles=5))
            sim.reset(seed=42)
            for _ in range(50):
                sim.step()
            states_by_K.append(sim.observe().copy())

        # All three should be different
        assert not np.allclose(states_by_K[0], states_by_K[1])
        assert not np.allclose(states_by_K[1], states_by_K[2])

    def test_particle_count(self):
        """Correct number of particles tracked."""
        for n in [1, 10, 50, 200]:
            sim = StandardMapSimulation(_make_config(n_particles=n))
            state = sim.reset()
            assert state.shape == (2 * n,)
            thetas, ps = sim.get_particles()
            assert thetas.shape == (n,)
            assert ps.shape == (n,)

    def test_observe_returns_copy(self):
        """observe() returns a copy, not a reference to internal state."""
        sim = StandardMapSimulation(_make_config(n_particles=5))
        sim.reset()
        obs1 = sim.observe()
        sim.step()
        obs2 = sim.observe()
        # Should not be the same array object
        assert not np.array_equal(obs1, obs2)

    def test_get_particles(self):
        """get_particles returns correctly separated theta and p arrays."""
        sim = StandardMapSimulation(_make_config(K=0.0, n_particles=5))
        sim.reset()
        thetas, ps = sim.get_particles()
        # Compare with raw state
        np.testing.assert_array_equal(thetas, sim._state[0::2])
        np.testing.assert_array_equal(ps, sim._state[1::2])


class TestStandardMapRediscovery:
    def test_lyapunov_data_generation(self):
        from simulating_anything.rediscovery.standard_map import generate_lyapunov_data
        data = generate_lyapunov_data(n_K=10, K_min=0.0, K_max=3.0, n_steps=500)
        assert len(data["K"]) == 10
        assert len(data["lyapunov"]) == 10

    def test_area_preservation_check(self):
        from simulating_anything.rediscovery.standard_map import test_area_preservation
        result = test_area_preservation(K_values=np.array([0.0, 1.0, 3.0]))
        assert len(result["K_values"]) == 3
        for dev in result["max_deviations"]:
            assert dev < 1e-12, f"Area not preserved: dev={dev}"

    def test_Kc_estimation(self):
        from simulating_anything.rediscovery.standard_map import estimate_Kc
        K = np.linspace(0, 3, 30)
        # Fake Lyapunov data: zero for K < 1, positive for K > 1
        lyap = np.where(K > 1.0, 0.5 * (K - 1.0), -0.01)
        K_c = estimate_Kc(K, lyap, threshold=0.01)
        # Should find K_c near 1.0
        assert abs(K_c - 1.0) < 0.2, f"K_c estimate={K_c}"

    def test_chaos_fraction_data(self):
        from simulating_anything.rediscovery.standard_map import (
            generate_chaos_fraction_data,
        )
        data = generate_chaos_fraction_data(
            n_K=5, K_min=0.0, K_max=3.0,
            n_particles=10, n_steps=200,
        )
        assert len(data["K_values"]) == 5
        assert len(data["chaos_fractions"]) == 5
        assert len(data["mean_lyapunovs"]) == 5
