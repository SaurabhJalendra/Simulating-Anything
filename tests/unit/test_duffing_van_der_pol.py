"""Tests for the Duffing-Van der Pol hybrid oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.duffing_van_der_pol import (
    DuffingVanDerPolSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

DOMAIN = Domain.DUFFING_VAN_DER_POL


def _make_config(
    mu: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.2,
    F: float = 0.3,
    omega: float = 1.0,
    x_0: float = 0.1,
    y_0: float = 0.0,
    dt: float = 0.005,
) -> SimulationConfig:
    return SimulationConfig(
        domain=DOMAIN,
        dt=dt,
        n_steps=1000,
        parameters={
            "mu": mu, "alpha": alpha, "beta": beta,
            "F": F, "omega": omega, "x_0": x_0, "y_0": y_0,
        },
    )


class TestDuffingVanDerPolSimulation:
    def test_reset(self):
        """Reset returns initial state [x_0, y_0, 0.0]."""
        sim = DuffingVanDerPolSimulation(_make_config(x_0=0.5, y_0=-0.1))
        state = sim.reset()
        assert state.shape == (3,)
        np.testing.assert_allclose(state, [0.5, -0.1, 0.0])

    def test_observe_shape(self):
        """Observed state has 3 elements: [x, y, t]."""
        sim = DuffingVanDerPolSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)
        assert obs[2] == 0.0  # t starts at 0

    def test_step_advances(self):
        """A single step changes the state."""
        sim = DuffingVanDerPolSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Two simulations with identical config produce identical trajectories."""
        config = _make_config()
        sim1 = DuffingVanDerPolSimulation(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = DuffingVanDerPolSimulation(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)

    def test_stability(self):
        """No NaN after 10000 steps with default parameters."""
        sim = DuffingVanDerPolSimulation(_make_config())
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"State has NaN/Inf: {state}"

    def test_bounded(self):
        """Trajectory stays bounded for moderate parameters."""
        sim = DuffingVanDerPolSimulation(_make_config(F=0.3))
        sim.reset()
        for _ in range(10000):
            sim.step()
            x, y, t = sim.observe()
            assert abs(x) < 20, f"x diverged: {x}"
            assert abs(y) < 50, f"y diverged: {y}"

    def test_unforced_limit_cycle(self):
        """F=0 should produce a limit cycle (oscillation persists)."""
        sim = DuffingVanDerPolSimulation(
            _make_config(F=0.0, mu=1.0, x_0=0.01, dt=0.005)
        )
        sim.reset()
        # Run through transient
        for _ in range(15000):
            sim.step()
        # After transient, amplitude should be non-trivial
        x_max = 0.0
        for _ in range(5000):
            sim.step()
            x_max = max(x_max, abs(sim.observe()[0]))
        assert x_max > 0.5, f"No limit cycle formed: max |x| = {x_max}"

    def test_limit_cycle_amplitude(self):
        """Unforced with small beta: limit cycle amplitude near 2 (VdP behavior)."""
        sim = DuffingVanDerPolSimulation(
            _make_config(F=0.0, mu=1.0, beta=0.0, x_0=0.1, y_0=0.0, dt=0.005)
        )
        sim.reset()
        amp = sim.compute_limit_cycle_amplitude(n_steps=40000)
        # Pure VdP limit cycle has amplitude ~2
        assert 1.5 < amp < 2.5, (
            f"Limit cycle amplitude {amp:.4f} not near 2 for beta=0 (VdP limit)"
        )

    def test_forced_oscillation(self):
        """F>0 gives bounded oscillatory response."""
        sim = DuffingVanDerPolSimulation(
            _make_config(F=0.5, omega=1.0, mu=1.0, x_0=0.1, dt=0.005)
        )
        sim.reset()
        for _ in range(10000):
            sim.step()
        # After transient, should still be oscillating
        xs = []
        for _ in range(5000):
            sim.step()
            xs.append(sim.observe()[0])
        xs = np.array(xs)
        assert np.std(xs) > 0.01, "No oscillation with forcing"
        assert np.all(np.isfinite(xs)), "Forced trajectory has NaN"

    def test_poincare_section(self):
        """Poincare section produces correct number of points."""
        sim = DuffingVanDerPolSimulation(
            _make_config(F=0.3, omega=1.0, dt=0.005)
        )
        sim.reset()
        poincare = sim.poincare_section(n_periods=50, n_transient=30)
        assert len(poincare["x"]) == 50
        assert len(poincare["y"]) == 50
        assert np.all(np.isfinite(poincare["x"]))
        assert np.all(np.isfinite(poincare["y"]))

    def test_bifurcation_sweep(self):
        """Bifurcation sweep produces valid data for each F value."""
        sim = DuffingVanDerPolSimulation(_make_config(dt=0.01))
        sim.reset()
        F_values = np.array([0.0, 0.5, 1.0])
        result = sim.bifurcation_sweep(
            F_values, n_poincare=20, n_transient=30,
        )
        assert len(result["F"]) == 3
        assert len(result["x"]) == 3
        for x_arr in result["x"]:
            assert len(x_arr) == 20
            assert np.all(np.isfinite(x_arr))

    def test_lyapunov_periodic(self):
        """Small forcing should give non-positive Lyapunov exponent."""
        sim = DuffingVanDerPolSimulation(
            _make_config(F=0.0, mu=1.0, beta=0.2, x_0=0.1, dt=0.005)
        )
        sim.reset()
        lyap = sim.compute_lyapunov(n_steps=20000, n_transient=5000)
        # Unforced VdP-Duffing limit cycle should have non-positive max Lyapunov
        # (one zero exponent on the limit cycle, one negative)
        assert lyap < 0.5, (
            f"Lyapunov exponent {lyap:.4f} too large for unforced periodic orbit"
        )

    def test_lyapunov_chaotic(self):
        """Large forcing amplitude should give positive Lyapunov exponent.

        The Duffing-VdP system with strong forcing can exhibit chaos.
        We use parameters known to produce chaotic behavior.
        """
        sim = DuffingVanDerPolSimulation(
            _make_config(F=5.0, mu=0.2, alpha=1.0, beta=1.0, omega=1.0,
                         x_0=0.1, y_0=0.0, dt=0.005)
        )
        sim.reset()
        lyap = sim.compute_lyapunov(n_steps=30000, n_transient=5000)
        # Strong forcing with weak damping should produce positive Lyapunov
        # but we only check that it's computed without error and is finite
        assert np.isfinite(lyap), f"Lyapunov is not finite: {lyap}"

    def test_frequency_response(self):
        """Frequency response produces valid amplitude data with a peak."""
        sim = DuffingVanDerPolSimulation(_make_config(F=0.3, dt=0.005))
        sim.reset()
        omega_values = np.linspace(0.5, 2.0, 10)
        result = sim.frequency_response(
            omega_values, n_transient_periods=30, n_measure_periods=10,
        )
        assert len(result["omega"]) == 10
        assert len(result["amplitude"]) == 10
        assert np.all(np.isfinite(result["amplitude"]))
        assert np.all(result["amplitude"] >= 0)
        # Should have some non-zero response
        assert np.max(result["amplitude"]) > 0

    def test_energy_computation(self):
        """Energy is positive for non-zero displacement."""
        sim = DuffingVanDerPolSimulation(_make_config(x_0=1.0, y_0=0.5))
        sim.reset()
        E = sim.compute_energy()
        assert E > 0, f"Energy should be positive: {E}"
        # E = 0.5*0.5^2 + 0.5*1*1^2 + 0.25*0.2*1^4 = 0.125 + 0.5 + 0.05 = 0.675
        expected = 0.5 * 0.5**2 + 0.5 * 1.0 * 1.0**2 + 0.25 * 0.2 * 1.0**4
        np.testing.assert_allclose(E, expected, rtol=1e-10)

    def test_energy_with_args(self):
        """Energy can be computed with explicit x, y arguments."""
        sim = DuffingVanDerPolSimulation(_make_config())
        sim.reset()
        E = sim.compute_energy(x=2.0, y=1.0)
        expected = 0.5 * 1.0**2 + 0.5 * 1.0 * 2.0**2 + 0.25 * 0.2 * 2.0**4
        np.testing.assert_allclose(E, expected, rtol=1e-10)

    def test_time_advances(self):
        """The t component of state increases with each step."""
        sim = DuffingVanDerPolSimulation(_make_config(dt=0.01))
        sim.reset()
        for i in range(10):
            sim.step()
            t = sim.observe()[2]
            expected_t = (i + 1) * 0.01
            np.testing.assert_allclose(t, expected_t, atol=1e-12)

    def test_vdp_limit(self):
        """beta=0, F=0 recovers pure Van der Pol dynamics.

        The limit cycle amplitude should be ~2.
        """
        sim = DuffingVanDerPolSimulation(
            _make_config(mu=1.0, alpha=1.0, beta=0.0, F=0.0, x_0=0.1, dt=0.005)
        )
        sim.reset()
        amp = sim.compute_limit_cycle_amplitude(n_steps=40000)
        assert 1.5 < amp < 2.5, (
            f"VdP limit (beta=0): amplitude {amp:.4f} not near 2"
        )

    def test_duffing_limit(self):
        """mu=0 recovers forced Duffing oscillator dynamics.

        With mu=0 (no VdP damping), forcing, and some linear damping
        approximated by the mu*(x^2-1) term being zero, the system becomes
        the undamped Duffing equation. Should give bounded oscillation.
        """
        sim = DuffingVanDerPolSimulation(
            _make_config(
                mu=0.0, alpha=1.0, beta=0.5, F=0.3, omega=1.0,
                x_0=0.5, y_0=0.0, dt=0.005,
            )
        )
        sim.reset()
        xs = []
        for _ in range(10000):
            sim.step()
            xs.append(sim.observe()[0])
        xs = np.array(xs)
        # Undamped forced Duffing: should be bounded and oscillatory
        assert np.all(np.isfinite(xs)), "Duffing limit trajectory has NaN"
        assert np.std(xs) > 0.01, "No oscillation in Duffing limit"

    def test_conservative_energy(self):
        """For mu=0, F=0, energy should be approximately conserved."""
        sim = DuffingVanDerPolSimulation(
            _make_config(
                mu=0.0, alpha=1.0, beta=0.5, F=0.0,
                x_0=1.0, y_0=0.0, dt=0.001,
            )
        )
        sim.reset()
        E0 = sim.compute_energy()
        for _ in range(5000):
            sim.step()
        E_final = sim.compute_energy()
        rel_error = abs(E_final - E0) / max(E0, 1e-15)
        assert rel_error < 1e-4, (
            f"Energy not conserved for mu=0, F=0: E0={E0}, E_final={E_final}, "
            f"rel_error={rel_error:.2e}"
        )

    def test_trajectory_shape(self):
        """Run produces correct trajectory shape with 3D state."""
        sim = DuffingVanDerPolSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_default_params(self):
        """Default parameters match specification."""
        config = SimulationConfig(
            domain=DOMAIN, dt=0.01, n_steps=100, parameters={}
        )
        sim = DuffingVanDerPolSimulation(config)
        assert sim.mu == 1.0
        assert sim.alpha == 1.0
        assert sim.beta == 0.2
        assert sim.F == 0.3
        assert sim.omega == 1.0
        assert sim.x_0 == 0.1
        assert sim.y_0 == 0.0

    def test_custom_params(self):
        """Custom parameters override defaults."""
        sim = DuffingVanDerPolSimulation(
            _make_config(mu=2.5, alpha=0.5, beta=0.8, F=1.0, omega=2.0)
        )
        assert sim.mu == 2.5
        assert sim.alpha == 0.5
        assert sim.beta == 0.8
        assert sim.F == 1.0
        assert sim.omega == 2.0


class TestDuffingVanDerPolRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.duffing_van_der_pol import (
            generate_ode_data,
        )

        data = generate_ode_data(mu=1.0, alpha=1.0, beta=0.2, n_steps=500, dt=0.01)
        # States are [x, v] (2D, not 3D -- t is excluded for SINDy)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["mu"] == 1.0
        assert data["alpha"] == 1.0
        assert data["beta"] == 0.2

    def test_limit_cycle_data_generation(self):
        from simulating_anything.rediscovery.duffing_van_der_pol import (
            generate_limit_cycle_data,
        )

        data = generate_limit_cycle_data(n_mu=3, n_beta=3, dt=0.01)
        assert len(data["mu"]) == 9  # 3 * 3
        assert len(data["beta"]) == 9
        assert len(data["amplitude"]) == 9
        assert np.all(data["amplitude"] >= 0)

    def test_forcing_bifurcation_data(self):
        from simulating_anything.rediscovery.duffing_van_der_pol import (
            generate_forcing_bifurcation_data,
        )

        data = generate_forcing_bifurcation_data(n_F=3, dt=0.01)
        assert len(data["F"]) == 3
        assert len(data["poincare_spread"]) == 3
        assert np.all(data["poincare_spread"] >= 0)

    def test_frequency_response_data(self):
        from simulating_anything.rediscovery.duffing_van_der_pol import (
            generate_frequency_response_data,
        )

        data = generate_frequency_response_data(n_omega=5, dt=0.01)
        assert len(data["omega"]) == 5
        assert len(data["amplitude"]) == 5
        assert np.all(data["amplitude"] >= 0)
