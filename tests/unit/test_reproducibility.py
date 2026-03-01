"""Tests for reproducibility verification across all 14 simulation domains.

Verifies determinism, finiteness, state dimensions, and domain-specific
physical invariants for every simulation in the project.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from simulating_anything.analysis.domain_statistics import DOMAIN_REGISTRY
from simulating_anything.types.simulation import Domain, SimulationBackend, SimulationConfig

# Gray-Scott is not in DOMAIN_REGISTRY but is one of the 14 core domains.
GRAY_SCOTT_SPEC = {
    "module": "simulating_anything.simulation.reaction_diffusion",
    "cls": "GrayScottSimulation",
    "domain": Domain.REACTION_DIFFUSION,
    "params": {"D_u": 0.16, "D_v": 0.08, "f": 0.035, "k": 0.065},
    "dt": 0.005,
    "n_steps": 100,
    "math_class": "PDE",
    "grid_resolution": (32, 32),
    "domain_size": (2.5, 2.5),
    "backend": SimulationBackend.JAX_FD,
}

# Combine into a full 14-domain registry for testing
FULL_REGISTRY = {**DOMAIN_REGISTRY, "gray_scott": GRAY_SCOTT_SPEC}

# Expected observation shapes for a single timestep
EXPECTED_OBS_SHAPES: dict[str, tuple[int, ...]] = {
    "projectile": (4,),
    "lotka_volterra": (2,),
    "gray_scott": (32, 32, 2),
    "sir_epidemic": (3,),
    "double_pendulum": (4,),
    "harmonic_oscillator": (2,),
    "lorenz": (3,),
    "navier_stokes": (1024,),  # Flattened 32x32 vorticity field
    "van_der_pol": (2,),
    "kuramoto": (20,),  # N=20 oscillator phases
    "brusselator": (2,),
    "fitzhugh_nagumo": (2,),
    "heat_equation": (64,),  # N=64 grid points
    "logistic_map": (1,),
}

# All domain names for parametrized tests
ALL_DOMAIN_NAMES = list(FULL_REGISTRY.keys())


def _make_config(spec: dict) -> SimulationConfig:
    """Build a SimulationConfig from a domain spec dictionary."""
    kwargs = {
        "domain": spec["domain"],
        "dt": spec["dt"],
        "n_steps": spec["n_steps"],
        "parameters": {k: float(v) for k, v in spec["params"].items()},
    }
    if spec["domain"] == Domain.REACTION_DIFFUSION:
        kwargs["backend"] = spec.get("backend", SimulationBackend.JAX_FD)
        kwargs["grid_resolution"] = spec.get("grid_resolution", (32, 32))
        kwargs["domain_size"] = spec.get("domain_size", (2.5, 2.5))
    return SimulationConfig(**kwargs)


def _make_sim(spec: dict):
    """Instantiate a simulation from a domain spec."""
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])
    config = _make_config(spec)
    return cls(config)


def _run_domain(domain_name: str) -> np.ndarray:
    """Run a domain simulation and return the full trajectory states."""
    spec = FULL_REGISTRY[domain_name]
    sim = _make_sim(spec)
    traj = sim.run(n_steps=spec["n_steps"])
    return traj.states


# ---------------------------------------------------------------------------
# Parametrized tests over all 14 domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain_name", ALL_DOMAIN_NAMES)
class TestDomainDeterminism:
    """Verify that all domains produce identical output when run twice."""

    def test_deterministic(self, domain_name: str) -> None:
        """Two runs with the same seed must produce bit-identical trajectories."""
        spec = FULL_REGISTRY[domain_name]
        sim1 = _make_sim(spec)
        sim2 = _make_sim(spec)

        traj1 = sim1.run(n_steps=spec["n_steps"])
        traj2 = sim2.run(n_steps=spec["n_steps"])

        np.testing.assert_array_equal(
            traj1.states,
            traj2.states,
            err_msg=f"{domain_name}: trajectories differ between identical runs",
        )


@pytest.mark.parametrize("domain_name", ALL_DOMAIN_NAMES)
class TestDomainFiniteness:
    """Verify that all domains produce finite (non-NaN, non-Inf) values."""

    def test_all_finite(self, domain_name: str) -> None:
        """Every state value in the trajectory must be finite."""
        states = _run_domain(domain_name)
        assert np.all(np.isfinite(states)), (
            f"{domain_name}: trajectory contains NaN or Inf values"
        )


@pytest.mark.parametrize("domain_name", ALL_DOMAIN_NAMES)
class TestDomainStateShape:
    """Verify that each domain produces states with the expected dimensions."""

    def test_obs_shape_matches_expected(self, domain_name: str) -> None:
        """Single-step observation shape must match the expected shape."""
        expected = EXPECTED_OBS_SHAPES.get(domain_name)
        if expected is None:
            pytest.skip(f"No expected shape defined for {domain_name}")

        spec = FULL_REGISTRY[domain_name]
        sim = _make_sim(spec)
        state = sim.reset()
        assert state.shape == expected, (
            f"{domain_name}: expected shape {expected}, got {state.shape}"
        )

    def test_trajectory_shape(self, domain_name: str) -> None:
        """Trajectory states array must be (n_steps+1, *obs_shape)."""
        expected = EXPECTED_OBS_SHAPES.get(domain_name)
        if expected is None:
            pytest.skip(f"No expected shape defined for {domain_name}")

        spec = FULL_REGISTRY[domain_name]
        n_steps = min(spec["n_steps"], 50)  # Short run for speed
        sim = _make_sim(spec)
        traj = sim.run(n_steps=n_steps)

        full_expected = (n_steps + 1,) + expected
        assert traj.states.shape == full_expected, (
            f"{domain_name}: expected trajectory shape {full_expected}, "
            f"got {traj.states.shape}"
        )


# ---------------------------------------------------------------------------
# Domain-specific invariant tests
# ---------------------------------------------------------------------------


class TestSIRConservation:
    """Verify SIR model conserves total population S + I + R = 1."""

    def _make_sir(
        self, beta: float = 0.3, gamma: float = 0.1, n_steps: int = 500,
    ):
        """Create an SIR simulation with given parameters."""
        spec = {
            "module": "simulating_anything.simulation.epidemiological",
            "cls": "SIRSimulation",
            "domain": Domain.EPIDEMIOLOGICAL,
            "params": {"beta": beta, "gamma": gamma, "S_0": 0.99, "I_0": 0.01},
            "dt": 0.1,
            "n_steps": n_steps,
        }
        return _make_sim(spec), spec

    def test_conservation_sum_equals_one(self) -> None:
        """S + I + R must equal 1.0 at every timestep."""
        sim, spec = self._make_sir()
        traj = sim.run(n_steps=spec["n_steps"])
        sums = traj.states.sum(axis=1)
        np.testing.assert_allclose(
            sums, 1.0, atol=1e-8,
            err_msg="SIR compartments do not sum to 1.0",
        )

    def test_non_negative_compartments(self) -> None:
        """All compartments must remain non-negative."""
        sim, spec = self._make_sir()
        traj = sim.run(n_steps=spec["n_steps"])
        assert np.all(traj.states >= -1e-12), (
            "SIR compartments went negative"
        )

    def test_conservation_high_r0(self) -> None:
        """Conservation holds even for large R0 (fast epidemic)."""
        sim, spec = self._make_sir(beta=2.0, gamma=0.1, n_steps=200)
        traj = sim.run(n_steps=spec["n_steps"])
        sums = traj.states.sum(axis=1)
        np.testing.assert_allclose(
            sums, 1.0, atol=1e-8,
            err_msg="SIR conservation violated at high R0",
        )


class TestPendulumEnergyConservation:
    """Verify double pendulum conserves total mechanical energy."""

    def _make_pendulum(self, n_steps: int = 500, dt: float = 0.001, **kwargs):
        """Create a double pendulum simulation."""
        params = {
            "m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
            "theta1_0": 1.0, "theta2_0": 1.5,
            "omega1_0": 0.0, "omega2_0": 0.0,
        }
        params.update(kwargs)
        spec = {
            "module": "simulating_anything.simulation.chaotic_ode",
            "cls": "DoublePendulumSimulation",
            "domain": Domain.CHAOTIC_ODE,
            "params": params,
            "dt": dt,
            "n_steps": n_steps,
        }
        return _make_sim(spec), spec

    def test_energy_drift_within_rk4_bounds(self) -> None:
        """Energy drift should be very small over 500 steps with dt=0.001."""
        from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation

        sim, spec = self._make_pendulum(n_steps=500, dt=0.001)
        traj = sim.run(n_steps=spec["n_steps"])

        # Compute energy at start and end
        config = _make_config(spec)
        energy_sim = DoublePendulumSimulation(config)
        energy_sim.reset()
        E0 = energy_sim.total_energy(traj.states[0])
        E_final = energy_sim.total_energy(traj.states[-1])

        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, (
            f"Energy drift {rel_drift:.2e} exceeds RK4 tolerance for dt=0.001"
        )

    def test_energy_finite_throughout(self) -> None:
        """Energy must remain finite at every step."""
        from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation

        sim, spec = self._make_pendulum(n_steps=200)
        traj = sim.run(n_steps=spec["n_steps"])

        config = _make_config(spec)
        energy_sim = DoublePendulumSimulation(config)
        energy_sim.reset()

        for i in range(traj.states.shape[0]):
            E = energy_sim.total_energy(traj.states[i])
            assert np.isfinite(E), f"Energy is not finite at step {i}"


class TestHarmonicOscillatorEnergyDecay:
    """Verify damped harmonic oscillator energy decreases with damping."""

    def _make_oscillator(self, c: float = 0.4, n_steps: int = 500):
        """Create a damped harmonic oscillator simulation."""
        spec = {
            "module": "simulating_anything.simulation.harmonic_oscillator",
            "cls": "DampedHarmonicOscillator",
            "domain": Domain.HARMONIC_OSCILLATOR,
            "params": {"k": 4.0, "m": 1.0, "c": c, "x_0": 2.0, "v_0": 0.0},
            "dt": 0.01,
            "n_steps": n_steps,
        }
        return _make_sim(spec), spec

    def test_energy_decreases_with_damping(self) -> None:
        """With c > 0, total energy E = 0.5*k*x^2 + 0.5*m*v^2 must decrease."""
        sim, spec = self._make_oscillator(c=0.4)
        traj = sim.run(n_steps=spec["n_steps"])

        k = spec["params"]["k"]
        m = spec["params"]["m"]
        x_vals = traj.states[:, 0]
        v_vals = traj.states[:, 1]
        energies = 0.5 * k * x_vals**2 + 0.5 * m * v_vals**2

        # Final energy should be less than initial
        assert energies[-1] < energies[0], (
            f"Energy did not decrease: E_0={energies[0]:.4f}, E_final={energies[-1]:.4f}"
        )

        # Energy should never increase (within numerical tolerance)
        diffs = np.diff(energies)
        max_increase = float(np.max(diffs))
        assert max_increase < 1e-8, (
            f"Energy increased by {max_increase:.2e} between consecutive steps"
        )

    def test_undamped_energy_conserved(self) -> None:
        """With c = 0, energy should be conserved."""
        sim, spec = self._make_oscillator(c=0.0, n_steps=1000)
        traj = sim.run(n_steps=spec["n_steps"])

        k = spec["params"]["k"]
        m = spec["params"]["m"]
        x_vals = traj.states[:, 0]
        v_vals = traj.states[:, 1]
        energies = 0.5 * k * x_vals**2 + 0.5 * m * v_vals**2

        rel_drift = abs(energies[-1] - energies[0]) / energies[0]
        assert rel_drift < 1e-8, (
            f"Undamped oscillator energy drift: {rel_drift:.2e}"
        )


class TestLogisticMapRange:
    """Verify logistic map stays in [0, 1] for valid parameters."""

    def test_range_r_3_9(self) -> None:
        """Logistic map with r=3.9 (chaotic) must stay in [0, 1]."""
        states = _run_domain("logistic_map")
        assert np.all(states >= 0.0), "Logistic map produced negative values"
        assert np.all(states <= 1.0), "Logistic map exceeded 1.0"

    def test_range_r_4(self) -> None:
        """Logistic map with r=4.0 (fully chaotic) stays in [0, 1]."""
        spec = {
            "module": "simulating_anything.simulation.logistic_map",
            "cls": "LogisticMapSimulation",
            "domain": Domain.LOGISTIC_MAP,
            "params": {"r": 4.0, "x_0": 0.5},
            "dt": 1.0,
            "n_steps": 1000,
        }
        sim = _make_sim(spec)
        traj = sim.run(n_steps=spec["n_steps"])
        assert np.all(traj.states >= 0.0), "r=4.0 produced negative values"
        assert np.all(traj.states <= 1.0 + 1e-12), "r=4.0 exceeded 1.0"


class TestLotkaVolterraPositivity:
    """Verify Lotka-Volterra populations remain non-negative."""

    def test_populations_non_negative(self) -> None:
        """Both prey and predator populations must stay >= 0."""
        states = _run_domain("lotka_volterra")
        assert np.all(states >= 0.0), (
            f"Negative populations found: min={np.min(states):.6f}"
        )


class TestHeatEquationConservation:
    """Verify heat equation conserves total heat with periodic boundary conditions."""

    def test_total_heat_conserved(self) -> None:
        """Sum of temperature across grid points should remain constant."""
        states = _run_domain("heat_equation")
        total_heat = states.sum(axis=1)
        max_drift = float(np.max(np.abs(total_heat - total_heat[0])))
        assert max_drift < 1e-10, (
            f"Total heat drifted by {max_drift:.2e} (spectral solver should be exact)"
        )

    def test_peak_temperature_decays(self) -> None:
        """Maximum temperature should decrease due to diffusion."""
        states = _run_domain("heat_equation")
        max_temps = np.max(states, axis=1)
        # After initial step, peak should decrease monotonically
        diffs = np.diff(max_temps[1:])
        max_increase = float(np.max(diffs)) if len(diffs) > 0 else 0.0
        assert max_increase < 1e-10, (
            f"Peak temperature increased by {max_increase:.2e}"
        )


class TestDomainStatisticsIntegration:
    """Verify domain_statistics.compute_all_stats() runs without error."""

    def test_compute_all_stats_returns_results(self) -> None:
        """compute_all_stats() should return stats for all non-skipped domains."""
        from simulating_anything.analysis.domain_statistics import compute_all_stats

        stats = compute_all_stats(skip_kuramoto=True)
        # Should have 12 results (13 in registry minus Kuramoto)
        assert len(stats) >= 10, (
            f"Expected at least 10 domain stats, got {len(stats)}"
        )

    def test_all_stats_are_finite(self) -> None:
        """All computed statistics must contain finite values."""
        from simulating_anything.analysis.domain_statistics import compute_all_stats

        stats = compute_all_stats(skip_kuramoto=True)
        for s in stats:
            assert np.isfinite(s.state_mean), f"{s.name}: non-finite mean"
            assert np.isfinite(s.state_std), f"{s.name}: non-finite std"
            assert np.isfinite(s.state_min), f"{s.name}: non-finite min"
            assert np.isfinite(s.state_max), f"{s.name}: non-finite max"
            assert s.is_finite, f"{s.name}: trajectory is not finite"

    def test_all_stats_deterministic(self) -> None:
        """Domains should be flagged as deterministic in stats."""
        from simulating_anything.analysis.domain_statistics import compute_all_stats

        stats = compute_all_stats(skip_kuramoto=True)
        for s in stats:
            assert s.is_deterministic, f"{s.name}: not deterministic"
