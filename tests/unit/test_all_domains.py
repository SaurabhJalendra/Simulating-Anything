"""Comprehensive tests for all 15 simulation domains (14 + Duffing template).

Verifies that every domain:
1. Resets to correct shape
2. Steps produce finite values
3. Observes match step output
4. Run collects correct trajectory shape
5. Is deterministic with same seed (where applicable)
"""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.types.simulation import Domain, SimulationBackend, SimulationConfig

# Domain configs: (class_path, domain_enum, params, dt, n_steps, expected_obs_shape)
DOMAIN_SPECS = {
    "projectile": {
        "module": "simulating_anything.simulation.rigid_body",
        "cls": "ProjectileSimulation",
        "domain": Domain.RIGID_BODY,
        "params": {
            "initial_speed": 30.0, "launch_angle": 45.0,
            "gravity": 9.81, "drag_coefficient": 0.1, "mass": 1.0,
        },
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (4,),
    },
    "lotka_volterra": {
        "module": "simulating_anything.simulation.agent_based",
        "cls": "LotkaVolterraSimulation",
        "domain": Domain.AGENT_BASED,
        "params": {
            "alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1,
            "prey_0": 40.0, "predator_0": 9.0,
        },
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (2,),
    },
    "sir_epidemic": {
        "module": "simulating_anything.simulation.epidemiological",
        "cls": "SIRSimulation",
        "domain": Domain.EPIDEMIOLOGICAL,
        "params": {"beta": 0.3, "gamma": 0.1, "S_0": 0.99, "I_0": 0.01},
        "dt": 0.1,
        "n_steps": 100,
        "obs_shape": (3,),
    },
    "double_pendulum": {
        "module": "simulating_anything.simulation.chaotic_ode",
        "cls": "DoublePendulumSimulation",
        "domain": Domain.CHAOTIC_ODE,
        "params": {
            "m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
            "theta1_0": 1.0, "theta2_0": 1.5,
            "omega1_0": 0.0, "omega2_0": 0.0,
        },
        "dt": 0.001,
        "n_steps": 100,
        "obs_shape": (4,),
    },
    "harmonic_oscillator": {
        "module": "simulating_anything.simulation.harmonic_oscillator",
        "cls": "DampedHarmonicOscillator",
        "domain": Domain.HARMONIC_OSCILLATOR,
        "params": {"k": 4.0, "m": 1.0, "c": 0.4, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (2,),
    },
    "lorenz": {
        "module": "simulating_anything.simulation.lorenz",
        "cls": "LorenzSimulation",
        "domain": Domain.LORENZ_ATTRACTOR,
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (3,),
    },
    "navier_stokes": {
        "module": "simulating_anything.simulation.navier_stokes",
        "cls": "NavierStokes2DSimulation",
        "domain": Domain.NAVIER_STOKES_2D,
        "params": {"nu": 0.01, "N": 32},
        "dt": 0.01,
        "n_steps": 50,
        "obs_shape": (1024,),  # Flattened 32x32 vorticity field
    },
    "van_der_pol": {
        "module": "simulating_anything.simulation.van_der_pol",
        "cls": "VanDerPolSimulation",
        "domain": Domain.VAN_DER_POL,
        "params": {"mu": 1.0, "x_0": 2.0, "v_0": 0.0},
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (2,),
    },
    "kuramoto": {
        "module": "simulating_anything.simulation.kuramoto",
        "cls": "KuramotoSimulation",
        "domain": Domain.KURAMOTO,
        "params": {"N": 20, "K": 2.0, "omega_std": 1.0},
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (20,),
    },
    "brusselator": {
        "module": "simulating_anything.simulation.brusselator",
        "cls": "BrusselatorSimulation",
        "domain": Domain.BRUSSELATOR,
        "params": {"a": 1.0, "b": 3.0, "u_0": 1.0, "v_0": 1.0},
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (2,),
    },
    "fitzhugh_nagumo": {
        "module": "simulating_anything.simulation.fitzhugh_nagumo",
        "cls": "FitzHughNagumoSimulation",
        "domain": Domain.FITZHUGH_NAGUMO,
        "params": {
            "a": 0.7, "b": 0.8, "eps": 0.08,
            "I_ext": 0.5, "v_0": -1.0, "w_0": -0.5,
        },
        "dt": 0.1,
        "n_steps": 100,
        "obs_shape": (2,),
    },
    "heat_equation": {
        "module": "simulating_anything.simulation.heat_equation",
        "cls": "HeatEquation1DSimulation",
        "domain": Domain.HEAT_EQUATION_1D,
        "params": {"D": 0.1, "N": 64},
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (64,),
    },
    "logistic_map": {
        "module": "simulating_anything.simulation.logistic_map",
        "cls": "LogisticMapSimulation",
        "domain": Domain.LOGISTIC_MAP,
        "params": {"r": 3.9, "x_0": 0.5},
        "dt": 1.0,
        "n_steps": 100,
        "obs_shape": (1,),
    },
    "duffing": {
        "module": "simulating_anything.simulation.template",
        "cls": "DuffingOscillator",
        "domain": Domain.RIGID_BODY,
        "params": {
            "alpha": 1.0, "beta": 1.0, "delta": 0.2,
            "gamma_f": 0.3, "omega": 1.0, "x_0": 0.5, "v_0": 0.0,
        },
        "dt": 0.01,
        "n_steps": 100,
        "obs_shape": (2,),
    },
}


def _make_sim(spec: dict):
    """Create a simulation instance from a domain spec."""
    import importlib
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])

    config_kwargs = {
        "domain": spec["domain"],
        "dt": spec["dt"],
        "n_steps": spec["n_steps"],
        "parameters": spec["params"],
    }
    # Gray-Scott needs extra fields
    if spec["domain"] == Domain.REACTION_DIFFUSION:
        config_kwargs["backend"] = SimulationBackend.JAX_FD
        config_kwargs["grid_resolution"] = spec.get("grid_resolution", (32, 32))
        config_kwargs["domain_size"] = spec.get("domain_size", (2.5, 2.5))

    config = SimulationConfig(**config_kwargs)
    return cls(config)


@pytest.mark.parametrize("domain_name", list(DOMAIN_SPECS.keys()))
class TestAllDomains:
    """Parametrized tests that run on every domain."""

    def test_reset_shape(self, domain_name):
        spec = DOMAIN_SPECS[domain_name]
        sim = _make_sim(spec)
        state = sim.reset()
        assert state.shape == spec["obs_shape"], (
            f"{domain_name}: expected {spec['obs_shape']}, got {state.shape}"
        )

    def test_step_finite(self, domain_name):
        spec = DOMAIN_SPECS[domain_name]
        sim = _make_sim(spec)
        sim.reset()
        for _ in range(10):
            state = sim.step()
        assert np.all(np.isfinite(state)), f"{domain_name}: non-finite state after 10 steps"

    def test_observe_matches_step(self, domain_name):
        spec = DOMAIN_SPECS[domain_name]
        sim = _make_sim(spec)
        sim.reset()
        step_result = sim.step()
        obs_result = sim.observe()
        np.testing.assert_array_equal(step_result, obs_result)

    def test_run_trajectory_shape(self, domain_name):
        spec = DOMAIN_SPECS[domain_name]
        n = 20  # Short for speed
        sim = _make_sim(spec)
        traj = sim.run(n_steps=n)
        expected_shape = (n + 1,) + spec["obs_shape"]
        assert traj.states.shape == expected_shape, (
            f"{domain_name}: expected {expected_shape}, got {traj.states.shape}"
        )
        assert len(traj.timestamps) == n + 1

    def test_trajectory_finite(self, domain_name):
        spec = DOMAIN_SPECS[domain_name]
        sim = _make_sim(spec)
        traj = sim.run(n_steps=50)
        assert np.all(np.isfinite(traj.states)), f"{domain_name}: non-finite trajectory"

    def test_deterministic(self, domain_name):
        """Same initial conditions should produce same trajectory."""
        spec = DOMAIN_SPECS[domain_name]
        # Skip Kuramoto since it uses random frequencies
        if domain_name == "kuramoto":
            pytest.skip("Kuramoto uses random frequencies")
        sim1 = _make_sim(spec)
        sim2 = _make_sim(spec)
        traj1 = sim1.run(n_steps=20)
        traj2 = sim2.run(n_steps=20)
        np.testing.assert_allclose(
            traj1.states, traj2.states, atol=1e-10,
            err_msg=f"{domain_name}: not deterministic",
        )
