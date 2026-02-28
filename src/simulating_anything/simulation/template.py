"""Template for adding a new simulation domain.

Copy this file and implement the three abstract methods:
  1. reset() -- set initial conditions
  2. step() -- advance by one timestep
  3. observe() -- return current observable state

That's it. Everything else (world model, exploration, PySR, SINDy,
cross-domain analysis, reporting) works automatically on the output
of these three methods.

Example: To add a new ODE domain (e.g., Duffing oscillator):
  1. Copy this template
  2. Implement _derivatives() with your ODE's right-hand side
  3. Use the RK4 integrator in step()
  4. Register a new Domain enum value in types/simulation.py

Lines of code needed: ~50-100 for a simple ODE, ~100-200 for a PDE.
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TemplateSimulation(SimulationEnvironment):
    """Template simulation -- replace with your domain name.

    Implements: [describe the dynamical system]
    State: [describe the state vector]
    Parameters: [list the parameters]
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Extract parameters from config
        # self.param1 = p.get("param1", default_value)
        # self.param2 = p.get("param2", default_value)

        # Initial conditions
        # self.x_0 = p.get("x_0", 0.0)

        # Internal state
        self._state_vec = np.zeros(2)  # Adjust dimension

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to initial conditions."""
        self._step_count = 0
        # self._state_vec = np.array([self.x_0, self.v_0])
        self._state_vec = np.zeros(2)  # Set initial state
        return self._state_vec.copy()

    def step(self) -> np.ndarray:
        """Advance by one timestep using RK4."""
        dt = self.config.dt
        y = self._state_vec

        # RK4 integration
        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)
        self._state_vec = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        self._step_count += 1
        return self._state_vec.copy()

    def observe(self) -> np.ndarray:
        """Return current state."""
        return self._state_vec.copy()

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dy/dt for the dynamical system.

        Replace this with your system's equations of motion.
        """
        # Example: simple harmonic oscillator
        # x, v = y
        # dx_dt = v
        # dv_dt = -self.omega**2 * x
        # return np.array([dx_dt, dv_dt])
        return np.zeros_like(y)


# ============================================================================
# EXAMPLE: Duffing Oscillator (complete 54-line implementation)
# ============================================================================

class DuffingOscillator(SimulationEnvironment):
    """Duffing oscillator: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t).

    A nonlinear oscillator that exhibits chaos, bifurcations, and strange attractors.

    State: [x, v] (position, velocity)
    Parameters: alpha, beta, delta (damping), gamma (forcing), omega (forcing freq)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.alpha = p.get("alpha", 1.0)
        self.beta = p.get("beta", 1.0)
        self.delta = p.get("delta", 0.2)
        self.gamma_f = p.get("gamma_f", 0.3)
        self.omega = p.get("omega", 1.0)
        self.x_0 = p.get("x_0", 0.5)
        self.v_0 = p.get("v_0", 0.0)
        self._state_vec = np.array([self.x_0, self.v_0])
        self._t = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._step_count = 0
        self._t = 0.0
        self._state_vec = np.array([self.x_0, self.v_0])
        return self._state_vec.copy()

    def step(self) -> np.ndarray:
        dt = self.config.dt
        y = self._state_vec
        t = self._t

        # RK4 with time-dependent forcing
        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)
        self._state_vec = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        self._t += dt
        self._step_count += 1
        return self._state_vec.copy()

    def observe(self) -> np.ndarray:
        return self._state_vec.copy()

    def _derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        x, v = y
        dx_dt = v
        dv_dt = (self.gamma_f * np.cos(self.omega * t)
                 - self.delta * v
                 - self.alpha * x
                 - self.beta * x**3)
        return np.array([dx_dt, dv_dt])

    @property
    def total_energy(self) -> float:
        """Total energy (kinetic + potential)."""
        x, v = self._state_vec
        return 0.5 * v**2 + 0.5 * self.alpha * x**2 + 0.25 * self.beta * x**4
