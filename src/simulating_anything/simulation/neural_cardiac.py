"""Coupled neural-cardiac oscillator system.

Novel coupled system combining a FitzHugh-Nagumo neural oscillator with
a cardiac pacemaker (modified Van der Pol) through autonomic coupling.

Physical motivation: The brain and heart communicate bidirectionally
through the autonomic nervous system. Sympathetic/parasympathetic tone
modulates heart rate, while cardiac afferents (baroreceptor feedback)
influence neural activity. This heart-brain axis is implicated in
conditions from anxiety to cardiac arrhythmia, but the coupled dynamics
are not analytically characterized.

State variables (4D):
  v: neural membrane potential (FHN)
  w: neural recovery variable (FHN)
  x: cardiac membrane potential (modified VdP)
  y: cardiac recovery variable (modified VdP)

Coupling:
  Neural -> Cardiac: heart rate modulation via sympathetic tone
  Cardiac -> Neural: baroreceptor feedback proportional to cardiac amplitude

No known analytical solution for the coupled system. Targets:
- Phase locking ratio between neural and cardiac oscillations
- Coupling-dependent frequency entrainment
- Critical coupling for synchronization
- SINDy ODE recovery of the 4D coupled system
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NeuralCardiacSimulation(SimulationEnvironment):
    """Coupled neural (FHN) + cardiac (VdP) oscillator simulation.

    Neural subsystem (FitzHugh-Nagumo):
        dv/dt = v - v^3/3 - w + I_ext + coupling_cv * x
        dw/dt = eps_n * (v + a_n - b_n * w)

    Cardiac subsystem (modified Van der Pol):
        dx/dt = mu_c * (1 - x^2) * x - y + coupling_nc * v
        dy/dt = eps_c * x

    Args:
        config: SimulationConfig with parameters.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Neural (FHN) parameters
        self.a_n = p.get("a_n", 0.7)
        self.b_n = p.get("b_n", 0.8)
        self.eps_n = p.get("eps_n", 0.08)
        self.I_ext = p.get("I_ext", 0.5)

        # Cardiac (VdP) parameters
        self.mu_c = p.get("mu_c", 1.0)
        self.eps_c = p.get("eps_c", 0.3)

        # Coupling parameters
        self.coupling_nc = p.get("coupling_nc", 0.1)  # neural -> cardiac
        self.coupling_cv = p.get("coupling_cv", 0.05)  # cardiac -> neural

        # Initial conditions
        self.v_0 = p.get("v_0", -1.0)
        self.w_0 = p.get("w_0", -0.5)
        self.x_0 = p.get("x_0", 0.5)
        self.y_0 = p.get("y_0", 0.0)

        self.dt = config.dt
        self._state = np.array(
            [self.v_0, self.w_0, self.x_0, self.y_0], dtype=np.float64
        )
        self._t = 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        """Compute right-hand side of the coupled system."""
        v, w, x, y = state

        # Neural dynamics (FHN) with cardiac feedback
        dv = v - v**3 / 3.0 - w + self.I_ext + self.coupling_cv * x
        dw = self.eps_n * (v + self.a_n - self.b_n * w)

        # Cardiac dynamics (VdP) with neural drive
        dx = self.mu_c * (1.0 - x**2) * x - y + self.coupling_nc * v
        dy = self.eps_c * x

        return np.array([dv, dw, dx, dy])

    def _rk4_step(self, state: np.ndarray) -> np.ndarray:
        """Fourth-order Runge-Kutta integration step."""
        k1 = self._rhs(state)
        k2 = self._rhs(state + 0.5 * self.dt * k1)
        k3 = self._rhs(state + 0.5 * self.dt * k2)
        k4 = self._rhs(state + self.dt * k3)
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to initial conditions."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                self.v_0 + rng.normal(0, 0.1),
                self.w_0 + rng.normal(0, 0.05),
                self.x_0 + rng.normal(0, 0.1),
                self.y_0 + rng.normal(0, 0.05),
            ])
        else:
            self._state = np.array(
                [self.v_0, self.w_0, self.x_0, self.y_0], dtype=np.float64
            )
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._state = self._rk4_step(self._state)
        self._t += self.dt
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [v, w, x, y]."""
        return self._state.copy()

    @property
    def neural_state(self) -> np.ndarray:
        """Return neural FHN state [v, w]."""
        return self._state[:2].copy()

    @property
    def cardiac_state(self) -> np.ndarray:
        """Return cardiac VdP state [x, y]."""
        return self._state[2:4].copy()

    def estimate_phase_ratio(
        self, n_steps: int = 5000, skip_transient: int = 1000,
    ) -> float:
        """Estimate neural/cardiac frequency ratio from zero-crossings.

        Args:
            n_steps: Total simulation steps.
            skip_transient: Steps to skip before analysis.

        Returns:
            Ratio of neural frequency to cardiac frequency.
        """
        self.reset(seed=0)
        trajectory = []
        for _ in range(n_steps):
            trajectory.append(self.step().copy())
        traj = np.array(trajectory)
        traj = traj[skip_transient:]

        # Count zero crossings of v (neural) and x (cardiac)
        v = traj[:, 0]
        x = traj[:, 2]

        v_crossings = np.sum(np.diff(np.sign(v)) != 0) / 2
        x_crossings = np.sum(np.diff(np.sign(x)) != 0) / 2

        if x_crossings < 1:
            return float("inf")
        return float(v_crossings / x_crossings)
