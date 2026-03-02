"""Double well potential (Kramers problem) simulation.

A particle in a symmetric double-well potential V(x) = x^4/4 - x^2/2
with damping and optional noise.

Minima at x = +/- 1, unstable equilibrium at x = 0.
Barrier height = V(0) - V(+/-1) = 0 - (-1/4) = 1/4.

Equations of motion:
    dx/dt = v
    dv/dt = -dV/dx - gamma*v + sigma*xi(t)
          = x - x^3 - gamma*v + sigma*xi(t)

Deterministic Hamiltonian (gamma=0, sigma=0):
    H = v^2/2 + x^4/4 - x^2/2

Target rediscoveries:
- ODE: dv/dt = x - x^3 - gamma*v  (via SINDy)
- Energy conservation (gamma=0, sigma=0)
- Barrier height = 1/4
- Small-oscillation frequency about minima: omega = sqrt(2)
- Equilibria: x = 0 (unstable), x = +/-1 (stable)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class DoubleWellSimulation(SimulationEnvironment):
    """Particle in a symmetric double-well potential.

    V(x) = x^4/4 - x^2/2, with damping and optional stochastic forcing.

    State vector: [x, v] where x = position, v = velocity (dx/dt).

    Parameters:
        gamma: damping coefficient (default 0.5)
        sigma: noise amplitude (default 0.0, deterministic)
        x_0: initial position (default 0.5)
        v_0: initial velocity (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.gamma = p.get("gamma", 0.5)
        self.sigma = p.get("sigma", 0.0)
        self.x_0 = p.get("x_0", 0.5)
        self.v_0 = p.get("v_0", 0.0)
        self._rng = np.random.default_rng(config.seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize position and velocity."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = np.array([self.x_0, self.v_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 (deterministic) or Euler-Maruyama (stochastic)."""
        if self.sigma > 0:
            self._euler_maruyama_step()
        else:
            self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, v]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta for the deterministic case."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _euler_maruyama_step(self) -> None:
        """Euler-Maruyama step for the stochastic case (sigma > 0)."""
        dt = self.config.dt
        y = self._state
        dydt = self._derivatives(y)
        noise = np.array([0.0, self.sigma * np.sqrt(dt) * self._rng.standard_normal()])
        self._state = y + dt * dydt + noise

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dy/dt for the double-well equations (deterministic part).

        Args:
            y: State vector [x, v].

        Returns:
            Derivatives [dx/dt, dv/dt].
        """
        x, v = y
        dx_dt = v
        # -dV/dx = -(x^3 - x) = x - x^3
        dv_dt = x - x**3 - self.gamma * v
        return np.array([dx_dt, dv_dt])

    @staticmethod
    def compute_potential(x: float | np.ndarray) -> float | np.ndarray:
        """Compute the double-well potential V(x) = x^4/4 - x^2/2.

        Args:
            x: Position(s) at which to evaluate the potential.

        Returns:
            Potential energy value(s).
        """
        return x**4 / 4.0 - x**2 / 2.0

    def compute_energy(self, state: np.ndarray | None = None) -> float:
        """Compute total mechanical energy H = v^2/2 + V(x).

        For the conservative system (gamma=0, sigma=0), this is a constant
        of motion.

        Args:
            state: State [x, v]. If None, uses the current state.

        Returns:
            Total energy.
        """
        if state is None:
            state = self._state
        x, v = state
        return float(0.5 * v**2 + self.compute_potential(x))

    @property
    def total_energy(self) -> float:
        """Current total mechanical energy."""
        return self.compute_energy()

    @property
    def kinetic_energy(self) -> float:
        """Current kinetic energy: KE = v^2/2."""
        return float(0.5 * self._state[1] ** 2)

    @property
    def potential_energy(self) -> float:
        """Current potential energy: V(x) = x^4/4 - x^2/2."""
        return float(self.compute_potential(self._state[0]))

    @staticmethod
    def find_equilibria() -> dict[str, dict]:
        """Find the equilibria of the double-well system.

        The potential V(x) = x^4/4 - x^2/2 has dV/dx = x^3 - x = 0
        at x = 0 (unstable, local max) and x = +/-1 (stable, local min).

        Returns:
            Dictionary of equilibria with stability information.
        """
        return {
            "x_minus": {"x": -1.0, "V": -0.25, "stable": True},
            "x_zero": {"x": 0.0, "V": 0.0, "stable": False},
            "x_plus": {"x": 1.0, "V": -0.25, "stable": True},
        }

    @staticmethod
    def barrier_height() -> float:
        """Compute the barrier height: V(0) - V(+/-1) = 0 - (-1/4) = 1/4.

        Returns:
            Barrier height (0.25).
        """
        return 0.25

    @staticmethod
    def well_frequency() -> float:
        """Small-oscillation angular frequency about either minimum.

        Near x = +/-1, V''(x) = 3x^2 - 1 = 2, so omega = sqrt(V''/m) = sqrt(2)
        for unit mass.

        Returns:
            Angular frequency sqrt(2).
        """
        return np.sqrt(2.0)

    def compute_oscillation_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period via zero crossings about a well minimum.

        The particle must be oscillating within one well (not crossing the
        barrier) for this measurement to be meaningful.

        Args:
            n_periods: Number of periods to average over.

        Returns:
            Mean period, or inf if no oscillation detected.
        """
        dt = self.config.dt
        T_theory = 2 * np.pi / self.well_frequency()

        # Skip initial transient
        transient_steps = max(int(20 / dt), int(10 * T_theory / dt))
        for _ in range(transient_steps):
            self.step()

        # Determine which well we are in
        x_eq = 1.0 if self._state[0] > 0 else -1.0

        # Detect upward crossings of (x - x_eq) through zero
        prev_dx = self._state[0] - x_eq
        crossings: list[float] = []
        measure_steps = int(n_periods * T_theory / dt * 3)
        for _ in range(measure_steps):
            self.step()
            dx = self._state[0] - x_eq
            if prev_dx < 0 and dx >= 0:
                frac = -prev_dx / (dx - prev_dx)
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_dx = dx

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def is_trapped_in_well(self) -> bool:
        """Check whether the particle has enough energy to cross the barrier.

        The particle is trapped in a well if its total energy E < V(0) = 0.

        Returns:
            True if trapped (cannot cross barrier), False otherwise.
        """
        return self.total_energy < 0.0

    def which_well(self) -> int:
        """Return which well the particle is currently in.

        Returns:
            -1 for left well (x < 0), +1 for right well (x > 0),
            0 if at the barrier top.
        """
        x = self._state[0]
        if x < 0:
            return -1
        elif x > 0:
            return 1
        return 0
