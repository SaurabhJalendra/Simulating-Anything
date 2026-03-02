"""Michaelis-Menten enzyme kinetics simulation.

Models the enzyme-substrate reaction scheme:
    E + S <-> ES -> E + P

Full ODE system (using conservation E = E_total - C):
    dS/dt  = -k1*(E_total - C)*S + k_1*C
    dC/dt  =  k1*(E_total - C)*S - (k_1 + k2)*C
    dP/dt  =  k2*C

where S = substrate, C = enzyme-substrate complex (ES), P = product,
E = free enzyme = E_total - C.

Target rediscoveries:
- Michaelis-Menten equation: v = V_max * S / (K_m + S)
- V_max = k2 * E_total
- K_m = (k_1 + k2) / k1
- Quasi-steady-state approximation: dC/dt ~ 0 after transient
- Conservation laws: E + C = E_total, S + C + P = S0
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MichaelisMentenSimulation(SimulationEnvironment):
    """Michaelis-Menten enzyme kinetics: E + S <-> ES -> E + P.

    State vector: [S, C, P] where S = substrate, C = ES complex, P = product.

    The free enzyme concentration is derived from conservation:
        E = E_total - C

    The system exhibits an initial transient where C builds up, followed by
    a quasi-steady-state regime where dC/dt ~ 0 and the reaction rate
    follows the Michaelis-Menten equation v = V_max*S/(K_m+S).

    Parameters:
        k1: forward binding rate constant (default 1.0)
        k_1: reverse unbinding rate constant (default 0.1)
        k2: catalytic rate constant (default 0.5)
        E_total: total enzyme concentration (default 1.0)
        S0: initial substrate concentration (default 10.0)
        C0: initial complex concentration (default 0.0)
        P0: initial product concentration (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.k1 = p.get("k1", 1.0)
        self.k_1 = p.get("k_1", 0.1)
        self.k2 = p.get("k2", 0.5)
        self.E_total = p.get("E_total", 1.0)
        self.S0 = p.get("S0", 10.0)
        self.C0 = p.get("C0", 0.0)
        self.P0 = p.get("P0", 0.0)

    @property
    def V_max(self) -> float:
        """Maximum reaction velocity: V_max = k2 * E_total."""
        return self.k2 * self.E_total

    @property
    def K_m(self) -> float:
        """Michaelis constant: K_m = (k_1 + k2) / k1."""
        return (self.k_1 + self.k2) / self.k1

    @property
    def free_enzyme(self) -> float:
        """Current free enzyme concentration E = E_total - C."""
        if self._state is None:
            return self.E_total
        return self.E_total - self._state[1]

    def steady_state_rate(self, S: float | None = None) -> float:
        """Michaelis-Menten steady-state rate: v = V_max * S / (K_m + S).

        Args:
            S: substrate concentration. If None, uses current state.

        Returns:
            Reaction rate under quasi-steady-state assumption.
        """
        if S is None:
            if self._state is None:
                S = self.S0
            else:
                S = float(self._state[0])
        if S <= 0:
            return 0.0
        return self.V_max * S / (self.K_m + S)

    def quasi_steady_state_check(self) -> float:
        """Check how well the quasi-steady-state approximation holds.

        Returns the absolute value of dC/dt, which should be near zero
        when the QSSA is valid.
        """
        if self._state is None:
            return float("inf")
        dy = self._derivatives(self._state)
        return float(abs(dy[1]))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize substrate, complex, and product concentrations."""
        self._state = np.array(
            [self.S0, self.C0, self.P0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [S, C, P]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce non-negativity (physical constraint)
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        S, C, P = y
        E = self.E_total - C

        # Binding/unbinding/catalysis rates
        binding = self.k1 * E * S
        unbinding = self.k_1 * C
        catalysis = self.k2 * C

        dS = -binding + unbinding
        dC = binding - unbinding - catalysis
        dP = catalysis
        return np.array([dS, dC, dP])

    def measure_initial_rate(
        self, n_steps: int = 10, transient_steps: int = 200,
    ) -> float:
        """Measure initial reaction rate dP/dt after pre-steady-state transient.

        The system starts with C=0, so the complex must first build up before
        the quasi-steady-state rate is reached. This method skips the
        pre-steady-state transient, then measures dP/dt while S is still
        approximately at S0.

        Args:
            n_steps: number of steps to average the rate over.
            transient_steps: steps to skip for complex build-up.

        Returns:
            Estimated initial rate of product formation.
        """
        self.reset()
        # Let complex build up to quasi-steady-state
        for _ in range(transient_steps):
            self.step()

        P_values = [self._state[2]]
        for _ in range(n_steps):
            self.step()
            P_values.append(self._state[2])

        P_arr = np.array(P_values)
        dt = self.config.dt
        # Use finite difference of P over the interval
        rates = np.diff(P_arr) / dt
        return float(np.mean(rates))

    def time_to_half_substrate(self) -> float:
        """Simulate until substrate drops to S0/2, return elapsed time.

        Returns inf if substrate never reaches half within 10*n_steps.
        """
        self.reset()
        half_S0 = self.S0 / 2.0
        max_steps = 10 * self.config.n_steps

        for i in range(1, max_steps + 1):
            self.step()
            if self._state[0] <= half_S0:
                return i * self.config.dt

        return float("inf")
