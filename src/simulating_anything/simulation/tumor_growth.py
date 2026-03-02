"""Tumor-immune interaction simulation (Kuznetsov-Taylor model).

Models the competition between tumor cells and immune effector cells:

    dT/dt = r * T * (1 - T/K) - a * T * I
    dI/dt = s + p * T * I / (g + T) - d * I - a_I * T * I

where:
    T   = tumor cell population
    I   = immune effector cell population (e.g., cytotoxic T-lymphocytes)
    r   = tumor intrinsic growth rate
    K   = tumor carrying capacity
    a   = immune killing rate (immune attack on tumor)
    s   = constant immune cell source rate
    p   = immune stimulation rate (tumor-dependent proliferation)
    g   = half-saturation constant for immune stimulation (Michaelis-Menten)
    d   = natural immune cell death rate
    a_I = tumor-induced immune inactivation rate

Three qualitative outcomes:
- Tumor elimination: immune system clears tumor (T -> 0)
- Tumor dormancy: stable coexistence at intermediate T and I
- Tumor escape: tumor grows to carrying capacity, immune response overwhelmed

Target rediscoveries:
- Elimination vs escape threshold in immune killing rate a
- Dormancy equilibrium conditions
- Saddle-node bifurcation in immune parameters
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class TumorGrowthSimulation(SimulationEnvironment):
    """Tumor-immune interaction (Kuznetsov-Taylor model).

    State vector: [T, I] where T = tumor cells, I = immune effector cells.

    Equations:
        dT/dt = r * T * (1 - T/K) - a * T * I
        dI/dt = s + p * T * I / (g + T) - d * I - a_I * T * I

    The system can exhibit tumor elimination, dormancy (stable coexistence),
    or tumor escape depending on immune parameters.

    Parameters:
        r:   tumor growth rate (default 0.5)
        K:   carrying capacity (default 1e6)
        a:   immune killing rate (default 1e-7)
        s:   immune source rate (default 1e4)
        p:   immune stimulation rate (default 0.1)
        g:   half-saturation constant (default 1e5)
        d:   immune death rate (default 0.03)
        a_I: immune inactivation rate by tumor (default 1e-7)
        T_0: initial tumor population
        I_0: initial immune population
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 0.5)
        self.K = p.get("K", 1e6)
        self.a = p.get("a", 1e-7)
        self.s = p.get("s", 1e4)
        self.p_stim = p.get("p", 0.1)
        self.g = p.get("g", 1e5)
        self.d = p.get("d", 0.03)
        self.a_I = p.get("a_I", 1e-7)
        self.T_0 = p.get("T_0", 1e4)
        self.I_0 = p.get("I_0", 1e5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize tumor and immune populations."""
        self._state = np.array([self.T_0, self.I_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with non-negativity clamp."""
        self._rk4_step()
        # Clamp to non-negative values (populations cannot be negative)
        self._state = np.maximum(self._state, 0.0)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [T, I]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute right-hand side of the tumor-immune ODE system.

        dT/dt = r * T * (1 - T/K) - a * T * I
        dI/dt = s + p * T * I / (g + T) - d * I - a_I * T * I
        """
        T, Im = y  # Im = immune cells (avoid E741 ambiguous name 'I')

        # Tumor dynamics: logistic growth - immune killing
        dT = self.r * T * (1.0 - T / self.K) - self.a * T * Im

        # Immune dynamics: source + stimulation - death - inactivation
        dI = (
            self.s
            + self.p_stim * T * Im / (self.g + T)
            - self.d * Im
            - self.a_I * T * Im
        )

        return np.array([dT, dI])

    @property
    def tumor_free_equilibrium(self) -> tuple[float, float]:
        """Tumor-free equilibrium: T = 0, I = s/d."""
        return (0.0, self.s / self.d)

    def coexistence_equilibria(self) -> list[tuple[float, float]]:
        """Find coexistence equilibria numerically.

        At equilibrium:
            r * (1 - T/K) = a * I             ... (from dT/dt = 0, T > 0)
            I = s / (d + a_I * T - p * T / (g + T))  ... (from dI/dt = 0)

        Substituting the first into the second and solving for T.
        We scan over T to find roots.
        """
        equilibria = []
        n_scan = 2000
        T_values = np.linspace(1.0, self.K * 0.99, n_scan)

        # From dT/dt = 0 (T > 0): I = r * (1 - T/K) / a
        # From dI/dt = 0: s + p*T*I/(g+T) - d*I - a_I*T*I = 0
        # Substitute I from first into second:
        prev_residual = None
        for i, T in enumerate(T_values):
            I_from_T = self.r * (1.0 - T / self.K) / self.a
            if I_from_T < 0:
                break

            residual = (
                self.s
                + self.p_stim * T * I_from_T / (self.g + T)
                - self.d * I_from_T
                - self.a_I * T * I_from_T
            )

            if prev_residual is not None and prev_residual * residual < 0:
                # Sign change detected -- interpolate
                T_prev = T_values[i - 1]
                # Linear interpolation for root
                frac = abs(prev_residual) / (abs(prev_residual) + abs(residual))
                T_root = T_prev + frac * (T - T_prev)
                I_root = self.r * (1.0 - T_root / self.K) / self.a
                if T_root > 0 and I_root > 0:
                    equilibria.append((float(T_root), float(I_root)))

            prev_residual = residual

        return equilibria

    def classify_outcome(
        self, n_steps: int = 5000, threshold_low: float = 10.0
    ) -> str:
        """Classify the long-term outcome of the tumor-immune dynamics.

        Returns one of:
            'elimination' -- tumor drops below threshold_low
            'dormancy'    -- tumor stabilizes at intermediate level
            'escape'      -- tumor approaches carrying capacity
        """
        self.reset()
        for _ in range(n_steps):
            self.step()

        T_final = self._state[0]

        if T_final < threshold_low:
            return "elimination"
        elif T_final > 0.5 * self.K:
            return "escape"
        else:
            return "dormancy"

    def measure_tumor_doubling_time(self) -> float:
        """Measure the effective tumor doubling time from initial growth phase.

        Returns the time for T to double from T_0, or inf if it does not.
        """
        self.reset()
        T_target = 2.0 * self.T_0
        dt = self.config.dt

        for step in range(50000):
            self.step()
            if self._state[0] >= T_target:
                return (step + 1) * dt
            if self._state[0] < self.T_0 * 0.1:
                # Tumor is shrinking, doubling won't happen
                return float("inf")

        return float("inf")

    def immune_strength_ratio(self) -> float:
        """Ratio of immune killing to tumor growth: a * I_0 / r.

        Higher values favor tumor elimination.
        """
        return self.a * self.I_0 / self.r
