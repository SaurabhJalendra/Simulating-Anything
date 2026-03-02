"""Semiconductor laser rate equations simulation -- carrier-photon dynamics.

Rate equations for a single-mode semiconductor laser:
    dN/dt = P - gamma_N*N - g*(N - N_tr)*S
    dS/dt = Gamma*g*(N - N_tr)*S - gamma_S*S + Gamma*beta*gamma_N*N

Parameters:
    P: pump current (injection rate, default 2.0)
    gamma_N: carrier decay rate (default 1.0, units 1/ns)
    gamma_S: photon decay rate (default 100.0, units 1/tau_p)
    g: differential gain (default 1000.0)
    N_tr: transparency carrier density (default 0.5)
    Gamma: confinement factor (default 0.3)
    beta: spontaneous emission factor (default 1e-4)

Target rediscoveries:
- Threshold pump: P_th = gamma_N*N_tr + gamma_S/(Gamma*g)
- Steady-state photon density: S_ss = Gamma*(P - P_th)/gamma_S
- Relaxation oscillation frequency: f_r ~ sqrt(gamma_S * g * S_ss)
- L-I curve (light output vs pump current)
- SINDy recovery of the two coupled ODEs
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LaserRateSimulation(SimulationEnvironment):
    """Semiconductor laser rate equations: carrier-photon coupled dynamics.

    State vector: [N, S] where N = carrier density, S = photon density.

    ODEs:
        dN/dt = P - gamma_N*N - g*(N - N_tr)*S
        dS/dt = Gamma*g*(N - N_tr)*S - gamma_S*S + Gamma*beta*gamma_N*N

    Above threshold (P > P_th), stimulated emission dominates and
    the laser output grows rapidly. The system exhibits relaxation
    oscillations around its steady state before settling.

    Parameters:
        P: pump current / injection rate
        gamma_N: carrier decay rate (non-radiative + spontaneous)
        gamma_S: photon decay rate (cavity loss, 1/tau_p)
        g: differential gain coefficient
        N_tr: transparency carrier density
        Gamma: optical confinement factor
        beta: spontaneous emission coupling factor
        N_0, S_0: initial carrier and photon densities
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.P = p.get("P", 2.0)
        self.gamma_N = p.get("gamma_N", 1.0)
        self.gamma_S = p.get("gamma_S", 100.0)
        self.g = p.get("g", 1000.0)
        self.N_tr = p.get("N_tr", 0.5)
        self.Gamma = p.get("Gamma", 0.3)
        self.beta = p.get("beta", 1e-4)
        self.N_0 = p.get("N_0", 0.5)
        self.S_0 = p.get("S_0", 0.01)

    @property
    def threshold_pump(self) -> float:
        """Lasing threshold pump current.

        P_th = gamma_N * N_tr + gamma_S / (Gamma * g)
        """
        return self.gamma_N * self.N_tr + self.gamma_S / (self.Gamma * self.g)

    @property
    def steady_state(self) -> tuple[float, float]:
        """Above-threshold steady state (N_ss, S_ss).

        N_ss = N_tr + gamma_S / (Gamma * g)
        S_ss = Gamma * (P - P_th) / gamma_S

        Returns (N_ss, S_ss). If below threshold, S_ss is clipped to 0.
        """
        N_ss = self.N_tr + self.gamma_S / (self.Gamma * self.g)
        P_th = self.threshold_pump
        S_ss = max(0.0, self.Gamma * (self.P - P_th) / self.gamma_S)
        return N_ss, S_ss

    @property
    def relaxation_frequency(self) -> float:
        """Relaxation oscillation angular frequency.

        omega_r = sqrt(gamma_S * g * S_ss)
        Returns 0 if below threshold (S_ss = 0).
        """
        _, S_ss = self.steady_state
        if S_ss <= 0:
            return 0.0
        return float(np.sqrt(self.gamma_S * self.g * S_ss))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize carrier and photon densities."""
        self._state = np.array([self.N_0, self.S_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        # Densities must be non-negative
        self._state = np.maximum(self._state, 0.0)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [N, S]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Laser rate equations."""
        N, S = state
        # Carrier equation
        dN = self.P - self.gamma_N * N - self.g * (N - self.N_tr) * S
        # Photon equation
        dS = (
            self.Gamma * self.g * (N - self.N_tr) * S
            - self.gamma_S * S
            + self.Gamma * self.beta * self.gamma_N * N
        )
        return np.array([dN, dS])

    def measure_relaxation_oscillations(
        self,
        n_transient: int = 2000,
        n_measure: int = 20000,
    ) -> float:
        """Measure relaxation oscillation frequency from photon density.

        Detects positive-going crossings of S through its mean value.
        Returns the angular frequency, or 0 if no oscillations detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect photon density values
        s_vals = []
        for _ in range(n_measure):
            state = self.step()
            s_vals.append(state[1])

        s_vals = np.array(s_vals)
        s_mean = np.mean(s_vals)

        # Detect positive-going crossings through the mean
        crossings: list[float] = []
        for i in range(1, len(s_vals)):
            if s_vals[i - 1] < s_mean and s_vals[i] >= s_mean:
                if s_vals[i] != s_vals[i - 1]:
                    frac = (s_mean - s_vals[i - 1]) / (
                        s_vals[i] - s_vals[i - 1]
                    )
                else:
                    frac = 0.5
                t_cross = (
                    self._step_count - n_measure + i - 1 + frac
                ) * dt
                crossings.append(t_cross)

        if len(crossings) < 2:
            return 0.0

        periods = np.diff(crossings)
        mean_period = float(np.mean(periods))
        if mean_period <= 0:
            return 0.0
        return 2.0 * np.pi / mean_period
