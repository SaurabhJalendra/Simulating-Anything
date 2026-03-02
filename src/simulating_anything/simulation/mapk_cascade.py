"""MAPK signaling cascade simulation (Goldbeter-Koshland ultrasensitivity).

Models a three-tier enzymatic phosphorylation cascade (MAPKKK -> MAPKK -> MAPK)
using Goldbeter-Koshland kinetics. Each tier is a covalent modification cycle
where an upstream kinase activates the next tier via Michaelis-Menten phosphorylation,
and a constitutive phosphatase deactivates it.

ODE system (fraction of active kinase at each tier):
    dX1/dt = V1*(1-X1)/(K1+1-X1) - V2*X1/(K2+X1)
    dX2/dt = k3*X1*(1-X2)/(K3+1-X2) - V4*X2/(K4+X2)
    dX3/dt = k5*X2*(1-X3)/(K5+1-X3) - V6*X3/(K6+X3)

where Xi is the fraction of activated kinase at tier i (bounded in [0, 1]).

Key biology:
- Small Michaelis constants (K << 1) produce ultrasensitive (switch-like) responses
- The cascade amplifies signal: small changes in V1 produce large changes in X3
- Each tier acts as a Goldbeter-Koshland zero-order ultrasensitive switch

Target rediscoveries:
- Ultrasensitive dose-response (Hill coefficient n_H >> 1 for small K)
- Signal amplification ratio X3/X1
- Steady-state response curve X3_ss(V1)
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class MAPKCascadeSimulation(SimulationEnvironment):
    """Three-tier MAPK enzymatic cascade with Goldbeter-Koshland kinetics.

    State vector: [X1, X2, X3] where Xi is the fraction of activated kinase
    at tier i. All values are bounded in [0, 1].

    The cascade exhibits ultrasensitive switch-like behavior when the Michaelis
    constants Ki are small relative to the total enzyme concentration (which is
    normalized to 1 here). Each tier amplifies the input signal, producing
    a very steep dose-response curve at the output (X3).

    Parameters:
        V1: input signal strength / max activation rate of tier 1 (default 1.0)
        V2: deactivation rate of tier 1 (default 0.3)
        k3: max activation rate of tier 2 by X1 (default 1.0)
        V4: deactivation rate of tier 2 (default 0.3)
        k5: max activation rate of tier 3 by X2 (default 1.0)
        V6: deactivation rate of tier 3 (default 0.3)
        K1: Michaelis constant for tier 1 activation (default 0.01)
        K2: Michaelis constant for tier 1 deactivation (default 0.01)
        K3: Michaelis constant for tier 2 activation (default 0.01)
        K4: Michaelis constant for tier 2 deactivation (default 0.01)
        K5: Michaelis constant for tier 3 activation (default 0.01)
        K6: Michaelis constant for tier 3 deactivation (default 0.01)
        X1_0: initial X1 (default 0.0)
        X2_0: initial X2 (default 0.0)
        X3_0: initial X3 (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        # Kinetic rate parameters
        self.V1 = p.get("V1", 1.0)
        self.V2 = p.get("V2", 0.3)
        self.k3 = p.get("k3", 1.0)
        self.V4 = p.get("V4", 0.3)
        self.k5 = p.get("k5", 1.0)
        self.V6 = p.get("V6", 0.3)
        # Michaelis constants (small = ultrasensitive)
        self.K1 = p.get("K1", 0.01)
        self.K2 = p.get("K2", 0.01)
        self.K3 = p.get("K3", 0.01)
        self.K4 = p.get("K4", 0.01)
        self.K5 = p.get("K5", 0.01)
        self.K6 = p.get("K6", 0.01)
        # Initial conditions
        self.X1_0 = p.get("X1_0", 0.0)
        self.X2_0 = p.get("X2_0", 0.0)
        self.X3_0 = p.get("X3_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize kinase activation fractions to [X1_0, X2_0, X3_0]."""
        self._state = np.array(
            [self.X1_0, self.X2_0, self.X3_0], dtype=np.float64
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [X1, X2, X3]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Clamp to physical range [0, 1]
        self._state = np.clip(self._state, 0.0, 1.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dXi/dt for the three-tier cascade."""
        X1, X2, X3 = y

        # Tier 1: activated by external signal V1, deactivated by V2
        dX1 = (
            self.V1 * (1.0 - X1) / (self.K1 + 1.0 - X1)
            - self.V2 * X1 / (self.K2 + X1)
        )

        # Tier 2: activated by X1, deactivated by V4
        dX2 = (
            self.k3 * X1 * (1.0 - X2) / (self.K3 + 1.0 - X2)
            - self.V4 * X2 / (self.K4 + X2)
        )

        # Tier 3: activated by X2, deactivated by V6
        dX3 = (
            self.k5 * X2 * (1.0 - X3) / (self.K5 + 1.0 - X3)
            - self.V6 * X3 / (self.K6 + X3)
        )

        return np.array([dX1, dX2, dX3])

    def signal_amplification(self) -> float:
        """Compute signal amplification ratio X3/X1 at current state.

        Returns the ratio of output (X3) to first tier (X1) activation.
        Returns 0.0 if X1 is near zero.
        """
        if self._state is None:
            return 0.0
        X1 = self._state[0]
        X3 = self._state[2]
        if X1 < 1e-10:
            return 0.0
        return float(X3 / X1)

    def response_time(self, threshold: float = 0.5) -> float:
        """Measure time for X3 to reach a threshold fraction of its steady state.

        Simulates from current state until X3 exceeds the threshold, or returns
        inf if the threshold is not reached within 10 * n_steps.

        Args:
            threshold: X3 value to reach (default 0.5).

        Returns:
            Time in simulation units to reach threshold.
        """
        max_steps = 10 * self.config.n_steps
        for i in range(1, max_steps + 1):
            self.step()
            if self._state[2] >= threshold:
                return i * self.config.dt
        return float("inf")

    def is_switch_like(self, n_points: int = 50) -> bool:
        """Check if the dose-response is ultrasensitive (Hill n_H > 1).

        Sweeps V1 and measures steady-state X3 to estimate the effective
        Hill coefficient from the steepness of the transition.

        Args:
            n_points: Number of V1 values to test.

        Returns:
            True if the estimated Hill coefficient exceeds 1.
        """
        result = self.dose_response(n_points=n_points)
        n_H = result.get("hill_coefficient", 1.0)
        return n_H > 1.0

    def dose_response(
        self,
        V1_range: tuple[float, float] = (0.0, 2.0),
        n_points: int = 50,
        n_steps: int = 5000,
    ) -> dict[str, np.ndarray | float]:
        """Sweep V1 (input signal) and measure steady-state X3.

        Args:
            V1_range: (min, max) for the signal strength sweep.
            n_points: Number of V1 values.
            n_steps: Steps to reach steady state at each V1.

        Returns:
            Dictionary with V1_values, X3_ss, X1_ss, X2_ss,
            amplification, and hill_coefficient.
        """
        V1_values = np.linspace(V1_range[0], V1_range[1], n_points)
        X1_ss = np.zeros(n_points)
        X2_ss = np.zeros(n_points)
        X3_ss = np.zeros(n_points)

        for i, V1 in enumerate(V1_values):
            params = dict(self.config.parameters)
            params["V1"] = V1
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=self.config.n_steps,
                parameters=params,
            )
            sim = MAPKCascadeSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            state = sim.observe()
            X1_ss[i] = state[0]
            X2_ss[i] = state[1]
            X3_ss[i] = state[2]

        # Estimate Hill coefficient from dose-response steepness
        # Use log-log slope around EC50
        hill_coeff = self._estimate_hill_coefficient(V1_values, X3_ss)

        # Amplification ratio where X1 > threshold
        mask = X1_ss > 0.01
        if np.any(mask):
            amp = float(np.mean(X3_ss[mask] / np.maximum(X1_ss[mask], 1e-10)))
        else:
            amp = 0.0

        return {
            "V1_values": V1_values,
            "X1_ss": X1_ss,
            "X2_ss": X2_ss,
            "X3_ss": X3_ss,
            "amplification": amp,
            "hill_coefficient": hill_coeff,
        }

    def _estimate_hill_coefficient(
        self, V1: np.ndarray, X3: np.ndarray,
    ) -> float:
        """Estimate effective Hill coefficient from dose-response curve.

        Uses the ratio of EC90 to EC10 concentrations:
            n_H = log(81) / log(EC90/EC10)

        Returns 1.0 if estimation fails.
        """
        X3_max = np.max(X3)
        if X3_max < 0.01:
            return 1.0

        X3_norm = X3 / X3_max

        # Find EC10 and EC90
        ec10_idx = np.argmin(np.abs(X3_norm - 0.1))
        ec90_idx = np.argmin(np.abs(X3_norm - 0.9))

        ec10 = V1[ec10_idx]
        ec90 = V1[ec90_idx]

        if ec10 <= 0 or ec90 <= ec10:
            return 1.0

        ratio = ec90 / ec10
        if ratio <= 1.0:
            return 1.0

        # Hill coefficient: n_H = log(81) / log(EC90/EC10)
        n_H = np.log(81.0) / np.log(ratio)
        return float(np.clip(n_H, 0.1, 100.0))

    def steady_state(self, n_steps: int = 10000) -> np.ndarray:
        """Run to steady state and return final [X1, X2, X3].

        Args:
            n_steps: Number of integration steps.

        Returns:
            Steady-state activation fractions.
        """
        self.reset()
        for _ in range(n_steps):
            self.step()
        return self._state.copy()
