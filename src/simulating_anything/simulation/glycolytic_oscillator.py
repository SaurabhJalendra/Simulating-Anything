"""Glycolytic oscillator simulation (Higgins two-variable model).

The Higgins model describes glycolytic oscillations in the phosphofructokinase
(PFK) reaction, where the product fructose-1,6-bisphosphate (P) allosterically
activates the enzyme through an autocatalytic feedback loop:

    dS/dt = v_in - k1 * S * P^2
    dP/dt = k1 * S * P^2 - k2 * P

where:
    S = substrate (glucose-6-phosphate)
    P = product (fructose-1,6-bisphosphate)
    v_in = constant substrate input flux
    k1  = autocatalytic rate constant (PFK activation by P^2)
    k2  = product removal rate

The system has a unique steady state at:
    P* = v_in / k2
    S* = k2^2 / (k1 * v_in)

The Jacobian trace at the fixed point is:
    tr(J) = k2 - k1 * v_in^2 / k2^2

Hopf bifurcation occurs at v_in_c = k2 * sqrt(k2/k1). For v_in < v_in_c the
fixed point is an unstable spiral and the trajectory oscillates with slowly
growing amplitude. For v_in > v_in_c the fixed point is a stable spiral and
transient oscillations decay.

Note: the pure Higgins model does not have a globally attracting limit cycle.
Near the Hopf threshold the growing spiral oscillates for many periods,
making the oscillatory dynamics observable and the Hopf boundary detectable.

Target rediscoveries:
- ODE recovery via SINDy (dS/dt, dP/dt terms including S*P^2 autocatalysis)
- Hopf bifurcation threshold v_in_c = k2 * sqrt(k2/k1)
- Oscillation frequency near the threshold
- Fixed point verification: P* = v_in/k2, S* = k2^2/(k1*v_in)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GlycolyticOscillatorSimulation(SimulationEnvironment):
    """Higgins two-variable glycolytic oscillator.

    State vector: [S, P] where S = substrate, P = product.

    The autocatalytic term S*P^2 models allosteric activation of
    phosphofructokinase by its own product. This positive feedback
    drives oscillatory dynamics near the Hopf bifurcation threshold.

    Default parameters are chosen near the Hopf threshold so that the
    unstable spiral produces many visible oscillation cycles.

    Parameters:
        v_in: substrate input flux (default 1.0)
        k1: autocatalytic rate constant (default 0.1)
        k2: product removal rate (default 0.5)
        S_0: initial substrate concentration (default 2.5)
        P_0: initial product concentration (default 2.1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.v_in = p.get("v_in", 1.0)
        self.k1 = p.get("k1", 0.1)
        self.k2 = p.get("k2", 0.5)
        self.S_0 = p.get("S_0", 2.5)
        self.P_0 = p.get("P_0", 2.1)

    @property
    def fixed_point(self) -> tuple[float, float]:
        """The unique steady state (S*, P*).

        Setting dS/dt = 0: v_in = k1 * S * P^2
        Setting dP/dt = 0: k1 * S * P^2 = k2 * P  =>  P* = v_in / k2
        Substituting back: S* = k2^2 / (k1 * v_in)
        """
        P_star = self.v_in / self.k2
        S_star = self.k2**2 / (self.k1 * self.v_in)
        return (S_star, P_star)

    @property
    def jacobian_trace(self) -> float:
        """Trace of the Jacobian at the fixed point.

        J = [[-k1*P*^2,         -2*k1*S**P*],
             [ k1*P*^2,  2*k1*S**P* - k2   ]]

        tr(J) = -k1*P*^2 + 2*k1*S**P* - k2
              = k2 - k1*v_in^2/k2^2
        """
        S_star, P_star = self.fixed_point
        return -self.k1 * P_star**2 + 2 * self.k1 * S_star * P_star - self.k2

    @property
    def jacobian_det(self) -> float:
        """Determinant of the Jacobian at the fixed point.

        det(J) = k1 * P*^2 * k2  (always positive for positive parameters)
        """
        _, P_star = self.fixed_point
        return self.k1 * P_star**2 * self.k2

    @property
    def eigenvalues(self) -> tuple[complex, complex]:
        """Eigenvalues of the Jacobian at the fixed point."""
        tr = self.jacobian_trace
        det = self.jacobian_det
        disc = tr**2 - 4 * det
        if disc < 0:
            real = tr / 2
            imag = np.sqrt(-disc) / 2
            return (complex(real, imag), complex(real, -imag))
        sqrt_disc = np.sqrt(disc)
        return (complex((tr + sqrt_disc) / 2), complex((tr - sqrt_disc) / 2))

    @property
    def oscillation_frequency(self) -> float:
        """Angular frequency of oscillation at the fixed point.

        Returns omega = Im(lambda) if the eigenvalues are complex, else 0.
        """
        lam = self.eigenvalues[0]
        return abs(lam.imag)

    @property
    def is_unstable_spiral(self) -> bool:
        """True if the fixed point is an unstable spiral (tr > 0, disc < 0).

        When this is true, the system oscillates with growing amplitude
        near the fixed point.
        """
        tr = self.jacobian_trace
        det = self.jacobian_det
        disc = tr**2 - 4 * det
        return tr > 0 and disc < 0

    @property
    def is_stable_spiral(self) -> bool:
        """True if the fixed point is a stable spiral (tr < 0, disc < 0)."""
        tr = self.jacobian_trace
        det = self.jacobian_det
        disc = tr**2 - 4 * det
        return tr < 0 and disc < 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize concentrations [S, P]."""
        self._state = np.array([self.S_0, self.P_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [S, P]."""
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
        """Compute dS/dt, dP/dt for the Higgins glycolytic model."""
        S, P = y
        autocatalytic = self.k1 * S * P**2
        dS = self.v_in - autocatalytic
        dP = autocatalytic - self.k2 * P
        return np.array([dS, dP])

    def measure_oscillation_period(self, n_cycles: int = 5) -> float:
        """Measure oscillation period from early trajectory near fixed point.

        Uses zero crossings of P - P* during the growing-oscillation phase.
        Returns mean period in time units, or inf if not oscillatory.
        """
        if not self.is_unstable_spiral:
            return float("inf")

        dt = self.config.dt
        P_star = self.fixed_point[1]

        # Start near fixed point with small perturbation
        S_star = self.fixed_point[0]
        self._state = np.array([S_star + 0.01, P_star + 0.01])

        # Detect upward crossings of P = P*
        crossings: list[float] = []
        prev_P = self._state[1]
        max_steps = int(n_cycles * 200 / dt)
        for _ in range(max_steps):
            self.step()
            P = self._state[1]
            if prev_P < P_star <= P:
                frac = (P_star - prev_P) / (P - prev_P) if P != prev_P else 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
                if len(crossings) >= n_cycles + 1:
                    break
            # Stop if P collapses (trajectory escaped)
            if P < P_star * 0.01 or P > P_star * 100:
                break
            prev_P = P

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))

    def measure_peak_amplitude(self, n_cycles: int = 3) -> tuple[float, float]:
        """Measure peak-to-peak amplitude in early oscillation cycles.

        Starts near the fixed point and measures amplitude before the
        trajectory escapes. Returns (S_amplitude, P_amplitude).
        """
        dt = self.config.dt
        S_star, P_star = self.fixed_point

        # Start near fixed point
        self._state = np.array([S_star + 0.01, P_star + 0.01])

        # Theoretical period for timing
        omega = self.oscillation_frequency
        if omega <= 0:
            return (0.0, 0.0)
        T = 2 * np.pi / omega
        measure_steps = int(n_cycles * T / dt)

        S_vals: list[float] = []
        P_vals: list[float] = []
        for _ in range(measure_steps):
            self.step()
            S_vals.append(self._state[0])
            P_vals.append(self._state[1])
            # Stop if escaped
            if self._state[1] < P_star * 0.001:
                break

        if not S_vals:
            return (0.0, 0.0)

        return (
            float(max(S_vals) - min(S_vals)),
            float(max(P_vals) - min(P_vals)),
        )


def compute_hopf_v_in(k1: float, k2: float) -> float:
    """Compute critical v_in for Hopf bifurcation at given k1, k2.

    The Jacobian trace at the fixed point is:
        tr(J) = k2 - k1*v_in^2/k2^2

    Hopf: tr(J) = 0  =>  v_in_c = k2 * sqrt(k2/k1)

    For v_in < v_in_c: unstable spiral (growing oscillations)
    For v_in > v_in_c: stable spiral (decaying oscillations)
    """
    return k2 * np.sqrt(k2 / k1)


def compute_hopf_v_in_numerical(
    k1: float, k2: float, v_in_range: tuple[float, float] = (0.01, 20.0)
) -> float:
    """Compute critical v_in numerically via root-finding on tr(J) = 0.

    This verifies the analytical formula and handles edge cases.
    """
    def trace_func(v_in: float) -> float:
        P_star = v_in / k2
        S_star = k2**2 / (k1 * v_in)
        return -k1 * P_star**2 + 2 * k1 * S_star * P_star - k2

    lo, hi = v_in_range
    f_lo = trace_func(lo)
    f_hi = trace_func(hi)

    # If the trace does not change sign, return the analytical value
    if f_lo * f_hi > 0:
        return compute_hopf_v_in(k1, k2)

    return float(brentq(trace_func, lo, hi))
