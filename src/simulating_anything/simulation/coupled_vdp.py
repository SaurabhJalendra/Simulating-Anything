"""Coupled Van der Pol oscillators simulation.

Two mutually coupled Van der Pol oscillators with diffusive position coupling:

    dx1/dt = y1
    dy1/dt = mu*(1 - x1^2)*y1 - x1 + k*(x2 - x1)
    dx2/dt = y2
    dy2/dt = mu*(1 - x2^2)*y2 - x2 + k*(x1 - x2)

The coupling is symmetric and diffusive in the position variable.
Depending on coupling strength k and initial conditions, the system exhibits:
- Independent limit cycles at k=0
- In-phase synchronization (x1 ~ x2) at moderate-to-strong coupling
- Anti-phase synchronization (x1 ~ -x2) at weak coupling with anti-phase ICs
- Amplitude death at very strong coupling for certain detuning

Target rediscoveries:
- Synchronization transition as a function of coupling k
- In-phase vs anti-phase mode selection
- ODE recovery via SINDy
- Sync error decay rate vs coupling strength
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CoupledVdPSimulation(SimulationEnvironment):
    """Two mutually coupled Van der Pol oscillators.

    State vector: [x1, y1, x2, y2]

    Parameters:
        mu: nonlinearity parameter controlling damping strength (default 1.0)
        k: coupling strength between oscillators (default 0.5)
        x1_0: initial displacement of oscillator 1 (default 0.1)
        y1_0: initial velocity of oscillator 1 (default 0.0)
        x2_0: initial displacement of oscillator 2 (default -0.1)
        y2_0: initial velocity of oscillator 2 (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.mu = p.get("mu", 1.0)
        self.k = p.get("k", 0.5)
        self.x1_0 = p.get("x1_0", 0.1)
        self.y1_0 = p.get("y1_0", 0.0)
        self.x2_0 = p.get("x2_0", -0.1)
        self.y2_0 = p.get("y2_0", 0.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize both oscillators with specified initial conditions."""
        self._state = np.array(
            [self.x1_0, self.y1_0, self.x2_0, self.y2_0],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x1, y1, x2, y2]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for the coupled VdP system.

        dx1/dt = y1
        dy1/dt = mu*(1 - x1^2)*y1 - x1 + k*(x2 - x1)
        dx2/dt = y2
        dy2/dt = mu*(1 - x2^2)*y2 - x2 + k*(x1 - x2)
        """
        x1, y1, x2, y2 = state

        dx1 = y1
        dy1 = self.mu * (1 - x1**2) * y1 - x1 + self.k * (x2 - x1)
        dx2 = y2
        dy2 = self.mu * (1 - x2**2) * y2 - x2 + self.k * (x1 - x2)

        return np.array([dx1, dy1, dx2, dy2])

    def compute_sync_error(self, state: np.ndarray | None = None) -> float:
        """Compute instantaneous synchronization error |x1 - x2|.

        Args:
            state: State vector [x1, y1, x2, y2]. Uses current state if None.

        Returns:
            Euclidean distance between oscillator positions and velocities.
        """
        s = state if state is not None else self._state
        diff = np.array([s[0] - s[2], s[1] - s[3]])
        return float(np.linalg.norm(diff))

    def compute_phase_difference(self) -> float:
        """Compute approximate phase difference between the two oscillators.

        Uses the arctangent of (y, x) for each oscillator to extract an
        instantaneous phase angle, then returns the difference mod 2*pi.

        Returns:
            Phase difference in radians, in the range [0, 2*pi).
        """
        x1, y1, x2, y2 = self._state
        phi1 = np.arctan2(y1, x1)
        phi2 = np.arctan2(y2, x2)
        dphi = (phi1 - phi2) % (2 * np.pi)
        return float(dphi)

    def coupling_sweep(
        self,
        k_values: np.ndarray,
        n_steps: int = 10000,
        n_transient: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep coupling strength and measure steady-state sync error.

        For each k value, creates a fresh simulation from the same initial
        conditions, discards transient, and computes the time-averaged
        synchronization error.

        Args:
            k_values: Array of coupling strengths to sweep.
            n_steps: Measurement steps after transient.
            n_transient: Transient steps to discard.

        Returns:
            Dict with 'k', 'mean_error', and 'mean_phase_diff' arrays.
        """
        mean_errors = np.empty(len(k_values))
        mean_phases = np.empty(len(k_values))

        for i, k_val in enumerate(k_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "mu": self.mu,
                    "k": float(k_val),
                    "x1_0": self.x1_0,
                    "y1_0": self.y1_0,
                    "x2_0": self.x2_0,
                    "y2_0": self.y2_0,
                },
            )
            sim = CoupledVdPSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure average error and phase difference
            error_sum = 0.0
            phase_sum = 0.0
            for _ in range(n_steps):
                sim.step()
                error_sum += sim.compute_sync_error()
                phase_sum += sim.compute_phase_difference()

            mean_errors[i] = error_sum / n_steps
            mean_phases[i] = phase_sum / n_steps

        return {
            "k": k_values,
            "mean_error": mean_errors,
            "mean_phase_diff": mean_phases,
        }

    def sync_error_trajectory(
        self, n_steps: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Record synchronization error and phase difference over time.

        Args:
            n_steps: Number of integration steps.

        Returns:
            Dict with 'time', 'error', and 'phase_diff' arrays.
        """
        dt = self.config.dt
        errors = np.empty(n_steps + 1)
        phases = np.empty(n_steps + 1)
        times = np.empty(n_steps + 1)

        errors[0] = self.compute_sync_error()
        phases[0] = self.compute_phase_difference()
        times[0] = 0.0

        for i in range(1, n_steps + 1):
            self.step()
            errors[i] = self.compute_sync_error()
            phases[i] = self.compute_phase_difference()
            times[i] = i * dt

        return {"time": times, "error": errors, "phase_diff": phases}

    def measure_period(self, n_periods: int = 5) -> float:
        """Measure the oscillation period of oscillator 1 via zero crossings.

        Runs through a transient then detects upward zero crossings of x1
        to estimate the period.

        Args:
            n_periods: Number of periods to average over.

        Returns:
            Mean period, or inf if no oscillation detected.
        """
        dt = self.config.dt
        T_approx = 2 * np.pi  # harmonic approximation

        # Transient
        transient_steps = max(int(50 / dt), int(20 * T_approx / dt))
        for _ in range(transient_steps):
            self.step()

        # Detect upward zero crossings of x1
        crossings: list[float] = []
        prev_x1 = self._state[0]
        measure_steps = int(n_periods * T_approx / dt * 2)
        for _ in range(measure_steps):
            self.step()
            x1 = self._state[0]
            if prev_x1 < 0 and x1 >= 0:
                t_cross = (
                    (self._step_count - 1) * dt
                    + dt * (-prev_x1) / (x1 - prev_x1)
                )
                crossings.append(t_cross)
            prev_x1 = x1

        if len(crossings) < 2:
            return float("inf")

        periods = np.diff(crossings)
        return float(np.mean(periods))

    def measure_amplitudes(self, n_periods: int = 3) -> tuple[float, float]:
        """Measure limit cycle amplitudes of both oscillators after transient.

        Args:
            n_periods: Number of periods to measure over.

        Returns:
            Tuple of (amplitude_1, amplitude_2) as max |x| values.
        """
        dt = self.config.dt
        T_approx = 2 * np.pi

        transient_steps = max(int(50 / dt), int(20 * T_approx / dt))
        for _ in range(transient_steps):
            self.step()

        x1_max = 0.0
        x2_max = 0.0
        measure_steps = int(n_periods * T_approx / dt * 2)
        for _ in range(measure_steps):
            self.step()
            x1_max = max(x1_max, abs(self._state[0]))
            x2_max = max(x2_max, abs(self._state[2]))

        return float(x1_max), float(x2_max)

    def compute_energy(self, state: np.ndarray | None = None) -> float:
        """Compute approximate total mechanical energy of both oscillators.

        E = 0.5*(y1^2 + y2^2) + 0.5*(x1^2 + x2^2) + 0.5*k*(x1-x2)^2

        This is the energy of the conservative part (ignoring VdP damping).
        Not conserved due to nonlinear damping, but useful for diagnostics.

        Args:
            state: State vector [x1, y1, x2, y2]. Uses current if None.

        Returns:
            Instantaneous total energy.
        """
        s = state if state is not None else self._state
        x1, y1, x2, y2 = s
        ke = 0.5 * (y1**2 + y2**2)
        pe_indiv = 0.5 * (x1**2 + x2**2)
        pe_coupling = 0.5 * self.k * (x1 - x2)**2
        return float(ke + pe_indiv + pe_coupling)

    def is_in_phase(self, threshold: float = 0.3) -> bool:
        """Check if oscillators are approximately in phase.

        In-phase means phase difference is near 0 (or 2*pi).

        Args:
            threshold: Maximum phase difference (radians) to count as in-phase.

        Returns:
            True if oscillators are approximately synchronized in phase.
        """
        dphi = self.compute_phase_difference()
        return bool(dphi < threshold or dphi > (2 * np.pi - threshold))

    def is_anti_phase(self, threshold: float = 0.3) -> bool:
        """Check if oscillators are approximately in anti-phase.

        Anti-phase means phase difference is near pi.

        Args:
            threshold: Maximum deviation from pi to count as anti-phase.

        Returns:
            True if oscillators are approximately in anti-phase.
        """
        dphi = self.compute_phase_difference()
        return bool(abs(dphi - np.pi) < threshold)
