"""Winfree model of pulse-coupled oscillators.

The Winfree model (1967) predates the Kuramoto model and describes coupled
oscillators interacting through pulse-like coupling rather than sinusoidal
phase differences. Each oscillator has a phase theta_i evolving as:

    d(theta_i)/dt = omega_i + (kappa/N) * P(theta_i) * sum_j R(theta_j)

where:
    P(theta) = (1 - cos(theta))  -- sensitivity function (how much osc i
                                     responds to incoming pulses)
    R(theta) = (1 + cos(theta))  -- pulse function (signal emitted by osc j)

The key difference from Kuramoto: coupling here is multiplicative (P*R),
not additive sin(theta_j - theta_i). This creates richer dynamics including
a sharp synchronization threshold ("death" of oscillations at strong coupling).

Target rediscoveries:
- Synchronization transition: order parameter r(kappa) from 0 to ~1
- Critical coupling kappa_c for onset of coherence
- Comparison with Kuramoto: Winfree has sharper transition
- Amplitude death at very strong coupling (oscillators quench)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class WinfreeSimulation(SimulationEnvironment):
    """Winfree model: pulse-coupled oscillators with multiplicative coupling.

    State vector: [theta_1, theta_2, ..., theta_N] (phases on [0, 2*pi)).

    The ODE for each oscillator i is:
        d(theta_i)/dt = omega_i + (kappa/N) * P(theta_i) * sum_j R(theta_j)

    where P(theta) = (1 - cos(theta)) is the sensitivity function and
    R(theta) = (1 + cos(theta)) is the pulse function.

    Parameters:
        N: number of oscillators (default 50)
        kappa: coupling strength (default 1.0)
        omega0: mean natural frequency (default 1.0)
        delta: frequency spread / half-width (default 0.1)
        distribution: 'lorentzian' or 'gaussian' (class attribute, default
            'lorentzian')
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 50))
        self.kappa = p.get("kappa", 1.0)
        self.omega0 = p.get("omega0", 1.0)
        self.delta = p.get("delta", 0.1)
        # String params cannot go in parameters dict
        self.distribution = "lorentzian"

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize phases uniformly on [0, 2*pi) and draw natural freqs."""
        rng = np.random.default_rng(seed)
        self._state = rng.uniform(0, 2 * np.pi, self.N).astype(np.float64)
        self._step_count = 0

        # Draw natural frequencies from chosen distribution
        if self.distribution == "gaussian":
            self.omega = (
                self.omega0 + self.delta * rng.standard_normal(self.N)
            ).astype(np.float64)
        else:
            # Lorentzian (Cauchy) distribution centered at omega0 with
            # half-width delta
            self.omega = (
                self.omega0
                + self.delta * np.tan(np.pi * (rng.uniform(size=self.N) - 0.5))
            ).astype(np.float64)

        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        # Wrap phases to [0, 2*pi)
        self._state = self._state % (2 * np.pi)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current phases."""
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

    def _derivatives(self, theta: np.ndarray) -> np.ndarray:
        """Compute d(theta)/dt for all oscillators.

        d(theta_i)/dt = omega_i + (kappa/N) * P(theta_i) * sum_j R(theta_j)
        """
        # Sensitivity: P(theta_i) = 1 - cos(theta_i)
        P = 1.0 - np.cos(theta)
        # Pulse: R(theta_j) = 1 + cos(theta_j)
        R = 1.0 + np.cos(theta)
        # Mean-field pulse signal
        R_sum = np.sum(R)
        # Coupling term
        coupling = (self.kappa / self.N) * P * R_sum
        return self.omega + coupling

    @staticmethod
    def sensitivity(theta: np.ndarray | float) -> np.ndarray | float:
        """Sensitivity function P(theta) = 1 - cos(theta)."""
        return 1.0 - np.cos(theta)

    @staticmethod
    def pulse(theta: np.ndarray | float) -> np.ndarray | float:
        """Pulse function R(theta) = 1 + cos(theta)."""
        return 1.0 + np.cos(theta)

    @property
    def order_parameter(self) -> tuple[float, float]:
        """Compute the Kuramoto-style order parameter r*exp(i*psi).

        Returns (r, psi) where r in [0, 1] measures synchronization
        and psi is the mean phase.
        """
        z = np.mean(np.exp(1j * self._state))
        return float(np.abs(z)), float(np.angle(z))

    @property
    def order_parameter_r(self) -> float:
        """Synchronization order parameter r in [0, 1]."""
        return self.order_parameter[0]

    def compute_order_parameter(
        self, state: np.ndarray | None = None,
    ) -> float:
        """Compute order parameter r for a given state.

        Args:
            state: Phase array of shape (N,). Uses current state if None.

        Returns:
            Order parameter r in [0, 1].
        """
        s = state if state is not None else self._state
        z = np.mean(np.exp(1j * s))
        return float(np.abs(z))

    def compute_mean_field(self) -> tuple[float, float]:
        """Compute mean-field quantities for the current state.

        Returns:
            Tuple of (mean_pulse_signal, mean_sensitivity) where:
            - mean_pulse_signal = (1/N) * sum_j R(theta_j)
            - mean_sensitivity = (1/N) * sum_i P(theta_i)
        """
        R_mean = float(np.mean(self.pulse(self._state)))
        P_mean = float(np.mean(self.sensitivity(self._state)))
        return R_mean, P_mean

    def compute_effective_frequency(self) -> np.ndarray:
        """Compute the instantaneous effective frequency of each oscillator.

        Returns:
            Array of shape (N,) with d(theta_i)/dt for each oscillator.
        """
        return self._derivatives(self._state)

    def compute_frequency_spread(self) -> float:
        """Compute the standard deviation of effective frequencies.

        Low spread indicates frequency entrainment (synchronization).
        """
        eff_freq = self.compute_effective_frequency()
        return float(np.std(eff_freq))

    def measure_steady_state_r(
        self,
        n_transient_steps: int = 5000,
        n_measure_steps: int = 2000,
        seed: int | None = None,
    ) -> float:
        """Measure the steady-state order parameter after transient.

        Args:
            n_transient_steps: Steps to discard as transient.
            n_measure_steps: Steps to average over.
            seed: Random seed for initial conditions.

        Returns:
            Time-averaged order parameter r.
        """
        self.reset(seed=seed)

        for _ in range(n_transient_steps):
            self.step()

        r_values = []
        for _ in range(n_measure_steps):
            self.step()
            r_values.append(self.order_parameter_r)

        return float(np.mean(r_values))

    def measure_steady_state_freq_spread(
        self,
        n_transient_steps: int = 5000,
        n_measure_steps: int = 2000,
        seed: int | None = None,
    ) -> float:
        """Measure steady-state frequency spread (entrainment indicator).

        Args:
            n_transient_steps: Steps to discard as transient.
            n_measure_steps: Steps to average over.
            seed: Random seed for initial conditions.

        Returns:
            Time-averaged frequency spread (std of effective frequencies).
        """
        self.reset(seed=seed)

        for _ in range(n_transient_steps):
            self.step()

        spreads = []
        for _ in range(n_measure_steps):
            self.step()
            spreads.append(self.compute_frequency_spread())

        return float(np.mean(spreads))

    def detect_amplitude_death(
        self,
        n_transient_steps: int = 10000,
        n_measure_steps: int = 2000,
        seed: int | None = None,
        threshold: float = 0.05,
    ) -> bool:
        """Detect whether oscillators have reached amplitude death.

        Amplitude death occurs when coupling is so strong that oscillators
        stop oscillating and cluster near a fixed phase. We detect this
        by checking if the frequency spread is very small.

        Args:
            n_transient_steps: Steps to discard as transient.
            n_measure_steps: Steps to average over.
            seed: Random seed for initial conditions.
            threshold: Frequency spread threshold below which we declare death.

        Returns:
            True if amplitude death is detected.
        """
        spread = self.measure_steady_state_freq_spread(
            n_transient_steps=n_transient_steps,
            n_measure_steps=n_measure_steps,
            seed=seed,
        )
        return bool(spread < threshold)
