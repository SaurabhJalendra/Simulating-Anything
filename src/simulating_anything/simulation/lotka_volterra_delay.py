"""Lotka-Volterra predator-prey with time delay in predator growth.

The delayed Lotka-Volterra equations with logistic prey growth:
    dN/dt = r*N*(1 - N/K) - a*N*P          (prey, logistic + predation)
    dP/dt = b*N(t-tau)*P - d*P              (predator, delayed numerical response)

N(t-tau) is the prey density at time t-tau, representing the gestation
period needed for predators to convert consumed prey into new predators.

Key dynamical properties:
- For tau=0, reduces to standard Lotka-Volterra with logistic prey growth
- Equilibrium point (N*, P*) is independent of tau
- Delay destabilizes the equilibrium via Hopf bifurcation at critical tau_c
- For tau > tau_c, sustained limit-cycle oscillations emerge
- Oscillation amplitude and period both increase with tau

Target rediscoveries:
- Equilibrium: N* = d/b, P* = r*(1 - N*/K)/a
- Critical delay tau_c for Hopf bifurcation
- Period and amplitude scaling with tau
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LotkaVolterraDelaySimulation(SimulationEnvironment):
    """Delayed Lotka-Volterra predator-prey model.

    State: [N, P] where N = prey density, P = predator density.
    Internal: ring buffer storing past prey values for delayed term N(t - tau).

    Parameters:
        r: prey intrinsic growth rate (default 1.0)
        K: prey carrying capacity (default 4.0)
        a: predation rate (default 0.5)
        b: predator conversion efficiency * attack rate (default 0.2)
        d: predator mortality rate (default 0.3)
        tau: gestation delay (default 1.0)
        N_0: initial prey density (default 3.0)
        P_0: initial predator density (default 1.0)

    Note on K: A moderate carrying capacity (K=4) keeps N*/K = 0.375, which
    ensures the no-delay system is stable while allowing the delay to cleanly
    destabilize the equilibrium via Hopf bifurcation. Large K (e.g. K=10)
    triggers the "paradox of enrichment" where delay-induced oscillations
    become so violent they drive populations to numerical extinction.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.r = p.get("r", 1.0)
        self.K = p.get("K", 4.0)
        self.a = p.get("a", 0.5)
        self.b = p.get("b", 0.2)
        self.d = p.get("d", 0.3)
        self.tau = p.get("tau", 1.0)
        self.N_0 = p.get("N_0", 3.0)
        self.P_0 = p.get("P_0", 1.0)

        # Ring buffer for delayed prey values
        self._buf_size = max(int(self.tau / config.dt) + 1, 1)
        self._buffer: np.ndarray = np.empty(0)
        self._buf_idx = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations and fill history buffer with N_0."""
        self._state = np.array([self.N_0, self.P_0], dtype=np.float64)
        self._step_count = 0

        # Fill history buffer with initial prey value (constant history)
        self._buffer = np.full(self._buf_size, self.N_0, dtype=np.float64)
        self._buf_idx = 0
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with fixed delayed term.

        The delayed prey value N(t - tau) is read from the ring buffer and
        held constant during the RK4 substeps (standard for DDE solvers).
        """
        dt = self.config.dt

        # Delayed prey value from ring buffer (fixed for this step)
        N_delayed = self._buffer[self._buf_idx]

        y = self._state
        k1 = self._derivatives(y, N_delayed)
        k2 = self._derivatives(np.maximum(y + 0.5 * dt * k1, 0.0), N_delayed)
        k3 = self._derivatives(np.maximum(y + 0.5 * dt * k2, 0.0), N_delayed)
        k4 = self._derivatives(np.maximum(y + dt * k3, 0.0), N_delayed)

        new_state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure non-negative populations
        new_state = np.maximum(new_state, 0.0)

        # Store new prey value in ring buffer (overwrite oldest)
        self._buffer[self._buf_idx] = new_state[0]
        self._buf_idx = (self._buf_idx + 1) % self._buf_size

        self._state = new_state
        self._step_count += 1
        return self._state.copy()

    def _derivatives(
        self, y: np.ndarray, N_delayed: float,
    ) -> np.ndarray:
        """Right-hand side of the delayed Lotka-Volterra system.

        Args:
            y: Current state [N, P].
            N_delayed: Prey density at time t - tau.
        """
        N, P = y
        dN = self.r * N * (1.0 - N / self.K) - self.a * N * P
        dP = self.b * N_delayed * P - self.d * P
        return np.array([dN, dP])

    def observe(self) -> np.ndarray:
        """Return current state [N, P]."""
        return self._state.copy()

    def equilibrium(self) -> tuple[float, float]:
        """Compute the interior equilibrium (N*, P*).

        At steady state (delay irrelevant for fixed points):
            dP/dt = 0  =>  b*N* - d = 0  =>  N* = d/b
            dN/dt = 0  =>  r*N*(1-N/K) = a*N*P  =>  P* = r*(1-N*/K)/a

        Returns:
            Tuple (N_star, P_star). Returns (0, 0) if no valid equilibrium.
        """
        if self.b <= 0:
            return (0.0, 0.0)

        N_star = self.d / self.b
        if N_star <= 0 or N_star >= self.K:
            return (0.0, 0.0)

        P_star = self.r * (1.0 - N_star / self.K) / self.a
        if P_star < 0:
            return (0.0, 0.0)

        return (float(N_star), float(P_star))

    def critical_tau(self) -> float:
        """Estimate critical delay tau_c for Hopf bifurcation onset.

        For the linearized system around (N*, P*), the characteristic
        equation has purely imaginary roots at the Hopf bifurcation.
        The critical delay satisfies:
            omega^2 = (a11)^2 + 2*a11*a22 + a22^2
            with appropriate Jacobian entries.

        For this simple system:
            a11 = r*(1 - 2*N*/K) - a*P*
            a22 = -d (evaluated at equilibrium where b*N*-d=0, so a22_eff=0)
            The cross terms involve b*P* (delayed coupling).

        We use the analytical Hopf condition for scalar DDE feedback:
            tau_c = arccos(-a11 / sqrt(a11^2 + (b*P*)^2)) / omega
            where omega = sqrt((b*P*)^2 - a11^2) if |b*P*| > |a11|

        Returns:
            Estimated critical delay. Returns float('inf') if no Hopf.
        """
        N_star, P_star = self.equilibrium()
        if N_star <= 0 or P_star <= 0:
            return float("inf")

        # Jacobian entries at equilibrium
        # dN/dt linearized: a11*dN + a12*dP  where
        #   a11 = r*(1 - 2*N*/K) - a*P*
        #   a12 = -a*N*
        # dP/dt linearized: a21_delayed*dN(t-tau) + 0*dP  where
        #   a21 = b*P*   (coupling through delay)
        #   Note: b*N*-d=0 at equilibrium, so the dP coefficient is 0.

        a11 = self.r * (1.0 - 2.0 * N_star / self.K) - self.a * P_star
        cross = self.b * P_star  # delayed coupling strength

        # For Hopf: need omega such that characteristic equation has
        # purely imaginary roots. The condition is:
        #   omega^2 = cross^2*a_N_star^2/... (simplified)
        # Using the standard 2D DDE Hopf condition:
        #   (i*omega - a11)*(-i*omega) = cross * a12 * exp(-i*omega*tau)
        # This is complex; use simpler sufficient condition:
        #   tau_c ~ pi / (2 * omega_nat) where omega_nat ~ sqrt(a*N* * b*P*)

        a12 = -self.a * N_star
        product = -a12 * cross  # = a*N* * b*P* > 0

        if product <= 0:
            return float("inf")

        omega_nat = np.sqrt(product)

        # Approximate critical delay from transcendental equation
        # tau_c ~ (1/omega_nat) * arctan(omega_nat / (-a11))
        if abs(a11) < 1e-12:
            tau_c = np.pi / (2.0 * omega_nat)
        else:
            tau_c = np.arctan2(omega_nat, -a11) / omega_nat

        return float(max(tau_c, 0.0))

    def measure_period(
        self,
        n_steps: int = 50000,
        n_transient: int = 20000,
    ) -> float:
        """Measure dominant oscillation period via FFT.

        Runs the current simulation for n_steps (after skipping transient),
        then extracts the dominant frequency from the prey time series.

        Returns:
            Period in time units. Returns 0.0 if no clear oscillation.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Collect prey data
        N_vals = []
        for _ in range(n_steps):
            state = self.step()
            N_vals.append(state[0])

        return self._fft_period(np.array(N_vals), dt)

    def delay_sweep(
        self,
        tau_values: np.ndarray,
        n_steps: int = 50000,
        n_transient: int = 20000,
    ) -> dict[str, np.ndarray]:
        """Sweep delay values and measure oscillation amplitude and period.

        Args:
            tau_values: Array of delay values to sweep.
            n_steps: Steps after transient for measurement.
            n_transient: Steps to discard as transient.

        Returns:
            Dict with tau, amplitude, and period arrays.
        """
        amplitudes = []
        periods = []

        for tau_val in tau_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "r": self.r, "K": self.K, "a": self.a,
                    "b": self.b, "d": self.d, "tau": tau_val,
                    "N_0": self.N_0, "P_0": self.P_0,
                },
            )
            sim = LotkaVolterraDelaySimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Collect prey values
            N_vals = []
            for _ in range(n_steps):
                state = sim.step()
                N_vals.append(state[0])

            N_arr = np.array(N_vals)
            amp = float(np.max(N_arr) - np.min(N_arr))
            amplitudes.append(amp)

            T = self._fft_period(N_arr, self.config.dt)
            periods.append(T)

        return {
            "tau": tau_values,
            "amplitude": np.array(amplitudes),
            "period": np.array(periods),
        }

    def hopf_bifurcation_detect(
        self,
        tau_values: np.ndarray,
        n_steps: int = 50000,
        n_transient: int = 20000,
        amplitude_threshold: float = 0.05,
    ) -> float | None:
        """Detect critical delay tau_c where Hopf bifurcation occurs.

        Sweeps tau values and finds the first value where oscillation
        amplitude exceeds the threshold.

        Returns:
            Estimated tau_c, or None if no bifurcation detected.
        """
        for tau_val in tau_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "r": self.r, "K": self.K, "a": self.a,
                    "b": self.b, "d": self.d, "tau": tau_val,
                    "N_0": self.N_0, "P_0": self.P_0,
                },
            )
            sim = LotkaVolterraDelaySimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure amplitude
            N_vals = []
            for _ in range(n_steps):
                state = sim.step()
                N_vals.append(state[0])

            N_arr = np.array(N_vals)
            amp = float(np.max(N_arr) - np.min(N_arr))

            if amp > amplitude_threshold:
                return float(tau_val)

        return None

    def no_delay_comparison(self, n_steps: int = 50000) -> dict[str, np.ndarray]:
        """Run the system with tau=0 for comparison (standard logistic LV).

        Returns trajectory data and final state for the no-delay case.
        """
        config = SimulationConfig(
            domain=self.config.domain,
            dt=self.config.dt,
            n_steps=n_steps,
            parameters={
                "r": self.r, "K": self.K, "a": self.a,
                "b": self.b, "d": self.d, "tau": 0.0,
                "N_0": self.N_0, "P_0": self.P_0,
            },
        )
        sim = LotkaVolterraDelaySimulation(config)
        sim.reset()

        N_vals = []
        P_vals = []
        for _ in range(n_steps):
            state = sim.step()
            N_vals.append(state[0])
            P_vals.append(state[1])

        return {
            "N": np.array(N_vals),
            "P": np.array(P_vals),
            "final_state": sim.observe(),
        }

    @staticmethod
    def _fft_period(signal: np.ndarray, dt: float) -> float:
        """Extract dominant period from a time series using FFT.

        Returns 0.0 if no clear oscillation is detected.
        """
        signal = signal - np.mean(signal)
        if np.std(signal) < 1e-10:
            return 0.0

        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=dt)

        # Skip DC component (index 0)
        if len(fft_vals) < 2:
            return 0.0

        peak_idx = np.argmax(fft_vals[1:]) + 1
        if freqs[peak_idx] <= 0:
            return 0.0

        return float(1.0 / freqs[peak_idx])
