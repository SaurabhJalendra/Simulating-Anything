"""Cable equation (passive neurite) simulation.

Models signal propagation along a passive dendrite/axon:
    tau_m * dV/dt = lambda^2 * d^2V/dx^2 - V + R_m * I_ext(x,t)

Target rediscoveries:
- Spatial decay: V(x) ~ exp(-|x - x0| / lambda) at steady state
- Space constant: lambda = sqrt(R_m / R_a)
- Time constant: tau_m = R_m * C_m
- Frequency-dependent attenuation (low-pass filtering)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CableEquationSimulation(SimulationEnvironment):
    """Cable equation: tau_m * dV/dt = lambda^2 * V_xx - V + R_m * I_ext.

    Finite cable with sealed ends (Neumann BCs: dV/dx = 0 at boundaries).
    Solved via Crank-Nicolson implicit scheme for stability.

    State: V(x) at N grid points representing membrane potential deviation
    from resting potential.

    Parameters:
        tau_m: membrane time constant in ms (default 10.0)
        lambda_e: electrotonic length constant in mm (default 0.5)
        L: cable length in mm (default 5.0)
        N: number of spatial grid points (default 100)
        R_m: membrane resistance (default 1.0)
        I0: injected current amplitude (default 1.0)
        inject_x: injection location as fraction of L (default 0.5 = center)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.tau_m = p.get("tau_m", 10.0)
        self.lambda_e = p.get("lambda_e", 0.5)
        self.L = p.get("L", 5.0)
        self.N = int(p.get("N", 100))
        self.R_m = p.get("R_m", 1.0)
        self.I0 = p.get("I0", 1.0)
        self.inject_x = p.get("inject_x", 0.5)

        # Derived quantities
        self.dx = self.L / (self.N - 1)
        self.x = np.linspace(0, self.L, self.N)
        self._inject_idx = int(self.inject_x * (self.N - 1))

        # Build Crank-Nicolson matrices for the equation:
        #   tau_m * (V^{n+1} - V^n)/dt = lambda^2 * V_xx^{avg} - V^{avg} + R_m*I_ext
        # where V^{avg} = (V^{n+1} + V^n)/2, V_xx^{avg} = (V_xx^{n+1} + V_xx^{n})/2
        self._build_cn_matrices()

        # External current: point injection at center by default
        self._I_ext = np.zeros(self.N)
        self._I_ext[self._inject_idx] = self.I0 / self.dx

    def _build_cn_matrices(self) -> None:
        """Build Crank-Nicolson implicit/explicit matrices.

        Rearranged form:
            A * V^{n+1} = B * V^n + dt * R_m * I_ext
        """
        dt = self.config.dt
        N = self.N
        r = (self.lambda_e ** 2 * dt) / (2.0 * self.tau_m * self.dx ** 2)
        s = dt / (2.0 * self.tau_m)

        # Implicit side: A * V^{n+1}
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 1.0 + 2.0 * r + s
            if i > 0:
                A[i, i - 1] = -r
            if i < N - 1:
                A[i, i + 1] = -r

        # Sealed ends (Neumann BCs): ghost points reflected
        # At i=0: V[-1] = V[1], so the -r contribution from ghost goes to i=1
        A[0, 0] = 1.0 + r + s  # Only one neighbor contributes r
        A[N - 1, N - 1] = 1.0 + r + s

        # Explicit side: B * V^n
        B = np.zeros((N, N))
        for i in range(N):
            B[i, i] = 1.0 - 2.0 * r - s
            if i > 0:
                B[i, i - 1] = r
            if i < N - 1:
                B[i, i + 1] = r

        B[0, 0] = 1.0 - r - s
        B[N - 1, N - 1] = 1.0 - r - s

        self._A = A
        self._B = B
        # Pre-factor for source term
        self._src_factor = dt * self.R_m / self.tau_m

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize membrane potential to zero (resting state)."""
        self._state = np.zeros(self.N, dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Crank-Nicolson."""
        rhs = self._B @ self._state + self._src_factor * self._I_ext
        self._state = np.linalg.solve(self._A, rhs)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current membrane potential profile V(x)."""
        return self._state

    def set_current(self, I_ext: np.ndarray) -> None:
        """Set arbitrary external current profile."""
        self._I_ext = I_ext.copy()

    def set_point_current(self, x_pos: float, amplitude: float) -> None:
        """Set point current injection at position x_pos."""
        self._I_ext = np.zeros(self.N)
        idx = int(np.round(x_pos / self.dx))
        idx = max(0, min(idx, self.N - 1))
        self._I_ext[idx] = amplitude / self.dx

    def steady_state_analytical(self, x: np.ndarray, I0: float) -> np.ndarray:
        """Analytical steady-state for infinite cable with point injection.

        V_ss(x) = (I0 * R_m / (2 * lambda)) * exp(-|x - x0| / lambda)

        Args:
            x: spatial positions.
            I0: injected current amplitude.

        Returns:
            Steady-state potential at each x.
        """
        x0 = self.x[self._inject_idx]
        prefactor = I0 * self.R_m / (2.0 * self.lambda_e)
        return prefactor * np.exp(-np.abs(x - x0) / self.lambda_e)

    def measure_space_constant(self) -> float:
        """Measure space constant by fitting exponential decay to profile.

        Fits V(x) = A * exp(-|x - x0| / lambda_fit) to the steady-state
        profile on the right side of injection.

        Returns:
            Measured lambda (space constant).
        """
        V = self._state
        x0_idx = self._inject_idx
        V_peak = V[x0_idx]
        if V_peak <= 0:
            return 0.0

        # Use right side of injection to avoid boundary effects
        right_x = self.x[x0_idx:] - self.x[x0_idx]
        right_V = V[x0_idx:]

        # Only use points where V > some fraction of peak (avoid noise floor)
        valid = right_V > 0.01 * V_peak
        if np.sum(valid) < 3:
            return 0.0

        log_ratio = np.log(right_V[valid] / V_peak)
        x_valid = right_x[valid]

        # Linear fit: log(V/V_peak) = -x / lambda
        if len(x_valid) < 2:
            return 0.0
        coeffs = np.polyfit(x_valid, log_ratio, 1)
        if coeffs[0] >= 0:
            return 0.0
        return float(-1.0 / coeffs[0])

    def measure_time_constant(self, n_steps: int = 5000) -> float:
        """Measure time constant from temporal approach to steady state.

        Runs simulation and fits V(t) = V_ss * (1 - exp(-t / tau_fit))
        at the injection point.

        Args:
            n_steps: number of steps to run.

        Returns:
            Measured tau (time constant).
        """
        self.reset()
        dt = self.config.dt
        times = []
        voltages = []

        for i in range(n_steps):
            self.step()
            times.append((i + 1) * dt)
            voltages.append(self._state[self._inject_idx])

        times = np.array(times)
        voltages = np.array(voltages)

        # Steady-state estimate from last 10% of trajectory
        V_ss = np.mean(voltages[-len(voltages) // 10:])
        if V_ss <= 0:
            return 0.0

        # Fit: V(t) = V_ss * (1 - exp(-t/tau))
        # => log(1 - V/V_ss) = -t/tau
        ratio = voltages / V_ss
        valid = (ratio > 0.01) & (ratio < 0.95)
        if np.sum(valid) < 3:
            return 0.0

        log_term = np.log(1.0 - ratio[valid])
        t_valid = times[valid]
        coeffs = np.polyfit(t_valid, log_term, 1)
        if coeffs[0] >= 0:
            return 0.0
        return float(-1.0 / coeffs[0])

    def frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        """Measure attenuation at different input frequencies.

        For each frequency, applies sinusoidal current and measures
        the ratio of response amplitude at a distant point to that
        at the injection point.

        Args:
            frequencies: array of frequencies to test (Hz).

        Returns:
            Attenuation ratio at each frequency.
        """
        dt = self.config.dt
        measure_idx = min(self._inject_idx + self.N // 4, self.N - 1)
        attenuations = []

        for freq in frequencies:
            self.reset()
            period = 1.0 / freq if freq > 0 else 1.0
            n_cycles = 5
            n_steps = int(n_cycles * period / dt)
            n_steps = max(n_steps, 100)

            inject_vals = []
            measure_vals = []

            for i in range(n_steps):
                t = (i + 1) * dt
                # Sinusoidal current
                self._I_ext = np.zeros(self.N)
                amp = self.I0 * np.sin(2.0 * np.pi * freq * t)
                self._I_ext[self._inject_idx] = amp / self.dx
                self.step()
                inject_vals.append(self._state[self._inject_idx])
                measure_vals.append(self._state[measure_idx])

            inject_amp = (np.max(inject_vals[-n_steps // 2:])
                          - np.min(inject_vals[-n_steps // 2:])) / 2.0
            measure_amp = (np.max(measure_vals[-n_steps // 2:])
                           - np.min(measure_vals[-n_steps // 2:])) / 2.0

            ratio = measure_amp / inject_amp if inject_amp > 1e-15 else 0.0
            attenuations.append(ratio)

        return np.array(attenuations)

    def length_constant_sweep(
        self, lambda_values: np.ndarray, n_steps: int = 5000
    ) -> dict[str, np.ndarray]:
        """Sweep lambda_e values and measure spatial decay constant.

        Args:
            lambda_values: array of lambda_e values to test.
            n_steps: steps to run for each value.

        Returns:
            Dict with lambda_set, lambda_measured arrays.
        """
        measured = []
        for lam in lambda_values:
            # Rebuild simulation with new lambda
            params = dict(self.config.parameters)
            params["lambda_e"] = lam
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters=params,
            )
            sim = CableEquationSimulation(config)
            sim.reset()
            for _ in range(n_steps):
                sim.step()
            lam_meas = sim.measure_space_constant()
            measured.append(lam_meas)

        return {
            "lambda_set": lambda_values.copy(),
            "lambda_measured": np.array(measured),
        }
