"""Wilson-Cowan neural population model simulation.

Describes the dynamics of interacting excitatory (E) and inhibitory (I)
neural populations:

    tau_E * dE/dt = -E + S(w_EE*E - w_EI*I + I_ext_E)
    tau_I * dI/dt = -I + S(w_IE*E - w_II*I + I_ext_I)

where S(x) = 1 / (1 + exp(-a*(x - theta))) is a sigmoid firing rate function.

Target rediscoveries:
- E-I oscillation cycle and frequency
- Hopf bifurcation: transition from steady state to oscillation vs I_ext_E
- Nullcline analysis: E-nullcline and I-nullcline intersection
- Hysteresis: multiple stable states depending on external input
- ODE recovery via SINDy
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class WilsonCowanSimulation(SimulationEnvironment):
    """Wilson-Cowan: interacting excitatory-inhibitory neural populations.

    State vector: [E, I] where E = excitatory activity, I = inhibitory activity.

    The sigmoid S(x) = 1/(1 + exp(-a*(x - theta))) converts net input to
    firing rate. Oscillations arise from the E-I feedback loop: excitation
    drives inhibition, which suppresses excitation.

    Parameters:
        tau_E: excitatory time constant (default 1.0)
        tau_I: inhibitory time constant (default 2.0)
        w_EE: E-to-E coupling weight (default 16.0)
        w_EI: I-to-E coupling weight (default 12.0)
        w_IE: E-to-I coupling weight (default 15.0)
        w_II: I-to-I coupling weight (default 3.0)
        a: sigmoid steepness (default 1.3)
        theta: sigmoid threshold (default 4.0)
        I_ext_E: external input to E population (default 1.5)
        I_ext_I: external input to I population (default 0.0)
        E_0: initial excitatory activity (default 0.1)
        I_0: initial inhibitory activity (default 0.05)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.tau_E = p.get("tau_E", 1.0)
        self.tau_I = p.get("tau_I", 2.0)
        self.w_EE = p.get("w_EE", 16.0)
        self.w_EI = p.get("w_EI", 12.0)
        self.w_IE = p.get("w_IE", 15.0)
        self.w_II = p.get("w_II", 3.0)
        self.a = p.get("a", 1.3)
        self.theta = p.get("theta", 4.0)
        self.I_ext_E = p.get("I_ext_E", 1.5)
        self.I_ext_I = p.get("I_ext_I", 0.0)
        self.E_0 = p.get("E_0", 0.1)
        self.I_0 = p.get("I_0", 0.05)

    def sigmoid(self, x: float | np.ndarray) -> float | np.ndarray:
        """Firing rate sigmoid: S(x) = 1 / (1 + exp(-a*(x - theta)))."""
        return 1.0 / (1.0 + np.exp(-self.a * (x - self.theta)))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize excitatory and inhibitory activities."""
        self._state = np.array([self.E_0, self.I_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [E, I]."""
        return self._state

    def _rk4_step(self) -> None:
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        exc, inh = y
        input_E = self.w_EE * exc - self.w_EI * inh + self.I_ext_E
        input_I = self.w_IE * exc - self.w_II * inh + self.I_ext_I
        dE = (-exc + self.sigmoid(input_E)) / self.tau_E
        dI = (-inh + self.sigmoid(input_I)) / self.tau_I
        return np.array([dE, dI])

    def nullclines(self, n_points: int = 100) -> dict[str, np.ndarray]:
        """Compute E and I nullclines for phase plane analysis.

        E-nullcline: E = S(w_EE*E - w_EI*I + I_ext_E)
        I-nullcline: I = S(w_IE*E - w_II*I + I_ext_I)

        Returns dict with 'E_null_E', 'E_null_I', 'I_null_E', 'I_null_I'
        arrays for plotting the two curves in (E, I) space.
        """
        E_range = np.linspace(0.0, 1.0, n_points)

        # E-nullcline: solve E = S(w_EE*E - w_EI*I + I_ext_E) for I
        # Rearrange: w_EI*I = w_EE*E - theta + ln(E/(1-E))/(-a) + I_ext_E
        # Actually solve numerically for each E
        I_for_E_null = np.full(n_points, np.nan)
        for i, E_val in enumerate(E_range):
            if E_val <= 0.001 or E_val >= 0.999:
                continue
            # E = S(w_EE*E - w_EI*I + I_ext_E)
            # Invert sigmoid: x = theta - ln(1/E - 1)/a
            x = self.theta - np.log(1.0 / E_val - 1.0) / self.a
            # x = w_EE*E - w_EI*I + I_ext_E => I = (w_EE*E + I_ext_E - x) / w_EI
            if abs(self.w_EI) > 1e-12:
                I_val = (self.w_EE * E_val + self.I_ext_E - x) / self.w_EI
                if 0.0 <= I_val <= 1.0:
                    I_for_E_null[i] = I_val

        # I-nullcline: solve I = S(w_IE*E - w_II*I + I_ext_I) for I at each E
        I_for_I_null = np.full(n_points, np.nan)
        for i, E_val in enumerate(E_range):
            def _i_null_eq(I_val: float) -> float:
                return I_val - self.sigmoid(
                    self.w_IE * E_val - self.w_II * I_val + self.I_ext_I
                )
            try:
                sol = fsolve(_i_null_eq, 0.5, full_output=True)
                I_val = sol[0][0]
                if 0.0 <= I_val <= 1.0 and abs(sol[1]["fvec"][0]) < 1e-8:
                    I_for_I_null[i] = I_val
            except Exception:
                pass

        return {
            "E_null_E": E_range,
            "E_null_I": I_for_E_null,
            "I_null_E": E_range,
            "I_null_I": I_for_I_null,
        }

    def find_fixed_points(self, n_guesses: int = 10) -> list[np.ndarray]:
        """Find fixed points by solving dE/dt = 0, dI/dt = 0 numerically."""
        fixed_points = []
        guesses = np.linspace(0.01, 0.99, n_guesses)

        for E_guess in guesses:
            for I_guess in guesses:
                def _residual(y: np.ndarray) -> np.ndarray:
                    return self._derivatives(y)

                try:
                    sol = fsolve(
                        _residual,
                        np.array([E_guess, I_guess]),
                        full_output=True,
                    )
                    y_sol = sol[0]
                    info = sol[1]
                    # Check convergence and valid range
                    residual = np.max(np.abs(info["fvec"]))
                    if residual < 1e-10 and np.all(y_sol >= -0.1) and np.all(y_sol <= 1.1):
                        # Check if this fixed point is new
                        is_new = True
                        for fp in fixed_points:
                            if np.allclose(fp, y_sol, atol=1e-6):
                                is_new = False
                                break
                        if is_new:
                            fixed_points.append(y_sol)
                except Exception:
                    pass

        return fixed_points

    def compute_jacobian(self, y: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at a given state [E, I]."""
        exc, inh = y
        input_E = self.w_EE * exc - self.w_EI * inh + self.I_ext_E
        input_I = self.w_IE * exc - self.w_II * inh + self.I_ext_I
        S_E = self.sigmoid(input_E)
        S_I = self.sigmoid(input_I)

        # S'(x) = a * S(x) * (1 - S(x))
        dS_E = self.a * S_E * (1.0 - S_E)
        dS_I = self.a * S_I * (1.0 - S_I)

        J = np.array([
            [(-1.0 + self.w_EE * dS_E) / self.tau_E,
             (-self.w_EI * dS_E) / self.tau_E],
            [(self.w_IE * dS_I) / self.tau_I,
             (-1.0 - self.w_II * dS_I) / self.tau_I],
        ])
        return J

    def compute_eigenvalues(self, y: np.ndarray | None = None) -> np.ndarray:
        """Compute eigenvalues of the Jacobian at a fixed point.

        If y is None, uses the first fixed point found.
        """
        if y is None:
            fps = self.find_fixed_points()
            if not fps:
                return np.array([np.nan, np.nan])
            y = fps[0]

        J = self.compute_jacobian(y)
        return np.linalg.eigvals(J)

    def hopf_bifurcation_sweep(
        self,
        I_ext_values: np.ndarray | None = None,
        n_test_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep I_ext_E and detect oscillation onset (Hopf bifurcation).

        Returns dict with I_ext values, amplitudes, and frequencies.
        """
        if I_ext_values is None:
            I_ext_values = np.linspace(0.0, 3.0, 25)

        amplitudes = []
        frequencies = []

        for I_ext in I_ext_values:
            self.I_ext_E = I_ext
            self.reset()

            # Transient
            for _ in range(n_test_steps):
                self.step()

            # Measure oscillation
            E_vals = []
            for _ in range(n_test_steps):
                self.step()
                E_vals.append(self._state[0])

            E_vals = np.array(E_vals)
            amp = float(np.max(E_vals) - np.min(E_vals))
            amplitudes.append(amp)

            # Frequency via zero crossings of E - mean(E)
            E_mean = np.mean(E_vals)
            crossings = []
            for j in range(1, len(E_vals)):
                if E_vals[j - 1] < E_mean and E_vals[j] >= E_mean:
                    crossings.append(j)
            if len(crossings) >= 2:
                periods = np.diff(crossings) * self.config.dt
                freq = 1.0 / np.mean(periods)
            else:
                freq = 0.0
            frequencies.append(float(freq))

        return {
            "I_ext": I_ext_values,
            "amplitude": np.array(amplitudes),
            "frequency": np.array(frequencies),
        }

    def frequency_spectrum(self, n_steps: int = 10000) -> dict[str, np.ndarray]:
        """Compute oscillation frequency via FFT of E(t).

        Returns dict with 'freq' and 'power' arrays, plus 'peak_freq'.
        """
        self.reset()

        # Transient
        for _ in range(n_steps // 2):
            self.step()

        # Collect signal
        E_vals = []
        for _ in range(n_steps):
            self.step()
            E_vals.append(self._state[0])

        E_vals = np.array(E_vals)
        E_detrend = E_vals - np.mean(E_vals)

        dt = self.config.dt
        fft_vals = np.fft.rfft(E_detrend)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(E_detrend), d=dt)

        # Find peak (skip DC)
        if len(power) > 1:
            peak_idx = np.argmax(power[1:]) + 1
            peak_freq = freqs[peak_idx]
        else:
            peak_freq = 0.0

        return {
            "freq": freqs,
            "power": power,
            "peak_freq": float(peak_freq),
        }
