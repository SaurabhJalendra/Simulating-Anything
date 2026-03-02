"""Goodwin oscillator simulation.

The Goodwin oscillator models a 3-variable gene regulatory negative feedback loop:

    dx/dt = a / (K^n + z^n) - b * x
    dy/dt = alpha * x - beta * y
    dz/dt = gamma * y - delta * z

where x = mRNA, y = enzyme, z = product (inhibitor).
The Hill function a / (K^n + z^n) provides negative feedback: z represses x production.

Target rediscoveries:
- Oscillation requires Hill coefficient n > 8 (Griffith's theorem for 3 variables)
- ODE recovery via SINDy
- Equilibrium (x*, y*, z*) computation
- Period dependence on parameters
- Oscillation onset at critical n
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GoodwinSimulation(SimulationEnvironment):
    """Goodwin oscillator: 3-variable gene regulatory negative feedback loop.

    State vector: [x, y, z] where x = mRNA, y = enzyme, z = product (inhibitor).

    The Hill function a / (K^n + z^n) models cooperative repression of mRNA
    synthesis by the product z. Oscillation requires a sufficiently steep
    Hill function: n > 8 for the 3-variable system (Griffith's theorem).

    At the unique equilibrium, the system undergoes a Hopf bifurcation as n
    increases past the critical value n_c ~ 8.

    Parameters:
        a: maximal mRNA production rate (default 1.0)
        K: Hill function half-saturation constant (default 1.0)
        n: Hill coefficient (default 9.0; needs n > 8 for oscillation)
        b: mRNA degradation rate (default 0.1)
        alpha: translation rate (default 0.1)
        beta: enzyme degradation rate (default 0.1)
        gamma: product synthesis rate (default 0.1)
        delta: product degradation rate (default 0.1)
        x_0: initial mRNA concentration (default: near equilibrium + perturbation)
        y_0: initial enzyme concentration
        z_0: initial product concentration
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.K = p.get("K", 1.0)
        self.n = p.get("n", 9.0)
        self.b = p.get("b", 0.1)
        self.alpha = p.get("alpha", 0.1)
        self.beta = p.get("beta", 0.1)
        self.gamma = p.get("gamma", 0.1)
        self.delta = p.get("delta", 0.1)
        # Initial conditions: default to near equilibrium with perturbation
        eq = self.equilibrium()
        self.x_0 = p.get("x_0", eq[0] * 1.1)
        self.y_0 = p.get("y_0", eq[1] * 1.1)
        self.z_0 = p.get("z_0", eq[2] * 0.9)

    def equilibrium(self) -> tuple[float, float, float]:
        """Compute the unique equilibrium point analytically.

        At equilibrium:
            dx/dt = 0: a / (K^n + z*^n) = b * x*
            dy/dt = 0: alpha * x* = beta * y*
            dz/dt = 0: gamma * y* = delta * z*

        From the last two: y* = (alpha/beta) * x*, z* = (gamma/delta) * y*
        Substituting into the first: a / (K^n + z*^n) = b * x*

        This reduces to a single equation in x* that we solve via
        Newton's method (or directly for the symmetric default case).
        """
        # From the linear chains: y* = (alpha/beta)*x*, z* = (gamma*alpha/(delta*beta))*x*
        ratio_yz = self.gamma * self.alpha / (self.delta * self.beta)

        # Solve: a / (K^n + (ratio_yz * x)^n) = b * x
        # => a = b * x * (K^n + (ratio_yz * x)^n)
        # Use Newton's method
        Kn = self.K ** self.n

        def f(x: float) -> float:
            zn = (ratio_yz * x) ** self.n
            return self.b * x * (Kn + zn) - self.a

        def df(x: float) -> float:
            zn = (ratio_yz * x) ** self.n
            # d/dx [b*x*(K^n + (r*x)^n)] = b*(K^n + (r*x)^n) + b*x*n*r^n*x^(n-1)
            # = b*(K^n + (r*x)^n) + b*n*(r*x)^n
            return self.b * (Kn + zn) + self.b * self.n * zn

        # Initial guess: if all rates equal, x* ~ (a/b)^(1/(n+1))
        x = (self.a / self.b) ** (1.0 / (self.n + 1.0))
        for _ in range(50):
            fx = f(x)
            dfx = df(x)
            if abs(dfx) < 1e-30:
                break
            dx = -fx / dfx
            x = max(x + dx, 1e-15)  # Keep positive
            if abs(fx) < 1e-14:
                break

        x_star = x
        y_star = (self.alpha / self.beta) * x_star
        z_star = (self.gamma / self.delta) * y_star
        return (x_star, y_star, z_star)

    def hill_function(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the Hill repression function a / (K^n + z^n).

        Args:
            z: product concentration (scalar or array).

        Returns:
            Hill function value(s).
        """
        return self.a / (self.K ** self.n + z ** self.n)

    @property
    def is_oscillatory(self) -> bool:
        """True if n > 8 (necessary condition for 3-variable Goodwin oscillation).

        For the 3-variable Goodwin model, Griffith's theorem states that
        the Hill coefficient must exceed 8 for the equilibrium to be unstable
        and a limit cycle to exist (when all degradation rates are equal).
        """
        return self.n > 8.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state near equilibrium with perturbation."""
        self._state = np.array([self.x_0, self.y_0, self.z_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x, y, z]."""
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
        """Compute dx/dt, dy/dt, dz/dt for the Goodwin system."""
        x, yc, z = y
        Kn = self.K ** self.n
        zn = z ** self.n if z > 0 else 0.0
        dx = self.a / (Kn + zn) - self.b * x
        dy = self.alpha * x - self.beta * yc
        dz = self.gamma * yc - self.delta * z
        return np.array([dx, dy, dz])

    def compute_period(
        self,
        n_transient: int = 3000,
        n_measure: int = 5000,
    ) -> float:
        """Measure the oscillation period via upward zero crossings of x - x*.

        Args:
            n_transient: number of steps to skip for transient decay.
            n_measure: number of steps to use for period measurement.

        Returns:
            Mean period, or inf if no oscillation detected.
        """
        if not self.is_oscillatory:
            return float("inf")

        dt = self.config.dt
        x_star = self.equilibrium()[0]

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect upward crossings of x = x_star
        crossings: list[float] = []
        prev_x = self._state[0]
        for _ in range(n_measure):
            self.step()
            x = self._state[0]
            if prev_x < x_star and x >= x_star:
                if x != prev_x:
                    frac = (x_star - prev_x) / (x - prev_x)
                else:
                    frac = 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")

        return float(np.mean(np.diff(crossings)))

    def measure_amplitude(self, transient_time: float = 500.0) -> float:
        """Measure peak-to-peak amplitude of x after transient.

        Args:
            transient_time: time units to skip before measuring.

        Returns:
            Peak-to-peak amplitude of x component.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(int(transient_time / dt)):
            self.step()

        # Collect x values
        x_vals: list[float] = []
        for _ in range(int(300 / dt)):
            self.step()
            x_vals.append(self._state[0])

        return float(max(x_vals) - min(x_vals))

    def hill_sweep(
        self,
        n_range: tuple[float, float] = (2.0, 15.0),
        n_points: int = 30,
        n_steps: int = 15000,
    ) -> dict[str, np.ndarray]:
        """Sweep the Hill coefficient n and measure oscillation amplitude.

        Args:
            n_range: (min_n, max_n) range for the sweep.
            n_points: number of n values to test.
            n_steps: simulation steps per sweep point.

        Returns:
            Dict with 'n_values', 'amplitudes', and 'periods' arrays.
        """
        n_values = np.linspace(n_range[0], n_range[1], n_points)
        amplitudes = []
        periods = []

        for n_val in n_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "a": self.a,
                    "K": self.K,
                    "n": float(n_val),
                    "b": self.b,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "delta": self.delta,
                },
            )
            sim = GoodwinSimulation(config)
            sim.reset()

            amp = sim.measure_amplitude(transient_time=300.0)
            amplitudes.append(amp)

            # Re-create for period measurement (measure_amplitude consumed steps)
            sim2 = GoodwinSimulation(config)
            sim2.reset()
            period = sim2.compute_period(
                n_transient=int(300.0 / self.config.dt),
                n_measure=int(500.0 / self.config.dt),
            )
            periods.append(period)

        return {
            "n_values": n_values,
            "amplitudes": np.array(amplitudes),
            "periods": np.array(periods),
        }
