"""Bernoulli differential equation simulation.

The Bernoulli ODE is:

    dy/dt = a*y + b*y^n    (n != 0, 1)

This is a classic nonlinear ODE solvable by the substitution v = y^(1-n),
which linearizes to:

    dv/dt = (1-n)*(a*v + b)

For n=2: v = 1/y, and the analytical solution is:
    v(t) = (v_0 + b/a)*exp(-a*t) - b/a
    y(t) = 1 / v(t)

For general n != 0,1: the substitution v = y^(1-n) gives:
    dv/dt = (1-n)*(a*v + b)
which is a first-order linear ODE with solution:
    v(t) = (v_0 + b/a)*exp((1-n)*a*t) - b/a

Key properties:
- Finite-time blowup possible when the denominator v(t) passes through zero
- For a > 0, b < 0, n=2: stable nontrivial steady state y_ss = -a/b
- For a < 0, b > 0, n=2: y_ss = -a/b is UNSTABLE; y=0 is the attractor
- Steady states: y=0 always; y = (-a/b)^(1/(n-1)) if -a/b > 0

Target rediscoveries:
- Analytical solution verification: y(t) vs exact formula
- Substitution-based linearization: v = y^(1-n) transforms to linear ODE
- Steady-state y_ss = (-a/b)^(1/(n-1)) via PySR parameter sweep
- SINDy ODE recovery: dy/dt = a*y + b*y^n
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BernoulliODESimulation(SimulationEnvironment):
    """Bernoulli differential equation: dy/dt = a*y + b*y^n.

    State vector: [y] (1D scalar ODE).

    The substitution v = y^(1-n) linearizes to dv/dt = (1-n)*(a*v + b),
    enabling exact analytical solutions for verification.

    Parameters:
        a: linear coefficient (default 1.0)
        b: nonlinear coefficient (default -1.0)
        n: nonlinear exponent, must not be 0 or 1 (default 2.0)
        y_0: initial value (default 0.1)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 1.0)
        self.b = p.get("b", -1.0)
        self.n = p.get("n", 2.0)
        self.y_0 = p.get("y_0", 0.1)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state to [y_0]."""
        self._state = np.array([self.y_0], dtype=np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [y]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integrator."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute dy/dt = a*y + b*y^n."""
        val = y[0]
        dydt = self.a * val + self.b * val ** self.n
        return np.array([dydt])

    def analytical_solution(self, t: float | np.ndarray) -> np.ndarray:
        """Compute the exact analytical solution at time(s) t.

        Uses the substitution v = y^(1-n) to solve the linearized ODE:
            dv/dt = (1-n)*(a*v + b)

        The linear ODE has solution:
            v(t) = (v_0 + b/a)*exp((1-n)*a*t) - b/a

        Then y(t) = v(t)^(1/(1-n)).

        Note: Requires a != 0 and y_0 > 0 for the standard formula.
        Returns NaN where the solution is undefined (e.g., blowup).
        """
        t = np.asarray(t, dtype=np.float64)
        a, b, n, y_0 = self.a, self.b, self.n, self.y_0

        if abs(a) < 1e-15:
            # Special case a=0: dy/dt = b*y^n
            # v = y^(1-n), dv/dt = (1-n)*b, v(t) = v_0 + (1-n)*b*t
            v_0 = y_0 ** (1.0 - n)
            v_t = v_0 + (1.0 - n) * b * t
            exp_inv = 1.0 / (1.0 - n)
            result = np.where(v_t > 0, v_t ** exp_inv, np.nan)
            return result

        v_0 = y_0 ** (1.0 - n)
        exp_factor = np.exp((1.0 - n) * a * t)
        v_t = (v_0 + b / a) * exp_factor - b / a

        exp_inv = 1.0 / (1.0 - n)
        # For n=2 (1-n = -1), v_t^(-1) = 1/v_t so y = 1/v_t
        # For n=3 (1-n = -2), v_t^(-1/2) = 1/sqrt(v_t)
        # Need v_t > 0 for even roots, handle sign carefully
        result = np.where(v_t > 0, v_t ** exp_inv, np.nan)
        return result

    @property
    def steady_state(self) -> float | None:
        """Non-trivial steady state y_ss where a*y + b*y^n = 0.

        Setting dy/dt = 0 and dividing by y (for y != 0):
            a + b*y^(n-1) = 0  =>  y^(n-1) = -a/b

        Returns the positive real steady state if it exists, else None.
        """
        ratio = -self.a / self.b
        if ratio <= 0:
            return None
        exp = 1.0 / (self.n - 1.0)
        # For non-integer exponent, ratio must be positive (already checked)
        return float(ratio ** exp)

    @property
    def blowup_time(self) -> float | None:
        """Estimated finite-time blowup for n=2 case.

        For n=2, blowup occurs when v(t) = 0:
            (v_0 + b/a)*exp(-a*t) = b/a
            t_blow = -ln(b / (a*(v_0 + b/a))) / a

        Returns None if no blowup (e.g., a < 0 with default params).
        """
        if abs(self.n - 2.0) > 1e-10:
            return None  # Only implemented for n=2
        a, b, y_0 = self.a, self.b, self.y_0
        if abs(a) < 1e-15:
            return None

        v_0 = 1.0 / y_0  # v = y^(1-n) = y^(-1) for n=2
        ratio_arg = b / (a * (v_0 + b / a))
        if ratio_arg <= 0:
            return None
        t_blow = -np.log(ratio_arg) / a
        if t_blow <= 0:
            return None
        return float(t_blow)

    def compute_linearized_trajectory(
        self, n_steps: int | None = None, dt: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute trajectory in the linearized v = y^(1-n) variable.

        Returns dict with 'time', 'y', 'v', and 'v_analytical' arrays.
        This demonstrates the substitution that transforms Bernoulli to linear.
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        if dt is None:
            dt = self.config.dt

        self.reset()
        times = np.arange(n_steps + 1) * dt
        y_vals = np.zeros(n_steps + 1)
        y_vals[0] = self.y_0

        for i in range(n_steps):
            self.step()
            y_vals[i + 1] = self._state[0]

        # Compute v = y^(1-n)
        valid = y_vals > 0
        v_vals = np.where(valid, y_vals ** (1.0 - self.n), np.nan)

        # Analytical v(t)
        a, b, n = self.a, self.b, self.n
        v_0 = self.y_0 ** (1.0 - n)
        if abs(a) < 1e-15:
            v_analytical = v_0 + (1.0 - n) * b * times
        else:
            exp_factor = np.exp((1.0 - n) * a * times)
            v_analytical = (v_0 + b / a) * exp_factor - b / a

        return {
            "time": times,
            "y": y_vals,
            "v": v_vals,
            "v_analytical": v_analytical,
        }

    def parameter_sweep_steady_state(
        self,
        a_values: np.ndarray | None = None,
        b_values: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep a and b, measure long-time y to compare with y_ss = (-a/b)^(1/(n-1)).

        Returns dict with 'a', 'b', 'y_final', and 'y_ss_theory' arrays.
        """
        if a_values is None:
            a_values = np.linspace(0.1, 2.0, 15)
        if b_values is None:
            b_values = np.linspace(-2.0, -0.1, 15)

        a_arr = []
        b_arr = []
        y_final_arr = []
        y_ss_theory_arr = []

        for a_val in a_values:
            for b_val in b_values:
                config = SimulationConfig(
                    domain=self.config.domain,
                    dt=self.config.dt,
                    n_steps=2000,
                    parameters={
                        "a": float(a_val),
                        "b": float(b_val),
                        "n": self.n,
                        "y_0": 0.5,
                    },
                )
                sim = BernoulliODESimulation(config)
                sim.reset()

                for _ in range(2000):
                    sim.step()

                y_final = float(sim.observe()[0])

                # Theory: y_ss = (-a/b)^(1/(n-1))
                ratio = -a_val / b_val
                if ratio > 0:
                    y_ss = ratio ** (1.0 / (self.n - 1.0))
                else:
                    y_ss = np.nan

                a_arr.append(a_val)
                b_arr.append(b_val)
                y_final_arr.append(y_final)
                y_ss_theory_arr.append(y_ss)

        return {
            "a": np.array(a_arr),
            "b": np.array(b_arr),
            "y_final": np.array(y_final_arr),
            "y_ss_theory": np.array(y_ss_theory_arr),
        }
