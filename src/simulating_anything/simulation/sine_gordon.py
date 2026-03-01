"""Sine-Gordon equation simulation -- nonlinear wave PDE with topological solitons.

The Sine-Gordon equation: u_tt = c^2 * u_xx - sin(u)

Target rediscoveries:
- Kink soliton: u(x,t) = 4*arctan(exp((x-vt)/sqrt(1-v^2/c^2)))
- Lorentz contraction: kink width ~ sqrt(1 - v^2/c^2)
- Energy conservation (integrable system)
- Topological charge: Q = (u(+inf) - u(-inf)) / (2*pi), integer for kinks
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SineGordonSimulation(SimulationEnvironment):
    """Sine-Gordon equation: u_tt = c^2 * u_xx - sin(u).

    Uses Stormer-Verlet (leapfrog) symplectic integrator for energy conservation
    on a periodic 1D domain.

    State vector: [u_0, ..., u_{N-1}, u_t_0, ..., u_t_{N-1}] (2*N array).
    u is the field, u_t is its time derivative.

    Parameters:
        c: wave speed (default 1.0)
        N: number of grid points (default 256)
        L: domain length (default 40.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.c = p.get("c", 1.0)
        self.N = int(p.get("N", 256))
        self.L = p.get("L", 40.0)

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Default init type (string params cannot go into parameters dict)
        self.init_type = "kink"

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize field u and its time derivative u_t."""
        if self.init_type == "kink":
            u, ut = self._kink_state(v=0.0, x0=self.L / 2)
        elif self.init_type == "breather":
            u, ut = self._breather_state(omega=0.5, x0=self.L / 2, t=0.0)
        elif self.init_type == "antikink":
            u, ut = self._antikink_state(v=0.0, x0=self.L / 2)
        elif self.init_type == "vacuum":
            u = np.zeros(self.N, dtype=np.float64)
            ut = np.zeros(self.N, dtype=np.float64)
        else:
            u = np.zeros(self.N, dtype=np.float64)
            ut = np.zeros(self.N, dtype=np.float64)

        self._state = np.concatenate([u, ut]).astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using Stormer-Verlet (leapfrog) integrator.

        The Stormer-Verlet scheme for u_tt = F(u):
            u_t_{n+1/2} = u_t_n + (dt/2) * F(u_n)
            u_{n+1}     = u_n   + dt * u_t_{n+1/2}
            u_t_{n+1}   = u_t_{n+1/2} + (dt/2) * F(u_{n+1})

        This is symplectic, preserving energy to high accuracy.
        """
        dt = self.config.dt
        N = self.N
        u = self._state[:N].copy()
        ut = self._state[N:].copy()

        # Half-step velocity
        accel = self._acceleration(u)
        ut_half = ut + 0.5 * dt * accel

        # Full-step position
        u_new = u + dt * ut_half

        # Half-step velocity with new position
        accel_new = self._acceleration(u_new)
        ut_new = ut_half + 0.5 * dt * accel_new

        self._state = np.concatenate([u_new, ut_new])
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u, u_t]."""
        return self._state

    def _acceleration(self, u: np.ndarray) -> np.ndarray:
        """Compute u_tt = c^2 * u_xx - sin(u) using finite differences.

        Uses wrapped second-order central differences to handle the 2*pi
        boundary jump for kink solutions on periodic domains. Since sin(u)
        is 2*pi-periodic, the physics lives on R/(2*pi*Z) and differences
        should be wrapped to [-pi, pi].
        """
        d_fwd = np.roll(u, -1) - u
        d_bwd = u - np.roll(u, 1)
        # Wrap to [-pi, pi] to handle topological boundary jumps
        d_fwd_w = (d_fwd + np.pi) % (2 * np.pi) - np.pi
        d_bwd_w = (d_bwd + np.pi) % (2 * np.pi) - np.pi
        u_xx = (d_fwd_w - d_bwd_w) / (self.dx ** 2)
        return self.c ** 2 * u_xx - np.sin(u)

    # --- Initial condition generators ---

    def _kink_state(
        self, v: float = 0.0, x0: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate kink soliton: u = 4*arctan(exp((x-x0)/gamma)).

        Args:
            v: kink velocity (|v| < c).
            x0: center position.

        Returns:
            (u, u_t) arrays.
        """
        if x0 is None:
            x0 = self.L / 2
        gamma_lor = np.sqrt(max(1 - v ** 2 / self.c ** 2, 1e-12))
        xi = (self.x - x0) / gamma_lor
        u = 4.0 * np.arctan(np.exp(xi))
        # Time derivative: du/dt = -v * du/dx
        u_x = (1.0 / gamma_lor) / np.cosh(xi)
        ut = -v * u_x
        return u.astype(np.float64), ut.astype(np.float64)

    def _antikink_state(
        self, v: float = 0.0, x0: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate antikink soliton: u = -4*arctan(exp((x-x0)/gamma))."""
        if x0 is None:
            x0 = self.L / 2
        gamma_lor = np.sqrt(max(1 - v ** 2 / self.c ** 2, 1e-12))
        xi = (self.x - x0) / gamma_lor
        u = -4.0 * np.arctan(np.exp(xi))
        u_x = -(1.0 / gamma_lor) / np.cosh(xi)
        ut = -v * u_x
        return u.astype(np.float64), ut.astype(np.float64)

    def _breather_state(
        self, omega: float = 0.5, x0: float | None = None, t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate breather solution (localized oscillating bound state).

        u(x,t) = 4*arctan( (sqrt(1-w^2)/w) * sin(w*t) / cosh(sqrt(1-w^2)*(x-x0)) )

        Valid for 0 < omega < 1 (with c=1).
        """
        if x0 is None:
            x0 = self.L / 2
        w = omega
        eps = np.sqrt(max(1 - w ** 2, 1e-12))
        denom = np.cosh(eps * (self.x - x0))
        numer_ratio = eps / w
        arg = numer_ratio * np.sin(w * t) / denom
        u = 4.0 * np.arctan(arg)

        # Time derivative via chain rule
        cos_wt = np.cos(w * t)
        d_arg_dt = numer_ratio * w * cos_wt / denom
        ut = 4.0 * d_arg_dt / (1 + arg ** 2)

        return u.astype(np.float64), ut.astype(np.float64)

    # --- Initial condition setters ---

    def kink_initial_condition(self, v: float = 0.0, x0: float | None = None) -> None:
        """Set the state to a kink soliton centered at x0 with velocity v."""
        u, ut = self._kink_state(v=v, x0=x0)
        self._state = np.concatenate([u, ut]).astype(np.float64)
        self._step_count = 0

    def breather_initial_condition(
        self, omega: float = 0.5, x0: float | None = None,
    ) -> None:
        """Set the state to a breather centered at x0 with frequency omega."""
        u, ut = self._breather_state(omega=omega, x0=x0)
        self._state = np.concatenate([u, ut]).astype(np.float64)
        self._step_count = 0

    # --- Physical observables ---

    def compute_energy(self) -> float:
        """Total energy: E = integral(u_t^2/2 + c^2*u_x^2/2 + (1-cos(u))) dx.

        Uses wrapped gradient differences to handle the 2*pi jump at the
        periodic boundary for kink solutions.
        """
        N = self.N
        u = self._state[:N]
        ut = self._state[N:]

        # Gradient energy with wrapping for kink boundary
        du = np.roll(u, -1) - np.roll(u, 1)
        du_wrapped = (du + np.pi) % (2 * np.pi) - np.pi
        u_x = du_wrapped / (2 * self.dx)

        kinetic = 0.5 * ut ** 2
        gradient = 0.5 * self.c ** 2 * u_x ** 2
        potential = 1.0 - np.cos(u)

        return float(np.sum(kinetic + gradient + potential) * self.dx)

    def compute_topological_charge(self) -> float:
        """Topological charge Q = (u(L) - u(0)) / (2*pi).

        For periodic BCs, we compute the winding number by summing the
        field increments modulo 2*pi.
        """
        N = self.N
        u = self._state[:N]
        # Sum the wrapped differences to get total winding
        du = np.diff(u)
        # Wrap to [-pi, pi] to handle the 2*pi jumps
        du_wrapped = (du + np.pi) % (2 * np.pi) - np.pi
        total_change = np.sum(du_wrapped)
        return float(total_change / (2 * np.pi))

    # --- Analysis methods ---

    def kink_velocity_sweep(
        self, velocities: np.ndarray | list[float],
    ) -> dict[str, np.ndarray]:
        """Measure kink profiles at different velocities.

        Returns dict with 'velocities', 'widths', 'energies'.
        """
        velocities = np.asarray(velocities)
        widths = []
        energies = []

        for v in velocities:
            u, ut = self._kink_state(v=v, x0=self.L / 2)
            self._state = np.concatenate([u, ut]).astype(np.float64)
            widths.append(self._measure_kink_width(u))
            energies.append(self.compute_energy())

        return {
            "velocities": velocities,
            "widths": np.array(widths),
            "energies": np.array(energies),
        }

    def measure_lorentz_contraction(
        self, velocities: np.ndarray | list[float],
    ) -> dict[str, np.ndarray]:
        """Verify kink width ~ sqrt(1 - v^2/c^2).

        Returns dict with 'velocities', 'measured_widths', 'theoretical_widths'.
        """
        velocities = np.asarray(velocities)
        measured = []
        theoretical = []

        # Width at rest (v=0)
        u_rest, _ = self._kink_state(v=0.0, x0=self.L / 2)
        w_rest = self._measure_kink_width(u_rest)

        for v in velocities:
            u, _ = self._kink_state(v=v, x0=self.L / 2)
            measured.append(self._measure_kink_width(u))
            gamma_factor = np.sqrt(max(1 - v ** 2 / self.c ** 2, 1e-12))
            theoretical.append(w_rest * gamma_factor)

        return {
            "velocities": velocities,
            "measured_widths": np.array(measured),
            "theoretical_widths": np.array(theoretical),
            "rest_width": w_rest,
        }

    def _measure_kink_width(self, u: np.ndarray) -> float:
        """Measure kink width as the inverse of the maximum gradient.

        Uses a wrapped gradient to avoid the 2*pi boundary jump on
        periodic domains. For a kink with Lorentz factor gamma,
        width = gamma / 2 (inverse of max du/dx = 2/gamma).
        """
        du = np.roll(u, -1) - np.roll(u, 1)
        # Wrap differences to [-pi, pi] to handle 2*pi boundary jumps
        du_wrapped = (du + np.pi) % (2 * np.pi) - np.pi
        u_x = np.abs(du_wrapped) / (2 * self.dx)
        max_grad = np.max(u_x)
        if max_grad < 1e-12:
            return float(self.L)
        return float(1.0 / max_grad)

    @staticmethod
    def analytical_kink_energy(c: float = 1.0, v: float = 0.0) -> float:
        """Analytical kink energy: E = 8*c / sqrt(1 - v^2/c^2).

        Rest energy of a kink (v=0) is 8*c.
        """
        gamma_factor = np.sqrt(max(1 - v ** 2 / c ** 2, 1e-12))
        return 8.0 * c / gamma_factor
