"""Sprott system simulation -- minimal chaotic flows.

Julien Clinton Sprott catalogued the simplest possible 3D autonomous ODE systems
exhibiting chaos. Each Sprott system (A through S) has a different set of ODEs,
all with minimal nonlinearity.

Target rediscoveries:
- SINDy recovery of Sprott ODEs
- Positive Lyapunov exponent confirming chaos
- Attractor characterization (bounded strange attractor)
- Comparison across Sprott system variants
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

# Sprott system definitions: each maps a system letter to its ODE function.
# All systems have state [x, y, z] and no free parameters in the canonical form.
SPROTT_SYSTEMS = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
}


class SprottSimulation(SimulationEnvironment):
    """Sprott minimal chaotic flows: the simplest ODEs exhibiting chaos.

    State vector: [x, y, z]

    Supported systems:
        A: dx=y, dy=-x+yz, dz=1-y^2
        B: dx=yz, dy=x-y, dz=1-xy
        C: dx=yz, dy=x-y, dz=1-x^2
        D: dx=-y, dy=x+z, dz=xz+3y^2
        E: dx=yz, dy=x^2-y, dz=1-4x
        F: dx=y+z, dy=-x+0.5*y, dz=x^2-z
        G: dx=0.4*x+z, dy=x*z-y, dz=-x+y
        H: dx=-y+z^2, dy=x+0.5*y, dz=x-z
        I: dx=-0.2*y, dy=x+z, dz=x+y^2-z
        J: dx=2*z, dy=-2*y+z, dz=-x+y+y^2
        K: dx=x*y-z, dy=x-y, dz=x+0.3*z
        L: dx=y+3.9*z, dy=0.9*x^2-y, dz=1-x
        M: dx=-z, dy=-x^2-z, dz=1.7+1.7*x+y
        N: dx=-2*y, dy=x+z^2, dz=1+y-2*z
        O: dx=y, dy=x-z, dz=x+x*z+2.7*y
        P: dx=2.7*y+z, dy=-x+y^2, dz=x+y
        Q: dx=-z, dy=x-y, dz=3.1*x+y^2+0.5*z
        R: dx=0.9-y, dy=0.4+z, dz=x*y-z
        S: dx=-x-4*y, dy=x+z^2, dz=1+x

    Parameters:
        system: Which Sprott system to use ('A' through 'S')
        x_0, y_0, z_0: initial conditions (default near origin)
    """

    # Class-level attribute for the boundary condition type
    boundary_type: str = "none"

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.system = p.get("system", "B")
        if isinstance(self.system, float):
            # Handle the case where system letter is stored as a float index
            # (SimulationConfig parameters are dict[str, float])
            # We use a string attribute instead
            self.system = "B"
        self.x_0 = p.get("x_0", 0.1)
        self.y_0 = p.get("y_0", 0.1)
        self.z_0 = p.get("z_0", 0.1)

    @classmethod
    def create(
        cls,
        system: str = "B",
        dt: float = 0.05,
        n_steps: int = 10000,
        x_0: float = 0.1,
        y_0: float = 0.1,
        z_0: float = 0.1,
    ) -> SprottSimulation:
        """Convenience factory that handles the system letter as a class attribute.

        Since SimulationConfig.parameters is dict[str, float], the system letter
        cannot be stored there. This factory sets it as an instance attribute.
        """
        from simulating_anything.types.simulation import Domain
        config = SimulationConfig(
            domain=Domain.SPROTT,
            dt=dt,
            n_steps=n_steps,
            parameters={"x_0": x_0, "y_0": y_0, "z_0": z_0},
        )
        sim = cls(config)
        sim.system = system
        return sim

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize Sprott state."""
        self._state = np.array(
            [self.x_0, self.y_0, self.z_0], dtype=np.float64
        )
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

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for the selected Sprott system."""
        x, y, z = state
        sys = self.system

        if sys == "A":
            dx = y
            dy = -x + y * z
            dz = 1.0 - y**2
        elif sys == "B":
            dx = y * z
            dy = x - y
            dz = 1.0 - x * y
        elif sys == "C":
            dx = y * z
            dy = x - y
            dz = 1.0 - x**2
        elif sys == "D":
            dx = -y
            dy = x + z
            dz = x * z + 3.0 * y**2
        elif sys == "E":
            dx = y * z
            dy = x**2 - y
            dz = 1.0 - 4.0 * x
        elif sys == "F":
            dx = y + z
            dy = -x + 0.5 * y
            dz = x**2 - z
        elif sys == "G":
            dx = 0.4 * x + z
            dy = x * z - y
            dz = -x + y
        elif sys == "H":
            dx = -y + z**2
            dy = x + 0.5 * y
            dz = x - z
        elif sys == "I":
            dx = -0.2 * y
            dy = x + z
            dz = x + y**2 - z
        elif sys == "J":
            dx = 2.0 * z
            dy = -2.0 * y + z
            dz = -x + y + y**2
        elif sys == "K":
            dx = x * y - z
            dy = x - y
            dz = x + 0.3 * z
        elif sys == "L":
            dx = y + 3.9 * z
            dy = 0.9 * x**2 - y
            dz = 1.0 - x
        elif sys == "M":
            dx = -z
            dy = -x**2 - z
            dz = 1.7 + 1.7 * x + y
        elif sys == "N":
            dx = -2.0 * y
            dy = x + z**2
            dz = 1.0 + y - 2.0 * z
        elif sys == "O":
            dx = y
            dy = x - z
            dz = x + x * z + 2.7 * y
        elif sys == "P":
            dx = 2.7 * y + z
            dy = -x + y**2
            dz = x + y
        elif sys == "Q":
            dx = -z
            dy = x - y
            dz = 3.1 * x + y**2 + 0.5 * z
        elif sys == "R":
            dx = 0.9 - y
            dy = 0.4 + z
            dz = x * y - z
        elif sys == "S":
            dx = -x - 4.0 * y
            dy = x + z**2
            dz = 1.0 + x
        else:
            raise ValueError(f"Unknown Sprott system: {sys}. Use A-S.")

        return np.array([dx, dy, dz])

    def estimate_lyapunov(
        self, n_steps: int = 50000, dt: float | None = None
    ) -> float:
        """Estimate the largest Lyapunov exponent via trajectory divergence.

        Uses the method of Wolf et al. (1985): track two nearby trajectories,
        renormalize when they diverge too far.
        """
        if dt is None:
            dt = self.config.dt

        eps = 1e-8
        state1 = self._state.copy()
        state2 = state1 + np.array([eps, 0, 0])

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance both states with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute distance
            dist = np.linalg.norm(state2 - state1)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize
                state2 = state1 + eps * (state2 - state1) / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def measure_period(
        self, n_transient: int = 5000, n_measure: int = 20000
    ) -> float:
        """Measure the oscillation period by detecting zero crossings of x.

        Returns the average period, or np.inf if no complete cycle is detected.
        """
        dt = self.config.dt

        # Skip transient
        for _ in range(n_transient):
            self.step()

        # Detect positive-going zero crossings of x
        crossings = []
        prev_x = self._state[0]
        for _ in range(n_measure):
            self.step()
            curr_x = self._state[0]
            if prev_x < 0 and curr_x >= 0:
                frac = -prev_x / (curr_x - prev_x) if curr_x != prev_x else 0.5
                t_cross = (self._step_count - 1 + frac) * dt
                crossings.append(t_cross)
            prev_x = curr_x

        if len(crossings) < 2:
            return np.inf

        periods = np.diff(crossings)
        return float(np.mean(periods))

    @property
    def sprott_b_fixed_points(self) -> list[np.ndarray]:
        """Compute fixed points of the Sprott-B system.

        For Sprott-B: dx=yz, dy=x-y, dz=1-xy

        Setting derivatives to zero:
            yz = 0  =>  y=0 or z=0
            x - y = 0  =>  x = y
            1 - xy = 0  =>  xy = 1

        Case y=0: x=0 from eq2, but xy=0 != 1 from eq3. No solution.
        Case z=0: x=y from eq2, x^2=1 from eq3 => x=+/-1.

        Fixed points: (1, 1, 0) and (-1, -1, 0)
        """
        if self.system != "B":
            return []
        return [
            np.array([1.0, 1.0, 0.0], dtype=np.float64),
            np.array([-1.0, -1.0, 0.0], dtype=np.float64),
        ]
