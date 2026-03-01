"""Bouncing ball on a vibrating table (impact map).

A ball bounces vertically on a sinusoidally vibrating table.
Between bounces the ball is in free fall; at impact the velocity is updated
using the coefficient of restitution and the table velocity.

Using the high-bounce approximation, the impact map in dimensionless
phase-velocity coordinates is:

    phi_{n+1} = phi_n + 2*v_n / (g*T)   (mod 1)
    v_{n+1}   = -e*v_n_impact + (1+e)*A*omega*cos(2*pi*phi_{n+1})

where T = 2*pi/omega is the table period.

Target rediscoveries:
- Period-1 orbits at low amplitude
- Period-doubling cascade as amplitude increases
- Chaotic bouncing at high amplitude
- Bifurcation diagram (amplitude vs bounce velocity)
- Lyapunov exponent as function of amplitude
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BouncingBallSimulation(SimulationEnvironment):
    """Bouncing ball impact map on a vibrating table.

    State: [phase, velocity] where phase is the table phase at impact (mod 1)
    and velocity is the ball velocity just after the bounce.

    Parameters:
        e: coefficient of restitution (0 < e < 1), default 0.5
        A: table vibration amplitude, default 0.1
        omega: table angular frequency, default 2*pi
        g: gravitational acceleration, default 9.81
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.e = p.get("e", 0.5)
        self.A = p.get("A", 0.1)
        self.omega = p.get("omega", 2 * np.pi)
        self.g = p.get("g", 9.81)
        self.T = 2 * np.pi / self.omega  # table period

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize state with phase=0 and velocity from first impact.

        The initial velocity is computed as the velocity just after
        the ball is released from rest at a small height and hits the
        table at phase=0.
        """
        phi_0 = 0.0
        # Initial velocity: ball dropped from small height, first bounce
        # Table velocity at phase 0: A*omega*cos(0) = A*omega
        # Assume ball arrives with velocity v_impact = sqrt(2*g*h) for small h
        # Simplify: use the table velocity as the initial kick
        v_table = self.A * self.omega
        v_0 = (1 + self.e) * v_table
        self._state = np.array([phi_0, v_0], dtype=np.float64)
        self._step_count = 0
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Apply one iteration of the impact map.

        phi_{n+1} = (phi_n + 2*v_n / (g*T)) mod 1
        v_{n+1}   = -e*v_n + (1+e)*A*omega*cos(2*pi*phi_{n+1})

        In the high-bounce approximation, v_n at impact equals -v_n
        (symmetric free-fall), so v_impact = -v_n and the restitution
        gives v_{n+1} = e*v_n + (1+e)*A*omega*cos(2*pi*phi_{n+1}).
        """
        phi = self._state[0]
        v = self._state[1]

        # Advance phase by flight time (high-bounce approximation)
        phi_new = (phi + 2.0 * v / (self.g * self.T)) % 1.0

        # Velocity after bounce: ball arrives with speed v (downward),
        # restitution reduces it, table adds momentum
        v_new = self.e * v + (1 + self.e) * self.A * self.omega * np.cos(
            2 * np.pi * phi_new
        )

        self._state = np.array([phi_new, v_new], dtype=np.float64)
        self._step_count += 1
        return self._state.copy()

    def observe(self) -> np.ndarray:
        """Return current state [phase, velocity]."""
        return self._state.copy()

    def is_sticking(self, velocity_threshold: float = 0.01) -> bool:
        """Check if the ball is chattering (velocity too low to leave table).

        When the bounce velocity drops below a threshold, the ball
        effectively sticks to the table surface.
        """
        return abs(self._state[1]) < velocity_threshold

    def bifurcation_diagram(
        self,
        A_values: np.ndarray,
        n_skip: int = 200,
        n_record: int = 100,
    ) -> dict[str, np.ndarray]:
        """Generate bifurcation diagram data (amplitude vs steady-state velocity).

        For each amplitude A, run the map for n_skip transient iterates,
        then record n_record steady-state velocities.

        Args:
            A_values: array of amplitude values to sweep.
            n_skip: transient iterates to discard.
            n_record: steady-state iterates to record.

        Returns:
            Dict with 'A' and 'velocity' arrays for plotting.
        """
        all_A = []
        all_v = []

        for A_val in A_values:
            # Set up with this amplitude
            phi = 0.0
            v_table = A_val * self.omega
            v = (1 + self.e) * v_table

            # Transient
            for _ in range(n_skip):
                phi = (phi + 2.0 * v / (self.g * self.T)) % 1.0
                v = self.e * v + (1 + self.e) * A_val * self.omega * np.cos(
                    2 * np.pi * phi
                )
                # Prevent runaway if velocity goes negative (sticking)
                if v < 0:
                    v = abs(v) * 0.1

            # Record steady state
            for _ in range(n_record):
                phi = (phi + 2.0 * v / (self.g * self.T)) % 1.0
                v = self.e * v + (1 + self.e) * A_val * self.omega * np.cos(
                    2 * np.pi * phi
                )
                if v < 0:
                    v = abs(v) * 0.1
                all_A.append(A_val)
                all_v.append(v)

        return {
            "A": np.array(all_A),
            "velocity": np.array(all_v),
        }

    def period_detection(
        self,
        n_steps: int = 500,
        max_period: int = 64,
        n_transient: int = 300,
        tol: float = 1e-4,
    ) -> int:
        """Detect the period of the orbit.

        Args:
            n_steps: iterates to use for detection after transient.
            max_period: maximum period to check.
            n_transient: transient iterates to discard.
            tol: tolerance for matching periodic points.

        Returns:
            Period (1, 2, 4, ...) or -1 if chaotic / period > max_period.
        """
        phi = self._state[0]
        v = self._state[1]

        # Transient
        for _ in range(n_transient):
            phi = (phi + 2.0 * v / (self.g * self.T)) % 1.0
            v = self.e * v + (1 + self.e) * self.A * self.omega * np.cos(
                2 * np.pi * phi
            )
            if v < 0:
                v = abs(v) * 0.1

        # Record orbit (velocities)
        orbit = []
        for _ in range(n_steps):
            phi = (phi + 2.0 * v / (self.g * self.T)) % 1.0
            v = self.e * v + (1 + self.e) * self.A * self.omega * np.cos(
                2 * np.pi * phi
            )
            if v < 0:
                v = abs(v) * 0.1
            orbit.append(v)

        # Check for period-p by comparing v_n with v_{n+p}
        for p in range(1, min(max_period + 1, len(orbit) // 2)):
            is_periodic = True
            # Check a window of max_period points
            check_len = min(max_period, len(orbit) - p)
            for i in range(check_len):
                if abs(orbit[i] - orbit[i + p]) > tol:
                    is_periodic = False
                    break
            if is_periodic:
                return p

        return -1  # Chaotic or period > max_period

    def compute_lyapunov(
        self,
        n_steps: int = 10000,
        n_transient: int = 500,
    ) -> float:
        """Compute the maximum Lyapunov exponent of the impact map.

        Uses the Jacobian of the map to accumulate the log of the
        largest singular value.

        The Jacobian of the 2D map (phi, v) -> (phi', v') is:
            dphi'/dphi = 1
            dphi'/dv   = 2 / (g*T)
            dv'/dphi   = -(1+e)*A*omega*2*pi*sin(2*pi*phi') * dphi'/dphi
            dv'/dv     = e + -(1+e)*A*omega*2*pi*sin(2*pi*phi') * dphi'/dv

        Returns:
            Maximum Lyapunov exponent.
        """
        phi = self._state[0]
        v = self._state[1]

        # Transient
        for _ in range(n_transient):
            phi = (phi + 2.0 * v / (self.g * self.T)) % 1.0
            v = self.e * v + (1 + self.e) * self.A * self.omega * np.cos(
                2 * np.pi * phi
            )
            if v < 0:
                v = abs(v) * 0.1

        log_sum = 0.0
        count = 0

        for _ in range(n_steps):
            phi_new = (phi + 2.0 * v / (self.g * self.T)) % 1.0

            # Jacobian entries
            dphi_dv = 2.0 / (self.g * self.T)
            sin_term = (
                -(1 + self.e) * self.A * self.omega
                * 2 * np.pi * np.sin(2 * np.pi * phi_new)
            )
            J = np.array([
                [1.0, dphi_dv],
                [sin_term, self.e + sin_term * dphi_dv],
            ])

            # Accumulate log of spectral radius
            eigenvalues = np.abs(np.linalg.eigvals(J))
            max_eigenval = np.max(eigenvalues)
            if max_eigenval > 0:
                log_sum += np.log(max_eigenval)
            count += 1

            v_new = self.e * v + (1 + self.e) * self.A * self.omega * np.cos(
                2 * np.pi * phi_new
            )
            phi = phi_new
            v = v_new
            if v < 0:
                v = abs(v) * 0.1

        if count == 0:
            return 0.0
        return log_sum / count
